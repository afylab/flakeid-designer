"""Finding a scanned chip again (moved on the stage, or taken off and put back) from the first scan's 10x images.

The viewer of the first scan keeps its 10x mosaic at full resolution (<piece>_tiles_viewer/tiles/image/<top level>), and
the flake list gives the mosaic's pixel <-> stage mapping (each flake's pixel and stage position).  A new 10x frame is
compared with the mosaic as it should look there under the current estimate of the chip's move (warped into the frame's
pixels, structure only: a band-pass of the brightness), and the best shift (and, without an estimate, the rotation) is
searched by normalized correlation; the frames of a few spots then give a rigid transform from the old stage
coordinates to the new ones, a few um accurate.  The same comparison on a 100x frame (shrunk to the 10x scale) checks
the 10x -> 100x offset.

Conventions as in the scan: a flake's stage position is the stage position that centres it, and the stage that centres
pixel p of a frame taken at stage s is s + M (p - c), c the frame centre (M: the 10x pixel -> stage matrix).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

TILE = 256                                  # the viewer's tile size before 2026-10-27 (each viewer's own: zoomview.tile_size)


def grey(rgb):
    """Brightness of a linear camera frame, relative to its median and gamma-encoded like the viewer's JPEGs."""
    a = np.maximum(np.asarray(rgb, np.float32), 0)
    g = a.mean(axis=2) if a.ndim == 3 else a
    return (g / max(float(np.median(g)), 1e-6)) ** (1 / 2.2)


def band(g, sig):
    import cv2
    return cv2.GaussianBlur(g, (0, 0), sig[0]) - cv2.GaussianBlur(g, (0, 0), sig[1])


class Mosaic:
    """The first scan's 10x mosaic (the viewer's full-resolution tiles) and its pixel <-> stage mapping, fitted to the
    flake list (target_x_px / target_y_px against stage_x_um / stage_y_um).  stage = B px + b."""

    def __init__(self, viewer_dir, rows):
        d = Path(viewer_dir) / "tiles" / "image"
        lv = [int(q.name) for q in d.iterdir() if q.is_dir() and q.name.isdigit()] if d.is_dir() else []
        if not lv:
            raise ValueError(f"no viewer tiles in {d}")
        self.dir = d / str(max(lv))
        from ..zoomview import tile_size
        self.T = tile_size(self.dir, TILE)
        px, st = [], []
        for r in rows:
            try:
                a = (float(r["target_x_px"]), float(r["target_y_px"]), float(r["stage_x_um"]), float(r["stage_y_um"]))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in a):
                px.append(a[:2]); st.append(a[2:])
        if len(px) < 3:
            raise ValueError("too few flakes with both a pixel and a stage position for the mosaic's mapping")
        A = np.c_[np.array(px), np.ones(len(px))]
        S = np.array(st)
        coef = np.linalg.lstsq(A, S, rcond=None)[0]
        self.residual_um = float(np.abs(A @ coef - S).max())
        self.B, self.b = coef[:2].T.copy(), coef[2].copy()
        self.Binv = np.linalg.inv(self.B)
        self.um_px = math.sqrt(abs(np.linalg.det(self.B)))
        self.cache = {}

    def to_px(self, stage):
        return (np.asarray(stage, float) - self.b) @ self.Binv.T

    def _tile(self, tx, ty):
        key = (tx, ty)
        if key not in self.cache:
            import cv2
            if len(self.cache) > 800:
                self.cache.clear()
            im = cv2.imread(str(self.dir / f"{tx}_{ty}.jpg"), cv2.IMREAD_GRAYSCALE)
            self.cache[key] = None if im is None else im.astype(np.float32) / 255.0
        return self.cache[key]

    def region(self, x0, y0, w, h):
        """Mosaic pixels [y0, y0 + h) x [x0, x0 + w), grey 0..1; NaN where there is no mosaic."""
        out = np.full((h, w), np.nan, np.float32)
        T = self.T
        for ty in range(max(y0 // T, 0), (y0 + h - 1) // T + 1):
            for tx in range(max(x0 // T, 0), (x0 + w - 1) // T + 1):
                im = self._tile(tx, ty)
                if im is None:
                    continue
                ys, xs = ty * T - y0, tx * T - x0
                a, b = max(ys, 0), max(xs, 0)
                c, d = min(ys + im.shape[0], h), min(xs + im.shape[1], w)
                if c > a and d > b:
                    out[a:c, b:d] = im[a - ys:c - ys, b - xs:d - xs]
        return out


def register(F, mos, g, Mf, cf, R, t, radius_um, angles=(0.0,), sig=(1.0, 6.0), crop=0.04):
    """Where frame F (grey(), its pixel u at stage g + Mf (u - cf)) really is, against the mosaic as the current
    estimate (new stage = R old + t) predicts it.  Searches shifts within radius_um (and the rotations `angles`, deg,
    about the frame).  None if the mosaic does not cover the place; else dict(peak, second (the best correlation more
    than ~15 um away), ang, d (frame pixels: the content at u was predicted at u + d), shift_um (stage), src / dst
    (old -> new stage pairs at 9 points of the frame), frame / ref (band-passed, aligned; for a picture))."""
    import cv2
    F = np.asarray(F, np.float32)
    Mf, g, R, t, cf = (np.asarray(v, float) for v in (Mf, g, R, t, cf))
    h, w = F.shape
    pxum = math.sqrt(abs(np.linalg.det(Mf)))
    m = int(math.ceil(radius_um / pxum))
    cr = int(crop * min(h, w))
    T = band(F, sig)[cr:h - cr, cr:w - cr]
    if float(T.std()) < 1e-6:
        return None
    Qg = np.linalg.solve(R, g - t)                     # the old position the frame's stage g is thought to show
    best = None
    for ang in angles:
        Ra = rot(ang) @ R
        ta = g - Ra @ Qg
        Ri = np.linalg.inv(Ra)
        # canvas pixel v -> frame pixel v - m -> stage g + Mf (v - m - cf) -> old Ri (s - ta) -> mosaic Binv (old - b)
        Aff = mos.Binv @ Ri @ Mf
        o = mos.Binv @ (Ri @ (g - Mf @ (np.array([m, m], float) + cf) - ta) - mos.b)
        Wc, Hc = w + 2 * m, h + 2 * m
        cor = np.array([[0, 0], [Wc, 0], [0, Hc], [Wc, Hc]], float) @ Aff.T + o
        x0, y0 = (np.floor(cor.min(0)) - 3).astype(int)
        x1, y1 = (np.ceil(cor.max(0)) + 3).astype(int)
        reg = mos.region(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        ok = np.isfinite(reg)
        if ok.mean() < 0.3:
            continue
        reg = np.where(ok, reg, float(np.median(reg[ok]))).astype(np.float32)
        k = math.sqrt(abs(np.linalg.det(Aff)))           # mosaic pixels per canvas pixel
        if k > 1.3:
            reg = cv2.GaussianBlur(reg, (0, 0), 0.5 * k)
        A2 = np.c_[Aff, o - np.array([x0, y0], float)]
        C = cv2.warpAffine(reg, A2, (Wc, Hc), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        Cb = band(C, sig)
        r = cv2.matchTemplate(Cb, T, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if best is None or mx > best["peak"]:
            r2 = r.copy()
            cv2.circle(r2, loc, max(3, int(round(15.0 / pxum))), -1.0, -1)
            lx, ly = loc
            sub = [0.0, 0.0]                             # sub-pixel peak (parabola through the neighbours)
            for ax, (a_, c_) in enumerate(((r[ly, lx - 1], r[ly, lx + 1]) if 0 < lx < r.shape[1] - 1 else (None, None),
                                           (r[ly - 1, lx], r[ly + 1, lx]) if 0 < ly < r.shape[0] - 1 else (None, None))):
                if a_ is not None:
                    den = a_ - 2 * mx + c_
                    sub[ax] = float(np.clip(0.5 * (a_ - c_) / den, -0.5, 0.5)) if den < 0 else 0.0
            d = np.array([lx + sub[0], ly + sub[1]]) - cr - m
            best = dict(peak=float(mx), second=float(r2.max()), ang=float(ang), R=Ra, t=ta, d=d,
                        frame=T, ref=Cb[ly:ly + T.shape[0], lx:lx + T.shape[1]])
    if best is None:
        return None
    Ri = np.linalg.inv(best["R"])
    us = [np.array([x, y]) for x in (0.2 * w, 0.5 * w, 0.8 * w) for y in (0.2 * h, 0.5 * h, 0.8 * h)]
    best["dst"] = [g + Mf @ (u - cf) for u in us]
    best["src"] = [Ri @ (g + Mf @ (u + best["d"] - cf) - best["t"]) for u in us]
    best["shift_um"] = -(Mf @ best["d"])
    return best



def match_crop(frame_grey, crop_grey, sig=(1.0, 8.0), crop=0.02, excl_um=15.0, um_px=1.09, edges=True):
    """Where the target is relative to a 100x frame: the frame (grey(), shrunk to the 10x scale) against the target's
    10x picture (<piece>_targets_crops/<label>.jpg, centred on the target; grey 0..1).  Returns None if the frame
    shows no structure or does not fit inside the picture; else dict(peak, second (the best correlation more than
    excl_um away), d (10x pixels: the target's position minus the frame centre; the stage that centres the target is
    the current one + M d)).  Same band-pass and normalized correlation as register(); edges: its magnitude, smoothed
    (a flake can be brighter than the substrate at 10x and darker at 100x, or the reverse: hBN colours differ between
    the two objectives, so the band-pass itself can be inverted between them)."""
    import cv2
    F = np.asarray(frame_grey, np.float32)
    C = np.asarray(crop_grey, np.float32)
    h, w = F.shape
    cr = int(crop * min(h, w))
    T = band(F, sig)
    Cb = band(C, sig)
    if edges:
        T, Cb = cv2.GaussianBlur(np.abs(T), (0, 0), 1.0), cv2.GaussianBlur(np.abs(Cb), (0, 0), 1.0)
    T = T[cr:h - cr, cr:w - cr]
    if float(T.std()) < 1e-6 or T.shape[0] >= C.shape[0] or T.shape[1] >= C.shape[1]:
        return None
    r = cv2.matchTemplate(Cb, T, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    r2 = r.copy()
    cv2.circle(r2, loc, max(3, int(round(excl_um / um_px))), -1.0, -1)
    lx, ly = loc
    sub = [0.0, 0.0]
    for ax, (a_, c_) in enumerate(((r[ly, lx - 1], r[ly, lx + 1]) if 0 < lx < r.shape[1] - 1 else (None, None),
                                   (r[ly - 1, lx], r[ly + 1, lx]) if 0 < ly < r.shape[0] - 1 else (None, None))):
        if a_ is not None:
            den = a_ - 2 * mx + c_
            sub[ax] = float(np.clip(0.5 * (a_ - c_) / den, -0.5, 0.5)) if den < 0 else 0.0
    # the frame's centre lands on crop pixel loc + (template centre); the target is the crop's centre
    fc = np.array([lx + sub[0] + T.shape[1] / 2.0, ly + sub[1] + T.shape[0] / 2.0])   # (T is centred on the frame)
    d = np.array([C.shape[1] / 2.0, C.shape[0] / 2.0]) - fc
    return dict(peak=float(mx), second=float(r2.max()), d=d, frame=T,
                ref=Cb[ly:ly + T.shape[0], lx:lx + T.shape[1]])

def picture(res, path):
    """The frame and the mosaic's prediction side by side (band-passed), for the log."""
    import cv2
    a, b = res["frame"], res["ref"]
    if b.shape != a.shape:
        return
    sd = max(float(a.std()), 1e-6), max(float(b.std()), 1e-6)
    im = np.hstack([a / sd[0], np.full((a.shape[0], 6), 3.0, np.float32), b / sd[1]])
    cv2.imwrite(str(path), np.clip(128 + 40 * im, 0, 255).astype(np.uint8))


def project(pts_stage, stage, M, c):
    """Pixel positions of stage points in a frame taken at `stage` (c: frame centre in pixels)."""
    Minv = np.linalg.inv(np.asarray(M, float))
    return (Minv @ (np.asarray(pts_stage, float) - np.asarray(stage, float)).T).T + np.asarray(c, float)


def rot(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def rigid_fit(src, dst):
    """Rotation R and translation t with dst ~ R src + t (least squares, no scaling), and the rms residual."""
    A, B = np.asarray(src, float), np.asarray(dst, float)
    if len(A) == 1:
        return np.eye(2), B[0] - A[0], 0.0
    ma, mb = A.mean(0), B.mean(0)
    Hm = (A - ma).T @ (B - mb)
    U, _, Vt = np.linalg.svd(Hm)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:                    # no mirroring
        Vt[1] *= -1
        Rm = Vt.T @ U.T
    t = mb - Rm @ ma
    res = B - (A @ Rm.T + t)
    return Rm, t, float(np.sqrt((res ** 2).sum(1).mean()))


def choose_spots(pts, fov, n_outer=4):
    """Reference spots (old stage coordinates) for the registration: where a 10x field holds the most flakes, one near
    the chip's centre and one in each quadrant far from it.  pts: flake positions (N, 2); fov: (w, h) of a 10x field."""
    P = np.asarray(pts, float)
    if len(P) == 0:
        return []
    c = np.median(P, axis=0)
    hw = np.asarray(fov, float) * 0.4
    cnt = np.zeros(len(P))
    for i in range(0, len(P), 500):
        d = np.abs(P[i:i + 500, None, :] - P[None, :, :])
        cnt[i:i + 500] = ((d[..., 0] < hw[0]) & (d[..., 1] < hw[1])).sum(1)
    r = np.hypot(*(P - c).T)
    rmax = max(float(r.max()), 1.0)
    spots = []
    inner = r < 0.3 * rmax
    if inner.any():
        spots.append(P[np.flatnonzero(inner)[np.argmax(cnt[inner])]])
    ang = np.arctan2(*(P - c).T[::-1])
    for q in range(n_outer):
        sel = (r > 0.45 * rmax) & (((ang + math.pi) // (2 * math.pi / n_outer)) % n_outer == q)
        if sel.any():
            w = cnt[sel] * (0.5 + r[sel] / rmax)               # many flakes, and far out (a better lever arm)
            spots.append(P[np.flatnonzero(sel)[np.argmax(w)]])
    return spots
