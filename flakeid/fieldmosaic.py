"""Mosaics of 100x / IR fields over a large terrace: the field plan (the scan) and the stitching (the export).

A target's chosen terrace (the uniform region the flake was picked for) is often larger than one 100x field (~157 x
119 um; graphite_20261019_1 flake 1: ~373 x 240 um).  The analysis writes the terrace's outline into the target list
(flakeid.live: terrace_poly_um, stage um); after the flake's main picture the scan images the other fields that the
terrace reaches into (plan_fields) and the export stitches them with the main one (stitch).

Plan: a grid in the camera's own directions anchored on the main picture (field (0, 0)), steps of (1 - overlap) of the
field.  Each grid point owns the cell of one step around it (the cells tile the plane and each lies inside its field),
so keeping every field whose cell holds terrace not already in the main picture covers the whole terrace.

Stitch: each field placed by its stage position through the camera's pixel -> stage matrix (mirroring and rotation
included), then corrected by phase correlation of the overlap strips (flat-fielded, high-passed; pairs with a weak
peak or whose aligned strips do not agree are not used), a least-squares solution of all the pair shifts (stage
positions as a weak prior: a field without a usable overlap stays where the stage put it), a per-field brightness
from the overlap medians (the main picture keeps its own), and feathered blending.
"""
from __future__ import annotations

import math

import numpy as np

POLY_MAX_VERTICES = 40          # terrace outline in the target list: at most this many vertices
CELL_MIN_FRAC = 0.01            # a field is planned when its cell holds at least this fraction of a cell of terrace
                                # (not already in the main picture): slivers from the 10x outline's ~1 um accuracy are not


# ---------------------------------------------------------------------------------------------- outlines
def parse_poly(s):
    """"x1 y1;x2 y2;..." -> (N, 2) float array, or None (empty, unreadable or fewer than 3 vertices)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        P = np.array([[float(v) for v in q.split()] for q in s.split(";") if q.strip()], float)
    except ValueError:
        return None
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 2 or not np.isfinite(P).all():
        return None
    return P


def format_poly(P, nd=1):
    return ";".join(f"{x:.{nd}f} {y:.{nd}f}" for x, y in np.asarray(P, float))


def simplify_poly(P, max_vertices=POLY_MAX_VERTICES):
    """Douglas-Peucker with its tolerance raised until at most max_vertices remain."""
    import cv2
    P = np.asarray(P, np.float64)
    if len(P) <= max_vertices:
        return P
    c = P.mean(0)
    q = (P - c).astype(np.float32).reshape(-1, 1, 2)
    per = float(cv2.arcLength(q, True))
    eps = max(per * 1e-3, 1e-6)
    out = q
    for _ in range(60):
        out = cv2.approxPolyDP(q, eps, True)
        if len(out) <= max_vertices:
            break
        eps *= 1.4
    return out.reshape(-1, 2).astype(np.float64) + c


def poly_area(P):
    P = np.asarray(P, float)
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------------------------- plan
def plan_fields(poly_rel, M, wh, overlap=0.15, max_fields=30, min_frac=CELL_MIN_FRAC):
    """The extra fields over a terrace.  poly_rel: its outline, stage um relative to the main picture's centre; M: the
    camera's stage um per pixel (2 x 2); wh: the frame (W, H) px.  Returns (fields, n_all): fields = [(i, j, dx_um,
    dy_um)] (grid index in the camera's directions, stage offset from the main picture), the main field left out, in a
    serpentine order (rows by stage y, alternating in x), at most max_fields - 1 of them (the nearest to the main
    picture when capped); n_all = the pictures the whole terrace needs, the main one included (1: it fits)."""
    import cv2
    M = np.asarray(M, float)
    W, H = float(wh[0]), float(wh[1])
    sx, sy = W * (1.0 - overlap), H * (1.0 - overlap)
    q = (np.linalg.inv(M) @ np.asarray(poly_rel, float).T).T          # camera px from the main field's centre
    lo, hi = q.min(0), q.max(0)
    # the terrace rasterised in the camera frame (at most ~1200 px across: cells are hundreds of px)
    f = min(1.0, 1200.0 / max(float((hi - lo).max()), 1.0))
    o = lo - 2.0 / f
    shape = (int(math.ceil((hi[1] - o[1]) * f)) + 3, int(math.ceil((hi[0] - o[0]) * f)) + 3)
    R = np.zeros(shape, np.uint8)
    cv2.fillPoly(R, [np.round((q - o) * f * 8).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=3)

    def box(x0, x1, y0, y1):                                        # raster slice of a camera-px rectangle
        a0, a1 = int(round((x0 - o[0]) * f)), int(round((x1 - o[0]) * f))
        b0, b1 = int(round((y0 - o[1]) * f)), int(round((y1 - o[1]) * f))
        return slice(max(b0, 0), max(min(b1, shape[0]), 0)), slice(max(a0, 0), max(min(a1, shape[1]), 0))
    R[box(-W / 2, W / 2, -H / 2, H / 2)] = 0                         # already in the main picture
    cell_px = sx * sy * f * f
    keep = []
    for j in range(int(math.floor(lo[1] / sy + 0.5)), int(math.floor(hi[1] / sy + 0.5)) + 1):
        for i in range(int(math.floor(lo[0] / sx + 0.5)), int(math.floor(hi[0] / sx + 0.5)) + 1):
            if i == 0 and j == 0:
                continue
            n = int(R[box(i * sx - sx / 2, i * sx + sx / 2, j * sy - sy / 2, j * sy + sy / 2)].sum())
            if n >= min_frac * cell_px:
                keep.append((i, j, n))
    n_all = len(keep) + 1
    if len(keep) > max_fields - 1:
        keep.sort(key=lambda c: (math.hypot(c[0], c[1]), -c[2]))
        keep = keep[:max(max_fields - 1, 0)]
    out = []
    rows = {}
    for i, j, _ in keep:
        d = M @ np.array([i * sx, j * sy])
        rows.setdefault(j, []).append((i, j, float(d[0]), float(d[1])))
    for n, j in enumerate(sorted(rows, key=lambda j_: float(np.mean([r[3] for r in rows[j_]])))):
        out.extend(sorted(rows[j], key=lambda r: r[2], reverse=bool(n % 2)))
    return out, n_all


def field_size_um(M, wh):
    """(X, Y) extent of a camera field in stage um."""
    M = np.abs(np.asarray(M, float))
    return float(M[0, 0] * wh[0] + M[0, 1] * wh[1]), float(M[1, 0] * wh[0] + M[1, 1] * wh[1])


def fmt_matrix(M):
    return ";".join(f"{v:.6g}" for v in np.ravel(np.asarray(M, float)))


def parse_matrix(s):
    try:
        v = [float(x) for x in str(s).split(";")]
        return np.array(v, float).reshape(2, 2) if len(v) == 4 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------- stitching
def _grey(a):
    a = np.asarray(a, np.float32)
    return a if a.ndim == 2 else a.mean(axis=2)


def _hp(g, sigma):
    """Registration image: brightness-normalised, high-passed (the illumination left after the flat field and the
    field-to-field brightness drop out), lightly smoothed (pixel noise).  Dead pixels (<= 0: the IR camera's border
    rows and columns) count as the median."""
    import cv2
    med = max(float(np.median(g[g > 0])) if (g > 0).any() else 1.0, 1e-6)
    g = np.where(g > 0, g, med) / med
    return cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), sigma)


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / d if d > 0 else 0.0


def pair_shift(ha, hb, d0, wh, min_px=24, max_shift=None):
    """Overlap of fields a and b (high-passed registration images) placed d0 = pos_b - pos_a apart (px): (correction e,
    response, ncc, overlap (w, h)) with the true offset d0 + e, or None when they do not overlap usefully."""
    import cv2
    W, H = wh
    D = np.round(np.asarray(d0, float)).astype(int)
    x0, x1 = max(0, D[0]), min(W, D[0] + W)                        # overlap in a's pixels
    y0, y1 = max(0, D[1]), min(H, D[1] + H)
    w, h = x1 - x0, y1 - y0
    if w < min_px or h < min_px:
        return None
    ca = np.ascontiguousarray(ha[y0:y1, x0:x1])
    cb = np.ascontiguousarray(hb[y0 - D[1]:y1 - D[1], x0 - D[0]:x1 - D[0]])
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    s, resp = cv2.phaseCorrelate(ca, cb, win)
    e = -np.array(s, float)                                         # (phaseCorrelate: cb(v) = ca(v - s); here cb(v) = ca(v + e))
    lim = max_shift if max_shift is not None else 0.4 * min(w, h)
    if not np.isfinite(e).all() or abs(e[0]) > min(lim, 0.45 * w) or abs(e[1]) > min(lim, 0.45 * h):
        return (e + D - np.asarray(d0, float), float(resp), 0.0, (w, h))
    ex, ey = int(round(e[0])), int(round(e[1]))
    u0, u1, v0, v1 = max(0, ex), min(w, w + ex), max(0, ey), min(h, h + ey)
    if u1 - u0 < 8 or v1 - v0 < 8:
        return (e + D - np.asarray(d0, float), float(resp), 0.0, (w, h))
    return (e + D - np.asarray(d0, float), float(resp), _ncc(ca[v0:v1, u0:u1], cb[v0 - ey:v1 - ey, u0 - ex:u1 - ex]), (w, h))


def stitch(images, xy_um, M, ref=0, hp_sigma=8.0, min_response=0.06, min_ncc=0.35, max_shift_px=None, prior_w=1e-3,
           feather=None, match_gain=True, log=None):
    """Fields (equal-size float images, (H, W) or (H, W, C)) at stage positions xy_um, M the camera's stage um per
    pixel: a dict with "image" (the mosaic, float32, uncovered pixels 0), "weight" (the blend weight sum: > 0 where
    covered), "pos" (each field's top-left in the mosaic, px), "pos0" (the same from the stage positions alone),
    "gain" (per field, per channel), "pairs" [(a, b, ex, ey, response, ncc, used)], "n_pairs_used", "ref_xy" (the ref
    field's centre in the mosaic).  ref: the field that keeps its position and brightness (the main picture)."""
    import cv2
    n = len(images)
    H, W = images[0].shape[:2]
    C = 1 if images[0].ndim == 2 else images[0].shape[2]
    Mi = np.linalg.inv(np.asarray(M, float))
    xy = np.asarray(xy_um, float)
    p0 = (Mi @ (xy - xy[ref]).T).T                                  # predicted top-left offsets from the ref field
    hp = [_hp(_grey(a), hp_sigma) for a in images]
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            r = pair_shift(hp[a], hp[b], p0[b] - p0[a], (W, H), max_shift=max_shift_px)
            if r is None:
                continue
            e, resp, ncc, ov = r
            ok = resp >= min_response and ncc >= min_ncc
            pairs.append([a, b, float(e[0]), float(e[1]), resp, ncc, ok, ov])

    def solve(use):
        # x_b - x_a = p0_b - p0_a + e_ab (weight: the pair's response), x_k = p0_k (prior_w), x_ref = p0_ref (fixed)
        rows, rhs, wts = [], [], []
        for a, b, ex, ey, resp, ncc, ok, ov in use:
            r = np.zeros(n); r[b], r[a] = 1.0, -1.0
            rows.append(r); rhs.append(p0[b] - p0[a] + np.array([ex, ey])); wts.append(max(resp, 0.05) * max(ncc, 0.05))
        for k in range(n):
            r = np.zeros(n); r[k] = 1.0
            rows.append(r); rhs.append(p0[k]); wts.append(1e3 if k == ref else prior_w)
        A, B, w_ = np.array(rows), np.array(rhs), np.sqrt(np.array(wts))[:, None]
        return np.linalg.lstsq(A * w_, B * w_, rcond=None)[0]
    use = [p for p in pairs if p[6]]
    pos = solve(use)
    for _ in range(3):                                              # pairs that disagree with the rest: out
        if not use:
            break
        res = np.array([np.hypot(*((pos[p[1]] - pos[p[0]]) - (p0[p[1]] - p0[p[0]] + np.array([p[2], p[3]])))) for p in use])
        lim = max(3.0, 3.0 * float(np.median(res)))
        if res.max() <= lim:
            break
        bad = int(np.argmax(res))
        use[bad][6] = False
        use = [p for p in use if p[6]]
        pos = solve(use)

    # brightness: per field and channel, from the medians of the aligned overlaps (log ratios, least squares, ref = 1)
    gain = np.ones((n, C))
    if match_gain and n > 1:
        eq = []
        for a, b, ex, ey, resp, ncc, ok, ov in pairs:
            d = np.round(pos[b] - pos[a]).astype(int)
            x0, x1, y0, y1 = max(0, d[0]), min(W, d[0] + W), max(0, d[1]), min(H, d[1] + H)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            ia = np.asarray(images[a], np.float32)[y0:y1, x0:x1].reshape(-1, C)
            ib = np.asarray(images[b], np.float32)[y0 - d[1]:y1 - d[1], x0 - d[0]:x1 - d[0]].reshape(-1, C)
            live = (ia.min(1) > 0) & (ib.min(1) > 0)
            if live.sum() < 200:
                continue
            ma, mb = np.median(ia[live][::7], axis=0), np.median(ib[live][::7], axis=0)
            if (ma > 0).all() and (mb > 0).all():
                eq.append((a, b, np.log(ma / mb), (x1 - x0) * (y1 - y0)))
        if eq:
            rows, rhs, wts = [], [], []
            for a, b, lr, area in eq:                              # g_b * B = g_a * A  ->  log g_b - log g_a = log(A / B)
                r = np.zeros(n); r[b], r[a] = 1.0, -1.0
                rows.append(r); rhs.append(lr); wts.append(math.sqrt(area))
            for k in range(n):
                r = np.zeros(n); r[k] = 1.0
                rows.append(r); rhs.append(np.zeros(C)); wts.append(1e4 if k == ref else 1e-2)
            A, B, w_ = np.array(rows), np.array(rhs), np.array(wts)[:, None]
            gain = np.exp(np.linalg.lstsq(A * w_, B * w_, rcond=None)[0])

    # feathered blend: weight ramps over the overlap width from each field's edges
    org = np.floor(pos.min(0)).astype(int)
    size = (np.ceil(pos.max(0) + [W, H]).astype(int) - org) + 1
    acc = np.zeros((size[1], size[0], C), np.float32)
    wsum = np.zeros((size[1], size[0]), np.float32)
    fw = feather if feather is not None else max(8.0, 0.5 * 0.15 * min(W, H))

    def ramp(m):
        t = np.clip((np.arange(m, dtype=np.float32) + 0.5) / fw, 0, 1)
        t = np.minimum(t, t[::-1])
        return t * t * (3 - 2 * t) + 1e-4                             # smoothstep (no kink at the ramp's ends)
    wt = np.outer(ramp(H), ramp(W)).astype(np.float32)
    for k in range(n):
        t = pos[k] - org
        it = np.floor(t).astype(int)
        fr = t - it
        img = np.asarray(images[k], np.float32).reshape(H, W, C) * gain[k].astype(np.float32)
        T = np.float32([[1, 0, fr[0]], [0, 1, fr[1]]])
        dead = _grey(images[k]) <= 0                                # (dead pixels: no weight)
        w0 = wt if not dead.any() else wt * cv2.erode((~dead).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        wk = cv2.warpAffine(w0, T, (W + 1, H + 1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ik = cv2.warpAffine(img, T, (W + 1, H + 1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ik = ik.reshape(H + 1, W + 1, C)
        sl = (slice(it[1], it[1] + H + 1), slice(it[0], it[0] + W + 1))
        acc[sl] += ik * wk[..., None]
        wsum[sl] += wk
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    out[wsum <= 1e-6] = 0
    if C == 1:
        out = out[..., 0]
    used = sum(1 for p in pairs if p[6])
    if log:
        log(f"    stitched {n} fields: {used} of {len(pairs)} overlaps matched, corrections up to "
            f"{float(np.abs(pos - p0).max()) if n > 1 else 0.0:.1f} px from the stage positions")
    return dict(image=out, weight=wsum, pos=pos - org, pos0=p0 - org, gain=gain, n_pairs_used=used,
                pairs=[tuple(p[:7]) for p in pairs], ref_xy=(pos[ref] - org + [W / 2.0, H / 2.0]))
