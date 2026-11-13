"""Flakes imported by hand (Library: Import flake): a rough thickness from the flake's colour against the substrate.

The flake's median colour (inside its drawn outline, a few pixels in from the edge) over the substrate's (a ring around
it, other flakes and debris left out by keeping the ring's most common colour) is compared with the thin-film model the
scan uses (flakeid.thick100.model100: the scan camera's colour response at 100x): the thickness whose model colour ratio fits
best, and other thicknesses that fit nearly as well (hBN colours repeat).  Pictures from another camera, or processed by
microscope software (white balance, gamma, auto contrast), make this rough; it is labelled an estimate.
"""
from __future__ import annotations


import numpy as np

def _model(material, oxide_nm):
    """The scan's 100x thin-film model for the material (flakeid.thick100.model100: graphite with its own fitted 100x
    optics since 2026-11-12, hBN on the scan camera at NA 0.9)."""
    from flakeid.thick100 import model100
    return model100(material, oxide_nm)


def colours(img_bgr, outline, edge_px=3, ring=(8, 40)):
    """(flake RGB, substrate RGB, number of flake / substrate pixels) of a picture (BGR uint8) and the flake's outline
    ([x0, y0, x1, y1, ...] in picture pixels): medians inside the outline and in a ring around it."""
    import cv2
    h, w = img_bgr.shape[:2]
    pts = np.asarray(outline, np.float64).reshape(-1, 2)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))      # noqa: E731
    inner = cv2.erode(m, k(edge_px)).astype(bool)
    if inner.sum() < 30:
        inner = m.astype(bool)
    near, far = cv2.dilate(m, k(ring[0])).astype(bool), cv2.dilate(m, k(ring[1])).astype(bool)
    ring_m = far & ~near
    rgb = img_bgr[..., ::-1]
    sub = rgb[ring_m].astype(np.float64)
    if len(sub) < 20:
        raise ValueError("too little bare substrate around the flake in this picture (the flake or others fill it): "
                         "no estimate; enter the thickness by hand")
    # the substrate: the ring's most common colour (other flakes, tape residue and debris left out)
    med = np.median(sub, axis=0)
    keep = np.abs(sub - med).max(axis=1) < 0.12 * max(float(med.max()), 1.0)
    if keep.sum() < 30:
        keep = np.ones(len(sub), bool)
    return np.median(rgb[inner].astype(np.float64), axis=0), np.median(sub[keep], axis=0), int(inner.sum()), int(keep.sum())


CLIP_HI, CLIP_LO = 236.0, 12.0     # a colour channel this close to white or black (flake or substrate) is left out:
                                   # microscope software compresses the highlights (Jordan's 2026-10-19 hBN: blue 240)
T_WINDOW = {"hbn": (2.0, 100.0), "graphite": (0.3, 30.0)}   # thicknesses searched by default (flakes for stacks)
GAMMA_DEFAULT = 2.2                # a picture saved by microscope software: brightness ~ light^(1/2.2) unless calibrated


def _ratios(fl, sb, linear, gamma):
    """(light ratio flake / substrate per channel, channels used): channels near clipping are left out."""
    ok = (np.maximum(fl, sb) < CLIP_HI) & (np.minimum(fl, sb) > CLIP_LO)
    if linear:
        r = fl / np.maximum(sb, 1e-6)
    else:
        r = (np.maximum(fl, 0.5) / np.maximum(sb, 0.5)) ** float(gamma or GAMMA_DEFAULT)
    return np.maximum(r, 1e-3), ok


def _grid(material, t_range):
    graphite = material == "graphite"
    lo, hi = t_range or T_WINDOW["graphite" if graphite else "hbn"]
    return np.arange(max(lo, 0.3 if graphite else 0.5), hi + 1e-9, 0.05 if graphite else 0.5)


def estimate(img_bgr, outline, material, oxide_nm, linear=False, t_max=None, gamma=None, t_range=None):
    """Thickness candidates from the flake's colour: dict(candidates=[{t_nm, err}], ratio=[r, g, b], note).
    The flake / substrate ratio per colour channel does not depend on the white balance; it does on the picture's
    gamma (gamma: a picture source's calibrated one, see calibrate(); default 2.2) and is lost where a channel is
    nearly white (left out).  Only t_range (nm; default T_WINDOW) is searched: hBN colours repeat every ~150 nm."""
    fl, sb, nf, ns = colours(img_bgr, outline)
    ratio, ok = _ratios(fl, sb, linear, gamma)
    if ok.sum() < 2:
        raise ValueError("too few usable colours: the flake or the substrate is nearly white or black in this picture "
                         "(over- or underexposed); no estimate")
    graphite = material == "graphite"
    if t_max and not t_range:
        t_range = (T_WINDOW["graphite" if graphite else "hbn"][0], t_max)
    grid = _grid(material, t_range)
    model = 1.0 + np.asarray(_model(material, oxide_nm).contrast(list(grid)))
    err = np.sqrt(((np.log(np.maximum(model[:, ok], 1e-3)) - np.log(ratio[ok])) ** 2).mean(axis=1))
    # local minima, best first; others only if nearly as good and clearly different
    mins = [i for i in range(len(grid)) if (i == 0 or err[i] <= err[i - 1]) and (i == len(grid) - 1 or err[i] <= err[i + 1])]
    mins.sort(key=lambda i: err[i])
    cands = []
    for i in mins:
        if cands and err[i] > err[mins[0]] + 0.06:
            break
        if all(abs(grid[i] - c["t_nm"]) > (1.0 if graphite else 15.0) for c in cands):
            c = dict(t_nm=round(float(grid[i]), 2 if graphite else 1), err=round(float(err[i]), 4))
            cands.append(c)
        if len(cands) >= 3:
            break
    best = cands[0]["err"] if cands else 9.0
    left = [n for n, k in zip("RGB", ok) if not k]
    note = ("good colour match" if best < 0.05 else "rough colour match" if best < 0.12 else
            "poor colour match: calibrate this picture source with a flake of known thickness, or check the oxide")
    if left:
        note += f" ({'/'.join(left)} left out: nearly white or black)"
    return dict(candidates=cands, ratio=[round(float(x), 4) for x in ratio], flake_rgb=[round(float(x), 1) for x in fl],
                substrate_rgb=[round(float(x), 1) for x in sb], pixels=[nf, ns], note=note, linear=bool(linear),
                gamma=None if linear else float(gamma or GAMMA_DEFAULT), used=[bool(k) for k in ok],
                window=[float(grid[0]), float(grid[-1])], material=material, oxide_nm=float(oxide_nm))


def calibrate(img_bgr, outline, material, oxide_nm, t_nm):
    """The gamma of a picture source from one flake of known thickness t_nm (AFM): the one that makes its colour ratios
    fit the optical model at t_nm best.  dict(gamma, err, channels)."""
    fl, sb, _, _ = colours(img_bgr, outline)
    model = 1.0 + np.asarray(_model(material, oxide_nm).contrast([float(t_nm)]))[0]
    best = None
    for g in np.arange(0.8, 3.501, 0.01):
        r, ok = _ratios(fl, sb, False, g)
        if ok.sum() < 2:
            raise ValueError("too few usable colours in this picture (nearly white or black)")
        e = float(np.sqrt(((np.log(model[ok]) - np.log(r[ok])) ** 2).mean()))
        if best is None or e < best[1]:
            best = (float(g), e, ok)
    return dict(gamma=round(best[0], 3), err=round(best[1], 4), channels=[n for n, k in zip("RGB", best[2]) if k])


def auto_outline(img_bgr, seed, thr=None, max_pts=160, mode="flake"):
    """The outline of the flake under seed ([x, y] in picture pixels) in a picture (BGR uint8): everything connected
    to the seed whose colour differs from the bare substrate (the most common colour that is not the seed's) by more
    than thr (relative, max over R, G, B; default: 40 % of the seed's own difference, 1.2 - 8 %).  Thicker parts
    of the same flake differ more, so they are included.  Returns dict(outline=[x0, y0, ...], thr, seed_contrast,
    substrate_rgb, area_px, touches_edge) or raises ValueError.
    mode "terrace": only what has the seed's colour (within thr; default 25 % of its difference from the substrate,
    3 - 8 %): one terrace, or a flake that touches another of a different thickness."""
    import cv2
    h0, w0 = img_bgr.shape[:2]
    k = min(1.0, 1600.0 / max(h0, w0))                  # (large pictures are outlined at most 1600 px wide)
    img = cv2.resize(img_bgr, (int(round(w0 * k)), int(round(h0 * k))), interpolation=cv2.INTER_AREA) if k < 1 else img_bgr
    h, w = img.shape[:2]
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5) + 1.0
    sx, sy = int(round(float(seed[0]) * k)), int(round(float(seed[1]) * k))
    if not (0 <= sx < w and 0 <= sy < h):
        raise ValueError("the point is outside the picture")
    sc = np.median(f[max(0, sy - 3):sy + 4, max(0, sx - 3):sx + 4].reshape(-1, 3), axis=0)
    flat = f[::3, ::3].reshape(-1, 3)
    other = flat[np.abs(np.log(flat / sc)).max(axis=1) > 0.04]       # not the seed's colour
    if len(other) < 50:
        raise ValueError("the whole picture has the flake's colour: no bare substrate to tell it from")
    q = np.round(np.log(other) / 0.01).astype(np.int64)
    key = (q[:, 0] * 4096 + q[:, 1]) * 4096 + q[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    top = other[key == vals[cnt.argmax()]]
    sub = np.median(other[np.abs(np.log(other / top.mean(axis=0))).max(axis=1) < 0.03], axis=0)
    d = np.abs(np.log(f / sub)).max(axis=2)
    dseed = float(np.abs(np.log(sc / sub)).max())
    if dseed <= 0.012:
        raise ValueError(f"the point clicked is no different from the substrate (contrast {dseed * 100:.1f} %): click inside the flake")
    if mode == "terrace":
        if thr is None:
            thr = max(0.03, min(0.25 * dseed, 0.08))
        thr = float(thr)
        m = (np.abs(np.log(f / sc)).max(axis=2) < thr).astype(np.uint8)
    else:
        if thr is None:
            thr = max(0.012, min(0.4 * dseed, 0.08))  # (capped: thinner parts of a thick flake stay in)
        thr = float(thr)
        if dseed <= thr:
            raise ValueError(f"the point clicked differs from the substrate by only {dseed * 100:.1f} %: lower the threshold or click inside the flake")
        m = (d > thr).astype(np.uint8)
    # opened with a disc ~0.4 % of the picture: residue and specks joined to the flake by thin bridges come off
    r = max(1, int(round(min(h, w) / 250.0)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3)))
    n, lab = cv2.connectedComponents(m, connectivity=8)
    li = lab[sy, sx]
    if li == 0:                                         # (the seed on a speck the opening removed: the nearest part)
        ys, xs = np.nonzero(lab)
        if not len(xs):
            raise ValueError("nothing found that differs from the substrate here")
        j = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
        li = lab[ys[j], xs[j]]
    comp = (lab == li).astype(np.uint8)
    inv = np.pad(1 - comp, 1, constant_values=1)        # holes (residue, bubbles inside the flake) filled
    cv2.floodFill(inv, None, (0, 0), 0)
    comp = comp | inv[1:-1, 1:-1]
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    eps = max(0.6, 0.002 * per)
    poly = cv2.approxPolyDP(c, eps, True)
    while len(poly) > max_pts:
        eps *= 1.4
        poly = cv2.approxPolyDP(c, eps, True)
    pts = poly.reshape(-1, 2).astype(np.float64) / k
    edge = bool(comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any())
    return dict(outline=[round(float(v), 1) for v in pts.ravel()], thr=round(thr, 4), mode=mode, seed_contrast=round(dseed, 4),
                substrate_rgb=[round(float(v) - 1, 1) for v in sub[::-1]], area_px=float(comp.sum()) / (k * k), touches_edge=edge)
