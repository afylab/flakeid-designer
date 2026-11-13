"""Screening of 100x flake images: features that decide whether a flake is worth keeping, fine-terrace outlines.

Fine terraces (hBN)
-------------------
Terraces a few nm thicker or thinner than the terrace of interest change its colour by only 1-3 % (they are
invisible in the natural image and are absorbed into the terrace by flakeid.hundred.analyse, which accepts 4 % in
every channel).  The thin-film model gives, at the flake's thickness, how the log-intensity of each channel changes
per nm of hBN (g, ~2-5.5 %/nm at 100x, NA 0.9; the most sensitive channel changes with thickness).  Projecting each
pixel's colour change onto g gives an equivalent thickness-change map dt (nm): fine terraces are connected regions
of the flake whose dt differs from the terrace level by more than FINE_DT_NM, are wider than FINE_MIN_W_UM and not
round (round spots are bubbles / debris).  The colour change not along g (orth) is residue / debris.

Graphite wrinkles and tears
---------------------------
Wrinkles and folds are long, thin, nearly straight lines across the flake, often only a few % darker or brighter in
one colour channel, far below the per-pixel noise.  wrinkle_lines() finds them with an oriented line filter: the
image (each channel relative to the flake's local level) is averaged along a direction over ~12 um and compared with
the two parallel bands 1-2 um to either side; a line differs from both sides alike, a terrace step or the flake edge
differs from one side only and is rejected.  Responses are scaled by their own spread on the flake (noise), and long,
thin components are kept as wrinkle lines (polylines, stored for the designer).  From them: the number of lines, the
largest wrinkle-free disc relative to the flake's own, and how parallel the lines are (1 = one direction; wrinkles
in one direction move less during stacking).  The older Hessian ridge density (ridges()) stays a model feature.
Tears and jagged edges show as holes and a rough outline of the terrace.

features() returns one dict per image; the screening model (profiles/screen100_<material>.json, fitted on the
user's 100x labels by tools/fit_screen100.py) turns it into P(useful or maybe).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .hundred import _M100, _flatten, _mode_rgb, analyse

# Graphite that is yellow at 100x: far brighter than the substrate in red and green (+0.67 / +0.79 / +0.47 in the
# picture), a thick piece hanging out of the plane (out of focus while the chip is in focus) or far too thick; not
# usable and not scannable (Jordan, 2026-10-19: graphite_20261019_3 F002, whose 10x colour had fitted 5.7 nm).  Every
# other graphite terrace of that chip is darker than the substrate, the thickest (21 nm) bluish.  Rejected at the
# screening (tier "reject"): never in the designer.
YELLOW_MIN = 0.30          # linear contrast to the substrate that red and green both exceed (blue lower than both)


def thick_yellow(terrace_rgb, substrate_rgb):
    """True when a graphite terrace's colour (linear RGB) is yellow: red and green both YELLOW_MIN above the
    substrate's, blue less than both."""
    c = np.asarray(terrace_rgb, float) / np.maximum(np.asarray(substrate_rgb, float), 1e-6) - 1.0
    return bool(c[0] > YELLOW_MIN and c[1] > YELLOW_MIN and c[2] < min(c[0], c[1]))


FINE_DT_NM = 0.3           # thickness change that makes a fine terrace (nm), after edge-preserving smoothing
FINE_MIN_W_UM = 2.0        # narrowest fine terrace (um; opening width)
FINE_MIN_UM2 = 15.0        # smallest fine terrace (um2)
FINE_EDGE_UM = 2.0         # rim of the terrace left out (blurred flake edge)
PROFILE_DIR = Path(__file__).parent / "profiles"
_G = {}


def sensitivity(material, oxide_nm, t_nm, na=0.9):
    """d ln I / d t per nm for (R, G, B) at thickness t (thin-film model, 100x NA, the RAW16 camera)."""
    key = (material, round(float(oxide_nm)), round(float(t_nm), 1), na)
    if key not in _G:
        from .optics import ContrastModel
        from .thickness import camera_from_profile, load_profile
        m = ContrastModel(material, float(oxide_nm), na=na, camera=camera_from_profile(load_profile("bx51_xc10_mm_raw16")))
        t = max(float(t_nm), 0.6)
        c = m.contrast([t])[0]
        _G[key] = (m.contrast([t + 0.5])[0] - m.contrast([t - 0.5])[0]) / (1.0 + c)
    return _G[key]


WRINKLE_DS = 0.5               # working scale of the line filter
WRINKLE_ALONG_UM = 6.0         # sigma along the line
WRINKLE_ACROSS_UM = 0.6        # sigma across (line ~1 um wide); side bands at 3 sigma
WRINKLE_Z = 6.0                # threshold in units of the response's spread on the flake
WRINKLE_MIN_UM = 10.0          # shortest line kept
WRINKLE_EDGE_UM = 3.0          # lines this close to the flake edge are not counted (edge artefacts)


def _flake_region(c8, T, ds):
    """The whole flake around terrace T (substrate is white in the enhanced view), at scale ds, shrunk off its edge."""
    from scipy import ndimage
    from skimage.morphology import remove_small_holes
    h, w = T.shape
    sz = (int(w * ds), int(h * ds))
    m = cv2.resize(c8.min(axis=2), sz) < 225
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    m = remove_small_holes(m, 3000)
    Ts = cv2.resize(T.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST)
    lab, _ = ndimage.label(m)
    ids = np.unique(lab[Ts > 0])
    ids = ids[ids > 0]
    return np.isin(lab, ids) | (Ts > 0)


def wrinkle_lines(lin, T, c8, px_um):
    """Wrinkle / fold lines on the flake: (features, polylines in original pixels)."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    ds = WRINKLE_DS
    px = px_um / ds
    F0 = _flake_region(c8, T, ds)
    e = int(round(WRINKLE_EDGE_UM / px))
    # (the frame border counts as an edge too: a flake filling the frame has no free edge there, but the filter does)
    F = cv2.erode(F0.astype(np.uint8), np.ones((2 * e + 1,) * 2, np.uint8), borderType=cv2.BORDER_CONSTANT,
                  borderValue=0).astype(bool)
    f = dict(wrinkle_n=0, wrinkle_len_density=0.0, wrinkle_free_ratio=1.0, wrinkle_coherence=1.0)
    if F.sum() < 500:
        return f, []
    h, w = F.shape
    small = cv2.resize(lin, (w, h), interpolation=cv2.INTER_AREA)
    Ff = F.astype(np.float32)
    sa, sc = WRINKLE_ALONG_UM / px, WRINKLE_ACROSS_UM / px
    off = 3 * sc
    D = int(np.ceil(np.hypot(h, w))) + 2
    best = np.zeros((h, w), np.float32)
    for ch in range(3):
        C = small[..., ch]
        lvl = cv2.GaussianBlur(C * Ff, (0, 0), 20) / np.maximum(cv2.GaussianBlur(Ff, (0, 0), 20), 1e-3)
        R = np.where(F, C / np.maximum(lvl, 1.0) - 1.0, 0).astype(np.float32)
        resp = np.zeros((h, w), np.float32)
        for a in np.arange(0.0, 180.0, 11.25):
            # rotate so that direction a runs along x, filter separably, rotate the response back
            M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
            M[:, 2] += [(D - w) / 2, (D - h) / 2]
            Rr = cv2.warpAffine(R, M, (D, D), flags=cv2.INTER_LINEAR)
            B = cv2.GaussianBlur(Rr, (0, 0), sigmaX=sa, sigmaY=sc)
            k = int(round(off))
            up, dn = np.roll(B, k, axis=0), np.roll(B, -k, axis=0)
            rr = np.abs(B - 0.5 * (up + dn)) - np.abs(up - dn)
            Mi = cv2.invertAffineTransform(M)
            resp = np.maximum(resp, cv2.warpAffine(rr, Mi, (w, h), flags=cv2.INTER_LINEAR))
        rv = resp[F]
        mad = 1.4826 * float(np.median(np.abs(rv - np.median(rv)))) + 1e-12
        best = np.maximum(best, (resp - float(np.median(rv))) / mad)
    mk = (best > WRINKLE_Z) & F
    lab, n = ndimage.label(mk, structure=np.ones((3, 3)))
    keep = np.zeros_like(mk)
    lines, angs, lens = [], [], []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        if len(ys) < 5:
            continue
        ev, evec = np.linalg.eigh(np.cov(np.c_[xs, ys].T))
        length = 4 * np.sqrt(max(ev[1], 0)) * px
        if length < WRINKLE_MIN_UM or ev[0] / max(ev[1], 1e-9) > 0.05:      # long and thin only
            continue
        keep[sl] |= lab[sl] == i
        v = evec[:, 1]
        angs.append(float(np.arctan2(v[1], v[0])))
        lens.append(float(length))
        # polyline along the line's axis: skeleton points ordered by their projection, ~1 point per 3 um
        sk = skeletonize(lab[sl] == i)
        sy, sx = np.nonzero(sk)
        if len(sy) < 2:
            sy, sx = ys, xs
        X = np.c_[sx + sl[1].start, sy + sl[0].start].astype(float)
        t = (X - X.mean(0)) @ v
        o = np.argsort(t)
        X, t = X[o], t[o]
        nb = max(2, int(length / 3))
        edges = np.linspace(t[0], t[-1] + 1e-6, nb + 1)
        pts = [X[(t >= edges[j]) & (t < edges[j + 1])].mean(0) for j in range(nb) if np.any((t >= edges[j]) & (t < edges[j + 1]))]
        lines.append([round(float(c) / ds, 1) for p in pts for c in p])
    area = float(F.sum()) * px * px
    f["wrinkle_n"] = len(lens)
    f["wrinkle_len_density"] = float(np.sum(lens) / area * 100) if lens else 0.0
    if lens:
        z = np.sum(np.array(lens) * np.exp(2j * np.array(angs))) / np.sum(lens)
        f["wrinkle_coherence"] = float(abs(z))
        free = F0 & ~cv2.dilate(keep.astype(np.uint8), np.ones((int(2 / px) + 1,) * 2, np.uint8)).astype(bool)
        d_free = cv2.distanceTransform(np.pad(free.astype(np.uint8), 1), cv2.DIST_L2, 5).max()
        d_all = cv2.distanceTransform(np.pad(F0.astype(np.uint8), 1), cv2.DIST_L2, 5).max()
        f["wrinkle_free_ratio"] = float(d_free / max(d_all, 1e-6))
    return f, lines


def _polys(mask, min_area=2.0):
    from .outline import polygons
    return polygons(mask.astype(np.uint8), 0, 0, 1.0, min_area=min_area)


def fine_terraces(rf, T, g, px_um):
    """dt map (nm), orth residual, fine-terrace label image and per-region stats inside the terrace mask T."""
    d = rf - 1.0
    gg = float(g @ g)
    dt = (d @ g) / gg
    orth = np.linalg.norm(d - dt[..., None] * g[None, None, :], axis=2)
    # edge-preserving smoothing: noise (~0.3 nm per pixel) averages out, terrace steps stay sharp
    dts = cv2.GaussianBlur(dt.astype(np.float32), (0, 0), 1.0)
    for _ in range(3):
        dts = cv2.bilateralFilter(dts, 15, 0.25, 6)
    e = max(1, int(round(FINE_EDGE_UM / px_um)))
    inside = cv2.erode(T.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2)).astype(bool)
    if not inside.any():
        inside = T.astype(bool)
    # terrace level: the most common dt (a flake may be half one terrace, half another)
    hst, edges = np.histogram(dts[inside], bins=np.arange(-3, 3.001, 0.05))
    hst = np.convolve(hst, np.ones(3) / 3, mode="same")
    lvl = float(edges[np.argmax(hst)] + 0.025)
    w = max(1, int(round(FINE_MIN_W_UM / px_um)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, w))
    regs, lab_all = [], np.zeros(T.shape, np.int32)
    n_lab = 0
    for sgn in (1, -1):
        m = ((sgn * (dts - lvl) > FINE_DT_NM) & inside).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            a_um2 = st[i, cv2.CC_STAT_AREA] * px_um ** 2
            if a_um2 < FINE_MIN_UM2:
                continue
            r = lab == i
            cnts, _ = cv2.findContours(r.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            per = sum(cv2.arcLength(c, True) for c in cnts)
            circ = 4 * np.pi * st[i, cv2.CC_STAT_AREA] / max(per, 1.0) ** 2
            if circ > 0.75 and a_um2 < 40:                     # round: bubble / debris, not a terrace
                continue
            n_lab += 1
            lab_all[r] = n_lab
            regs.append(dict(dt_nm=round(float(np.median(dts[r]) - lvl), 2), area_um2=round(float(a_um2), 1),
                             orth=round(float(np.median(orth[r])), 4)))
    return dts - lvl, orth, lab_all, regs


def ridges(L, T, px_um):
    """Wrinkle / fold line density: Hessian ridge strength of the flattened brightness, length per terrace area."""
    s = 1.5
    Ls = cv2.GaussianBlur(L.astype(np.float32), (0, 0), s)
    dxx = cv2.Sobel(Ls, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(Ls, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(Ls, cv2.CV_32F, 1, 1, ksize=3)
    tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
    l1 = 0.5 * (dxx + dyy + tmp)
    l2 = 0.5 * (dxx + dyy - tmp)
    ridge = np.maximum(np.abs(l1), np.abs(l2)) * (np.abs(np.abs(l1) - np.abs(l2)) / (np.abs(l1) + np.abs(l2) + 1e-6))
    e = max(1, int(round(1.5 / px_um)))
    inner = cv2.erode(T.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2)).astype(bool)
    if not inner.any():
        return 0.0, np.zeros(T.shape, bool)
    rv = ridge[inner]
    thr = max(0.004, float(np.median(rv)) + 6.0 * 1.4826 * float(np.median(np.abs(rv - np.median(rv)))))   # noise-scaled
    line = (ridge > thr) & inner
    line = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 1), np.uint8)).astype(bool)
    # keep elongated components only (lines, not specks)
    n, lab, st, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
    keep = np.zeros(T.shape, bool)
    for i in range(1, n):
        wv, hv = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if max(wv, hv) * px_um >= 3.0 and st[i, cv2.CC_STAT_AREA] < 0.35 * wv * hv:
            keep |= lab == i
    length_um = keep.sum() * px_um / 2.0                       # ~2 px wide lines
    return float(length_um / max(inner.sum() * px_um ** 2, 1.0) * 100.0), keep   # um of line per 100 um2


def bubbles(rf, T, px_um, dev_thr):
    """Round blobs on the terrace (bubbles under / on the flake): connected areas whose colour deviates from the
    terrace by more than dev_thr, 2-300 um2, circularity > 0.6.  Returns (count, area fraction, outlines)."""
    e = max(1, int(round(1.0 / px_um)))
    inner = cv2.erode(T.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2))
    dev = np.sqrt(((cv2.GaussianBlur(rf, (0, 0), 1.5) - 1.0) ** 2).mean(axis=2))
    m = ((dev > dev_thr) & (inner > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(T.shape, bool)
    cnt = 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA] * px_um ** 2
        if not 2.0 <= a <= 300.0:
            continue
        r = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = sum(cv2.arcLength(c, True) for c in cs)
        if 4 * np.pi * st[i, cv2.CC_STAT_AREA] / max(per, 1.0) ** 2 > 0.6:
            cnt += 1
            keep |= r.astype(bool)
    return cnt, float(keep.sum() / max(T.sum(), 1)), keep


# 100x score (Avram, 2026-10-27): the largest connected part of the terrace with no step edge, glue / debris, bubble or
# fold, kept CLEAN_GAP_UM away from each and CLEAN_EDGE_UM from the terrace edge (blurred at 100x)
CLEAN_GAP_UM = 0.5
CLEAN_EDGE_UM = 1.0


def clean_score(T, reg, flab, defects, px_um, lines=()):
    """The 100x score of a frame: dict(clean_um2 (the largest clean part's area, the score), clean_d_um (the largest
    disc inside it), poly (its outline, frame pixels)).  T: the terrace mask; reg: hundred.analyse's regions (debris and
    steps: other-thickness areas, residue haze); flab: fine-terrace labels (screen100.fine_terraces: their borders are
    step edges); defects: other masks to keep out (bubbles, fold lines); lines: polylines to keep out (wrinkles)."""
    h, w = T.shape
    bad = np.zeros((h, w), np.uint8)
    for key in ("debris", "steps"):
        for p in reg.get(key) or []:
            cv2.fillPoly(bad, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], 1)
    if flab is not None and flab.any():                     # step edges: where the fine-terrace label changes
        lab = flab.astype(np.float32)
        edge = cv2.dilate(lab, np.ones((3, 3), np.uint8)) != cv2.erode(lab, np.ones((3, 3), np.uint8))
        bad |= edge.astype(np.uint8)
    for m in defects:
        if m is not None:
            bad |= np.asarray(m, bool).astype(np.uint8)
    lw = max(1, int(round(1.0 / px_um)))
    for ln in lines or ():
        q = np.round(np.array(ln).reshape(-1, 2)).astype(np.int32)
        if len(q) >= 2:
            cv2.polylines(bad, [q], False, 1, lw)
    g = max(1, int(round(CLEAN_GAP_UM / px_um)))
    bad = cv2.dilate(bad, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1,) * 2))
    e = max(1, int(round(CLEAN_EDGE_UM / px_um)))
    inner = cv2.erode(T.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1,) * 2))
    ok = ((inner > 0) & (bad == 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ok, connectivity=4)
    if n <= 1:
        return dict(clean_um2=0.0, clean_d_um=0.0, poly=[])
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    best = (lab == k).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(best, 1), cv2.DIST_L2, 5)
    return dict(clean_um2=round(float(best.sum()) * px_um ** 2, 1), clean_d_um=round(2.0 * float(dist.max()) * px_um, 1),
                poly=_polys(best, min_area=4.0))


def features(lin, material, oxide_nm, t_nm, substrate=None, expect=None):
    """Screening features of one 100x frame (linear RGB, optics pattern divided out); also the fine-terrace /
    wrinkle outlines for the viewer and designer.  substrate, expect: hundred.analyse (off-centre flakes)."""
    px_um = float(np.sqrt(abs(np.linalg.det(_M100()))))
    h, w = lin.shape[:2]
    c8, sg, reg = analyse(lin, substrate, expect)
    tc = np.asarray(reg["terrace_rgb"], np.float32)          # the colour analyse took the terrace from (its seed)
    rf = _flatten(cv2.GaussianBlur(lin / np.maximum(tc, 1.0), (0, 0), 1.0))
    T = np.zeros((h, w), np.uint8)
    for p in reg["terrace"]:
        cv2.fillPoly(T, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], 1)
    a_px = px_um ** 2
    ta = max(float(T.sum()) * a_px, 1.0)
    f = dict(terrace_um2=reg["terrace_um2"], clean_um2=reg["clean_um2"], clean_d_um=reg["clean_d_um"],
             clean_frac=reg["clean_um2"] / ta, debris_n=reg["debris_n"], debris_per_1000=reg["debris_n"] / ta * 1000,
             debris_frac=reg["debris_um2"] / ta, steps_frac=reg["steps_um2"] / ta, cut_by_frame=float(reg["cut_by_frame"]),
             t_nm=float(t_nm))
    extra = dict(fine=[], wrinkles=[], c8=c8, sg=sg)
    if material == "graphite":
        sub = np.asarray(substrate, float) if substrate is not None else _mode_rgb(lin)[0]
        cs = tc / np.maximum(sub, 1e-6) - 1.0
        f.update(con_r=float(cs[0]), con_g=float(cs[1]), con_b=float(cs[2]), thick_yellow=float(thick_yellow(tc, sub)))
    reg["score100"] = dict(clean_um2=0.0, clean_d_um=0.0, poly=[])
    if T.sum() < 100 or (reg.get("seed") or {}).get("found") is False:
        return f, reg, extra                    # (no terrace, or the frame missed the flake: 100x score 0)
    # outline roughness and holes (tears) of the terrace
    cnts, _ = cv2.findContours(T, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    f["solidity"] = float(cv2.contourArea(c) / max(cv2.contourArea(hull), 1.0))
    f["roughness"] = float(cv2.arcLength(c, True) / max(cv2.arcLength(hull, True), 1.0))
    L = rf.mean(axis=2)
    f["ridge_density"], rmask = ridges(L, T, px_um)
    inside = T.astype(bool)
    f["focus"] = float(np.var(cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 1.0), cv2.CV_32F)[cv2.dilate(T, np.ones((15, 15), np.uint8)) > 0]) * 1e6)
    nb, fb, bmask = bubbles(rf, T, px_um, (reg.get("noise") or {}).get("dev_thr", 0.025))
    f["bubble_per_1000"] = nb / ta * 1000
    f["bubble_frac"] = fb
    extra["bubbles"] = _polys(bmask, min_area=2.0)
    g = sensitivity(material, oxide_nm, t_nm)
    dt, orth, flab, fregs = fine_terraces(rf, T, g, px_um)
    f["g_norm"] = float(np.linalg.norm(g))
    f["dt_iqr"] = float(np.subtract(*np.percentile(dt[inside], [75, 25])))
    f["dt_p95"] = float(np.percentile(np.abs(dt[inside]), 95))
    f["orth_p90"] = float(np.percentile(orth[inside], 90))
    f["fine_n"] = len(fregs)
    f["fine_frac"] = float((flab > 0).sum() * a_px / ta)
    extra["fine"] = [dict(r, poly=_polys(flab == i + 1, min_area=4.0)) for i, r in enumerate(fregs)]
    if material == "graphite":                  # (validated on graphite only; on hBN it also finds terrace steps)
        wf, extra["wrinkles"] = wrinkle_lines(lin, T, c8, px_um)   # polylines (original pixels)
        f.update(wf)
    extra["dt"] = dt
    reg["score100"] = clean_score(T, reg, flab, (bmask, rmask), px_um, extra.get("wrinkles") or ())
    return f, reg, extra


FEATURES = ["log_terrace", "log_clean", "clean_d_um", "clean_frac", "debris_per_1000", "debris_frac", "steps_frac",
            "cut_by_frame", "solidity", "roughness", "ridge_density", "focus_log", "dt_iqr", "dt_p95", "orth_p90",
            "fine_n", "fine_frac", "bubble_per_1000", "bubble_frac"]


def vector(f, features=None):
    v = dict(f, log_terrace=np.log10(max(f.get("terrace_um2", 1.0), 1.0)), log_clean=np.log10(max(f.get("clean_um2", 1.0), 1.0)),
             focus_log=np.log10(max(f.get("focus", 1e-3), 1e-3)))
    return np.array([float(v.get(k, np.nan)) for k in (features or FEATURES)], float)


def _logistic(d, f):
    fs = d.get("features") or FEATURES
    x = vector(f, fs)
    t = f.get("t_terrace_nm")                       # (terrace a's thickness, set by the catalog; t_nm is the target's
    for k, v in (d.get("omit_above_t_nm") or {}).items():  # label): left out of thick flakes, filled (graphite focus_log
        if k in fs and t is not None and t > v:         # above 5 nm, 2026-10-22: the sharp edge of clean thick flakes)
            x[fs.index(k)] = np.nan
    for k, v in (d.get("cap") or {}).items():       # (raw features limited from above: graphite pick, 2026-10-22)
        if k in fs:
            x[fs.index(k)] = min(x[fs.index(k)], v)
    x = np.where(np.isfinite(x), x, np.array(d["fill"]))
    zs = (x - np.array(d["mean"])) / np.array(d["std"])
    if d.get("clip"):                               # (standardised features clipped to +-clip: graphite pick, 2026-10-22)
        zs = np.clip(zs, -d["clip"], d["clip"])
    z = float(zs @ np.array(d["coef"]) + d["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


class Screen:
    """The screening of a material's 100x images (profiles/screen100_<material>.json; ok = False: no profile).

    p_quality: P(useful or maybe) from the model fitted on the 100x labels; below reject_below a flake is dropped
    before the designer.  p: the score shown in the designer and used for its tiers: the "pick" model fitted on the
    designer's keep / discard decisions when the profile has one (tools/fit_screen100.py fit-decisions), else
    p_quality.  tier: "reject" (p_quality < reject_below), else "keep" (p >= keep_above) or "review".
    A pick model with "inputs" (hBN, tools/fit_screen100.py fit-defects) scores a run's flakes together from p_quality
    and the terrace's defects in the 100x picture (flakeid.defects100.score_flakes, when the designer bundle is made:
    tier "reject" there keeps the flake in the designer, as screened out); for one image p and its tier stay
    p_quality's.  A pick model on the image's features with its own reject_below (graphite, 2026-10-22): p below it
    is tier "reject" too, but the flake stays in the designer, as screened out (keep() true: only the label model's
    gate and thick yellow graphite drop a flake)."""

    def __init__(self, material):
        p = PROFILE_DIR / f"screen100_{material}.json"
        self.ok = p.is_file()
        if self.ok:
            d = json.loads(p.read_text())
            self.d = d
            self.threshold = d["threshold"]
            self.features = d.get("features") or FEATURES
            self.pick = d.get("pick")
            # (a pick model on the defects scores whole runs, flakeid.defects100: not one image's features)
            self.pick_f = self.pick if self.pick and not self.pick.get("inputs") else None
            # tiers: below reject_below the flake is clearly not useful (dropped); at or above keep_above it is kept;
            # in between it goes to the user's review (designer).  Profiles without tiers: keep / reject at threshold.
            self.reject_below = d.get("reject_below", self.threshold)
            self.keep_above = (self.pick_f or {}).get("keep_above", d.get("keep_above", self.threshold))
            self.screen_below = (self.pick_f or {}).get("reject_below")     # (screened out, shown in the designer)

    def p_quality(self, f):
        return _logistic(self.d, f)

    def p(self, f):
        return _logistic(self.pick_f, f) if self.pick_f else self.p_quality(f)

    def dropped(self, f):
        """never in the designer: thick yellow graphite only.  (2026-11-11, Avram: a flake must be very bad to be discarded
        automatically, discarding by hand is quick: the label model's gate, p_quality below reject_below, dropped 60 of 94
        flakes of BN7_run_1 before anyone saw them; such flakes are now in the designer, automatically discarded only
        when their keep probability is below the pick model's reject_below, 0.03)"""
        return bool(f.get("thick_yellow"))

    def tier(self, f):
        if self.dropped(f):                         # (thick yellow graphite: out of plane, never usable)
            return "reject"
        p = self.p(f)
        if self.screen_below is not None and p < self.screen_below:
            return "reject"
        return "keep" if p >= self.keep_above else "review"

    def keep(self, f):
        """in the designer (tier "reject" by the pick model's reject_below: there, as screened out)"""
        return not self.dropped(f)
