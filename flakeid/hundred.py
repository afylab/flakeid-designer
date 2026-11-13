"""100x images (and stacking-domain images) into a substrate's viewer, then the TIFFs deleted.

    python -m flakeid.hundred <piece>_100x --viewer <results>/<piece>_tiles_viewer [--flat flat_100x.npz] [--keep-tiles]

Every 100x frame of mm_chip_scan.bsh (<piece>_100x/<piece>_100x_index.csv + TIFFs) is demosaiced, dark / flat
corrected and white-balanced so the bare substrate has the colour of the demo 100x images of the material (as on
the scoring pages); the white point is twice that scale (the whole range of the scoring pages' default), so only
nearly clipped pixels are white.  Written as <viewer>/100x/<file>.jpg (full resolution) and a 480 px preview.

Stacking-domain images (<piece>_domains/, graphite) are kept losslessly as 16-bit PNG (the domain contrast is ~1 %,
which 8-bit JPEG can erase) plus a JPEG preview with the illumination flattened and the contrast stretched.

Each image is matched to its flake in the viewer (the flake number in the 100x label, checked against the stage
position; nearest flake otherwise), the viewer page is updated (selecting a flake shows its images), and the TIFFs
are deleted once every JPEG / PNG is written, unless --keep-tiles.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from .look import look

# substrate colour (8-bit display) of the demo 100x images: hBN on 285 nm, graphite on 90 nm SiO2
DEMO_SUBSTRATE = {"hbn": (220.0, 151.0, 192.0), "graphite": (180.0, 147.5, 163.0)}
WHITE = 510.0                     # display value shown as white (2 x the demo scale)
# per material: graphite chips are shown brighter (substrate ~120 of 255 instead of 90): graphene is darker than the
# substrate, and the extra levels keep its faint contrast from being flattened into 8 x 8 blocks by the JPEG
# (2026-10-04); thick graphite (~2x the substrate on 90 nm oxide) still stays below white
WHITE_BY_MATERIAL = {"hbn": 510.0, "graphite": 380.0}
OFFSET_100X = (24.0, -18.3)       # 10x -> 100x stage offset (flakeid.live --offset default)
LABEL_MATCH_UM = 250.0            # a 100x picture goes to the flake its label names when within this of it (attach)
RATIO_100X = 9.54                 # 100x pixels per 10x pixel (BX51 + XC10)
# debris on the terrace of interest
TERRACE_TOL = 0.04                # terrace: within 4 % of the centre colour (flattened, 1 px smoothed)
DEBRIS_DEV = 0.025                # debris: colour distance from the terrace above 2.5 % ...
DEBRIS_DOG = 0.012                # ... or fine detail (difference of Gaussians 1.5-10 px) above 1.2 %
DEBRIS_MIN_PX = 8                 # smaller spots are ignored (8 px = 0.1 um^2, about the 100x resolution)
CLEAN_MARGIN_UM = 0.5             # clean area: terrace minus debris grown by this
STEP_MIN_UM2 = 10.0               # a deviating area at least this large is a step / other thickness, not debris
EDGE_EXCLUDE_UM = 1.0             # debris is not looked for this close to the terrace edge (the edge is ~1 um wide at 100x)
# terrace seed when the frame centre is bare substrate (an off-centre 100x shot)
SEED_SUB_TOL = 0.025              # centre colour within 2.5 % of the substrate: the centre is bare substrate
SEED_FLAKE_TOL = 0.04             # flake: colour more than 4 % from the substrate (smoothed)
SEED_MIN_UM2 = 80.0               # smallest flake the seed moves to
SEED_AREA_LO, SEED_AREA_HI = 0.2, 4.0   # a 100x terrace fits the 10x one within these area ratios (the 10x outline
                                        # overstates thin, long flakes: 100x / 10x 0.4-1.3 on normal frames)
SEED_DS = 4                       # the search runs at 1/4 scale


def _M100():
    from .mm import DEFAULT_M
    return np.asarray(DEFAULT_M, float) / RATIO_100X


def _flatten(r, tol=0.06, sigma=60.0, k=8):
    """r / its smooth background, estimated from pixels close to the terrace colour only (slow illumination
    changes across the terrace removed).  The background varies over tens of um, so it is computed on a k x smaller
    copy and scaled back up (identical to the full-resolution result to ~0.01 %, about 50x faster)."""
    h, w = r.shape[:2]
    m = (np.abs(r - 1.0).max(axis=2) < tol).astype(np.float32)
    sw, sh = max(1, w // k), max(1, h // k)
    rs = cv2.resize(r * m[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    s1, s4 = sigma / k, 4 * sigma / k
    num = cv2.GaussianBlur(rs, (0, 0), s1)
    den = cv2.GaussianBlur(ms, (0, 0), s1)
    wide = cv2.GaussianBlur(rs, (0, 0), s4) / np.maximum(cv2.GaussianBlur(ms, (0, 0), s4), 1e-6)[..., None]
    a = np.clip(den / 0.3, 0, 1)[..., None]
    bg = a * num / np.maximum(den, 1e-6)[..., None] + (1 - a) * wide
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return r / np.where(np.isfinite(bg) & (bg > 0.2), bg, 1.0)


def _u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _hue_match(c, ref):
    """c is bare substrate of colour ref seen brighter or a little darker (2026-10-27): the same hue within 2 %, 3 %
    darker to 15 % brighter.  Frames, and the parts of a frame, differ in brightness (illumination, vignetting,
    residue haze over the field): a frame 8 % brighter than the run, its centre 4 % brighter again, made bare substrate
    a flake (Avram's "66 nm" picture of residue on substrate; the simulated frames).  No flake passes: at 100x every
    hBN of 1 nm or more shifts the hue by 3 % or more, and graphene that keeps the hue is darker (monolayer: -5 % on
    300 nm oxide, -10 % on 85 nm)."""
    c, ref = np.asarray(c, float), np.maximum(np.asarray(ref, float), 1e-6)
    b = float(c.mean() / ref.mean())
    hue = float(np.abs((c / max(c.mean(), 1e-6)) / (ref / ref.mean()) - 1.0).max())
    return 0.97 <= b <= 1.15 and hue < 0.02


def _same_substrate(fm, sub):
    """A frame's commonest colour fm is the bare substrate of the run's colour sub: within 6 % (as before), or the
    same hue a little brighter (_hue_match)."""
    fm, sub = np.asarray(fm, float), np.maximum(np.asarray(sub, float), 1e-6)
    return float(np.abs(fm / sub - 1.0).max()) < 0.06 or _hue_match(fm, sub)


def _centre_is_substrate(lin, sub, win=90):
    """(centre is bare substrate, the substrate colour of this frame).  The frame's own commonest colour stands for
    the substrate while it is the run's substrate (_same_substrate: frames differ in brightness)."""
    h, w = lin.shape[:2]
    tc, _ = _mode_rgb(lin[h // 2 - win:h // 2 + win, w // 2 - win:w // 2 + win], step=3)
    fm, ffrac = _mode_rgb(lin[::4, ::4], step=3)
    subf = fm if _same_substrate(fm, sub) and ffrac > 0.2 else sub
    return bool(float(np.abs(tc / np.maximum(subf, 1.0) - 1.0).max()) < SEED_SUB_TOL or _hue_match(tc, subf)), subf


def _flakes_near_centre(lin, subf, min_um2, n_max=3):
    """The flakes (colour more than SEED_FLAKE_TOL from the substrate, at least min_um2) nearest the frame centre:
    masks at 1/SEED_DS scale, nearest first."""
    h, w = lin.shape[:2]
    k = SEED_DS
    small = cv2.GaussianBlur(cv2.resize(lin, (w // k, h // k), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    fg = (np.abs(small / np.maximum(subf, 1.0) - 1.0).max(axis=2) > SEED_FLAKE_TOL).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    a_um2 = (float(np.sqrt(abs(np.linalg.det(_M100())))) * k) ** 2
    yy, xx = np.mgrid[0:lab.shape[0], 0:lab.shape[1]]
    d0 = np.hypot(yy - h / 2 / k, xx - w / 2 / k)
    # (a region of the substrate's own hue, a little brighter, is substrate: _hue_match)
    comps = sorted((float(d0[lab == i].min()), i) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] * a_um2 >= min_um2
                   and not _hue_match(np.median(small[lab == i], axis=0), subf))
    return [(lab == i).astype(np.uint8) for _, i in comps[:n_max]]


def _terrace_at(lin, tc, cy, cx, win=90):
    """(r, rf, terrace mask) of the connected area of colour tc around (cy, cx), holes (debris, bubbles) filled."""
    h, w = lin.shape[:2]
    r = cv2.GaussianBlur(lin / np.maximum(tc, 1.0), (0, 0), 1.0)      # 1 px: below the 100x resolution, removes sensor noise
    rf = _flatten(r)
    m0 = (np.abs(rf - 1.0).max(axis=2) < TERRACE_TOL).astype(np.uint8)
    m0 = cv2.morphologyEx(m0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(m0, connectivity=8)
    c = lab[max(cy - win, 0):min(cy + win, h), max(cx - win, 0):min(cx + win, w)]
    ids, cnt = np.unique(c[c > 0], return_counts=True)
    if not len(ids):
        return r, rf, None
    T = (lab == ids[cnt.argmax()]).astype(np.uint8)
    T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # holes: the pieces of the rest that do not reach the frame edge.  (A flood fill from the corner pixel filled the
    # whole frame when the terrace covered that corner, and the far side of a flake that crosses the frame: substrate
    # counted as terrace made large flakes look uneven and dirty, 2026-10-14.)
    nb, bl = cv2.connectedComponents((T == 0).astype(np.uint8), connectivity=4)
    edge = np.unique(np.concatenate([bl[0], bl[-1], bl[:, 0], bl[:, -1]]))
    return r, rf, (T.astype(bool) | ((bl > 0) & ~np.isin(bl, edge))).astype(np.uint8)


def analyse(lin, substrate=None, expect=None):
    """Boosted renderings of one 100x frame and the debris on its centre terrace.

    lin: linear RGB (dark subtracted, flat corrected).  Returns (colour8 RGB, signgray, regions): colour8 = each
    channel stretched to the terrace colour +-8 %; signgray = flattened green vs the terrace +-4 % (thicker darker);
    regions = the terrace of interest and its debris outlines (frame pixels) and the clean-area figures.

    substrate: the run's bare-substrate colour in lin's units; expect: the target terrace at 10x, dict(area_um2=,
    contrast=(r, g, b)).  With them, an off-centre shot is handled: when the centre is bare substrate, or its terrace's
    size or colour does not fit the 10x terrace, the terraces of the flakes nearest the centre are tried and the one
    that fits best is taken (size, then colour: the 100x contrast is ~0.75 x the 10x one, then distance from the
    centre).  An off-centre shot otherwise made a terrace of the substrate or of a neighbour, and the screen judged
    that instead of the flake (2026-10-14: 214 of 1574 frames had bare substrate at the centre).  regions["seed"]:
    where the terrace was taken; found = False: no flake that fits is in the frame (the 100x shot missed it)."""
    px_um = float(np.sqrt(abs(np.linalg.det(_M100()))))
    a_px = px_um ** 2
    h, w = lin.shape[:2]
    win = 90
    exp_a = float((expect or {}).get("area_um2") or 0.0)
    exp_c = (expect or {}).get("contrast")
    sub = None if substrate is None else np.maximum(np.asarray(substrate, np.float32), 1.0)

    def cost(T, tc, cy, cx, moved):
        """(how badly a terrace fits the 10x one (0: perfectly), its size is plausible, its colour is plausible)."""
        area = float(T.sum()) * a_px if T is not None else 0.0
        c_size = abs(np.log(max(area, 1.0) / exp_a)) if exp_a else 0.0
        c_col, strong = 0.0, True
        ref = subf if subf is not None else sub
        if exp_c is not None and ref is not None:
            ce = 0.75 * np.asarray(exp_c, float)
            cc = np.asarray(tc, float) / ref - 1.0
            g_exp, g = float(ce[1]), float(cc[1])
            c_col = min(abs(g - g_exp) / (0.1 + abs(g_exp)), 2.0)
            # (2026-10-27) the channel the 10x colour changes most must change the same way, by at least 35 % of it:
            # residue haze or a brightness change of the substrate (a few % to -15 %) passed for a 66 nm flake
            k = int(np.argmax(np.abs(ce)))
            strong = abs(ce[k]) < 0.1 or (cc[k] * ce[k] > 0 and abs(cc[k]) >= 0.35 * abs(ce[k]))
        size_ok = not exp_a or SEED_AREA_LO <= area / exp_a <= SEED_AREA_HI
        return (c_size + c_col + np.hypot(cy - h / 2, cx - w / 2) * px_um / 60.0 + (0.1 if moved else 0.0), size_ok,
                c_col < 1.0 and (strong or not moved))      # (the centre's own terrace: as before)

    subf = None
    centre_sub, subf = (False, None) if sub is None else _centre_is_substrate(lin, sub, win)
    best, found = None, True
    if not centre_sub:
        cy, cx = h // 2, w // 2
        tc = _mode_rgb(lin[cy - win:cy + win, cx - win:cx + win], step=3)[0]
        r, rf, T = _terrace_at(lin, tc, cy, cx, win)
        cst, size_ok, col_ok = cost(T, tc, cy, cx, False)
        ok = size_ok and col_ok
        best = (cst, cy, cx, tc, False, r, rf, T, ok)
        if ok or not (exp_a or exp_c is not None):
            sub = None                              # (a centre terrace that fits is kept, as before: no search)
    if sub is not None:
        k = SEED_DS
        min_a = max(SEED_MIN_UM2, SEED_AREA_LO * exp_a) if exp_a else SEED_MIN_UM2
        loose = None                                # the best one whose colour fits, whatever its size
        for comp in _flakes_near_centre(lin, subf, min_a):
            m = cv2.erode(comp, np.ones((3, 3), np.uint8))
            for _ in range(3):                      # the flake's terraces, largest interior first
                if m.sum() * (px_um * k) ** 2 < SEED_MIN_UM2:
                    break
                dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
                y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
                sy, sx = int(y * k + k // 2), int(x * k + k // 2)
                y0, y1, x0, x1 = max(sy - win, 0), min(sy + win, h), max(sx - win, 0), min(sx + win, w)
                mw = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)[y0:y1, x0:x1] > 0
                px = lin[y0:y1, x0:x1][mw]
                if len(px) < 50:
                    break
                tc = _mode_rgb(px.reshape(-1, 1, 3), step=3)[0]
                r, rf, T = _terrace_at(lin, tc, sy, sx, win)
                cst, size_ok, col_ok = cost(T, tc, sy, sx, True)
                if size_ok and col_ok and (best is None or not best[8] or cst < best[0]):
                    best = (cst, sy, sx, tc, True, r, rf, T, True)
                elif col_ok and T is not None and (loose is None or cst < loose[0]):
                    loose = (cst, sy, sx, tc, True, r, rf, T, False)
                if T is None:
                    break
                Ts = cv2.resize(T, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = (m.astype(bool) & ~cv2.dilate(Ts, np.ones((5, 5), np.uint8)).astype(bool)).astype(np.uint8)
        if best is None and loose is not None:      # (centre on bare substrate: a flake of the right colour is there,
            best = loose                            # but its size does not fit the 10x terrace)
        if best is None or (centre_sub and not best[4]):
            found = False
    if best is None:                                # bare substrate and no flake that fits: the centre, as before
        cy, cx = h // 2, w // 2
        tc = _mode_rgb(lin[cy - win:cy + win, cx - win:cx + win], step=3)[0]
        r, rf, T = _terrace_at(lin, tc, cy, cx, win)
        best = (0.0, cy, cx, tc, False, r, rf, T, False)
        found = False
    _, cy, cx, tc, moved, r, rf, T, fits = best
    c8 = _u8((r - 0.92) / 0.16)
    sg = _u8(0.5 + (rf[..., 1] - 1.0) / 0.08)
    dev = np.sqrt(((rf - 1.0) ** 2).mean(axis=2))
    L = rf.mean(axis=2)
    dog = cv2.GaussianBlur(L, (0, 0), 1.5) - cv2.GaussianBlur(L, (0, 0), 10.0)
    regions = dict(terrace=[], debris=[], steps=[], terrace_um2=0.0, debris_n=0, debris_um2=0.0, steps_um2=0.0, clean_um2=0.0, clean_d_um=0.0,
                   cut_by_frame=False, terrace_rgb=[round(float(v), 1) for v in tc],
                   seed=dict(x=int(cx), y=int(cy), moved=bool(moved), found=bool(found), fits=bool(fits) if (exp_a or exp_c is not None) else None,
                             centre_substrate=bool(centre_sub),
                             expect_um2=round(exp_a, 1) if exp_a else None,
                             offset_um=[round((cx - w / 2) * px_um, 1), round((cy - h / 2) * px_um, 1)]))
    if T is None:
        return c8, sg, regions
    e = 2 * int(round(EDGE_EXCLUDE_UM / px_um)) + 1
    inner = cv2.erode(T, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))   # not the blurred terrace edge
    # thresholds at least DEBRIS_DEV / DEBRIS_DOG, raised to ~5 noise sigma on noisy frames (dark graphite terraces,
    # single frames without the 3-position median): noise must not count as debris
    ii = inner > 0
    if ii.sum() > 200:
        mad = lambda v: 1.4826 * float(np.median(np.abs(v - np.median(v))))          # noqa: E731
        dev_thr = max(DEBRIS_DEV, float(np.median(dev[ii])) + 5.0 * mad(dev[ii]))
        dog_thr = max(DEBRIS_DOG, 5.0 * mad(dog[ii]))
    else:
        dev_thr, dog_thr = DEBRIS_DEV, DEBRIS_DOG
    regions.update(noise=dict(dev_thr=round(dev_thr, 4), dog_thr=round(dog_thr, 4)))
    deb = (((dev > dev_thr) | (np.abs(dog) > dog_thr)) & (inner > 0)).astype(np.uint8)
    deb = cv2.morphologyEx(deb, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, dl, dst, _ = cv2.connectedComponentsWithStats(deb, connectivity=8)
    a_px = px_um ** 2
    smin = STEP_MIN_UM2 / a_px
    big = [k for k in range(1, n) if DEBRIS_MIN_PX <= dst[k, cv2.CC_STAT_AREA] < smin]
    stp = [k for k in range(1, n) if dst[k, cv2.CC_STAT_AREA] >= smin]
    keep = np.isin(dl, big).astype(np.uint8)
    steps = np.isin(dl, stp).astype(np.uint8)
    grow = max(1, int(round(CLEAN_MARGIN_UM / px_um)))
    clean = (T.astype(bool) & ~cv2.dilate(keep | steps, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)).astype(bool)).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(clean, 1), cv2.DIST_L2, 5)
    from .outline import polygons
    regions.update(terrace=polygons(T, 0, 0, 1.0), debris=polygons(keep, 0, 0, 0.7, min_area=2.0), steps=polygons(steps, 0, 0, 1.0),
                   terrace_um2=round(float(T.sum()) * a_px, 1), debris_n=len(big), debris_um2=round(float(keep.sum()) * a_px, 2),
                   steps_um2=round(float(steps.sum()) * a_px, 1),
                   clean_um2=round(float(clean.sum()) * a_px, 1), clean_d_um=round(2.0 * float(dist.max()) * px_um, 1),
                   cut_by_frame=bool(T[0].any() or T[-1].any() or T[:, 0].any() or T[:, -1].any()))
    return c8, sg, regions


def _align_gray(lin):
    """log brightness with slow variations removed: flake edges and texture, for alignment."""
    g = np.log(np.maximum(lin.mean(axis=2), 1.0)).astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 25)


def frame_shift(ref, other, margin=200):
    """Pixel displacement (dx, dy) of other's content relative to ref and the match score: the central part of ref
    (inside `margin` px) template-matched over other (normalised correlation of _align_gray), sub-pixel by a
    parabola through the peak.  Frames from mm_chip_scan.bsh are shifted ~10 um (~90 px) at 100x."""
    A, B = _align_gray(ref), _align_gray(other)
    h, w = A.shape
    tpl = A[margin:h - margin, margin:w - margin]
    res = cv2.matchTemplate(B, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, (x, y) = cv2.minMaxLoc(res)

    def sub(v0, v1, v2):
        d = v0 - 2 * v1 + v2
        return 0.5 * (v0 - v2) / d if abs(d) > 1e-9 else 0.0
    fx = sub(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < res.shape[1] - 1 else 0.0
    fy = sub(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < res.shape[0] - 1 else 0.0
    return (x + fx - margin, y + fy - margin), float(mx)


SHIFT_MIN_SCORE = 0.9          # a shifted frame joins the median only if it matches the frame this well


def shift_median(lin0, shifted, min_score=SHIFT_MIN_SCORE):
    """(kept name) shift_combine(..., method="median")."""
    return shift_combine(lin0, shifted, "median", min_score)


def shift_combine(lin0, shifted, method="mean", min_score=SHIFT_MIN_SCORE, weights=None):
    """The frame and the same flake at shifted stage positions, aligned on the flake and combined.  method "mean":
    weighted mean (weights: frames averaged into each, e.g. 4 for the main frame and 1 for each shift; noise falls as
    1/sqrt(total frames), optics artifacts fixed in the frame are spread out over the three positions); "median":
    optics artifacts land in a different place in each aligned frame and drop out (less noise reduction).

    shifted: [(linear frame, (dx, dy) stage um)].  The displacement of each shifted frame is measured (frame_shift),
    not taken from the commanded stage move: the stage lands within ~10 % of it and the image moves opposite to
    what the 100x pixel -> stage matrix gives.  (Until 2026-10-03 the commanded shift was assumed and a phase
    correlation refinement failed on most frames, so frames ~90-175 px apart were combined: ghost edges.)
    The median is formed only when every shifted frame aligns (score >= min_score), else the frame is returned
    unchanged.  Pixels not covered by all frames keep the first frame.
    Returns (frame, [(dx, dy, score) per shifted frame])."""
    if not shifted:
        return lin0, []
    h, w = lin0.shape[:2]
    stack, valid, info = [lin0], [np.ones((h, w), np.float32)], []
    for lin, _d in shifted:
        (dx, dy), sc = frame_shift(lin0, lin)
        info.append((round(dx, 2), round(dy, 2), round(sc, 3)))
        if sc < min_score:
            continue
        A = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stack.append(cv2.warpAffine(lin, A, (w, h), flags=cv2.INTER_LINEAR, borderValue=0))
        valid.append(cv2.warpAffine(np.ones((h, w), np.float32), A, (w, h), flags=cv2.INTER_NEAREST, borderValue=0))
    if len(stack) < len(shifted) + 1 or len(stack) < 3:
        return lin0, info
    allv = np.prod(np.stack(valid), axis=0) > 0.5
    if method == "median":
        comb = np.median(np.stack(stack), axis=0).astype(np.float32)
    else:
        w = np.asarray(weights if weights is not None and len(weights) == len(stack) else [1.0] * len(stack), np.float32)
        comb = (np.tensordot(w / w.sum(), np.stack(stack), axes=1)).astype(np.float32)
    return np.where(allv[..., None], comb, lin0), info


PATTERN_MIN_FRAMES = 20          # static_pattern: fewer frames give no pattern


def _pattern_terms(lin):
    """One frame's part of static_pattern: its terrace-relative deviation where it is terrace (else 0), and that mask."""
    h, w = lin.shape[:2]
    tc, _ = _mode_rgb(lin[h // 2 - 90:h // 2 + 90, w // 2 - 90:w // 2 + 90], step=3)
    rf = _flatten(cv2.GaussianBlur(lin / np.maximum(tc, 1.0), (0, 0), 1.0))
    m = (np.abs(rf - 1.0).max(axis=2) < TERRACE_TOL).astype(np.float32)
    return (rf - 1.0) * m[..., None], m


def static_pattern(frames, min_frames=PATTERN_MIN_FRAMES, min_cover=5):
    """Optics artifacts (dust in the light path, camera specks) from a run's own 100x frames: every frame is centred
    on a different flake, so what stays at the same pixel in all of them is not on the sample.  The terrace-relative
    brightness of each frame (flattened, terrace pixels only) is averaged per pixel and channel.  Returns a (H, W, 3)
    factor to divide the frames by (1 where too few terraces cover a pixel), or None for small runs.

    On bigbntest_2 (72 frames) the pattern (soft dark blobs and a lattice of small specks, 0.7 % rms) accounted
    for most of the texture on a terrace (single frames correlate 0.78 with it)."""
    acc = cnt = None
    n = 0
    for lin in frames:                                          # (an iterator: one frame in memory at a time)
        n += 1
        d, m = lin if isinstance(lin, tuple) else _pattern_terms(lin)      # (or the frame's _pattern_terms already)
        if acc is None:
            acc, cnt = np.zeros_like(d), np.zeros(m.shape, np.float32)
        acc += d
        cnt += m
    if n < min_frames or acc is None:
        return None
    avg = acc / np.maximum(cnt, 1.0)[..., None]
    wgt = np.clip((cnt - min_cover + 1) / 4.0, 0, 1)[..., None]       # fade out where few terraces cover a pixel
    return (1.0 + avg * wgt).astype(np.float32)


def to_stage(polys, stage_xy, frame_wh, offset=OFFSET_100X):
    """Frame-pixel polygons -> stage um in the 10x frame (the viewer's and the designer's coordinates)."""
    M = _M100()
    c = np.array(frame_wh, float) / 2.0
    base = np.asarray(stage_xy, float) - np.asarray(offset)
    return [np.round((np.array(p).reshape(-1, 2) - c) @ M.T + base, 2).ravel().tolist() for p in polys]


def _viewer_oxide(vdir: Path):
    """The analysis's oxide thickness from a chip viewer's data (const D = {...}), or None."""
    try:
        h = (Path(vdir) / "index.html").read_text()
        i = h.index("const D=") + 8
        return float(json.loads(h[i:h.index(";\n", i)]).get("oxide"))
    except Exception:
        return None


def _thickness(row):
    for k in ("thickness_nm",):
        try:
            return float(row[k])
        except (KeyError, TypeError, ValueError):
            pass
    m = re.search(r"_(\d+(?:\.\d+)?)nm", row.get("label") or row.get("file") or "")
    return float(m.group(1)) if m else 30.0


def overlay(rgb8, regions):
    """BGR copy of the natural image with the terrace of interest (white), its debris (magenta) and larger areas of
    another thickness (steps, cyan) outlined, and the clean area the 100x score measures (green; screen100.clean_score)."""
    o = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    for p in regions["terrace"]:
        cv2.polylines(o, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    for p in regions.get("steps", []):
        cv2.polylines(o, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    for p in regions["debris"]:
        cv2.polylines(o, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    for f in regions.get("fine", []):                       # fine terraces: yellow thicker, blue thinner (BGR)
        for p in f["poly"]:
            cv2.polylines(o, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], True,
                          (0, 230, 255) if f["dt_nm"] > 0 else (255, 160, 0), 2, cv2.LINE_AA)
    for p in (regions.get("score100") or {}).get("poly", []):
        cv2.polylines(o, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], True, (60, 220, 60), 3, cv2.LINE_AA)
    return o


def _mode_rgb(rgb, step=8):
    q = (np.clip(rgb, 0, 255 * step * 16 - 1) // step).astype(np.int64)
    k = (q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]).ravel()
    v, c = np.unique(k, return_counts=True)
    sel = k == v[c.argmax()]
    flat = rgb.reshape(-1, 3)
    m = flat[sel].mean(0)
    near = np.abs(flat - m).max(1) < 3 * step
    return np.median(flat[near], 0), float(near.mean())


def _material(folder: Path):
    from .mm import scan_params
    name = re.sub(r"_100x$", "", folder.name)
    sp = scan_params(folder.parent / f"{name}_tiles")
    return (sp or {}).get("material", "hbn")


# per-image work of export(), run in worker processes: the context (flat field, optics pattern, gains) is set once per
# process by _init_ctx, then each 100x frame is combined, screened and written by _write_one

DIAG_QUALITY = 90                # JPEG quality of the diagnostic views (debris, thickness change, sg, avg_sg, preview):
                                 # ~1/2.5 of the size at 100; the 100x pictures themselves (main, avg, c8, avg_c8) keep
                                 # export's quality (100)


def jpeg_params(quality):
    """JPEG settings for the 100x images: the given quality and full-resolution colour (4:4:4; OpenCV's default 4:2:0
    halves the colour resolution, which showed as coloured fringes and blocks in the high-contrast views)."""
    p = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        p += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    return p

_CTX = {}


def _init_ctx(ctx):
    from .mm import FlatField
    from . import screen100
    _CTX.clear()
    _CTX.update(ctx)
    _CTX["FF"] = FlatField.load(ctx["flat"]) if ctx.get("flat") else None
    _CTX["screen"] = screen100.Screen(ctx["material"])
    cv2.setNumThreads(1)


def _linear(raw):
    from .mm import BAYER_CODE
    rgb = cv2.cvtColor(raw, BAYER_CODE).astype(np.float32)
    FF = _CTX.get("FF")
    return FF.apply(rgb) if FF is not None else rgb - _CTX["dark"]


def _combined(r):
    """The flake's frame: with shifted frames (HUNDRED_SHIFTS_UM), combined with them aligned on the flake.  The optics
    pattern (dust in the light path, camera specks: fixed to the camera) is divided out of EACH frame before they are
    aligned on the flake; divided out of the combined frame instead, the shifted frames' copies of it stayed, offset by
    the shift: every speck three times (pleasework_1 flake 11, high-contrast view, 2026-10-04)."""
    folder, combine, pat = Path(_CTX["folder"]), _CTX["combine"], _CTX.get("pat")

    def load(name):
        lin = _linear(cv2.imread(str(folder / name), cv2.IMREAD_UNCHANGED))
        return lin / pat if pat is not None else lin
    lin0 = load(r["file"])
    _CTX["lin0"] = lin0                             # (the averaged main frame alone: also exported, _write_one)
    if combine == "none":                           # flat field + optics pattern only
        return lin0, []
    sh = []
    for f_, d_ in zip((r.get("shift_files") or "").split(";"), (r.get("shift_um") or "").split(";")):
        if f_ and (folder / f_).is_file() and ":" in d_:
            sh.append((load(f_), tuple(float(v) for v in d_.split(":"))))
    try:
        n0 = float(r.get("frames") or 1)            # frames averaged into the main image (flakeid.scope: hundred_frames)
    except ValueError:
        n0 = 1.0
    return shift_combine(lin0, sh, combine, weights=[n0] + [1.0] * len(sh))


def _read_sub(path, terms=False):
    """A frame's substrate colour (_mode_rgb) and, with terms, its _pattern_terms (computed here, in export's reading
    threads, rather than one frame after another in static_pattern)."""
    lin = _linear(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    return _mode_rgb(lin[::2, ::2]), (_pattern_terms(lin) if terms else None)


def _flake_label_id(label):
    m = re.search(r"_F(\d+)", label or "")
    return int(m.group(1)) if m else None


def dt_view(dt, terrace_polys, d8):
    """Thickness-change map in colour (+-1.5 nm), the rest of the frame dimmed (BGR, for the viewer and designer)."""
    dtv = np.clip(128 + dt / 1.5 * 127, 0, 255).astype(np.uint8)
    dtc = cv2.applyColorMap(dtv, cv2.COLORMAP_TWILIGHT_SHIFTED)
    T0 = np.zeros(dtv.shape, np.uint8)
    for q in terrace_polys:
        cv2.fillPoly(T0, [np.round(np.array(q).reshape(-1, 2)).astype(np.int32)], 1)
    dtc[T0 == 0] = (cv2.cvtColor(d8, cv2.COLOR_RGB2BGR)[T0 == 0] * 0.35).astype(np.uint8)
    return dtc


def screen_record(screen, feat, material):
    """The screening verdict stored with a 100x image (regions.json "screen")."""
    rec = dict(features={k: (round(v, 5) if isinstance(v, float) else v) for k, v in feat.items()},
               p_keep=None, keep=None, tier=None, model=None)
    if screen.ok:
        rec.update(p_keep=round(screen.p(feat), 4), p_quality=round(screen.p_quality(feat), 4), tier=screen.tier(feat),
                   model=f"screen100_{material}.json")
        rec["keep"] = screen.keep(feat)             # (tier "reject" with keep true: screened out, in the designer)
    if feat.get("thick_yellow"):
        rec.update(tier="reject", keep=False, reason="thick graphite: yellow at 100x (out of plane or far too thick)")
    return rec


def _flake_expect(vdir):
    """{flake id: dict(area_um2=, contrast=, t=, alt=, tid=)} of each flake's target terrace at 10x, from the viewer
    page: what the 100x terrace is looked for as (analyse), its thickness and other readings (branch100), or {}."""
    try:
        html = (Path(vdir) / "index.html").read_text(encoding="utf-8")
        m = re.search(r"const D=(\{.*?\});\n", html, re.S)
        if not m:
            return {}
        D = json.loads(m.group(1))
        T = D.get("terraces") or {}
        out = {}
        for f in D["flakes"]:
            terr = f.get("terr") or []
            t = (T.get(str(terr[0])) if terr else None) or {}
            out[int(f["id"])] = dict(area_um2=t.get("area") or f.get("area"), contrast=t.get("c"), t=t.get("t"), alt=t.get("alt"),
                                     tid=terr[0] if terr else None)
        return out
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return {}


# the 10x colour of an hBN terrace can fit two thicknesses about equally (e.g. ~25 and ~156 nm on ~295 nm oxide:
# "two_branch"); at 100x (NA 0.9) they differ (red: -0.5 thin, -0.1 thick).  The 100x colour of the terrace picks the
# reading whose 100x model colour it is nearest, when that is clear: the nearest within BRANCH100_MAX_D and the other
# at least BRANCH100_RATIO x as far (20261027_run_6: 4 of 10 such flakes ~156 nm, read ~25 nm at 10x; Avram's F047)
BRANCH100_MAX_D = 0.30
BRANCH100_RATIO = 1.8
_BR_MODELS = {}


def branch100(lin, reg, material, oxide_nm, expect):
    """For an hBN terrace with more than one 10x thickness reading (its thickness t and alt, from the viewer): which
    reading its 100x colour fits (the median colour of the terrace against the frame's substrate, vs the thin-film model
    at 100x NA 0.9 and the run's oxide).  dict(t10, readings, d, chosen, decided, changed, contrast) or None."""
    if material != "hbn" or not expect or expect.get("t") is None or not expect.get("alt"):
        return None
    if (reg.get("seed") or {}).get("found") is False or not reg.get("terrace"):
        return None
    try:
        t10 = float(expect["t"])
        readings = [t10] + [float(v) for v in str(expect["alt"]).replace(",", ";").split(";") if v.strip()]
    except (TypeError, ValueError):
        return None
    h, w = lin.shape[:2]
    T = np.zeros((h, w), np.uint8)
    for p in reg["terrace"]:
        cv2.fillPoly(T, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], 1)
    if T.sum() < 2000:
        return None
    tc = np.median(lin[T > 0], axis=0)
    fm, ffrac = _mode_rgb(lin[::4, ::4], step=3)
    if ffrac < 0.15:
        return None                                   # (no clear substrate in the frame)
    c = np.asarray(tc, float) / np.maximum(np.asarray(fm, float), 1e-6) - 1.0
    key = round(float(oxide_nm), 1)
    if key not in _BR_MODELS:
        from .optics import ContrastModel
        from .thickness import camera_from_profile, load_profile
        _BR_MODELS[key] = ContrastModel("hbn", float(oxide_nm), na=0.9, camera=camera_from_profile(load_profile("bx51_xc10_mm_raw16")))
    m = _BR_MODELS[key]
    d = [float(np.linalg.norm(c - m.contrast([t])[0])) for t in readings]
    k = int(np.argmin(d))
    others = [x for i, x in enumerate(d) if i != k]
    decided = d[k] <= BRANCH100_MAX_D and all(x >= BRANCH100_RATIO * d[k] for x in others)
    chosen = readings[k] if decided else t10
    return dict(t10=t10, tid=expect.get("tid"), readings=readings, d=[round(x, 3) for x in d], chosen=chosen, decided=bool(decided),
                changed=bool(decided and k != 0), contrast=[round(float(v), 3) for v in c], oxide_nm=float(oxide_nm))


def _write_one(r):
    from . import screen100
    out, gain = Path(_CTX["out"]), _CTX["gain"]
    material, oxide_nm, quality, screen = _CTX["material"], _CTX["oxide_nm"], _CTX["quality"], _CTX["screen"]
    lin, resid = _combined(r)                       # (optics pattern divided out of each frame)
    white = WHITE_BY_MATERIAL.get(material, WHITE)
    d8 = np.clip(lin * gain * (255.0 / white) + 0.5, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(d8, cv2.COLOR_RGB2BGR)
    stem = Path(r["file"]).stem
    cv2.imwrite(str(out / f"{stem}.jpg"), bgr, jpeg_params(quality))
    # the copy people look at (flakeid.look: the reference picture's substrate colour); <stem>.jpg stays linear for the
    # analyses that read it back
    demo = np.asarray(DEMO_SUBSTRATE.get(material, DEMO_SUBSTRATE["hbn"]), float)
    lk = cv2.cvtColor(look(lin * gain, demo), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / f"{stem}_look.jpg"), lk, jpeg_params(quality))
    # the bare substrate in lin's units (the white balance maps it to the demo colour) and the flake's 10x area: an
    # off-centre shot's terrace is the flake nearest the centre (analyse / _seed)
    sub = np.asarray(DEMO_SUBSTRATE.get(material, DEMO_SUBSTRATE["hbn"]), float) / np.asarray(gain, float)
    expect = (_CTX.get("expect") or {}).get(_flake_label_id(r.get("label")))
    if resid:
        # the frame-averaged main image alone (no shifted frames), natural and high contrast: free of anything the
        # combination of the three positions may add (pleasework_1, 2026-10-04: marks repeated at the shifts)
        lin0 = _CTX["lin0"]
        a8 = np.clip(lin0 * gain * (255.0 / white) + 0.5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{stem}_avg.jpg"), cv2.cvtColor(a8, cv2.COLOR_RGB2BGR), jpeg_params(quality))
        cv2.imwrite(str(out / f"{stem}_avg_look.jpg"), cv2.cvtColor(look(lin0 * gain, demo), cv2.COLOR_RGB2BGR), jpeg_params(quality))
        ac8, asg, _ = analyse(lin0, sub, expect)
        cv2.imwrite(str(out / f"{stem}_avg_c8.jpg"), cv2.cvtColor(ac8, cv2.COLOR_RGB2BGR), jpeg_params(quality))
        cv2.imwrite(str(out / f"{stem}_avg_sg.jpg"), asg, jpeg_params(min(quality, DIAG_QUALITY)))
    # boosted contrast around the terrace of interest, and its debris (also in stage um, for the designer)
    # terrace, debris, fine terraces (thickness-change map), wrinkles and the screening verdict
    feat, reg, extra = screen100.features(lin, material, oxide_nm, _thickness(r), substrate=sub, expect=expect)
    b100 = branch100(lin, reg, material, oxide_nm, expect)
    if b100 is not None:
        reg["branch100"] = b100
    c8, sg = extra["c8"], extra["sg"]
    reg["fine"] = extra["fine"]
    reg["wrinkles"] = extra["wrinkles"]
    reg["screen"] = screen_record(screen, feat, material)
    if (reg.get("seed") or {}).get("fits") is False and reg["screen"]["tier"] is not None:
        # the terrace does not fit the 10x one: the screen may have judged the wrong thing, so the user decides
        reg["screen"].update(tier="review", keep=True)
    if "dt" in extra:
        cv2.imwrite(str(out / f"{stem}_dt.jpg"), dt_view(extra["dt"], reg["terrace"], d8), jpeg_params(min(quality, DIAG_QUALITY)))
    cv2.imwrite(str(out / f"{stem}_c8.jpg"), cv2.cvtColor(c8, cv2.COLOR_RGB2BGR), jpeg_params(quality))
    cv2.imwrite(str(out / f"{stem}_sg.jpg"), sg, jpeg_params(min(quality, DIAG_QUALITY)))
    cv2.imwrite(str(out / f"{stem}_debris.jpg"), overlay(d8, reg), jpeg_params(min(quality, DIAG_QUALITY)))
    sxy, wh = (float(r["stage_x_um"]), float(r["stage_y_um"])), (lin.shape[1], lin.shape[0])
    (out / f"{stem}_regions.json").write_text(json.dumps(dict(
        reg, label=r["label"], file=r["file"], stage_um=list(sxy), frame_wh=list(wh),
        px_um=float(np.sqrt(abs(np.linalg.det(_M100())))), offset_100x_um=list(OFFSET_100X),
        terrace_stage_um=to_stage(reg["terrace"], sxy, wh), debris_stage_um=to_stage(reg["debris"], sxy, wh),
        steps_stage_um=to_stage(reg["steps"], sxy, wh),
        fine_stage_um=[dict(f, poly=to_stage(f["poly"], sxy, wh)) for f in reg.get("fine", [])],
        oxide_nm=oxide_nm,
        shift_frames=len(resid), shift_alignment=resid,
        shift_used=bool(resid) and all(q[2] >= SHIFT_MIN_SCORE for q in resid),
        thresholds=dict(terrace_tol=TERRACE_TOL, debris_dev=DEBRIS_DEV, debris_dog=DEBRIS_DOG,
                        debris_min_px=DEBRIS_MIN_PX, step_min_um2=STEP_MIN_UM2, edge_exclude_um=EDGE_EXCLUDE_UM,
                        clean_margin_um=CLEAN_MARGIN_UM)), separators=(",", ":")))
    cv2.imwrite(str(out / "preview" / f"{stem}.jpg"), cv2.resize(bgr, (480, int(480 * bgr.shape[0] / bgr.shape[1])),
                                                                interpolation=cv2.INTER_AREA), jpeg_params(min(quality, DIAG_QUALITY)))
    cv2.imwrite(str(out / "preview" / f"{stem}_look.jpg"), cv2.resize(lk, (480, int(480 * lk.shape[0] / lk.shape[1])),
                                                                      interpolation=cv2.INTER_AREA), jpeg_params(min(quality, DIAG_QUALITY)))
    return stem


DOMAIN_VIEW_HALF = 0.08          # IR preview: the central flake's level +- this (contrast to the substrate); 2026-10-13:
                                 # -6.11 .. +7.32 % around a +0.5 % flake (grtest_2 F002) showed its stacking domains;
                                 # +-8 % chosen 2026-10-15


def domain_contrast(g, centre=None):
    """(contrast to the substrate, the central flake's level) of an IR picture: divided by a smooth fit of the
    substrate pixels (the frame's most common level, away from edges), so slow illumination left after the flat field
    goes too; the level is the median at the frame's centre (the scan centres the flake).  A flake that fills the frame
    is its own reference (level ~0).  centre: (x, y) px of the flake instead of the frame centre (a mosaic: the main
    picture's centre).  Pixels <= 0 (dead, or a mosaic's uncovered parts) are not used."""
    g = np.asarray(g, np.float32).copy()
    good = g > 0
    med = float(np.median(g[good])) if good.any() else 1.0
    g[~good] = med                                    # (dead border rows / columns)
    sm = cv2.GaussianBlur(g, (0, 0), 2)
    grad = cv2.morphologyEx(sm, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    subm = (np.abs(sm - med) < 0.04 * med) & (grad < np.percentile(grad, 60))
    w = cv2.GaussianBlur(subm.astype(np.float32), (0, 0), 40)
    bg = cv2.GaussianBlur(np.where(subm, sm, 0).astype(np.float32), (0, 0), 40) / np.maximum(w, 1e-3)
    bg[(w < 1e-3) | ~np.isfinite(bg) | (bg <= 0)] = med
    c = g / bg - 1.0
    h, wd = c.shape
    cs = cv2.GaussianBlur(c, (0, 0), 1.5)
    cx, cy = (wd // 2, h // 2) if centre is None else (int(round(centre[0])), int(round(centre[1])))
    lev = float(np.median(cs[max(cy - 15, 0):cy + 15, max(cx - 15, 0):cx + 15]))
    return c, lev


def domain_preview(g, half=DOMAIN_VIEW_HALF, centre=None):
    """The IR preview (8-bit): contrast to the substrate, lightly smoothed, over the central flake's level +- half."""
    c, lev = domain_contrast(g, centre)
    c = cv2.GaussianBlur(c, (0, 0), 1.0)
    return np.clip((c - (lev - half)) / (2 * half) * 255, 0, 255).astype(np.uint8), lev


def load_domain_flat(path, log=print):
    """(flat, dark or None) of the IR flat field .npz, or None if it cannot be read.  A dark in the file is not used:
    the IR camera's automatic level control reads a picture with the light blocked at another gain, and subtracting it
    put a band into every domain image (flat_domain_20261013_1522)."""
    try:
        z = np.load(str(path))
        dark = z["dark"] if "dark" in z.files and z["dark"].size else None
        log(f"IR flat field {path}" + (" (its dark picture is not used)" if dark is not None else ""))
        return z["flat"].astype(np.float32), (dark.astype(np.float32) if dark is not None else None)
    except Exception as e:
        log(f"IR flat field {path} not used ({type(e).__name__}: {e})")
        return None


IR_FIT_MIN = 0.45                # correlation of the two pictures' edge maps below which an IR picture is not laid over
                                 # the 100x one (good fits: 0.6-0.9, grtest / test_ir 2026-10-15)
IR_SCALES = (0.22, 0.6)          # IR pixels per 100x pixel searched (measured 0.358, the cameras turned 5.8 deg)


def _edges(g, sig):
    g = cv2.GaussianBlur(np.asarray(g, np.float32), (0, 0), sig)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (float(np.percentile(m, 99.5)) + 1e-6)


def ir_fit(c100, ir, prior=None):
    """(M, correlation) with M (2 x 3) taking 100x pixels to IR pixels, or None: the IR picture of a flake fitted to its
    100x picture on their edges (flake outlines, folds, debris look alike in both).  Searched over the IR picture's
    mirror images (a camera mounted mirrored), scales and positions, then refined (scale, rotation, shear, shift).
    prior: an M from other flakes of the run: only the shift is fitted (a flake with few edges)."""
    g100 = c100 if c100.ndim == 2 else cv2.cvtColor(c100, cv2.COLOR_BGR2GRAY)
    g100 = g100.astype(np.float32)
    irg = ir.astype(np.float32) if ir.ndim == 2 else cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY).astype(np.float32)
    irg = irg / max(float(np.median(irg)), 1e-6)
    hi, wi = irg.shape
    ds = 0.5                                       # the fit runs at half the 100x resolution
    T = _edges(cv2.resize(g100, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA), 2.0)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    def ecc(I, W, motion):
        try:
            cc, W = cv2.findTransformECC(T, I, W.astype(np.float32), motion, crit, None, 5)
            return float(cc), W
        except cv2.error:
            return None

    if prior is not None:
        P = np.asarray(prior, np.float32).reshape(2, 3).copy()
        P[:, :2] /= ds
        r = ecc(_edges(irg, 1.0), P, cv2.MOTION_TRANSLATION)
        if r is None:
            return None
        W = r[1].copy()
        W[:, :2] *= ds
        return W.astype(np.float64), r[0]
    best = None
    for fl in (None, 1, 0, -1):                    # as taken, mirrored left-right, top-bottom, both
        I = _edges(irg if fl is None else cv2.flip(irg, fl), 1.5)
        for s in np.exp(np.linspace(np.log(IR_SCALES[0]), np.log(IR_SCALES[1]), 36)):
            w, h = int(g100.shape[1] * s), int(g100.shape[0] * s)
            if w >= wi or h >= hi:
                continue
            t = _edges(cv2.resize(g100, (w, h), interpolation=cv2.INTER_AREA), 1.5)
            _, v, _, loc = cv2.minMaxLoc(cv2.matchTemplate(I, t, cv2.TM_CCOEFF_NORMED))
            if best is None or v > best[0]:
                best = (v, fl, s, loc)
    if best is None:
        return None
    _, fl, s, loc = best
    I = _edges(irg if fl is None else cv2.flip(irg, fl), 1.0)
    r = ecc(I, np.array([[s / ds, 0, loc[0]], [0, s / ds, loc[1]]]), cv2.MOTION_AFFINE)
    if r is None:
        return None
    W = np.vstack([r[1].astype(np.float64), [0, 0, 1]])
    W[:2, :2] *= ds
    F = np.eye(3)                                  # back to the IR picture as taken (a mirror is its own inverse)
    if fl in (1, -1):
        F[0] = [-1, 0, wi - 1]
    if fl in (0, -1):
        F[1] = [0, -1, hi - 1]
    return (F @ W)[:2], r[0]


def _ir_view(vdir, e, d, M):
    """Write the IR preview of domain entry d (from its 16-bit picture: the current display range) in the frame of
    100x entry e (its file name + _ir.jpg)."""
    c = cv2.imread(str(vdir / e["f"]), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    h, w = c.shape[0] * 2, c.shape[1] * 2
    raw = cv2.imread(str(vdir / d["raw"]), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    pv = domain_preview(raw.astype(np.float32) if raw.ndim == 2 else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32))[0]
    out = re.sub(r"\.jpg$", "_ir.jpg", e["f"])
    a = cv2.warpAffine(pv, np.asarray(M, np.float64), (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(str(vdir / out), a, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


def align_ir(vdir, H, log=print):
    """Lay each flake's IR (stacking-domain) picture over its 100x pictures: e["ir"] = the IR preview in the 100x frame
    (100x/<image>_ir.jpg), e["ir_m"] its fit (100x px -> IR px), e["ir_fit"] the correlation.  Flakes that fit poorly
    are fitted again with the run's typical scale and rotation (shift only); still poor: left without.  Returns the
    number of 100x pictures with an IR view."""
    vdir = Path(vdir)
    todo = []
    for k, v in H.items():
        doms = [x for x in v if x.get("domain") and x.get("raw")]
        if not doms:
            continue
        for e in v:
            if e.get("domain") or (e.get("ir") and (vdir / e["ir"]).is_file()):
                continue
            stem = Path(e["f"]).stem
            # the same flake's picture: the same name (before 2026-10-19), or the same <chip>_F<flake>_<thickness> (names
            # <..>_100x_<order> / <..>_IR_<order> since); else the latest
            key = re.sub(r"_(100x|IR)(_[A-Za-z0-9]+)?_\d{3}$", "", stem)
            d = next((x for x in doms if Path(x["f"]).stem == stem), None) or \
                next((x for x in doms if re.sub(r"_(100x|IR)(_[A-Za-z0-9]+)?_\d{3}$", "", Path(x["f"]).stem) == key), doms[-1])
            todo.append((e, d))
    if not todo:
        return sum(1 for v in H.values() for e in v if e.get("ir"))
    fits = []
    for e, d in todo:
        c = cv2.imread(str(vdir / e["f"]), cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(str(vdir / d["raw"]), cv2.IMREAD_UNCHANGED)
        fits.append(ir_fit(c, ir) if c is not None and ir is not None else None)
    good = [f[0] for f in fits if f and f[1] >= 0.6]
    prior = np.median(np.array(good), axis=0) if good else None
    n_ok, n_prior = 0, 0
    for (e, d), f in zip(todo, fits):
        if prior is not None and (f is None or f[1] < IR_FIT_MIN or np.abs(f[0][:, :2] - prior[:, :2]).max() > 0.02 * np.abs(prior[:, :2]).max()):
            c = cv2.imread(str(vdir / e["f"]), cv2.IMREAD_GRAYSCALE)
            ir = cv2.imread(str(vdir / d["raw"]), cv2.IMREAD_UNCHANGED)
            g = ir_fit(c, ir, prior=prior)
            if g is not None and g[1] >= IR_FIT_MIN * 0.8:
                f, n_prior = g, n_prior + 1
        if f is None or f[1] < IR_FIT_MIN * 0.8:
            continue
        out = _ir_view(vdir, e, d, f[0])
        if out:
            e.update(ir=out, ir_m=[round(float(x), 5) for x in np.ravel(f[0])], ir_fit=round(float(f[1]), 3))
            n_ok += 1
    if prior is not None:
        sc = float(np.sqrt(abs(np.linalg.det(prior[:, :2]))))
        rot = float(np.degrees(np.arctan2(prior[1, 0], prior[0, 0])))
        log(f"  IR pictures laid over the 100x ones: {n_ok} of {len(todo)} ({n_prior} with the run's typical fit; "
            f"{sc:.4f} IR px per 100x px, turned {rot:.1f} deg{', mirrored' if np.linalg.det(prior[:, :2]) < 0 else ''})")
    else:
        log(f"  IR pictures laid over the 100x ones: {n_ok} of {len(todo)} (no good fit to learn the run's typical one)")
    return sum(1 for v in H.values() for e in v if e.get("ir"))


MOSAIC_QUALITY = 95             # JPEG quality of the terrace mosaics (a 30-field mosaic is ~7000 x 4500 px)


def _mosaic_groups(path):
    """{main file: [rows]} of a mosaic index (flakeid.scope: <piece>_100x_mosaic.csv / <piece>_domains_mosaic.csv), each
    group in field order (k = 1: the main picture); the last group of a main file wins (a retake)."""
    G = {}
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("k") == "1":
                    G[r["main_file"]] = []
                G.setdefault(r["main_file"], []).append(r)
    except (OSError, ValueError):
        return {}
    return {k: sorted(v, key=lambda r: int(r["k"])) for k, v in G.items()}


def _mosaic_write_info(path, rows, res, M, files):
    H, W = res["weight"].shape
    px = float(np.sqrt(abs(np.linalg.det(M))))
    path.write_text(json.dumps(dict(main=rows[0]["main_file"], label=rows[0].get("label", ""), fields=files,
                                    stage_um=[[float(r["stage_x_um"]), float(r["stage_y_um"])] for r in rows if r["file"] in files],
                                    z_um=[float(r["z_um"]) if r.get("z_um") else None for r in rows if r["file"] in files],
                                    focus=[r.get("focus", "") for r in rows if r["file"] in files],
                                    um_per_px=np.asarray(M, float).tolist(), size_px=[W, H], size_um=[round(W * px, 1), round(H * px, 1)],
                                    pos_px=np.round(res["pos"], 2).tolist(), main_centre_px=[round(float(v), 1) for v in res["ref_xy"]],
                                    pairs_used=res["n_pairs_used"], pairs=len(res["pairs"]),
                                    gain=np.round(res["gain"], 4).tolist()), separators=(",", ":")))


def stitch_100x_mosaics(folder, out, mains, gain, material, quality=100, log=print):
    """Each flake's terrace mosaic (flakeid.scope: <piece>_100x_mosaic.csv, fields <main>_mNN.tif) stitched
    (flakeid.fieldmosaic.stitch) and written as <viewer>/100x/<main>_mosaic.jpg (+ preview, + _mosaic.json) with the main
    JPEG's look: the same dark / flat field, optics pattern and white balance (_CTX, gain).  mains: the main files
    exported now.  Returns {main file: [field files]} of the mosaics written."""
    from .fieldmosaic import parse_matrix, stitch
    mp = next(Path(folder).glob("*_100x_mosaic.csv"), None)
    done = {}
    if mp is None:
        return done
    white = WHITE_BY_MATERIAL.get(material, WHITE)
    pat = _CTX.get("pat")
    for main, rows in _mosaic_groups(mp).items():
        if main not in mains or len(rows) < 2:
            continue
        rows = [r for r in rows if (Path(folder) / r["file"]).is_file()]
        if len(rows) < 2 or rows[0]["k"] != "1":
            continue
        try:
            imgs = []
            for r in rows:
                lin = _linear(cv2.imread(str(Path(folder) / r["file"]), cv2.IMREAD_UNCHANGED))
                imgs.append((lin / pat if pat is not None else lin).astype(np.float32))
            M = parse_matrix(rows[0].get("um_per_px")) if rows[0].get("um_per_px") else None
            M = M if M is not None else _M100()
            res = stitch(imgs, [(float(r["stage_x_um"]), float(r["stage_y_um"])) for r in rows], M, ref=0)
            del imgs
            d8 = np.clip(res["image"] * gain * (255.0 / white) + 0.5, 0, 255).astype(np.uint8)
            d8[res["weight"] <= 1e-6] = 0
            bgr = cv2.cvtColor(d8, cv2.COLOR_RGB2BGR)
            stem = Path(main).stem
            cv2.imwrite(str(Path(out) / f"{stem}_mosaic.jpg"), bgr, jpeg_params(min(quality, MOSAIC_QUALITY)))
            cv2.imwrite(str(Path(out) / "preview" / f"{stem}_mosaic.jpg"),
                        cv2.resize(bgr, (480, max(1, int(480 * bgr.shape[0] / bgr.shape[1]))), interpolation=cv2.INTER_AREA),
                        jpeg_params(min(quality, DIAG_QUALITY)))
            # the copy people look at (flakeid.look), as the single pictures
            lk = look(res["image"] * gain, DEMO_SUBSTRATE.get(material, DEMO_SUBSTRATE["hbn"]))
            lk[res["weight"] <= 1e-6] = 0
            lk = cv2.cvtColor(lk, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(Path(out) / f"{stem}_mosaic_look.jpg"), lk, jpeg_params(min(quality, MOSAIC_QUALITY)))
            cv2.imwrite(str(Path(out) / "preview" / f"{stem}_mosaic_look.jpg"),
                        cv2.resize(lk, (480, max(1, int(480 * lk.shape[0] / lk.shape[1]))), interpolation=cv2.INTER_AREA),
                        jpeg_params(min(quality, DIAG_QUALITY)))
            _mosaic_write_info(Path(out) / f"{stem}_mosaic.json", rows, res, M, [r["file"] for r in rows])
            done[main] = [r["file"] for r in rows[1:]]
            log(f"  {stem}: mosaic of {len(rows)} fields ({res['n_pairs_used']} of {len(res['pairs'])} overlaps matched) -> {stem}_mosaic.jpg")
        except Exception as e:                       # (the main picture is exported all the same)
            log(f"  {Path(main).stem}: mosaic not stitched ({type(e).__name__}: {e})")
    return done


def stitch_ir_mosaics(ddir, out, ffd, log=print):
    """The IR pictures' terrace mosaics (<piece>_domains_mosaic.csv): stitched as the 100x ones (the IR camera's own
    matrix, its mirroring included), IR flat field divided out, written as <viewer>/100x/domains/<IR main>_mosaic.png
    (16-bit) + _mosaic.jpg (domain_preview's +-8 % around the flake's level at the main picture's centre, from the
    mosaic's own substrate).  Returns {main file: [field files]} of the mosaics written."""
    from .fieldmosaic import parse_matrix, stitch
    mp = next(Path(ddir).glob("*_domains_mosaic.csv"), None)
    done = {}
    if mp is None:
        return done
    for main, rows in _mosaic_groups(mp).items():
        rows = [r for r in rows if (Path(ddir) / r["file"]).is_file()]
        if len(rows) < 2 or rows[0]["k"] != "1":
            continue
        stem = Path(main).stem
        try:
            imgs = []
            for r in rows:
                a = cv2.imread(str(Path(ddir) / r["file"]), cv2.IMREAD_UNCHANGED)
                g = a.astype(np.float32) if a.ndim == 2 else a.astype(np.float32).mean(axis=2)
                if ffd is not None and g.shape == ffd[0].shape:
                    g = g / np.maximum(ffd[0], 0.05)
                imgs.append(g)
            M = parse_matrix(rows[0].get("um_per_px"))
            if M is None:
                log(f"  {stem}: IR mosaic not stitched (no pixel matrix in {mp.name})")
                continue
            res = stitch(imgs, [(float(r["stage_x_um"]), float(r["stage_y_um"])) for r in rows], M, ref=0)
            del imgs
            g, cov = res["image"], res["weight"] > 1e-6
            a16 = np.clip(g * (65535.0 / max(float(g[cov].max()), 1.0)), 0, 65535).astype(np.uint16)
            a16[~cov] = 0
            cv2.imwrite(str(Path(out) / f"{stem}_mosaic.png"), a16, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            pv = domain_preview(a16.astype(np.float32), centre=res["ref_xy"])[0]
            pv[~cov] = 0
            cv2.imwrite(str(Path(out) / f"{stem}_mosaic.jpg"), pv, [cv2.IMWRITE_JPEG_QUALITY, 92])
            _mosaic_write_info(Path(out) / f"{stem}_mosaic.json", rows, res, M, [r["file"] for r in rows])
            done[main] = [r["file"] for r in rows[1:]]
            log(f"  {stem}: IR mosaic of {len(rows)} fields ({res['n_pairs_used']} of {len(res['pairs'])} overlaps matched)")
        except Exception as e:
            log(f"  {stem}: IR mosaic not stitched ({type(e).__name__}: {e})")
    return done


def _delete_fields(folder, done, out, ext, log):
    """The mosaic fields' TIFFs, once their mosaic is written."""
    n = 0
    for main, files in done.items():
        if not (Path(out) / (Path(main).stem + "_mosaic" + ext)).is_file():
            continue
        for f in files:
            p = Path(folder) / f
            if p.is_file():
                p.unlink()
                n += 1
    if n:
        log(f"  deleted {n} mosaic field TIFFs from {folder}")


def export(folder, viewer_dir, flat=None, material=None, dark=128.0, keep_tiles=False, log=print, quality=100, verbose=True,
           oxide_nm=None, median=True, combine=None, workers=None, skip_done=False, replace=False, transform=None,
           flat_domain=None):
    """Export one piece's 100x (and domain) images into a viewer; returns the number of images written.
    combine: how the frames of the 3 stage positions are combined: "mean" (aligned, weighted by the frames averaged into
    each; the default), "median", or "none" (the main frame only); median=False is the older spelling of "none".
    workers: processes for the per-image work (default: cores - 1, at most 4; 1 = in this process).
    skip_done: leave out images already exported (a second pass after the stacking-domain images).
    replace: a re-image (scan mode "retarget"): each flake's new image replaces its earlier ones (see attach).  Images
    already exported (no TIFF left) are attached again, so this also redoes the attachment of an earlier re-image.
    transform: (R, t) of the re-image (its <piece>_retakeN_chip.json): 10x stage positions now = R . then + t; the images'
    positions are taken back to the first scan's (the viewer's) before matching.
    flat_domain: the IR camera's flat field (.npz of the scan's Flat field, objective "IR (domain camera)"): the domain
    images are divided by it before they are written."""
    combine = combine or ("mean" if median else "none")
    folder, vdir = Path(folder), Path(viewer_dir)
    idx_path = next(folder.glob("*_100x_index.csv"), None)
    if idx_path is None:
        log(f"{folder}: no *_100x_index.csv")
        return 0
    mos100, mosir = {}, {}
    rows = [r for r in csv.DictReader(open(idx_path)) if (folder / r["file"]).is_file() or (vdir / "100x" / (Path(r["file"]).stem + ".jpg")).is_file()]
    material = material or _material(folder)
    oxide_nm = oxide_nm or _viewer_oxide(vdir) or (85.0 if material == "graphite" else 300.0)
    out = vdir / "100x"
    (out / "preview").mkdir(parents=True, exist_ok=True)
    import os
    workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))

    from .progress import Progress
    todo = [r for r in rows if (folder / r["file"]).is_file()
            and not (skip_done and (out / f"{Path(r['file']).stem}_regions.json").is_file())]
    ctx = dict(folder=str(folder), out=str(out), flat=str(flat) if flat else None, dark=dark, combine=combine,
               material=material, oxide_nm=oxide_nm, quality=quality, pat=None, gain=None, expect=_flake_expect(vdir))
    _init_ctx(ctx)
    subs = []

    def first_pass(terms):
        # one read of every frame: its substrate colour (white balance) and its part of the optics pattern (threads: the
        # reads, the demosaicking and the pattern's filtering release the GIL; the pattern's per-frame work ran one frame
        # after another in static_pattern: 176 -> 57 ms per frame, same pattern).  The write pass reads the frames again
        # in its worker processes, but reading, demosaicking and flat-fielding a frame is ~10 ms of its ~1.9 s; handing
        # the workers the frames read here (17 MB each as float) would cost more than that
        from concurrent.futures import ThreadPoolExecutor
        pr = Progress("100x: reading frames", len(todo), verbose)
        from collections import deque
        with ThreadPoolExecutor(min(4, workers + 1)) as ex:
            q, it = deque(), iter(todo)
            for r in it:                                   # a few frames ahead (one frame is ~17 MB as float)
                q.append(ex.submit(_read_sub, folder / r["file"], terms))
                if len(q) >= 6:
                    break
            while q:
                sub, pt = q.popleft().result()
                r = next(it, None)
                if r is not None:
                    q.append(ex.submit(_read_sub, folder / r["file"], terms))
                subs.append(sub)
                pr.update()
                yield pt
    # optics artifacts that stay put from frame to frame, estimated from this run's frames and divided out (none from
    # fewer than PATTERN_MIN_FRAMES frames: then only the substrate colours are read)
    pat = None
    if len(todo) >= PATTERN_MIN_FRAMES:
        pat = static_pattern(first_pass(True))
    else:
        for _ in first_pass(False):
            pass
    if pat is not None:
        cv2.imwrite(str(out / "optics_pattern.jpg"), _u8(0.5 + (pat.mean(axis=2) - 1.0) / 0.04))
        log(f"{folder.name}: optics pattern from {len(todo)} frames ({100 * float((pat - 1).std()):.2f} % rms) divided out")
    if todo:
        fr = np.array([f for _, f in subs])
        good = np.array([m for m, f in subs if f >= min(0.25, float(np.median(fr)))])
        # substrate colour and brightness against the blank-substrate reference for this objective and oxide
        from . import bgcheck

        def _exp(r):
            try:
                return float(r.get("exposure_ms") or "nan") if r.get("exposure_ms") else None
            except ValueError:
                return None
        exps = [_exp(r) for r in todo]
        known = [e for e in exps if e]
        cut = min(0.25, float(np.median(fr)))
        bgres = bgcheck.check(np.median(good, 0), "100x", oxide_nm, exposure_ms=(float(np.median(known)) if known else None),
                              flat_info=(_CTX["FF"].info if _CTX["FF"] is not None else None),
                              frames=[(m, e) for (m, f), e in zip(subs, exps) if f >= cut])
        bgcheck.report(bgres, log=log, prefix=f"{folder.name}: ")
        (out / f"bgcheck_{folder.name}.json").write_text(json.dumps(bgres, indent=1))
        gain = np.asarray(DEMO_SUBSTRATE.get(material, DEMO_SUBSTRATE["hbn"])) / np.median(good, 0)
        ctx.update(pat=pat, gain=gain)
        pr = Progress("100x: writing images", len(todo), verbose)
        if workers <= 1 or len(todo) < 4:
            _init_ctx(ctx)
            for r in todo:
                _write_one(r)
                pr.update()
        else:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_ctx,
                                     initargs=(ctx,)) as ex:
                for _ in ex.map(_write_one, todo):
                    pr.update()
        log(f"{folder.name}: {len(todo)} 100x images -> {out} (white balance for {material}, gains {np.round(gain, 4).tolist()}"
            f"{f'; {workers} processes' if workers > 1 and len(todo) >= 4 else ''})")
        # terrace mosaics (flakeid.scope hundred_mosaic): stitched here, with this pass's white balance and pattern
        _init_ctx(ctx)
        mos100 = stitch_100x_mosaics(folder, out, {r["file"] for r in todo}, gain, material, quality, log)

    # stacking-domain images: 16-bit PNG + stretched preview
    name = re.sub(r"_100x$", "", folder.name)
    ddir = folder.parent / f"{name}_domains"
    drows = []
    if ddir.is_dir():
        di = next(ddir.glob("*_domains_index.csv"), None)
        drows = list(csv.DictReader(open(di))) if di else []
        (out / "domains").mkdir(exist_ok=True)
        ffd = load_domain_flat(flat_domain, log) if flat_domain and drows else None
        prd = Progress("domain images", max(len(drows), 1), verbose and bool(drows))
        for r in drows:
            prd.update()
            src, stem = ddir / r["file"], Path(r["file"]).stem
            if not src.is_file():
                continue
            a = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if ffd is not None:
                if a.shape[:2] == ffd[0].shape:
                    g0 = a.astype(np.float32) if a.ndim == 2 else a.astype(np.float32).mean(axis=2)
                    a = g0 / np.maximum(ffd[0], 0.05)     # (no dark: see load_domain_flat)
                else:
                    log(f"  {stem}: {a.shape[1]} x {a.shape[0]} px, the IR flat field is {ffd[0].shape[1]} x {ffd[0].shape[0]}: not flat-corrected")
            if a.dtype != np.uint16:                  # 32-bit average of several frames: keep the range in 16 bit
                a = np.clip(a.astype(np.float64) * (65535.0 / max(float(a.max()), 1.0)), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(out / "domains" / f"{stem}.png"), a, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            g = a.astype(np.float32) if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cv2.imwrite(str(out / "domains" / f"{stem}.jpg"), domain_preview(g)[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
        log(f"{ddir.name}: {len(drows)} domain images -> {out / 'domains'} (16-bit PNG + JPEG at the flake's level +-{DOMAIN_VIEW_HALF * 100:g} %"
            + (", IR flat field applied)" if ffd is not None else ")"))
        if drows:
            mosir = stitch_ir_mosaics(ddir, out / "domains", ffd, log)

    n = attach(vdir, rows, drows, log=log, replace=replace, transform=transform)
    if not keep_tiles:
        _delete(folder, rows, out, ".jpg", log)
        _delete_fields(folder, mos100, out, ".jpg", log)
        if drows:
            _delete(ddir, drows, out / "domains", ".png", log)
            _delete_fields(ddir, mosir, out / "domains", ".png", log)
    return n


def _delete(folder, rows, out, ext, log):
    gone, size = 0, 0
    for r in rows:
        if not (out / (Path(r["file"]).stem + ext)).is_file():
            continue
        for f in [r["file"]] + [x for x in (r.get("shift_files") or "").split(";") if x]:
            src = folder / f
            if src.is_file():
                size += src.stat().st_size
                src.unlink()
                gone += 1
    if gone:
        (folder / "TIFFS_DELETED.txt").write_text(f"{gone} TIFFs ({size / 1e6:.0f} MB) deleted after export to {out} "
                                                  f"(flakeid.hundred; --keep-tiles keeps them).\n")
        log(f"  deleted {gone} TIFFs ({size / 1e6:.0f} MB) from {folder}")


def _apply_branch(D, fid, b, log=print):
    """The viewer's data (D) with the thickness the 100x colour chose (branch100) for flake fid's terrace: t, the 10x
    one kept as t10 and alt, quality ;100x_branch; the flake's best t if it is its best terrace.  1 if changed."""
    if not b or not b.get("changed"):
        return 0
    f_ = next((f for f in D.get("flakes") or () if int(f["id"]) == int(fid)), None)
    tid = b.get("tid") if b.get("tid") is not None else ((f_ or {}).get("terr") or [None])[0]
    T_ = (D.get("terraces") or {}).get(str(tid))
    if T_ is None or T_.get("t100") is not None:
        return 0
    T_["t10"], T_["t100"] = T_.get("t"), b["chosen"]
    T_["alt"] = f"{T_['t10']}"
    T_["t"] = T_["tlo"] = T_["thi"] = b["chosen"]
    T_["q"] = (T_.get("q") or "good").replace("good", "").strip(";") + ";100x_branch"
    if f_ is not None and str((f_.get("terr") or [None])[0]) == str(tid):
        f_["t"] = b["chosen"]
    log(f"  F{fid}: the 100x colour fits {b['chosen']:g} nm, not {b['t10']:g} nm (10x readings {b['readings']}, distances "
        f"{b['d']}): thickness changed")
    return 1


def attach(viewer_dir, rows, drows=(), offset=OFFSET_100X, log=print, replace=False, transform=None):
    """Match images to the viewer's flakes and write the mapping into its index.html.  Returns the number matched.
    replace: the images are a re-image of their flakes: a flake's earlier 100x images are dropped from the page (the
    files stay) when the new one is focused, or when none of the earlier ones was (a new unfocused image of a flake
    that has a focused one is only added).  An earlier entry of another flake that used the same file name is dropped
    too (the file now holds the new image), and an entry of the same file is refreshed (focus, statistics).
    transform: (R, t): a re-image of a chip that had moved (10x stage positions now = R . then + t); the images'
    positions are taken back to the viewer's before matching.  A re-image without one (or whose position does not
    fit) is matched by the flake number in its label: re-image labels are those of the final target list."""
    vdir = Path(viewer_dir)
    page = vdir / "index.html"
    html = page.read_text()
    m = re.search(r"const D=(\{.*?\});\n", html, re.S)
    if not m:
        raise RuntimeError(f"{page}: no viewer data")
    D = json.loads(m.group(1))
    fl = [f for f in D["flakes"] if f.get("sx") is not None]
    by_id = {f["id"]: f for f in fl}
    P = np.array([[f["sx"], f["sy"]] for f in fl]) if fl else np.zeros((0, 2))

    Rt = None
    if transform is not None:
        Rt = (np.asarray(transform[0], float).reshape(2, 2), np.asarray(transform[1], float).reshape(2))

    def match(r):
        xy = np.array([float(r["stage_x_um"]), float(r["stage_y_um"])]) - np.asarray(offset)
        rc = re.match(r"\s*(-?[\d.]+):(-?[\d.]+)\s*$", r.get("recentre_um") or "")
        if rc:                                       # (re-centred at 100x: back to the position the list planned)
            xy = xy - np.array([float(rc.group(1)), float(rc.group(2))])
        if Rt is not None:
            xy = Rt[0].T @ (xy - Rt[1])                  # (R is a rotation: its inverse is its transpose)
        mm_ = re.search(r"_F(\d+)", r["label"])
        f = by_id.get(int(mm_.group(1))) if mm_ else None
        # the flake named in the label (the piece's own target list) unless the picture is far from it: a re-centring
        # or a hand move while focusing puts the picture up to a field (~160 um) from the 10x position; only 40 um was
        # allowed before 2026-10-21 (bn_20261020_1: 18 of 80 pictures, flakes 1, 5, 6, 14 and 16 among them, were
        # left off the page and so out of the designer)
        if f is not None and np.hypot(f["sx"] - xy[0], f["sy"] - xy[1]) < (40.0 if Rt is not None else LABEL_MATCH_UM):
            return f["id"]
        if f is not None and replace and Rt is None:
            return f["id"]
        if len(P):
            k = int(np.argmin(np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1])))
            if np.hypot(*(P[k] - xy)) < 60.0:
                return fl[k]["id"]
        return None

    # images already attached (an earlier 100x folder of the same piece, e.g. a resumed run) are kept
    H = {}
    mh = re.search(r"const H100=(.*?);\n", html, re.S)
    if mh and mh.group(1).strip() not in ("null", ""):
        try:
            H = json.loads(mh.group(1))
        except ValueError:
            H = {}
    have = {e["f"] for v in H.values() for e in v}
    n, n_rep, n_br = 0, 0, 0
    focused = ("auto", "manual")
    for r in rows:
        fid = match(r)
        if fid is None:
            continue
        stem = Path(r["file"]).stem
        e = dict(f=f"100x/{stem}.jpg", p=f"100x/preview/{stem}.jpg", lab=r["label"], focus=r.get("focus", ""), z=r.get("z_um", ""))
        # the copies people look at (flakeid.look), shown in place of f / p / a; the page's "linear" view shows these
        lk = {k: f"100x/{n_}" for k, n_ in (("f", f"{stem}_look.jpg"), ("p", f"preview/{stem}_look.jpg"),
                                             ("a", f"{stem}_avg_look.jpg")) if (vdir / "100x" / n_).is_file()}
        if lk:
            e["look"] = lk
        if replace:
            for k in list(H):                        # the file name used before (by another flake, or by this one: refreshed)
                H[k] = [x for x in H[k] if x["f"] != e["f"]]
                if not H[k]:
                    del H[k]
            old = [x for x in H.get(str(fid), []) if not x.get("domain")]
            if old and (e["focus"] in focused or not any(x.get("focus") in focused for x in old)):
                H[str(fid)] = [x for x in H[str(fid)] if x not in old]
                n_rep += len(old)
            have = {x["f"] for v in H.values() for x in v}
        rj = vdir / "100x" / f"{stem}_regions.json"
        if rj.is_file():
            g = json.loads(rj.read_text())
            if (vdir / "100x" / f"{stem}_avg.jpg").is_file():
                e.update(a=f"100x/{stem}_avg.jpg", a_c8=f"100x/{stem}_avg_c8.jpg", a_sg=f"100x/{stem}_avg_sg.jpg")
            n_br += _apply_branch(D, fid, g.get("branch100"), log)
            if g.get("score100") is not None:           # the 100x score: largest clean area (um2) and its disc (um)
                e["s100"] = [g["score100"].get("clean_um2", 0.0), g["score100"].get("clean_d_um", 0.0)]
            e.update(c8=f"100x/{stem}_c8.jpg", sg=f"100x/{stem}_sg.jpg", deb=f"100x/{stem}_debris.jpg",
                     stats=dict(terrace_um2=g["terrace_um2"], debris_n=g["debris_n"], debris_um2=g["debris_um2"], steps_um2=g.get("steps_um2", 0),
                                clean_um2=g["clean_um2"], clean_d_um=g["clean_d_um"], cut=g.get("cut_by_frame", False)))
        if e["f"] not in have:
            H.setdefault(str(fid), []).append(e)
        n += 1
    for r in drows:
        fid = match(r)
        if fid is None:
            continue
        stem = Path(r["file"]).stem
        if f"100x/domains/{stem}.jpg" in have:
            continue
        H.setdefault(str(fid), []).append(dict(f=f"100x/domains/{stem}.jpg", raw=f"100x/domains/{stem}.png", lab=r["label"],
                                               domain=True, t=r.get("thickness_nm", "")))
    n_mos = _attach_mosaics(vdir, H)
    if any(e.get("domain") for v in H.values() for e in v):
        try:
            align_ir(vdir, H, log)
        except Exception as ex:                  # (the 100x images are attached all the same)
            log(f"  IR pictures not laid over the 100x ones ({type(ex).__name__}: {ex})")
    if n_br:                                    # (the page's flake data with the thicknesses the 100x colour chose)
        html = html.replace(m.group(0), "const D=" + json.dumps(D, separators=(",", ":")) + ";\n", 1)
    line = "const H100=" + json.dumps(H, separators=(",", ":")) + ";\n"
    if re.search(r"const H100=.*?;\n", html, re.S):
        html = re.sub(r"const H100=.*?;\n", lambda _: line, html, count=1, flags=re.S)
    else:                                   # a page written before the 100x support: data only (the page cannot show them)
        html = html.replace(m.group(0), m.group(0) + line, 1)
        log(f"  note: {page} predates the 100x display; rebuild the viewer to see the images")
    page.write_text(html)
    log(f"  viewer {page}: 100x images for {len(H)} flakes ({n} of {len(rows)} images matched"
        + (f"; {n_rep} earlier images replaced" if replace else "") + (f"; {n_mos} terrace mosaics" if n_mos else "") + ")")
    if n_mos and "'mos','mosaic'" not in html:
        from .zoomview import upgrade_page              # (a page older than the mosaic view)
        if upgrade_page(page, backup=False):
            log(f"  {page}: page brought up to date (terrace mosaics, key W)")
    return n


def _attach_mosaics(vdir, H):
    """Terrace mosaics written by export onto the viewer's entries: a 100x entry gets mos (its mosaic), mp (preview),
    mos_n / mos_um (fields, size); a domain entry its IR mosaic (mos, mos_raw), which the flake's 100x entries carry
    as irmos / irmos_raw too (the IR view's mosaic).  Returns the number of mosaics attached."""
    n = 0
    for v in H.values():
        irm = None
        for e in v:
            stem = Path(e["f"]).stem
            if e.get("domain"):
                if (vdir / "100x" / "domains" / f"{stem}_mosaic.jpg").is_file():
                    e.update(mos=f"100x/domains/{stem}_mosaic.jpg", mos_raw=f"100x/domains/{stem}_mosaic.png")
                    irm = e
                    n += 1
            elif (vdir / "100x" / f"{stem}_mosaic.jpg").is_file():
                e.update(mos=f"100x/{stem}_mosaic.jpg", mp=f"100x/preview/{stem}_mosaic.jpg")
                if (vdir / "100x" / f"{stem}_mosaic_look.jpg").is_file():
                    e.setdefault("look", {}).update(mos=f"100x/{stem}_mosaic_look.jpg", mp=f"100x/preview/{stem}_mosaic_look.jpg")
                try:
                    g = json.loads((vdir / "100x" / f"{stem}_mosaic.json").read_text())
                    e.update(mos_n=len(g["fields"]), mos_um=g["size_um"])
                except (OSError, ValueError, KeyError):
                    pass
                n += 1
        if irm is not None:
            for e in v:
                if not e.get("domain"):
                    e.update(irmos=irm["mos"], irmos_raw=irm["mos_raw"])
    return n


def add_ir_views(viewer_dir, log=print):
    """align_ir on a viewer already written; returns the number of 100x pictures with an IR view (None: no viewer data)."""
    page = Path(viewer_dir) / "index.html"
    html = page.read_text()
    mh = re.search(r"const H100=(.*?);\n", html, re.S)
    if not mh or mh.group(1).strip() in ("null", ""):
        log(f"{page}: no 100x images")
        return None
    H = json.loads(mh.group(1))
    vdir = Path(viewer_dir)
    nd = 0
    for v in H.values():                              # the IR previews again at the current display range
        for d in v:
            if d.get("domain") and d.get("raw") and (vdir / d["raw"]).is_file():
                g = cv2.imread(str(vdir / d["raw"]), cv2.IMREAD_UNCHANGED)
                g = g.astype(np.float32) if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
                cv2.imwrite(str(vdir / d["f"]), domain_preview(g)[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
                nd += 1
            if not d.get("domain"):
                d.pop("ir", None)                     # (laid over again at the current range)
    if nd:
        log(f"  {nd} IR previews redrawn at the flake's level +-{DOMAIN_VIEW_HALF * 100:g} %")
    n = align_ir(vdir, H, log)
    line = "const H100=" + json.dumps(H, separators=(",", ":")) + ";\n"
    page.write_text(re.sub(r"const H100=.*?;\n", lambda _: line, html, count=1, flags=re.S))
    if n:
        from .zoomview import upgrade_page              # (a page older than the IR view: its data in the current page)
        if upgrade_page(page, backup=False):
            log(f"  {page}: page brought up to date (IR view, key I)")
        import os
        for d in (page.parent.parent, page.parent.parent.parent):   # (the designer rebuilds the run: IR view there too)
            if (d / "copy_done.json").is_file():
                os.utime(str(d / "copy_done.json"))
    return n


def score_viewer(viewer_dir, log=print):
    """The 100x score (screen100.clean_score: the largest clean area) of every 100x picture already in a viewer, from
    its natural JPEGs (linear and white-balanced, as make_catalog_100x re-analyses them: the raw TIFFs are deleted after
    the export): into each picture's regions.json (score100) and the page (s100), and its debris picture redrawn with
    the scored area in green.  For viewers exported before the score (2026-10-27).  Returns the number scored."""
    from . import screen100
    vdir = Path(viewer_dir)
    page = vdir / "index.html"
    html = page.read_text(encoding="utf-8")
    D = json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1))
    mh = re.search(r"const H100=(.*?);\n", html, re.S)
    H = json.loads(mh.group(1)) if mh and mh.group(1).strip() not in ("null", "") else {}
    material = D.get("material") if D.get("material") in DEMO_SUBSTRATE else "hbn"
    oxide = float(D.get("oxide") or (85.0 if material == "graphite" else 300.0))
    sub = np.asarray(DEMO_SUBSTRATE[material], float) * 255.0 / WHITE_BY_MATERIAL.get(material, WHITE)   # (JPEG units)
    expect = _flake_expect(vdir)
    n = n_br = 0
    for es in H.values():
        for e in es:
            src = vdir / e["f"]
            if e.get("domain") or not src.is_file():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(src), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            try:
                _, reg, extra = screen100.features(rgb.astype(np.float32), material, oxide, _thickness(dict(label=e.get("lab"), file=e["f"])),
                                                   substrate=sub, expect=expect.get(_flake_label_id(e.get("lab"))))
            except Exception as ex:                   # (one picture must not stop the rest)
                log(f"  {src.name}: not scored ({type(ex).__name__}: {ex})")
                continue
            sc = reg["score100"]
            e["s100"] = [sc["clean_um2"], sc["clean_d_um"]]
            ex = expect.get(_flake_label_id(e.get("lab")))
            b100 = branch100(rgb.astype(np.float32), reg, material, oxide, ex)   # (JPEG colours are linear: ratios hold)
            rj = src.with_name(src.stem + "_regions.json")
            if rj.is_file():
                G = json.loads(rj.read_text())
                G["score100"] = sc
                if b100 is not None:
                    G["branch100"] = b100
                rj.write_text(json.dumps(G, separators=(",", ":")))
            fid_ = _flake_label_id(e.get("lab"))
            if fid_ is not None:
                n_br += _apply_branch(D, fid_, b100, log)
            reg["fine"] = extra["fine"]
            cv2.imwrite(str(src.with_name(src.stem + "_debris.jpg")), overlay(rgb, reg), jpeg_params(DIAG_QUALITY))
            n += 1
    line = "const H100=" + json.dumps(H, separators=(",", ":")) + ";\n"
    if n_br:
        html = re.sub(r"const D=\{.*?\};\n", lambda _: "const D=" + json.dumps(D, separators=(",", ":")) + ";\n", html, count=1, flags=re.S)
    if mh:
        html = re.sub(r"const H100=.*?;\n", lambda _: line, html, count=1, flags=re.S)
        page.write_text(html, encoding="utf-8")
    log(f"{vdir.name}: 100x score of {n} pictures (largest clean area: no step edges, glue / debris, bubbles or folds)"
        + (f"; {n_br} thicknesses changed by the 100x colour (of two 10x readings)" if n_br else ""))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flakeid.hundred", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("folder", nargs="?", help="<piece>_100x folder written by mm_chip_scan.bsh")
    ap.add_argument("--viewer", required=True, help="the piece's viewer folder (<name>_viewer with index.html)")
    ap.add_argument("--flat", help="100x flat field .npz (python -m flakeid.mm flat); else a fixed black level")
    ap.add_argument("--flat-domain", help="IR flat field .npz (scan: Flat field, IR (domain camera)) for the domain images")
    ap.add_argument("--material", choices=("hbn", "graphite"), help="default: from the scan parameters")
    ap.add_argument("--keep-tiles", action="store_true", help="keep the 100x TIFFs (deleted by default once exported)")
    ap.add_argument("--combine", choices=("mean", "median", "none"), default="mean",
                    help="frames of the 3 stage positions: aligned weighted mean (default; least noise), median, or none")
    ap.add_argument("--no-median", action="store_true", help="= --combine none (flat field and optics pattern only)")
    ap.add_argument("--replace", action="store_true", help="a re-image: each flake's new image replaces its earlier ones "
                    "on the page (a focused one always; an unfocused one only earlier unfocused ones)")
    ap.add_argument("--retake-chip", help="a re-image's <piece>_retakeN_chip.json: its transform takes the images' positions "
                    "back to the first scan's (the chip had moved)")
    ap.add_argument("--align-ir", action="store_true", help="only lay the viewer's IR (stacking-domain) pictures over its "
                    "100x ones (a viewer exported before 2026-10-15); no folder needed")
    ap.add_argument("--score100", action="store_true", help="only score the viewer's 100x pictures (the largest clean area) from "
                    "their JPEGs: a viewer exported before the 100x score (2026-10-27); no folder needed")
    a = ap.parse_args(argv)
    if a.score100:
        from .zoomview import upgrade_page
        n = score_viewer(a.viewer)
        upgrade_page(Path(a.viewer) / "index.html")       # (the page with the 100x score column and 100x mode)
        return 0 if n else 1
    if a.align_ir:
        return 0 if add_ir_views(a.viewer) is not None else 1
    if not a.folder:
        ap.error("the <piece>_100x folder is needed (or --align-ir)")
    tr = None
    if a.retake_chip:
        T = json.loads(Path(a.retake_chip).read_text()).get("transform") or {}
        if T.get("R") and T.get("t"):
            tr = (T["R"], T["t"])
            print(f"positions taken back to the first scan's with the transform in {a.retake_chip}")
        else:
            print(f"(no transform in {a.retake_chip}: matched by the flake numbers in the labels)")
    export(a.folder, a.viewer, flat=a.flat, flat_domain=a.flat_domain, material=a.material, keep_tiles=a.keep_tiles,
           combine="none" if a.no_median else a.combine, replace=a.replace, transform=tr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
