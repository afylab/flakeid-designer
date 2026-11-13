"""Defects of an hBN flake's terrace a in its natural 100x picture, and the designer-decision screening built on them.

Jordan's discards of hBN flakes the 100x label model scored well (2026-10-21: many steps across terrace a, residue
reaching in from the flake edge, ripped-looking residue in the centre) show in the thickness-change map of the
picture (the colour change along the thin-film model's direction at terrace a's thickness, nm, as tools/terraces100.py
makes it): areas off terrace a's level, step edges and mottled texture.  From them, for terrace a:

  clean_d            diameter (um) of the largest disc on terrace a clear of all three (0.6 um margin)
  edge_um_per_1000   length (um) of step edges (map gradient > 0.25 nm/um) per 1000 um2 of terrace a
  off_frac, tex_frac, defect_frac, tex_p90, a_d   (stored for scripts; not in the model)
  res_a_frac, res_touch_frac, res_bleed_um       residue by its colour (residue(): share of terrace a, share of its
                     boundary touched, how far edge residue reaches in, um; stored, not in the model)
  v                  the version of the measurement (below)

Version 2 (2026-10-22, defects(); version 1 before, _defects_v1): the flake's edge looks thicker in the map for 1 - 3 um
all along the outline (an optical halo, on the flakes Jordan kept as on the others), and version 1 counted it, smeared
4 - 7 um inwards by the texture filter, as a defect band along the whole outline: it shrank clean_d of clean
medium-size flakes (6 of the keeps the screening rejected).  Version 2 leaves the outermost RIM_UM out, takes the
halo's profile along the outline out, averages its filters over the measured part only and measures off-level from a
robust quadratic surface (a smooth drift is no defect).  On the 1009 decided flakes (fit-defects, same inputs):
leave-one-run-out AUC 0.848 against 0.836, discards removed keeping 95 % of the keeps on held-out runs 404 against
369; with the training runs' cutoff 420 discards removed and 7 of 124 keeps lost, against 367 and 6 (at 96 %: 398 and
6).  The residue features did not add over all runs (0.841 - 0.848, more keeps lost; on bn_20261020_run_1 alone the
residue share raised the leave-one-chip-out AUC from 0.891 to 0.910).  A pick model records the version it was fitted
on ("defects", 1 when missing): the build and the re-score command measure that way (defects_version), and
score_flakes leaves flakes measured otherwise as they are.

The hBN "pick" model of profiles/screen100_hbn.json (tools/fit_screen100.py fit-defects, on 1009 decided flakes of 7
runs, 2026-10-22; version 1 of the defects) combines the label model's p_quality (its logit) with clean_d and
edge_um_per_1000 (leave-one-run-out AUC 0.836; p_quality alone 0.806).  Tiers: reject below reject_below (95 % of the
keep / used flakes above it), keep from keep_above, else review.  A model may take the two features relative to the run's (or chip's) median ("centre":
a run's focus and exposure shift them), the median shrunk to the training runs' by n0 flakes since a run's bundle
grows by chips while it scans (tools/make_designer_bundle.py --update scores every hBN flake again); the fitted one
does not: run-centred was 0.839 when scored on whole runs, but 0.836 as the bundle grows, and its cutoff lost 11
instead of 6 of 124 keeps on held-out runs.

Works on the picture at the cut-out's scale (half resolution), as the bundle stores it (VIEW_JPEG), and only around
the flake: ~85 ms per flake with the residue (i7-7700; version 1 with the residue 88 ms in a catalog build).  Python 3.8.

    python -m flakeid.defects100 <designer folder or bundle.json> [--map pictures.json] [--pictures DIR ...] [-o out.json]

re-scores a built bundle in place (patch: the pictures are not rebuilt): defects of the hBN flakes that have none or
another version than the pick model's, from the natural 100x pictures (the bundle's own x100_nat_* files, or --map
{flake id: picture} / --pictures folders holding them), then p_keep / tier of every hBN flake.  Run it on a copy on this computer, not on the share.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OFF_NM = 0.3                   # off terrace a's level (nm; from a robust quadratic surface fitted to terrace a)
TEX_NM = 0.2                   # local texture (robust std in ~2 um windows, nm)
GRAD_NM_UM = 0.25              # step edge: map gradient (nm per um)
RIM_UM = 2.0                   # the flake's outermost band: not measured (its edge halo is no defect)
RIM_PROFILE_UM = 4.0           # the halo's mean profile along the whole outline is taken out up to this far in
VERSION = 2                    # the measurement (defects): 2 from 2026-10-22 (rim, level surface); 1 before.  A pick
                               # model says which one it was fitted on ("defects", default 1): defects_version()
KEYS = ("clean_d", "edge_um_per_1000", "off_frac", "tex_frac", "defect_frac", "tex_p90", "a_d", "res_a_frac", "res_touch_frac",
        "res_bleed_um")
# The defects are measured on the picture as the designer bundle has it (tools/make_designer_bundle.JPEG_VIEW: JPEG
# quality 95, full colour resolution; the scan's own pictures are ~5x larger): a bundle re-scored from its own pictures
# then gets the build's values (on the originals clean_d was 1.7 um larger and the step-edge density 3 lower on
# average, 10 of 208 flakes changed tier, 2026-10-22)
VIEW_JPEG = [cv2.IMWRITE_JPEG_QUALITY, 95] + ([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
                                              if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") else [])


def view_bgr(bgr, n_bytes=None):
    """A natural 100x picture (BGR) as the designer bundle stores it: re-encoded with VIEW_JPEG, unless that is not
    smaller than the source's n_bytes (tools/make_designer_bundle.view_bytes keeps it then)."""
    ok, enc = cv2.imencode(".jpg", bgr, VIEW_JPEG)
    if not ok or (n_bytes is not None and len(enc) >= n_bytes):
        return bgr
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _k(um, px):
    return max(3, int(round(um / px)) | 1)


def _flatten_fast(r, tol=0.06, sigma=60.0, k=8, box=None):
    """flakeid.hundred._flatten's background (r / it), computed with OpenCV (numpy's max over the colour axis took
    20 of its 50 ms): returns the background, full size or inside box = (y0, y1, x0, x1) only (the same values: the
    bilinear interpolation of cv2.resize for those pixels); the division is left to the caller."""
    h, w = r.shape[:2]
    near = cv2.inRange(r, (1.0 - tol,) * 3, (1.0 + tol,) * 3)          # (every channel within tol of the terrace)
    rm = np.zeros_like(r)
    cv2.copyTo(r, near, rm)
    sw, sh = max(1, w // k), max(1, h // k)
    rs = cv2.resize(rm, (sw, sh), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(near.astype(np.float32) * np.float32(1 / 255.0), (sw, sh), interpolation=cv2.INTER_AREA)
    s1, s4 = sigma / k, 4 * sigma / k
    num = cv2.GaussianBlur(rs, (0, 0), s1)
    den = cv2.GaussianBlur(ms, (0, 0), s1)
    wide = cv2.GaussianBlur(rs, (0, 0), s4) / np.maximum(cv2.GaussianBlur(ms, (0, 0), s4), 1e-6)[..., None]
    al = np.clip(den / 0.3, 0, 1)[..., None]
    bg = al * num / np.maximum(den, 1e-6)[..., None] + (1 - al) * wide
    bg = bg.astype(np.float32)
    if box is None:
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    else:                                   # (dst pixel x -> source (x + x0 + 0.5) sw / w - 0.5, as cv2.resize)
        y0, y1, x0, x1 = box
        fx, fy = sw / w, sh / h
        M = np.array([[fx, 0, (x0 + 0.5) * fx - 0.5], [0, fy, (y0 + 0.5) * fy - 0.5]], np.float64)
        bg = cv2.warpAffine(bg, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
    return np.where(np.isfinite(bg) & (bg > 0.2), bg, np.float32(1.0))


def masks(shape, flake):
    """The flake outline F and terrace a A (uint8) in the picture at the cut-out's scale, from the designer flake's
    frame (x0, y0, k), outline and terrace a polygon (cut-out px), or (None, None)."""
    fr = flake.get("frame") or {}
    t = (flake.get("terraces") or [{}])[0]
    poly = t.get("poly_10x") or t.get("poly")          # (the catalog polygon: the model was fitted on it)
    if not fr or not flake.get("outline") or not poly:
        return None, None
    k = float(fr.get("k", 0.5))
    off = np.array([fr["x0"] * k, fr["y0"] * k])
    F = np.zeros(shape, np.uint8)
    cv2.fillPoly(F, [np.round(np.array(flake["outline"], float).reshape(-1, 2) + off).astype(np.int32)], 1)
    A = np.zeros(shape, np.uint8)
    cv2.fillPoly(A, [np.round(np.array(poly, float).reshape(-1, 2) + off).astype(np.int32)], 1)
    return F, A


def thickness_map(lin, F, A, material, oxide_nm, t_a, box, rel=False):
    """tools/terraces100.thickness_change_map + its 0.8 px smoothing, inside box = (y0, y1, x0, x1) only (the
    background from the whole picture, as there).  rel: also the picture relative to terrace a's colour in the box
    (RGB, terrace a = 1: the residue detector's colour)."""
    from .screen100 import sensitivity
    y0, y1, x0, x1 = box
    tc = np.median(lin[y0:y1, x0:x1][A[y0:y1, x0:x1] > 0], axis=0)
    inv = tuple(float(1.0 / max(v, 1e-6)) for v in tc)
    rb = cv2.GaussianBlur(cv2.multiply(lin, inv + (0.0,)), (0, 0), 1.0)
    rf = rb[y0:y1, x0:x1] / _flatten_fast(rb, box=box)
    g = np.asarray(sensitivity(material, oxide_nm, t_a), np.float64)
    # (rf - 1) . g / |g|^2, per pixel
    d = (cv2.transform(rf, (g / float(g @ g)).reshape(1, 3)) - float(g.sum() / (g @ g))).astype(np.float32)
    d = cv2.GaussianBlur(d, (0, 0), 0.8)
    return (d, cv2.multiply(np.ascontiguousarray(lin[y0:y1, x0:x1]), inv + (0.0,))) if rel else d


def _mblur(v, w, sigma):
    """Gaussian blur of v over the pixels of weight w only (normalised convolution; outside the array counts as
    weight 0).  A wide one (sigma > 6 px) on a grid binned ~sigma / 3, interpolated back."""
    vw = v * w
    f = int(sigma // 3) if sigma > 6 else 1
    if f > 1:
        H, W = v.shape
        sz = (max(1, -(-W // f)), max(1, -(-H // f)))
        vw, w = cv2.resize(vw, sz, interpolation=cv2.INTER_AREA), cv2.resize(w, sz, interpolation=cv2.INTER_AREA)
        sigma = sigma * sz[0] / W
    num = cv2.GaussianBlur(vw, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    den = cv2.GaussianBlur(w, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    out = num / np.maximum(den, np.float32(1e-3))
    return cv2.resize(out, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR) if f > 1 else out


def rim_profile(e, dF, A, px, rim_um=RIM_PROFILE_UM, n_min=30):
    """The median of e over terrace a A per distance from the flake outline (dF, px; 1 px bins) up to rim_um, relative
    to the next 2 px and tapered to 0 over the last um: the edge halo all along the outline (an image of it, to
    subtract; a defect at one place of the outline hardly moves the median)."""
    nb = int(np.ceil(rim_um / px))
    sel = A & (dF < nb + 2)
    out = np.zeros(e.shape, np.float32)
    if not sel.any():
        return out
    b = np.minimum(dF[sel].astype(np.int32), nb + 1)
    o = np.argsort(b, kind="stable")
    v = e[sel][o]
    n = np.bincount(b, minlength=nb + 2)
    c = np.concatenate([[0], np.cumsum(n)])
    prof = np.zeros(nb + 2, np.float32)
    for i in np.nonzero(n >= n_min)[0]:
        prof[i] = np.median(v[c[i]:c[i + 1]])
    ref = float(np.median(v[c[nb]:])) if n[nb:].sum() >= n_min else 0.0
    prof = (prof - ref) * np.clip((nb - np.arange(nb + 2)) * px, 0.0, 1.0).astype(np.float32)
    prof[n < n_min] = 0.0
    out[sel] = prof[b]
    return out


def level_surface(e, W, n_max=3000, it=3, start_nm=1.0, max_nm=1.0):
    """A quadratic surface fitted to e over the pixels W, robustly: from the points within start_nm of the median
    (terrace a's polygon can hold whole areas of another thickness), then without those beyond 2.5 sigma (at least
    0.25 nm); within +-max_nm.  Terrace a's own level, a smooth drift across it (illumination, focus) included."""
    st = max(1, int(np.sqrt(np.count_nonzero(W) / n_max)))         # (points on a grid of st px)
    ys, xs = np.nonzero(W[::st, ::st])
    if ys.size < 50:
        return np.zeros(e.shape, np.float32)
    ys, xs = ys * st, xs * st
    yc, xc = float(ys.mean()), float(xs.mean())
    sc = max(float(ys.std()), float(xs.std()), 1.0)
    y, x = (ys - yc) / sc, (xs - xc) / sc
    B = np.stack([np.ones_like(y), x, y, x * x, x * y, y * y], -1)
    v = e[ys, xs].astype(np.float64)
    ok = np.abs(v - np.median(v)) < start_nm
    c = np.zeros(6)
    for _ in range(it):
        if ok.sum() < 20:
            break
        c = np.linalg.lstsq(B[ok], v[ok], rcond=None)[0]
        r = np.abs(v - B @ c)
        ok = r < 2.5 * max(1.4826 * float(np.median(r[ok])), 0.1)
    H, Wd = e.shape
    yy = ((np.arange(H, dtype=np.float32) - yc) / sc)[:, None]
    xx = ((np.arange(Wd, dtype=np.float32) - xc) / sc)[None, :]
    c = c.astype(np.float32)
    S = c[0] + c[1] * xx + c[2] * yy + c[3] * xx * xx + c[4] * yy * xx + c[5] * yy * yy
    return np.clip(S, -max_nm, max_nm).astype(np.float32)


def _defects_v1(d, A, px, off_nm=OFF_NM, tex_nm=TEX_NM, grad_nm_um=GRAD_NM_UM):
    """Version 1 (2026-10-21): as defects, without the rim and with the median level, the filters over the whole box
    (which needs a margin of ~4 x 3 um around the flake)."""
    A = A.astype(bool)
    Ai = cv2.erode(A.astype(np.uint8), np.ones((_k(1.0, px),) * 2, np.uint8)).astype(bool)   # off the blurred rim
    nA = int(Ai.sum())
    if nA < 200:
        return None
    lev = float(np.median(d[Ai]))
    e = cv2.medianBlur((d - lev).astype(np.float32), 5)
    s = cv2.GaussianBlur(e, (0, 0), 0.5 / px)
    grad = cv2.magnitude(cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)) / (8 * px)
    hp = e - cv2.GaussianBlur(e, (0, 0), 3.0 / px)
    tex = np.sqrt(np.maximum(cv2.blur(hp * hp, (_k(2.0, px),) * 2), 0))
    off = np.abs(e) > off_nm
    edge = grad > grad_nm_um
    rough = tex > tex_nm
    D = (off | edge | rough) & A
    area = nA * px * px
    f = dict(off_frac=float(off[Ai].sum()) / nA, edge_um_per_1000=float(edge[Ai].sum() * px / area * 1000 / 3.0),
             tex_frac=float(rough[Ai].sum()) / nA, tex_p90=float(np.percentile(tex[Ai], 90)),
             defect_frac=float(D[Ai].sum()) / nA)
    clean = A & ~cv2.dilate(D.astype(np.uint8), np.ones((_k(0.6, px),) * 2, np.uint8)).astype(bool)
    f["clean_d"] = 2 * float(cv2.distanceTransform(np.pad(clean.astype(np.uint8), 1), cv2.DIST_L2, 5).max()) * px
    f["a_d"] = 2 * float(cv2.distanceTransform(np.pad(A.astype(np.uint8), 1), cv2.DIST_L2, 5).max()) * px
    return f, dict(e=e, D=D, clean=clean, off=off & A, edge=edge & A, rough=rough & A)


def defects(d, A, F, px, off_nm=OFF_NM, tex_nm=TEX_NM, grad_nm_um=GRAD_NM_UM, rim_um=RIM_UM, rim_profile_um=RIM_PROFILE_UM,
            level="quad", version=VERSION):
    """Defect features of terrace A in the thickness-change map d (all the same size, cut around the flake with a
    margin wider than the filters): (dict, masks dict(e, D, clean, off, edge, rough)) or None (terrace too small).

    The flake's edge looks thicker in the map for 1 - 3 um (a halo all along the outline, on the flakes Jordan kept
    as on the others; up to ~1 nm at the edge): the band rim_um wide inside the outline is not measured (no defect,
    no clean area: the clean disc stays inside it), the halo's profile along the whole outline is taken out up to
    rim_profile_um, and the filters average over the measured part only (the 3 um high-pass took the halo 4 - 7 um
    into the flake as texture, 2026-10-22).  Off-level is measured from a robust quadratic surface over terrace a
    (level "quad"; "flat": its median), so a smooth drift is no defect; steps (gradient) and texture are local.
    version 1: the measurement before (_defects_v1; its box needs the wider margin of flake_defects)."""
    if int(version) == 1:
        return _defects_v1(d, A, px, off_nm, tex_nm, grad_nm_um)
    A = A.astype(bool)
    dF = cv2.distanceTransform(np.pad(F.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    Am = A & (dF >= rim_um / px) if rim_um > 0 else A                                    # measured part of terrace a
    Ai = cv2.erode(A.astype(np.uint8), np.ones((_k(1.0, px),) * 2, np.uint8)).astype(bool) & Am   # off the blurred rim
    nA = int(Ai.sum())
    if nA < 200:
        return None
    e = cv2.medianBlur((d - np.float32(np.median(d[Ai]))).astype(np.float32), 5)
    if rim_profile_um > 0:
        e -= rim_profile(e, dF, A, px, rim_profile_um)
    W = Ai.astype(np.float32)
    s = _mblur(e, Am.astype(np.float32), 0.5 / px)
    grad = cv2.magnitude(cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)) / (8 * px)
    # local texture: std of the map in ~2 um windows after removing the ~6 um level (steps add little, mottling a lot)
    hp = e - _mblur(e, W, 3.0 / px)
    kb = (_k(2.0, px),) * 2
    tex = np.sqrt(np.maximum(cv2.blur(hp * hp * W, kb) / np.maximum(cv2.blur(W, kb), np.float32(1e-3)), 0))
    off = np.abs(e - level_surface(e, Ai) if level == "quad" else e) > off_nm
    edge = grad > grad_nm_um
    rough = tex > tex_nm
    D = (off | edge | rough) & Am
    area = nA * px * px
    f = dict(off_frac=float(off[Ai].sum()) / nA, edge_um_per_1000=float(edge[Ai].sum() * px / area * 1000 / 3.0),
             tex_frac=float(rough[Ai].sum()) / nA, tex_p90=float(np.percentile(tex[Ai], 90)),
             defect_frac=float(D[Ai].sum()) / nA)
    # largest clean disc (and terrace a's own)
    clean = Am & ~cv2.dilate(D.astype(np.uint8), np.ones((_k(0.6, px),) * 2, np.uint8)).astype(bool)
    f["clean_d"] = 2 * float(cv2.distanceTransform(np.pad(clean.astype(np.uint8), 1), cv2.DIST_L2, 5).max()) * px
    f["a_d"] = 2 * float(cv2.distanceTransform(np.pad(A.astype(np.uint8), 1), cv2.DIST_L2, 5).max()) * px
    return f, dict(e=e, D=D, clean=clean, off=off & Am, edge=edge & Am, rough=rough & Am)


# -- residue (tape residue on the flake: dark specks and streaks, yellow / orange at edges), by its colour.  From the
# 2026-10-22 prototype on Jordan's residue polygons of bn_20261020_run_1 (pixel precision 0.2 - 0.5; on that run the
# residue share of terrace a raised the leave-one-chip-out AUC of the screening from 0.881 to 0.910).  Residue is found
# locally (top-hats: compared with its own surroundings, so terraces of another thickness, slow illumination changes and
# the camera's colour calibration drop out), on the near-level part of the flake (|d - terrace a| < RES_CLIP_NM), on
# the flake's box binned RES_DS x RES_DS (~0.46 um per pixel):
#   dark    black top-hat (disc 1.5 um) of the grey relative to terrace a: dark specks and streaks;
#   yellow  top-hat (disc 3 um) of the colour change orthogonal to the thin-film direction at terrace a's thickness,
#           towards yellow / orange (a thickness change moves the colour along the thin-film direction: nothing here).
# Seeds (hysteresis, at least RES_MIN_UM2; long thin ones are a step's dark rim, a crack or a fold: left out) are grown
# by RES_GROW_UM (residue regions are clusters of specks, with the small terraces and steps they make around them);
# small off-level islands enclosed in the near-level part (saturated blobs) are residue too.
RES_DS = 2
RES_CLIP_NM = 2.0
RES_SMOOTH_UM = 0.5
RES_DARK_UM, RES_DARK_K, RES_DARK_MIN = 1.5, 4.0, 0.004     # disc; threshold max(MIN, K x the grey's noise in terrace a)
RES_YELLOW_UM, RES_YELLOW_MIN = 3.0, 0.007
RES_HYST = 0.5
RES_MIN_UM2 = 1.5
RES_GROW_UM = 2.0
RES_LINE = (8.0, 2.5, 6.0)      # a seed at least 8 um long, at most 2.5 um wide on average, length^2 / area >= 6: a line
RES_ISLAND_UM2 = 30.0
_YDIR = {}


def _disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)


def _hyst(v, thr, grow):
    lo = (v > thr * grow).astype(np.uint8)
    n, lab = cv2.connectedComponents(lo, connectivity=8)
    if n < 2:
        return np.zeros(v.shape, bool)
    seed = np.zeros(n, bool)
    seed[np.unique(lab[v > thr])] = True
    seed[0] = False
    return seed[lab]


def _blobs(m, q, amin):
    """the components of m of at least amin pixels that are not lines (RES_LINE)"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    a = st[:, cv2.CC_STAT_AREA].astype(float)
    L = np.hypot(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]).astype(float)
    line = (a / np.maximum(L, 1) * q <= RES_LINE[1]) & (L * L / np.maximum(a, 1) >= RES_LINE[2]) & (L * q >= RES_LINE[0])
    ok = (a >= amin) & ~line
    ok[0] = False
    return ok[lab]


def _yellow_dir(material, oxide_nm, t_a):
    """unit RGB direction towards yellow (R + G - 2B) with the thin-film direction at t_a projected out"""
    key = (material, round(float(oxide_nm)), round(float(t_a)))
    if key not in _YDIR:
        # (from the rounded key itself: from the first flake's own t_a the result depended on the order flakes came in)
        from .screen100 import sensitivity
        g = np.asarray(sensitivity(material, float(key[1]), float(key[2])), np.float64)
        g = g / np.linalg.norm(g)
        y = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
        y = y - (y @ g) * g
        _YDIR[key] = (y / np.linalg.norm(y)).astype(np.float32)
    return _YDIR[key]


def residue(d, F, A, px, rel, material="hbn", oxide_nm=300.0, t_a=50.0, full=False):
    """Residue on terrace a: (residue_features, the residue mask on the flake (bool, the size of d) if full else
    None).  d thickness-change map (nm), F, A flake / terrace a, rel the picture relative to terrace a's colour (RGB
    float, terrace a = 1), all the same size; px um per pixel."""
    H, W = F.shape
    Fb, Ab = F.astype(bool), A.astype(bool)
    out = np.zeros((H, W), bool) if full else None
    ys, xs = np.nonzero(Fb)
    if ys.size == 0 or not Ab.any():
        return residue_features(np.zeros(1, bool), np.zeros(1, bool), px), out
    mg = int(round(3.0 / px))
    y0, y1, x0, x1 = max(ys.min() - mg, 0), min(ys.max() + mg + 1, H), max(xs.min() - mg, 0), min(xs.max() + mg + 1, W)
    h, w = y1 - y0, x1 - x0
    sz = (max(1, w // RES_DS), max(1, h // RES_DS))
    q = px * w / sz[0]
    Fu = (cv2.resize(Fb[y0:y1, x0:x1].astype(np.float32), sz, interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    Au = (cv2.resize(Ab[y0:y1, x0:x1].astype(np.float32), sz, interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    Ai = cv2.erode(Au, _disk(1.0 / q)).astype(bool)
    if Ai.sum() < 30:
        Ai = Au.astype(bool)
    if not Ai.any():
        return residue_features(np.zeros(1, bool), np.zeros(1, bool), px), out
    dd = cv2.resize(d[y0:y1, x0:x1].astype(np.float32), sz, interpolation=cv2.INTER_AREA)
    dc = cv2.GaussianBlur(np.clip(dd - np.float32(np.median(dd[Ai])), -RES_CLIP_NM, RES_CLIP_NM), (0, 0), RES_SMOOTH_UM / q)
    off = ((np.abs(dc) >= RES_CLIP_NM * 0.95) & (Fu > 0)).astype(np.uint8)
    lvl = (Fu > 0) & (off == 0)
    r8 = np.clip(cv2.resize(rel[y0:y1, x0:x1], sz, interpolation=cv2.INTER_AREA), 0.92, 1.08)   # (as the c8 picture)
    r8 -= np.median(r8[Ai], axis=0).astype(np.float32)
    g8 = r8.mean(axis=2)
    hp = (g8 - cv2.GaussianBlur(g8, (0, 0), 1.5 / q))[Ai]
    noise = float(1.4826 * np.median(np.abs(hp - np.median(hp))))
    g8 = cv2.GaussianBlur(g8, (0, 0), 0.4 / q)
    dark = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, _disk(RES_DARK_UM / q))
    seeds = _hyst(np.where(lvl, dark, 0), max(RES_DARK_MIN, RES_DARK_K * noise), RES_HYST)
    yel = cv2.GaussianBlur(r8, (0, 0), RES_SMOOTH_UM / q) @ _yellow_dir(material, oxide_nm, t_a)
    yel = cv2.morphologyEx(yel, cv2.MORPH_TOPHAT, _disk(RES_YELLOW_UM / q))
    seeds |= _hyst(np.where(lvl, yel, 0), RES_YELLOW_MIN, RES_HYST)
    res = _blobs(seeds, q, RES_MIN_UM2 / (q * q))
    if RES_GROW_UM > 0 and res.any():
        res = cv2.dilate(res.astype(np.uint8), _disk(RES_GROW_UM / q)).astype(bool) & lvl
    if RES_ISLAND_UM2 > 0 and off.any():        # small off-level islands inside the near-level part: saturated residue
        n, lab, st, _ = cv2.connectedComponentsWithStats(off, connectivity=8)
        rim = ~cv2.erode(Fu, _disk(1.0 / q)).astype(bool)
        touch = np.zeros(n, bool)
        touch[np.unique(lab[rim & (off > 0)])] = True
        isl = (st[:, cv2.CC_STAT_AREA] * q * q < RES_ISLAND_UM2) & ~touch
        isl[0] = False
        res = res | isl[lab]
    if full:
        out[y0:y1, x0:x1] = cv2.resize(res.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out &= Fb
    return residue_features(res, Au, q), out


def residue_features(res, A, px):
    """Terrace a's residue (res, A: residue and terrace a masks, the same size, with a margin around A; px um per
    pixel): res_a_frac (share of its area), res_touch_frac (share of its boundary band, 0.7 um, with residue within
    0.7 um, inside or out), res_bleed_um (how far residue that reaches a's boundary goes in, um: residue at the
    flake's edge bleeding in)."""
    Ab = A.astype(bool)
    if not Ab.any():
        return dict(res_a_frac=0.0, res_touch_frac=0.0, res_bleed_um=0.0)
    rb = res.astype(bool)
    Au = Ab.astype(np.uint8)
    k = _disk(0.7 / px)
    bnd = Ab & ~cv2.erode(Au, k).astype(bool)                          # terrace a's boundary band (0.7 um)
    near = cv2.dilate(rb.astype(np.uint8), k).astype(bool)
    inA = rb & Ab
    f = dict(res_a_frac=float(inA.sum()) / max(float(Ab.sum()), 1.0),
             res_touch_frac=float((near & bnd).sum()) / max(float(bnd.sum()), 1.0), res_bleed_um=0.0)
    n, lab = cv2.connectedComponents(inA.astype(np.uint8), connectivity=8)
    if n > 1:
        tb = np.zeros(n, bool)
        tb[np.unique(lab[inA & bnd])] = True
        tb[0] = False
        m = tb[lab]
        if m.any():
            depth = cv2.distanceTransform(np.pad(Au, 1), cv2.DIST_L2, 3)[1:-1, 1:-1] * px
            f["res_bleed_um"] = float(depth[m].max())
    return f


def defects_version(material="hbn"):
    """The defects version the material's pick model was fitted on (profile pick "defects"; 1 when it does not say,
    VERSION when there is no pick model)."""
    if material not in _PICK_V:
        v = VERSION
        try:
            from .screen100 import Screen
            sc = Screen(material)
            pick = getattr(sc, "pick", None) if getattr(sc, "ok", False) else None
            if pick and pick.get("inputs"):
                v = int(pick.get("defects", 1))
        except Exception:
            pass
        _PICK_V[material] = v
    return _PICK_V[material]


_PICK_V = {}


def flake_defects(rgb, flake, keep_masks=False, version=None):
    """Defects of a designer flake's terrace a (rounded dict, KEYS + ms), from its natural 100x picture: rgb = the
    scan's picture (RGB uint8, full resolution as taken) or a path to it or to the bundle's copy (x100_nat_*: used as it
    is); the scan's picture is first made the bundle's view (view_bgr).  None when the picture or the outlines do not
    allow it.  keep_masks: also (dict, masks with box (y0, y1, x0, x1) in the half-resolution picture, and that picture h).
    version: of the measurement (defects; default: the one the hBN pick model was fitted on), stored as "v"."""
    t0 = time.perf_counter()
    version = defects_version() if version is None else int(version)
    if isinstance(rgb, np.ndarray):
        rgb = cv2.cvtColor(view_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)
    else:
        b = np.fromfile(str(rgb), np.uint8)
        a = cv2.imdecode(b, cv2.IMREAD_COLOR)
        if a is None:
            return None
        if not Path(rgb).name.startswith("x100_"):
            a = view_bgr(a, len(b))
        rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    t = (flake.get("terraces") or [{}])[0]
    fr = flake.get("frame") or {}
    if not fr or t.get("t") is None:
        return None
    k = float(fr.get("k", 0.5))
    h = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if abs(k - 1.0) > 1e-6 else rgb
    F, A = masks(h.shape[:2], flake)
    if F is None or A.sum() < 50:
        return None
    px = float(flake["px_um"])
    bx, by, bw, bh = cv2.boundingRect(F)
    if bw == 0:
        return None
    # the box: the flake and 3 um (the residue detector's margin; the defects' filters average over terrace a only,
    # the thickness map's background is the whole picture's); version 1: its 3 um Gaussian (radius ~4 sigma) + 5x5
    # median + 2 um box + 1 um erosion stay inside the margin, so the values on the flake are those of the whole picture
    m = int(np.ceil(3.0 / px)) + 8 if version >= 2 else int(np.ceil(4 * 3.0 / px)) + _k(2.0, px) + _k(1.0, px) + 8
    y0, y1 = max(0, by - m), min(h.shape[0], by + bh + m)
    x0, x1 = max(0, bx - m), min(h.shape[1], bx + bw + m)
    lin = h.astype(np.float32)
    lin += 0.5
    mat, ox = flake.get("material", "hbn"), flake.get("oxide_nm") or 300.0
    d, rel = thickness_map(lin, F, A, mat, ox, float(t["t"]), (y0, y1, x0, x1), rel=True)
    Fc, Ac = F[y0:y1, x0:x1], A[y0:y1, x0:x1]
    r = defects(d, Ac, Fc, px, version=version)
    if r is None:
        return None
    rf, res = residue(d, Fc, Ac, px, rel, mat, ox, float(t["t"]), full=keep_masks)
    r[0].update(rf)
    f = {kk: round(float(r[0][kk]), 4 if kk.endswith("frac") or kk == "tex_p90" else 2) for kk in KEYS}
    f["v"] = version
    f["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if keep_masks:
        return f, dict(r[1], residue=res, box=(y0, y1, x0, x1), F=F, A=A), h
    return f


def _logit(p):
    p = min(max(float(p), 1e-3), 1 - 1e-3)
    return float(np.log(p / (1 - p)))


def pick_p(pick, p_quality, f, centre):
    """The pick model's P(keep) of one flake: p_quality (label model), its defects f, centre {feature: run centre}."""
    v = {"p_quality_logit": _logit(p_quality)}
    for kk in pick["inputs"][1:]:
        v[kk] = float(f[kk]) - float(centre.get(kk, 0.0))
    x = np.array([v[kk] for kk in pick["inputs"]], float)
    z = float(((x - np.array(pick["mean"])) / np.array(pick["std"])) @ np.array(pick["coef"]) + pick["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def centres(pick, fs):
    """The run centre of each centred feature over the defect dicts fs (one run's hBN flakes): the median, shrunk to
    the training runs' median by n0 flakes (a bundle with its first chip only is half-way)."""
    c = pick.get("centre") or {}
    if not c or c.get("by") in (None, "none"):
        return {}
    n0 = float(c.get("n0", 0))
    out = {}
    for kk, ref in zip(c["features"], c["ref"]):
        v = np.array([float(f[kk]) for f in fs], float)
        out[kk] = float(ref) if not len(v) else float((len(v) * np.median(v) + n0 * float(ref)) / (len(v) + n0))
    return out


def score_flakes(flakes, screen=None, log=None):
    """p_keep / tier of every hBN flake (terrace a) with defects, from the hBN pick model, centred on these flakes (one
    run: a bundle or a run's catalog); flakes without defects keep theirs.  p_quality stays the label model's (filled
    from p_keep for terraces scored before p_quality was stored).  Returns {tier: count} of the flakes scored."""
    if screen is None:
        from .screen100 import Screen
        screen = Screen("hbn")
    pick = getattr(screen, "pick", None) if getattr(screen, "ok", False) else None
    if not pick or not pick.get("inputs"):
        return {}
    ts = []
    pv, n_other = int(pick.get("defects", 1)), 0
    for f in flakes:
        if f.get("material") != "hbn" or not f.get("terraces"):
            continue
        t = f["terraces"][0]
        if t.get("p_quality") is None and t.get("p_keep") is not None and "p_pick" not in t:
            t["p_quality"] = t["p_keep"]
        if t.get("defects") and t.get("p_quality") is not None:
            if int(t["defects"].get("v", 1)) != pv:     # (measured otherwise than the model was fitted on: keeps its tier)
                n_other += 1
                continue
            ts.append(t)
    cen = centres(pick, [t["defects"] for t in ts])
    n = {}
    for t in ts:
        p = pick_p(pick, t["p_quality"], t["defects"], cen)
        tier = "reject" if p < pick["reject_below"] else "keep" if p >= pick["keep_above"] else "review"
        if (t.get("seed") or {}).get("fits") is False:
            tier = "review"                 # (its terrace does not fit the 10x one: the screen may have judged the wrong thing)
        t["p_keep"], t["p_pick"], t["tier"] = round(p, 4), True, tier
        n[tier] = n.get(tier, 0) + 1
    if log and ts:
        log(f"hBN screening (decision model, run centre {', '.join(f'{k} {v:.1f}' for k, v in cen.items()) or 'none'}): "
            + ", ".join(f"{v} {k}" for k, v in sorted(n.items())) + f" of {len(ts)} flakes")
    if log and n_other:
        log(f"hBN screening: {n_other} flakes measured with another defects version than the model's ({pv}) keep their "
            "tier (python -m flakeid.defects100 <bundle> measures them again)")
    return n


def _replace(tmp, dst, tries=6):
    """os.replace with retries (Windows: a virus scanner or indexer may hold the file a moment)."""
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def rescore(bundle_path, pictures=(), mapping=None, out=None, recompute=False, log=print):
    """Defects of the hBN flakes of a built bundle that have none or another version than the pick model's (or all:
    recompute), from local natural 100x pictures, then p_keep / tier of every hBN flake (score_flakes); bundle.json
    written (out, default in place).  A flake's picture is the one the bundle was made from: mapping[flake id] (a
    JSON file {id: path}), else the bundle's own file f["img"] in the bundle's folder or in one of `pictures` (never a
    picture found by its F number: they repeat within a chip and in resumed scans)."""
    p = Path(bundle_path)
    p = p / "bundle.json" if p.is_dir() else p
    b = json.loads(p.read_text(encoding="utf-8"))
    mp = json.loads(Path(mapping).read_text(encoding="utf-8")) if mapping else {}
    dirs = [p.parent] + [Path(q) for q in pictures]
    n_new = n_miss = n_fail = 0
    ms = []
    ver = defects_version()
    for f in b["flakes"]:
        if f.get("material") != "hbn" or not f.get("terraces"):
            continue
        dfx = f["terraces"][0].get("defects")
        if dfx and int(dfx.get("v", 1)) == ver and not recompute:
            continue
        img = mp.get(f["id"]) or mp.get(f["id"].split(":", 1)[-1])
        if not img and isinstance(f.get("img"), str):
            name = Path(f["img"]).name
            img = next((str(q) for d in dirs for q in (d / f["img"], d / name, d / "100x" / name) if q.is_file()), None)
        if not img or not Path(img).is_file():
            n_miss += 1
            continue
        r = flake_defects(img, f)
        if r is None:
            n_fail += 1
            continue
        ms.append(r.pop("ms"))
        f["terraces"][0]["defects"] = r
        n_new += 1
    n = score_flakes(b["flakes"], log=log)
    dst = Path(out) if out else p
    tmp = dst.with_name("." + dst.name + ".part")
    tmp.write_text(json.dumps(b, separators=(",", ":")), encoding="utf-8")
    _replace(tmp, dst)
    log(f"{p}: defects of {n_new} hBN flakes ({np.median(ms) if ms else 0:.0f} ms each)"
        + (f", {n_miss} without a local picture" if n_miss else "") + (f", {n_fail} not measurable" if n_fail else "")
        + f"; tiers {', '.join(f'{v} {k}' for k, v in sorted(n.items())) or 'unchanged (no hBN decision model)'} -> {dst}")
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="a designer folder (bundle.json in it) or a bundle.json, on this computer")
    ap.add_argument("--pictures", action="append", default=[], help="a folder holding the bundle's x100_nat_* pictures "
                    "(also <folder>/100x; repeat)")
    ap.add_argument("--map", help="JSON {flake id: natural 100x picture} (ids with or without the run prefix)")
    ap.add_argument("-o", "--out", help="write the re-scored bundle.json here (default: in place)")
    ap.add_argument("--recompute", action="store_true", help="defects of every hBN flake again, also those that have them")
    a = ap.parse_args(argv)
    rescore(a.bundle, a.pictures, a.map, a.out, a.recompute)
    return 0


if __name__ == "__main__":
    sys.exit(main())
