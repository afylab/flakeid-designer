"""The designer's flake outline refined on its 100x picture (tools/make_catalog_100x.py; 2026-10-22).

The catalog's colour mask (tools/make_catalog_100x.flake_mask) is the 100x picture blurred with sigma 2 px and
thresholded at a fixed 10 % from the substrate, limited to the 10x outline grown by 5 um and closed with 7 x 7 px.  The
blur and the low threshold put its edge in the dark halo outside the flake (a rim of substrate inside the outline); the
10x limit cuts chords through flake parts the 10x outline missed; the closing joins neighbouring fragments across
narrow gaps.  Here, in a band (BAND_UM) around that outline, at the cut-out's scale (half resolution):

  * contrast C = picture / substrate - 1 (Gaussian 0.7 px), D = max over channels |C|; substrate: the most common
    colour of the bare area beyond the band, noise floor from the spread of that colour's cluster;
  * the edge is where D falls to half the flake's own local contrast (D of the flake interior nearby: normalised
    Gaussian over the old outline's inside), but never above T_ABS (darker flake parts next to a bright terrace stay
    in) nor below the noise floor: a step edge is found near its middle, the halo (10-20 % against 100-200 % on the
    flake) stays outside;
  * terrace a must be flake by that rule (TA_ON of it), else the old outline is kept ("terrace a off the flake");
  * the connected part holding terrace a, after an opening of open_um (thin bridges to neighbouring fragments and
    residue strands cut), grown back by the opening radius inside the mask (corners kept);
  * soft flaps (lifted / folded, contrast ~0.15-0.35 against the local substrate) inside the old outline added to the
    kept part (SOFT_FLAPS; residue streaks and the halo left out: see there);
  * a part added outside the old outline whose body (wider than ~2 um) reaches the band's outer edge is taken off
    again: it goes on beyond what is looked at, mostly a neighbouring flake joined along a strip or across a crack (the
    old outline is kept there, so the refinement never takes in more of a neighbour than that);
  * cracks and inlets narrower than CLOSE_UM closed (closing of the kept part only: no neighbour is joined by it),
    holes ignored (the outline is one polygon), smoothed (Gaussian smooth_um, cut at 0.5), traced, simplified
    (approxPolyDP eps_um) and, given crystal_deg, long straight edges (>= 5 um) turned onto the crystal directions
    (crystal_deg + n x 30 deg) when within 3 deg (vertices moved <= 1.5 um, area change < 3 %);
  * parts of the old outline beyond the picture (or beyond a mosaic's covered area) are kept as they were.
The old outline is kept (used False, why: the reason) when the flake contrast is too low, the picture is out of focus
(edge sharpness < min_sharp), terrace a is not on the flake, or the area ratio new / old is outside
[min_area_ratio, max_area_ratio] (shrinking is allowed further: it is mostly neighbours the old outline swallowed).

Verified on bn_20261020_run_1 (2026-10-22, 15 flakes not used while tuning, judged on the pictures): 13 better, 1 mixed,
1 unclear, none worse; ~22 ms per flake (i7-7700).  Known limits, as with the old outline: neighbours joined across a
dark gap inside the old outline stay joined; where a 10x straight cut missed more than BAND_UM of the flake, the rest
stays missing.  The catalog refines without crystal directions (tools/terraces100.py finds them later); the designer
bundle turns the refined outline's edges onto them then (snap_flakes, tools/make_designer_bundle.py).

Coordinates: refine_outline works in picture px with pixel centres at integers; its outline runs through the centres
of the refined mask's boundary pixels (as cv2.findContours).  The designer's outlines are in the canvas convention
(pixel j spans [j, j+1]; make_catalog_100x.poly_of adds 0.5): subtract 0.5 going in, add 0.5 coming out.

    python -m flakeid.outline100 <designer folder or bundle.json> [--pictures DIR ...] [-o out.json] [--recompute]

refines the outlines and flake areas of a built bundle in place (patch: the pictures and cut-outs are not rebuilt) from
its own natural 100x views (x100_nat_*, x100_mos_* for flakes with a terrace mosaic).  Run it on a copy on this
computer, not on the share.  Python 3.8, numpy + OpenCV only.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BAND_UM = 8.0                  # the band around the old outline that is looked at
CLOSE_UM = 3.0                 # cracks and inlets narrower than this are closed
T_ABS = 0.35                   # the edge threshold is never above this contrast (cap: darker parts next to bright stay in)
TA_ON = 0.9                    # this much of terrace a must be above the flake threshold (else the old outline is kept)
# Soft low-contrast flaps attached to a flake (lifted / folded; contrast ~0.15-0.35 next to a flake of 1-2, under the
# capped threshold) are flake (Jordan, 2026-10-22): kept when inside the old outline.  Contrast against the local
# substrate (SOFT_BG_UM around, outside the old outline) above max(SOFT_MIN_D, SOFT_FLOOR_K x noise floor), bodies wider
# than SOFT_BODY_UM (the edge halo, 10-20 % and 1-2 um wide, does not pass), grown into their fainter parts down to
# SOFT_LOW_D; joined when within CLOSE_UM of the flake and mostly soft (SOFT_SHARE), not streaky (SOFT_TEX), not in a
# field of the like around (SOFT_RING), and, when bluish as residue streaks are (SOFT_TINT), along the flake's edge for
# SOFT_TINT_CONTACT_UM.  bn_20261020_4-f0057: its 475 um2 top flap kept.  False: left out.
SOFT_FLAPS = True
SOFT_MIN_D = 0.12
SOFT_FLOOR_K = 1.5
SOFT_BODY_UM = 2.0
SOFT_BG_UM = 6.0
SOFT_LOW_D = 0.06
SOFT_SHARE = 0.5
SOFT_RING = 0.08
SOFT_TEX = 0.18
SOFT_TINT = 0.1
SOFT_TINT_CONTACT_UM = 12.0
# Small thick fragments touching the flake's edge (outside the old outline) are added.  False: an added part smaller
# than FRAG_MAX_UM2 whose contrast is off the flake's next to it by more than half of it is taken off again.
EDGE_FRAGMENTS = True
FRAG_MAX_UM2 = 50.0


def _disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _pts(p):
    return np.asarray(p, np.float64).reshape(-1, 2)


def _fill(shape, pts, origin):
    m = np.zeros(shape, np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(m, [np.round(pts - origin).astype(np.int32)], 1)
    return m


def _crop(a, x0, y0, x1, y1, fill=0):
    """a[y0:y1, x0:x1] of an image, the parts beyond it filled."""
    h, w = a.shape[:2]
    out = np.full((y1 - y0, x1 - x0) + a.shape[2:], fill, a.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


def substrate_mode(rgb, valid=None, q=4):
    """Most common colour (bins of q levels; median of the pixels in the peak bin and its neighbours) and the share
    of the pixels it covers."""
    a = rgb.reshape(-1, 3)
    if valid is not None:
        a = a[valid.ravel()]
    a = a[a.max(axis=1) > 8]
    if len(a) < 50:
        return None, 0.0
    nb = 256 // q
    qa = np.minimum(a // q, nb - 1).astype(np.int64)
    key = (qa[:, 0] * nb + qa[:, 1]) * nb + qa[:, 2]
    kb = int(np.argmax(np.bincount(key, minlength=nb ** 3)))
    c = np.array([kb // (nb * nb), (kb // nb) % nb, kb % nb], np.float64) * q + q / 2.0
    near = np.abs(a.astype(np.int16) - np.round(c).astype(np.int16)).max(axis=1) <= 1.5 * q
    return np.median(a[near], axis=0).astype(np.float32), float(near.mean())


def _trace(mask, sigma):
    """The smoothed mask's outline (boundary pixel centres of the mask smoothed by a Gaussian sigma and cut at 0.5):
    Nx2 float, the largest outer contour."""
    bx, by, bw, bh = cv2.boundingRect(mask.view(np.uint8))
    if bw == 0:
        return None
    p = int(3 * sigma) + 2
    h, w = mask.shape
    x0, y0, x1, y1 = max(bx - p, 0), max(by - p, 0), min(bx + bw + p, w), min(by + bh + p, h)
    g = cv2.GaussianBlur(mask[y0:y1, x0:x1].view(np.uint8) * np.uint8(255), (0, 0), max(sigma, 0.5))
    cnts, _ = cv2.findContours((g > 127).view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) + [x0, y0]


def _sharpness(D8, T8, dense, edge_frac):
    """Mean steepness of the contrast across the outline: |grad D| (Sobel over 5 x 5 px means, at the points only)
    over the flake's local contrast, at every 3rd contour point; ~1 / edge width in px (sharp pictures ~0.5, out of
    focus < 0.1)."""
    p = np.round(dense[::3]).astype(np.int64)
    h, w = D8.shape
    S = cv2.integral(D8)                                     # (box means from the integral image)

    def box(dx, dy):
        xa, ya = np.clip(p[:, 0] + dx - 2, 0, w), np.clip(p[:, 1] + dy - 2, 0, h)
        xb, yb = np.clip(p[:, 0] + dx + 3, 0, w), np.clip(p[:, 1] + dy + 3, 0, h)
        n = np.maximum((xb - xa) * (yb - ya), 1)
        return (S[yb, xb] - S[ya, xb] - S[yb, xa] + S[ya, xa]) / n * 0.01
    gx = (box(1, -1) + 2 * box(1, 0) + box(1, 1) - box(-1, -1) - 2 * box(-1, 0) - box(-1, 1)) / 8.0
    gy = (box(-1, 1) + 2 * box(0, 1) + box(1, 1) - box(-1, -1) - 2 * box(0, -1) - box(1, -1)) / 8.0
    G = np.hypot(gx, gy)
    dl = np.maximum(T8[p[:, 1].clip(0, h - 1), p[:, 0].clip(0, w - 1)].astype(np.float32) * 0.01 / edge_frac, 0.05)
    return float(np.mean(G / dl))


def snap_polygon(P, dense, idx, crystal_deg, px_um, min_len_um=5.0, tol_deg=3.0, max_move_um=1.5):
    """Edges of the closed polygon P (Nx2; vertex i = dense[idx[i]]) at least min_len_um long and within tol_deg of
    crystal_deg + n x 30 deg turned onto that direction, through the mean of the contour points they stand for;
    vertices between two turned edges moved to their crossing, next to one turned edge onto it (moves over
    max_move_um not made).  Returns (P', n edges turned)."""
    n = len(P)
    if crystal_deg is None or n < 3:
        return P, 0
    lines = [None] * n
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        L = float(np.hypot(*(b - a))) * px_um
        if L < min_len_um:
            continue
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        d = (ang - crystal_deg + 15.0) % 30.0 - 15.0
        if abs(d) > tol_deg:
            continue
        t = math.radians(ang - d)
        u = np.array([math.cos(t), math.sin(t)])
        i0, i1 = idx[i], idx[(i + 1) % n]
        seg = dense[i0:i1 + 1] if i1 >= i0 else np.vstack([dense[i0:], dense[:i1 + 1]])
        if len(seg) < 2:
            seg = np.vstack([a, b])
        lines[i] = (seg.mean(axis=0), u)
    out = P.copy()
    for j in range(n):
        la, lb = lines[j - 1], lines[j]                      # edge j-1 ends at vertex j, edge j starts there
        v = P[j]
        if la is not None and lb is not None:
            (p1, u1), (p2, u2) = la, lb
            cr = u1[0] * u2[1] - u1[1] * u2[0]
            if abs(cr) > math.sin(math.radians(10.0)):
                s = ((p2 - p1)[0] * u2[1] - (p2 - p1)[1] * u2[0]) / cr
                q = p1 + s * u1
            else:                                            # (nearly parallel: the mean of the two projections)
                q = 0.5 * (p1 + np.dot(v - p1, u1) * u1 + p2 + np.dot(v - p2, u2) * u2)
        elif la is not None or lb is not None:
            p1, u1 = la if la is not None else lb
            q = p1 + np.dot(v - p1, u1) * u1
        else:
            continue
        if float(np.hypot(*(q - v))) * px_um <= max_move_um:
            out[j] = q
    return out, sum(1 for l in lines if l is not None)


def snap_outline(outline, crystal_deg, px_um, min_len_um=5.0, tol_deg=3.0, max_move_um=1.5, max_area_change=0.03):
    """snap_polygon for a stored outline (flat [x, y, ...] or Nx2; e.g. once tools/terraces100.py has found crystal_deg
    for a flake whose outline was refined without it): each edge stands for its own points every px.  Returns (flat
    list, n edges turned); unchanged when the area would change by more than max_area_change."""
    P = _pts(outline)
    if crystal_deg is None or len(P) < 3:
        return [round(float(v), 2) for v in P.ravel()], 0
    dense, idx = [], []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))))))
        idx.append(len(dense))
        dense.extend(a + (b - a) * (j / float(n)) for j in range(n))
    Q, nsn = snap_polygon(P, np.array(dense), idx, float(crystal_deg), px_um, min_len_um, tol_deg, max_move_um)
    a0, a1 = abs(cv2.contourArea(P.astype(np.float32))), abs(cv2.contourArea(Q.astype(np.float32)))
    if a0 <= 0 or abs(a1 / a0 - 1.0) > max_area_change:
        Q, nsn = P, 0
    return [round(float(v), 2) for v in Q.ravel()], nsn


def refine_outline(picture, outline_px, terrace_a, px_um, substrate_rgb=None, crystal_deg=None, cover=None,
                   band_um=BAND_UM, open_um=1.5, close_um=CLOSE_UM, smooth_um=0.35, eps_um=0.4, snap_min_um=5.0,
                   snap_tol_deg=3.0, min_area_ratio=0.35, max_area_ratio=1.4, min_sharp=0.1, edge_frac=0.5, t_abs=T_ABS,
                   interior_um=1.5, local_um=4.0, body_um=1.0, ta_on=TA_ON, soft_flaps=SOFT_FLAPS,
                   edge_fragments=EDGE_FRAGMENTS, keep_maps=False):
    """The flake outline refined on its 100x picture (see the module docstring).

    picture: HxWx3 RGB (uint8) of the picture (or of a terrace mosaic around it), at the cut-out's scale; outline_px:
    the old outline in that picture's px (flat [x, y, ...] or Nx2; may reach beyond the picture); terrace_a: terrace a,
    a polygon as the outline or an HxW bool mask of the picture; px_um: um per picture px; substrate_rgb: the run's
    substrate colour (else the mode of the bare area around the flake: what was verified); crystal_deg: edges at
    crystal_deg + n x 30 deg (picture frame), None: no snapping; cover: HxW bool, the part of the picture that is
    image (a mosaic's covered area).
    Returns dict(outline (flat list, picture px), used (bool: the refined outline is given), why (the reason the old
    one is kept, or None), area_ratio (new / old polygon area), n_old, n_new, n_snapped, n_cut, cut_by_frame (the new
    outline runs along the edge of the picture / covered area), floor, substrate, sharpness, ta_on); keep_maps: also
    mask (the refined outline filled), origin (its top-left in the picture), D, T, cov (canvas)."""
    old = _pts(outline_px)
    H, W = picture.shape[:2]
    res = dict(outline=[round(float(v), 2) for v in old.ravel()], used=False, why=None, area_ratio=1.0,
               n_old=len(old), n_new=len(old), n_snapped=0, cut_by_frame=False)
    ta_mask = isinstance(terrace_a, np.ndarray) and terrace_a.dtype == bool and terrace_a.shape == (H, W)
    ta = None if ta_mask else _pts(terrace_a)
    if len(old) < 3 or (ta is not None and len(ta) < 3):
        res["why"] = "no outline / terrace"
        return res
    band = max(2, int(round(band_um / px_um)))
    marg = band + max(4, int(round(4.0 / px_um)))           # (bare substrate around the band for the substrate colour)
    x0, y0 = int(math.floor(old[:, 0].min())) - marg, int(math.floor(old[:, 1].min())) - marg
    x1, y1 = int(math.ceil(old[:, 0].max())) + marg + 1, int(math.ceil(old[:, 1].max())) + marg + 1
    origin = np.array([x0, y0], np.float64)
    ch, cw = y1 - y0, x1 - x0
    if min(x1, W) <= max(x0, 0) or min(y1, H) <= max(y0, 0):
        res["why"] = "outline off the picture"
        return res
    rgb = _crop(picture, x0, y0, x1, y1)
    cov = _crop(np.ones((H, W), bool) if cover is None else cover.astype(bool), x0, y0, x1, y1, False)
    M0 = _fill((ch, cw), old, origin)
    if ta_mask:
        A = _crop(terrace_a, x0, y0, x1, y1, False)
        n_ta = int(np.count_nonzero(terrace_a))
    else:
        A = _fill((ch, cw), ta, origin) > 0
        n_ta = int(abs(cv2.contourArea(ta.astype(np.float32))))
    nA = int(np.count_nonzero(A))
    if nA < 20:
        # (terrace a drawn mostly beyond the area around the old outline: not on this flake)
        res["why"] = "terrace a outside the flake outline" if n_ta >= 20 else "terrace too small"
        return res
    if np.count_nonzero(A & (M0 > 0)) < 0.5 * nA:
        res["why"] = "terrace a outside the flake outline"
        return res
    # distance outside the old outline (px; worked out at 1/4 scale: only compared with the band's width)
    q4 = (max(1, cw // 4), max(1, ch // 4))
    d4 = cv2.distanceTransform((cv2.resize(M0, q4, interpolation=cv2.INTER_AREA) == 0).view(np.uint8), cv2.DIST_L2, 3)
    d_out = cv2.resize(d4 * (cw / float(q4[0])), (cw, ch), interpolation=cv2.INTER_LINEAR)
    d_out[M0 > 0] = 0
    bandm = d_out <= band
    cov8 = cov.view(np.uint8)
    far = cov & ~bandm
    if substrate_rgb is None:
        st = max(4, int(math.sqrt(ch * cw / 20000.0)))     # (~20000 samples)
        substrate_rgb, _ = substrate_mode(rgb[::st, ::st], far[::st, ::st])
        if substrate_rgb is None:
            res["why"] = "no bare substrate"
            return res
    sub = np.maximum(np.asarray(substrate_rgb, np.float32), 1.0)
    bl = cv2.GaussianBlur(rgb, (3, 3), 0.7)
    # D = max over channels |colour / substrate - 1|, in steps of 0.01 (uint8) through a per-channel lookup table
    lut = np.clip(np.round(np.abs(np.arange(256, dtype=np.float32)[:, None] / sub[None, :] - 1.0) * 100.0), 0, 255)
    d3 = cv2.LUT(bl, lut.astype(np.uint8).reshape(256, 1, 3))
    D8 = cv2.max(cv2.max(d3[..., 0], d3[..., 1]), d3[..., 2])
    # noise floor: robust spread of the contrast over the bare substrate around the band (its colour cluster only:
    # other flakes, residue and a mosaic's seams there would make it the spread of everything)
    cs = bl[::3, ::3][far[::3, ::3]].astype(np.float32) / sub - 1.0
    cs = cs[np.abs(cs).max(axis=1) < 0.1]
    if len(cs) > 100:
        mad = np.median(np.abs(cs - np.median(cs, axis=0)), axis=0) * 1.4826
        floor = max(0.04, 5.0 * float(mad.max()))
    else:
        floor = 0.06
    # the flake's own contrast nearby: D over the old outline's inside (eroded), normalised Gaussian at 1/8 scale
    ri = max(1, int(round(interior_um / px_um)))
    inner = cv2.erode(M0, np.ones((2 * ri + 1, 2 * ri + 1), np.uint8))
    I8 = cv2.bitwise_and(inner, cov8)
    I8[D8 <= int(round(200 * floor))] = 0
    Ac = A & cov
    nI = int(cv2.countNonZero(I8))
    if nI < 50:
        res["why"] = "low contrast"
        return res
    if not Ac.any():
        res["why"] = "terrace a beyond the picture"
        return res
    da, di = float(np.median(D8[Ac][::5])) * 0.01, float(np.median(D8[I8 > 0][::5])) * 0.01
    if da < 3 * floor:
        # (the flake around it stands out: terrace a is not on it, or hardly differs from the substrate)
        res["why"] = ("terrace a hardly differs from the substrate (contrast %.2f, flake %.2f)" % (da, di)
                      if di >= 3 * floor else "low contrast")
        return res
    s8 = 8
    sz = (max(1, cw // s8), max(1, ch // s8))
    sw = cv2.resize(I8 * np.uint8(255), sz, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    sv = cv2.resize(cv2.bitwise_and(D8, D8, mask=I8), sz, interpolation=cv2.INTER_AREA).astype(np.float32)
    sg = local_um / px_um / s8
    bw = cv2.GaussianBlur(sw, (0, 0), sg)
    Dl = np.where(bw > 1e-3, cv2.GaussianBlur(sv, (0, 0), sg) / np.maximum(bw, 1e-6), float(np.median(D8[I8 > 0][::5])))
    T8s = np.clip(np.round(np.minimum(edge_frac * Dl, 100 * t_abs)), int(round(100 * floor)), 255).astype(np.uint8)
    T8 = cv2.resize(T8s, (cw, ch), interpolation=cv2.INTER_LINEAR)
    fg = (D8 > T8) & cov & bandm
    # terrace a must be flake by the edge rule itself (the mask is joined with terrace a below: a test after that could
    # not fail)
    on = float(np.count_nonzero(fg & Ac)) / max(int(np.count_nonzero(Ac)), 1)
    res["ta_on"] = round(on, 3)
    if on < ta_on:
        res["why"] = "terrace a off the flake (%.0f %% of it above the flake threshold)" % (100 * on)
        return res
    sb = None
    if soft_flaps:
        # soft flaps inside the old outline: contrast to the local substrate (bare surroundings outside the old outline,
        # normalised Gaussian SOFT_BG_UM at 1/8 scale: a mosaic's tiles and vignetting differ in colour) above the soft
        # threshold, bodies wider than SOFT_BODY_UM (the edge halo is narrower), with their fainter parts (hysteresis
        # down to SOFT_LOW_D); joined to the flake below.  Worked on the bounding box of the candidates only.
        mx, my, mw, mh = cv2.boundingRect(M0)
        bs = (slice(my, my + mh), slice(mx, mx + mw))
        cand = np.zeros((ch, cw), np.uint8)
        cand[bs] = (D8[bs] > int(round(100 * floor))) & (M0[bs] > 0) & cov[bs] & ~fg[bs]
        qx, qy, qw, qh = cv2.boundingRect(cand)
        if qw > 0:
            wb = ((M0 == 0) & cov & (D8 < int(round(100 * t_abs)))).view(np.uint8)
            bwb = cv2.GaussianBlur(cv2.resize(wb * np.float32(1), sz, interpolation=cv2.INTER_AREA), (0, 0),
                                   SOFT_BG_UM / px_um / s8)
            bvb = cv2.GaussianBlur(cv2.resize(cv2.bitwise_and(bl, bl, mask=wb).astype(np.float32), sz,
                                              interpolation=cv2.INTER_AREA), (0, 0), SOFT_BG_UM / px_um / s8)
            loc = np.where(bwb[..., None] > 1e-3, bvb / np.maximum(bwb, 1e-6)[..., None], sub[None, None, :])
            pb = int(math.ceil(SOFT_BODY_UM / px_um)) + 2
            x0q, y0q, x1q, y1q = max(qx - pb, 0), max(qy - pb, 0), min(qx + qw + pb, cw), min(qy + qh + pb, ch)
            iy, ix = np.nonzero(cand[y0q:y1q, x0q:x1q])
            gy, gx = iy + y0q, ix + x0q
            # (bilinear at the candidates, as cv2.resize to the canvas would give)
            fy = np.clip((gy + 0.5) * (sz[1] / float(ch)) - 0.5, 0, sz[1] - 1)
            fx = np.clip((gx + 0.5) * (sz[0] / float(cw)) - 0.5, 0, sz[0] - 1)
            y0i, x0i = fy.astype(np.int64), fx.astype(np.int64)
            y1i, x1i = np.minimum(y0i + 1, sz[1] - 1), np.minimum(x0i + 1, sz[0] - 1)
            wy, wx = (fy - y0i)[:, None], (fx - x0i)[:, None]
            lq = ((loc[y0i, x0i] * (1 - wx) + loc[y0i, x1i] * wx) * (1 - wy)
                  + (loc[y1i, x0i] * (1 - wx) + loc[y1i, x1i] * wx) * wy)
            ds = np.abs(bl[gy, gx].astype(np.float32) / np.maximum(lq, 1.0) - 1.0).max(axis=1)
            kb = _disk(0.5 * SOFT_BODY_UM / px_um)
            soft = np.zeros((y1q - y0q, x1q - x0q), np.uint8)
            hi = ds > max(SOFT_MIN_D, SOFT_FLOOR_K * floor)
            soft[iy[hi], ix[hi]] = 1
            sbq = cv2.morphologyEx(soft, cv2.MORPH_OPEN, kb)
            if cv2.countNonZero(sbq):
                if SOFT_LOW_D is not None:
                    lo = ds > max(SOFT_LOW_D, floor)
                    soft[:] = 0
                    soft[iy[lo], ix[lo]] = 1
                    nL, labL = cv2.connectedComponents(cv2.morphologyEx(soft, cv2.MORPH_OPEN, kb) | sbq, connectivity=8)
                    kl = np.zeros(nL, bool)
                    kl[labL[sbq > 0]] = True
                    kl[0] = False
                    sbq = kl[labL].view(np.uint8)
                sb = np.zeros((ch, cw), np.uint8)
                sb[y0q:y1q, x0q:x1q] = sbq
    unk = ~cov                                               # beyond the picture: the old outline as it was
    fg |= (M0 > 0) & unk
    # the part holding terrace a, thin bridges cut
    r = max(1, int(round(0.5 * open_um / px_um)))
    fo = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _disk(r))
    nl, lab = cv2.connectedComponents(fo, connectivity=8)
    ov = np.bincount(lab[A], minlength=nl)
    ov[0] = 0
    if ov.max() == 0:
        res["why"] = "terrace a not in the flake mask"
        return res
    keep = ov >= max(1, 0.02 * ov.max())
    comp = keep[lab]
    grown = (cv2.dilate(comp.astype(np.uint8), _disk(r + 1)) > 0) & fg
    nl2, lab2 = cv2.connectedComponents(grown.astype(np.uint8), connectivity=8)
    keep2 = np.zeros(nl2, bool)
    keep2[np.unique(lab2[comp])] = True
    keep2[0] = False
    m = keep2[lab2] | A                                      # (holes: the outer contour is traced)
    # a part added outside the old outline whose body (wider than 2 body_um) reaches the band's outer edge goes on
    # beyond what is looked at: mostly a neighbouring flake joined along a strip or across a crack (a dark crack line
    # counts as flake, so a neck test does not separate it): taken off again, the old outline kept there
    n_cut = 0
    add = (m & (M0 == 0) & cov).view(np.uint8)
    rim = d_out > band - 1.5
    if (add.view(bool) & rim).any() or (not edge_fragments and add.any()):
        rb = max(r + 1, int(round(body_um / px_um)))
        body = cv2.morphologyEx(add, cv2.MORPH_OPEN, _disk(rb))
        nb, lb = cv2.connectedComponents(body, connectivity=8)
        hit = np.zeros(nb, bool)
        hit[lb[rim & (body > 0)]] = True
        if not edge_fragments and nb > 1:
            # (small parts whose contrast is off the flake's next to it: thick fragments)
            Dlc = cv2.resize(Dl.astype(np.float32), (cw, ch), interpolation=cv2.INTER_LINEAR)
            area = np.bincount(lb.ravel(), minlength=nb)
            for j in range(1, nb):
                if hit[j] or area[j] * px_um * px_um >= FRAG_MAX_UM2:
                    continue
                sel = lb == j
                dj, lj = float(np.median(D8[sel])), float(np.median(Dlc[sel]))
                if abs(dj - lj) > 0.5 * max(lj, 1.0):
                    hit[j] = True
        hit[0] = False
        if hit.any():
            rm = (cv2.dilate(hit[lb].view(np.uint8), _disk(rb + 1)) > 0) & add.view(bool)
            m = m & ~rm
            n3, lab3 = cv2.connectedComponents(m.view(np.uint8), connectivity=8)
            ov3 = np.bincount(lab3[A], minlength=n3)
            ov3[0] = 0
            m = (ov3 >= max(1, 0.02 * ov3.max()))[lab3] | A
            n_cut = int(hit.sum())
    rc = max(1, int(round(0.5 * close_um / px_um)))
    if sb is not None:
        # soft flap bodies within close_um of the kept part joined, with the flake-contrast bodies of the old outline
        # they hold (a flap's darker folds), when mostly soft: a thick piece reached only across soft residue (cut off
        # by the opening above) stays off
        rb = max(r + 1, int(round(0.5 * SOFT_BODY_UM / px_um)))
        mx, my, mw, mh = cv2.boundingRect(M0)                # (all of it is inside the old outline)
        mx0, my0, mx1, my1 = max(mx - 2, 0), max(my - 2, 0), min(mx + mw + 2, cw), min(my + mh + 2, ch)
        rest = (fg[my0:my1, mx0:mx1] & ~m[my0:my1, mx0:mx1] & (M0[my0:my1, mx0:mx1] > 0)
                & cov[my0:my1, mx0:mx1]).view(np.uint8)
        c = np.zeros((ch, cw), np.uint8)
        c[my0:my1, mx0:mx1] = ((sb[my0:my1, mx0:mx1] > 0) | (cv2.morphologyEx(rest, cv2.MORPH_OPEN, _disk(rb)) > 0))
        nS, labS, stS, _ = cv2.connectedComponentsWithStats(c, connectivity=8)
        near = (cv2.dilate(m.view(np.uint8), _disk(2 * rc)) > 0) & (c > 0)
        hitS = np.zeros(nS, bool)
        hitS[labS[near]] = True
        hitS &= np.bincount(labS[sb > 0], minlength=nS) >= SOFT_SHARE * stS[:, cv2.CC_STAT_AREA]
        hitS[0] = False
        # residue is not a flap: streaky (texture: median |D - D smoothed over 1 um| inside, over its median D, above
        # SOFT_TEX) or lying in a field of the like on the bare substrate around (share of the substrate 2-10 um outside
        # the old outline and within 12 um of the part at contrast 0.1-0.6 above SOFT_RING); a flap is smooth on clean
        # substrate
        rr = int(round(12.0 / px_um))
        for j in np.nonzero(hitS)[0]:
            bx, by, bw_, bh = stS[j, :4]
            ya, yb, xa, xb = max(by - rr, 0), min(by + bh + rr, ch), max(bx - rr, 0), min(bx + bw_ + rr, cw)
            sc = (labS[ya:yb, xa:xb] == j).view(np.uint8)
            dc, dq = D8[ya:yb, xa:xb], d_out[ya:yb, xa:xb]
            ring = ((cv2.distanceTransform(1 - sc, cv2.DIST_L2, 3) <= rr) & (dq > 2.0 / px_um) & (dq < 10.0 / px_um)
                    & cov[ya:yb, xa:xb])
            nr = int(np.count_nonzero(ring))
            res_share = float(np.count_nonzero(ring & (dc > 10) & (dc < 60))) / nr if nr > 50 else 0.0
            # (texture over the part's box and a margin of 3 sigma)
            g = int(math.ceil(3.0 / px_um))
            ya2, yb2, xa2, xb2 = max(by - g, 0), min(by + bh + g, ch), max(bx - g, 0), min(bx + bw_ + g, cw)
            df = D8[ya2:yb2, xa2:xb2].astype(np.float32)
            hp = np.abs(df - cv2.GaussianBlur(df, (0, 0), 1.0 / px_um))
            core = cv2.erode(sc[ya2 - ya:yb2 - ya, xa2 - xa:xb2 - xa], _disk(1.0 / px_um)) > 0
            if not core.any():
                core = sc[ya2 - ya:yb2 - ya, xa2 - xa:xb2 - xa] > 0
            tex = float(np.median(hp[core])) / max(float(np.median(df[core])), 1.0)
            if res_share > SOFT_RING or tex > SOFT_TEX:
                hitS[j] = False
                continue
            # residue streaks are bluish against the substrate (blue up, red down, green as the substrate), as are thin
            # dark flaps: a part of that tint is kept only when it lies along the flake's edge for SOFT_TINT_CONTACT_UM
            # (flaps: 17-97 um on bn_20261020; residue touching the flake: 0-8 um)
            col = bl[ya:yb, xa:xb][sc > 0].astype(np.float32).mean(axis=0) / sub
            if col[2] - max(col[0], col[1]) > SOFT_TINT:
                mc = cv2.dilate(m[ya:yb, xa:xb].view(np.uint8), np.ones((3, 3), np.uint8)) > 0
                if np.count_nonzero(mc & (sc > 0)) * px_um < SOFT_TINT_CONTACT_UM:
                    hitS[j] = False
        if hitS.any():
            m |= hitS[labS]
    # cracks and narrow inlets (narrower than close_um) closed: the outline is the flake's edge, not every slit in it
    bx, by, bw_, bh = cv2.boundingRect(m.view(np.uint8))
    bx0, by0, bx1, by1 = max(bx - rc - 2, 0), max(by - rc - 2, 0), min(bx + bw_ + rc + 2, cw), min(by + bh + rc + 2, ch)
    m[by0:by1, bx0:bx1] = cv2.morphologyEx(m[by0:by1, bx0:bx1].view(np.uint8), cv2.MORPH_CLOSE, _disk(rc)) > 0
    dense = _trace(m, smooth_um / px_um)
    if dense is None or len(dense) < 8:
        res["why"] = "no contour"
        return res
    approx = cv2.approxPolyDP(dense.astype(np.float32).reshape(-1, 1, 2), eps_um / px_um, True).reshape(-1, 2)
    approx = approx.astype(np.float64)
    P, nsn = approx, 0
    if crystal_deg is not None and len(approx) >= 3:
        # vertex -> index on the dense contour (approxPolyDP keeps input points, whole px)
        ug = np.round(dense).astype(np.int64)
        key = dict(zip((ug[:, 1] * (cw + 8) + ug[:, 0]).tolist(), range(len(ug))))
        ua = np.round(approx).astype(np.int64)
        idx = [key.get(int(v), 0) for v in (ua[:, 1] * (cw + 8) + ua[:, 0])]
        Ps, nsn = snap_polygon(approx, dense, idx, float(crystal_deg), px_um, snap_min_um, snap_tol_deg)
        a0, a1 = abs(cv2.contourArea(approx.astype(np.float32))), abs(cv2.contourArea(Ps.astype(np.float32)))
        if a0 > 0 and abs(a1 / a0 - 1.0) < 0.03:
            P = Ps
        else:
            nsn = 0
    a_old = abs(cv2.contourArea(old.astype(np.float32)))
    a_new = abs(cv2.contourArea(P.astype(np.float32)))
    ratio = a_new / max(a_old, 1e-6)
    newm = _fill((ch, cw), P + origin, origin) > 0
    edge = cv2.dilate(unk.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    res.update(area_ratio=round(ratio, 3), n_snapped=nsn, n_cut=n_cut, floor=round(floor, 3),
               substrate=[round(float(v), 1) for v in sub], cut_by_frame=bool((newm & edge).any()))
    if keep_maps:
        res.update(mask=newm, origin=origin, D=D8.astype(np.float32) * 0.01, T=T8.astype(np.float32) * 0.01, cov=cov)
    sharp = _sharpness(D8, T8, dense, edge_frac)
    res["sharpness"] = round(sharp, 3)
    if sharp < min_sharp:
        res["why"] = "blurred picture (edge sharpness %.2f)" % sharp
        return res
    if not min_area_ratio <= ratio <= max_area_ratio:
        res["why"] = "area change x%.2f" % ratio
        return res
    if float(np.count_nonzero(newm & A)) < 0.9 * nA:
        res["why"] = "terrace a not inside the traced outline"
        return res
    res.update(outline=[round(float(v), 2) for v in (P + origin).ravel()], used=True, n_new=len(P))
    return res


def refine_mask(src, fm, terrace, k, px_cut, cover=None, **kw):
    """The catalog's flake mask refined (tools/make_catalog_100x.py, right after flake_mask): src the picture (or
    terrace mosaic) the mask fm (HxW bool) was made on, terrace (HxW bool) terrace a there, k cut-out px per src px
    (1 / --downscale), px_cut um per cut-out px, cover (HxW bool or None) a mosaic's covered area.  Worked on the part
    of src around the mask, scaled to the cut-out's scale (as verified).  Returns (mask, res): mask the refined
    outline filled in src px (fm itself when the old outline is kept), res as refine_outline (without the maps) with
    outline_src = the refined outline in src px, canvas convention (pixel j spans [j, j+1]), and ms."""
    t0 = time.perf_counter()
    ys, xs = np.nonzero(fm)
    if not len(xs):
        return fm, dict(used=False, why="no flake mask", ms=0.0)
    Hs, Ws = fm.shape
    step = int(round(1.0 / k)) if abs(1.0 / k - round(1.0 / k)) < 1e-6 else 1
    m = int(math.ceil(((kw.get("band_um", BAND_UM) + 4.0) / px_cut + 8) / k))
    # (the crop: the canvas refine_outline looks at; on the grid of the scaled picture, so that a cut-out px is the
    # mean of whole picture px as in the cut-out)
    x0, y0 = max((int(xs.min()) - m) // step * step, 0), max((int(ys.min()) - m) // step * step, 0)
    x1, y1 = min(int(xs.max()) + m + 1, Ws), min(int(ys.max()) + m + 1, Hs)
    w, h = x1 - x0, y1 - y0
    sw, sh = max(8, int(round(w * k))), max(8, int(round(h * k)))
    fx, fy = sw / float(w), sh / float(h)

    def small(a, thr=127):
        return cv2.resize(a[y0:y1, x0:x1].astype(np.uint8) * np.uint8(255), (sw, sh), interpolation=cv2.INTER_AREA) > thr
    pic = cv2.resize(src[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_AREA)
    fs = small(fm)
    cnts, _ = cv2.findContours(fs.view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return fm, dict(used=False, why="no flake mask", ms=0.0)
    old = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.5, True).reshape(-1, 2).astype(np.float64)
    cs = small(cover, 250) if cover is not None else None
    res = refine_outline(pic, old, small(terrace), px_cut, cover=cs, keep_maps=True, **kw)
    mk, org = res.pop("mask", None), res.pop("origin", None)
    for kk in ("D", "T", "cov"):
        res.pop(kk, None)
    if res["used"]:
        Ms = _crop(mk.view(np.uint8), -int(org[0]), -int(org[1]), -int(org[0]) + sw, -int(org[1]) + sh)
        out = np.zeros_like(fm)
        out[y0:y1, x0:x1] = cv2.resize(Ms, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        fm = out | terrace
        P = _pts(res["outline"])
        res["outline_src"] = [round(float(v), 2) for v in
                              np.column_stack([x0 + (P[:, 0] + 0.5) / fx, y0 + (P[:, 1] + 0.5) / fy]).ravel()]
    res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return fm, res


def snap_flakes(flakes, log=None):
    """The refined outlines (outline_refined) of designer flakes whose crystal directions tools/terraces100.py found
    (crystal_deg, cut-out frame) turned onto them (snap_outline; outline_snapped: the edges turned), before the bundle
    rounds its outlines (tools/make_designer_bundle.py).  Any material.  Returns the number of flakes changed."""
    n_f = n_e = 0
    for f in flakes:
        if f.get("crystal_deg") is None or not f.get("outline_refined") or len(f.get("outline") or []) < 6:
            continue
        ol, n = snap_outline(f["outline"], f["crystal_deg"], float(f["px_um"]))
        f["outline_snapped"] = n
        if n:
            f["outline"] = ol
            n_f += 1
            n_e += n
    if log:
        log(f"outlines turned onto the crystal directions: {n_e} edges on {n_f} flakes")
    return n_f


# ------------------------------------------------------------------ patch of a built bundle (no rebuild)

def _view(dirs, name):
    for d in dirs:
        for q in (d / name, d / Path(name).name, d / "100x" / Path(name).name):
            if q.is_file():
                return q
    return None


def _read(path):
    a = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def patch_flake(f, dirs):
    """One designer flake's outline (and area) refined from the bundle's own 100x view (the terrace mosaic's when it
    has one): refine_outline's result (with crystal_deg when tools/terraces100.py found it: snapped as it refines), or
    None without a local picture.  The flake is changed when the refinement is used."""
    fr, t = f.get("frame") or {}, (f.get("terraces") or [{}])[0]
    ta = t.get("poly_10x") or t.get("poly")                     # (the catalog's terrace a, as the build refines with)
    if not fr or len(f.get("outline") or []) < 6 or len(ta or []) < 6:
        return None
    k = float(fr.get("k", 0.5))
    px = float(f["px_um"])
    off = np.array([fr["x0"], fr["y0"]], np.float64) * k        # cut-out px -> picture px at the cut-out's scale
    mos = f.get("mos") or {}
    q = _view(dirs, f["img_mos"]) if isinstance(f.get("img_mos"), str) and mos.get("w") else None
    img = _read(q) if q is not None else None
    cov = None
    if img is not None:                                          # (picture px + (mos x, y) = mosaic px)
        s = k * float(mos["w"]) / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cov = cv2.erode((img.max(axis=2) > 8).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        off = off + np.array([mos["x"], mos["y"]], np.float64) * k
    else:
        q = _view(dirs, f["img"]) if isinstance(f.get("img"), str) else None
        img = _read(q) if q is not None else None
        if img is None:
            return None
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if abs(k - 1.0) > 1e-6 else img
    old = _pts(f["outline"]) + off - 0.5
    R = refine_outline(img, old, _pts(ta) + off, px, crystal_deg=f.get("crystal_deg"), cover=cov, keep_maps=True)
    mk, org = R.pop("mask", None), R.pop("origin", None)
    for kk in ("D", "T", "cov"):
        R.pop(kk, None)
    f["outline_refined"] = bool(R["used"])
    f.pop("outline_why", None)
    if not R["used"]:
        f["outline_why"] = R["why"]
        return R
    n0 = int(np.count_nonzero(_fill(mk.shape, old, org)))
    if f.get("area") and n0:                                     # (the area as the build counts it: mask px)
        f["area"] = round(float(f["area"]) * np.count_nonzero(mk) / n0, 1)
    P = _pts(R["outline"]) + 0.5 - off
    f["outline"] = [int(round(v)) for v in P.ravel()]            # (whole cut-out px, as the bundle's)
    f["outline_snapped"] = R["n_snapped"]
    return R


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


def patch_bundle(bundle_path, pictures=(), out=None, recompute=False, only=None, log=print):
    """refine the outlines of a built bundle's flakes that have not been (outline_refined missing; all: recompute) from
    its own pictures (patch_flake), bundle.json written (out, default in place).  The step terraces
    (tools/terraces100.py --bundle) and hBN defects (python -m flakeid.defects100 --recompute) were measured inside the
    old outline: run them again after it for their values on the new one."""
    p = Path(bundle_path)
    p = p / "bundle.json" if p.is_dir() else p
    b = json.loads(p.read_text(encoding="utf-8"))
    dirs = [p.parent] + [Path(q) for q in pictures]
    n_used = n_miss = 0
    why, ms = {}, []
    for f in b["flakes"]:
        if (only and only not in f["id"]) or ("outline_refined" in f and not recompute):
            continue
        t0 = time.perf_counter()
        R = patch_flake(f, dirs)
        if R is None:
            n_miss += 1
            continue
        ms.append((time.perf_counter() - t0) * 1000)
        if R["used"]:
            n_used += 1
        else:
            w = R["why"].split(" (")[0]
            why[w] = why.get(w, 0) + 1
    dst = Path(out) if out else p
    tmp = dst.with_name("." + dst.name + ".part")
    tmp.write_text(json.dumps(b, separators=(",", ":")), encoding="utf-8")
    _replace(tmp, dst)
    log(f"{p}: outlines refined on {n_used} of {len(ms)} flakes ({np.median(ms) if ms else 0:.0f} ms each)"
        + (f"; old outline kept: {', '.join(f'{v} {k}' for k, v in sorted(why.items()))}" if why else "")
        + (f"; {n_miss} without a local picture" if n_miss else "") + f" -> {dst}")
    return n_used


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="a designer folder (bundle.json in it) or a bundle.json, on this computer")
    ap.add_argument("--pictures", action="append", default=[], help="a folder holding the bundle's x100_nat_* / x100_mos_* "
                    "pictures (also <folder>/100x; repeat)")
    ap.add_argument("-o", "--out", help="write the patched bundle.json here (default: in place)")
    ap.add_argument("--recompute", action="store_true", help="every flake again, also those already refined")
    ap.add_argument("--only", help="flakes whose id contains this only")
    a = ap.parse_args(argv)
    patch_bundle(a.bundle, a.pictures, a.out, a.recompute, a.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
