"""Clean areas: large areas of one colour of the right thickness, whatever the flake's outline (live --target-mode
avram_new1; Avram, 2026-10-25).

The flake identification (pipeline.find_flakes) starts from the flakes' outlines and rejects regions by rules about
them (residue, colour fit, flatness, ...), so a large clean flake can be missed.  Here only the colour counts:

1. every pixel's thickness from its colour: the nearest point of the thin-film model's colour curve inside the
   thickness window (so a colour that a much thicker film also has counts as the thinner, in-window thickness), and how
   well the colour fits it;
2. the pixels of the right colour: the fit within max(fit_abs, fit_rel |C|) (the identification's own tolerance for a
   colour match), the thickness inside the window, clearly not substrate (|C| >= min_contrast), and flat (a low
   colour gradient: not a step or an edge);
3. connected areas of them; an area's clean area = its pixels within max(same_abs_nm, same_rel x thickness) of its
   median thickness, x the pixel area.

The flake outline is looked up only afterwards (the identified flake under the area's middle, if any), for the viewer
and the 100x labels.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .segment import gradient_magnitude, smooth_contrast


def clean_areas(contrast, chip_mask, est, px_um, t_range, min_contrast, min_area_um2=100.0, fit_abs=0.20, fit_rel=0.25,
                grad_max=0.035, same_abs_nm=1.0, same_rel=0.06):
    """Clean areas of the working-resolution contrast map (H, W, 3): [dict(tx, ty (working px, the middle: the point
    deepest inside the area's clean pixels), area_um2, thickness_nm, d_um (largest circle inside), n_px, contrast)],
    largest first.  est: the piece's ThicknessEstimator; t_range: (min, max) nm."""
    cs = smooth_contrast(np.asarray(contrast, np.float32))
    g = gradient_magnitude(cs)
    mag = np.linalg.norm(cs, axis=2)
    cand = np.asarray(chip_mask, bool) & (mag >= min_contrast) & (g < grad_max)
    th = np.full(cand.shape, np.nan, np.float32)
    win = (est.d >= t_range[0]) & (est.d <= t_range[1])
    if cand.any() and win.any():
        # the colour against the window's part of the model curve only: a colour that a much thicker film has nearly as
        # well (interference colours repeat) counts as the thickness inside the window (the thinner one: errs thin)
        from scipy.spatial import cKDTree
        res, idx = cKDTree(est.curve[win]).query(cs[cand])
        d = est.d[win][idx]
        th[cand] = np.where(res <= np.maximum(fit_abs, fit_rel * mag[cand]), d, np.nan).astype(np.float32)
    good = np.isfinite(th)
    lab, n = ndimage.label(good)
    out = []
    if not n:
        return out
    min_px = max(4, int(min_area_um2 / px_um ** 2))
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == k
        if m.sum() < min_px:
            continue
        t = th[sl][m]
        med = float(np.median(t))
        same = np.zeros_like(m)
        same[m] = np.abs(t - med) <= max(same_abs_nm, same_rel * med)
        n_px = int(same.sum())
        area = n_px * px_um ** 2
        if area < min_area_um2:
            continue
        dist = cv2.distanceTransform(np.pad(same.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        ty, tx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        out.append(dict(tx=int(tx + sl[1].start), ty=int(ty + sl[0].start), area_um2=round(float(area), 1),
                        thickness_nm=round(med, 1), d_um=round(2.0 * float(dist.max()) * px_um, 1), n_px=n_px,
                        contrast=[round(float(v), 4) for v in np.median(cs[sl][same], axis=0)]))
    out.sort(key=lambda c: -c["area_um2"])
    return out
