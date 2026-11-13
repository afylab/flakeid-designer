"""Segmentation of flakes into regions of uniform optical contrast.

1. Edge-preserving smoothing of the contrast map.
2. Foreground = chip pixels whose contrast differs from the substrate.
3. "Cores" = foreground pixels where the local contrast gradient is small
   (flat terraces), eroded and labelled.
4. Each core (largest first) is grown into the connected pixels whose
   contrast matches the core's median within a tolerance.  Terraces of
   different thickness therefore become separate regions, and pixels that
   match nothing (edges, blurred junk, gradients) stay unassigned instead of
   being forced into a neighbouring region.
5. The connected foreground component a region belongs to is its parent
   flake (so terraces of the same flake share a parent id).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_holes


@dataclass
class SegmentParams:
    fg_sigma: float = 4.0          # foreground threshold in units of smoothed noise
    fg_min: float = 0.035          # absolute minimum contrast for foreground
    grad_max: float = 0.035        # max contrast change per pixel inside a core
    core_erode_px: int = 1
    min_core_px: int = 6
    tol_abs: float = 0.045         # region growing tolerance: max(tol_abs, tol_rel*|C|)
    tol_rel: float = 0.08
    min_region_px: int = 10
    max_hole_px: int = 25          # holes up to this size are filled


@dataclass
class Region:
    label: int
    parent: int
    slices: tuple                  # bounding box slices in the working image
    mask: np.ndarray               # bool mask within the bounding box
    median: np.ndarray             # (3,) median contrast
    spread: float                  # robust spread of contrast (norm of per-channel MAD)
    uniformity: float              # fraction of pixels close to the median
    saturated_frac: float
    saturated_ch: np.ndarray       # (3,) fraction of clipped pixels per channel
    touches_chip_edge: bool
    edge_width: float              # 20-80 % edge width in working pixels (sharp ~1-2)
    thick_border: float            # fraction of the surrounding ring with strong contrast (|C|>0.25)
    ring_similar: float            # fraction of the surroundings with similar hue/strength (haze fields)


def smooth_contrast(contrast: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(contrast, d=5, sigmaColor=0.08, sigmaSpace=1.5)


def gradient_magnitude(c: np.ndarray) -> np.ndarray:
    g = np.zeros(c.shape[:2], np.float32)
    for k in range(3):
        gx = cv2.Sobel(c[..., k], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(c[..., k], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        g = np.maximum(g, np.sqrt(gx * gx + gy * gy))
    return g


def edge_width(c, m, med, reach=6):
    """Width (px) of the edge of region ``m``: the boundary-averaged contrast
    profile (projected on the region's contrast vector) versus signed distance
    from the boundary, measured between its 20 % and 80 % levels.  Averaging
    over the whole boundary makes this robust even for faint regions.  Sharp
    flake edges give ~1-2 px; diffuse smudges and defocused areas give more."""
    n2 = float(np.dot(med, med))
    if n2 < 1e-8:
        return float("nan")
    m8 = np.pad(m.astype(np.uint8), reach + 1)
    cp = np.pad(c, ((reach + 1,) * 2, (reach + 1,) * 2, (0, 0)), mode="edge")
    din = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
    d = np.where(m8 > 0, din - 0.5, -(dout - 0.5))
    sel = np.abs(d) <= reach
    # outside the region keep only pixels whose colour lies between the region
    # and the bare substrate, so that neighbouring terraces or other flakes do
    # not masquerade as a gradual edge (computed on the band within reach only: the same pixels, in the same order)
    cs_, ds_ = cp[sel], d[sel]
    t_all = (cs_ @ med) / n2
    perp = np.linalg.norm(cs_ - t_all[:, None] * med, axis=1)
    between = (t_all > -0.3) & (t_all < 1.05) & (perp < 0.35 * np.sqrt(n2) + 0.03)
    keep = (ds_ > 0) | between
    if (keep & (ds_ <= -reach + 1)).sum() < 5:
        return float("nan")        # no substrate-facing boundary to measure
    proj = t_all[keep]
    bins = np.round(ds_[keep]).astype(int)
    # each distance bin's median (as np.median: the mean of the two middle values), from one sort (a median call per
    # bin was a third of the segmentation's time)
    o = np.lexsort((proj, bins))
    sb, sp = bins[o], proj[o]
    xs, start, cnt = np.unique(sb, return_index=True, return_counts=True)
    ys = (sp[start + (cnt - 1) // 2] + sp[start + cnt // 2]) / 2.0
    outer = ys[xs <= -reach + 1]
    inner = ys[xs >= max(1, xs.max() - 1)]
    if len(outer) == 0 or len(inner) == 0:
        return float("nan")
    lo, hi = float(np.median(outer)), float(np.median(inner))
    if hi - lo < 0.2:
        return float(reach * 2)
    yn = (ys - lo) / (hi - lo)
    yn = np.maximum.accumulate(yn)          # monotone envelope from outside to inside
    x20 = np.interp(0.2, yn, xs)
    x80 = np.interp(0.8, yn, xs)
    return float(max(x80 - x20, 0.0))


def _ring_similarity(c, m, med, inner=2, outer=6):
    """Fraction of the ring (inner..outer px outside ``m``) whose contrast has
    the region's hue and a similar strength (0.4-1.6x).  A real flake stands
    out from bare substrate or from clearly different neighbours; a patch cut
    out of a field of residue haze is surrounded by more of the same."""
    k3 = np.ones((3, 3), np.uint8)
    m8 = m.astype(np.uint8)
    ring = cv2.dilate(m8, k3, iterations=outer).astype(bool) & ~cv2.dilate(m8, k3, iterations=inner).astype(bool)
    if not ring.any():
        return 0.0
    v = c[ring]
    nm = float(np.linalg.norm(med))
    nv = np.linalg.norm(v, axis=1)
    cosang = (v @ med) / np.maximum(nv * nm, 1e-9)
    sim = (nv > 0.4 * nm) & (nv < 1.6 * nm) & (cosang > 0.8)
    return float(sim.mean())


def segment(contrast: np.ndarray, chip_mask: np.ndarray, saturated: np.ndarray,
            noise: np.ndarray, params: SegmentParams = SegmentParams(),
            min_parent_px: int = 20):
    """Return (regions, smoothed contrast, label image)."""
    cs = smooth_contrast(contrast)
    grad = gradient_magnitude(cs)

    # foreground: contrast significantly different from substrate in any channel
    thr = np.maximum(params.fg_sigma * noise * 0.6, params.fg_min)   # smoothing ~halves noise
    fg = (np.abs(cs) > thr[None, None, :]).any(axis=2) & chip_mask
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    # fill only small holes (dirt / pinholes on a flake); large enclosed areas
    # are usually bare substrate surrounded by debris
    fg = remove_small_holes(fg, area_threshold=params.max_hole_px) & chip_mask

    chip_edge = chip_mask & ~cv2.erode(chip_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    parents, n_par = ndimage.label(fg, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(parents)
    par_sizes = np.bincount(parents.ravel(), minlength=n_par + 1)

    labels = np.zeros(contrast.shape[:2], np.int32)
    regions: list[Region] = []
    next_label = 1
    H, W = fg.shape
    for pid, sl in enumerate(objs, start=1):
        if sl is None or par_sizes[pid] < min_parent_px:
            continue
        # pad bounding box (room for the ring used to measure edge sharpness)
        sl = tuple(slice(max(s.start - 8, 0), min(s.stop + 8, lim)) for s, lim in zip(sl, (H, W)))
        pmask = parents[sl] == pid
        g = grad[sl]
        c = cs[sl]
        core = pmask & (g < params.grad_max)
        if params.core_erode_px:
            core = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8),
                             iterations=params.core_erode_px).astype(bool)
        mk, nk = ndimage.label(core)
        if nk == 0:
            continue
        sizes = ndimage.sum(np.ones_like(mk), mk, index=np.arange(1, nk + 1))
        keep = np.zeros(nk + 1, bool)
        keep[1:] = sizes >= params.min_core_px
        mk = np.where(keep[mk], mk, 0)
        if not mk.any():
            continue
        # grow cores (largest first) into connected pixels of matching contrast
        core_ids = np.arange(1, nk + 1)[keep[1:]]
        order = core_ids[np.argsort(-sizes[core_ids - 1])]
        core_boxes = ndimage.find_objects(mk)
        assigned = np.zeros(pmask.shape, bool)
        grown = []
        PH, PW = pmask.shape
        for k in order:
            bb = core_boxes[k - 1]
            ck_full = (mk[bb] == k) & ~assigned[bb]
            if ck_full.sum() < params.min_core_px:
                continue
            med_k = np.median(c[bb][ck_full], axis=0)
            margin = 24
            while True:
                win = (slice(max(bb[0].start - margin, 0), min(bb[0].stop + margin, PH)),
                       slice(max(bb[1].start - margin, 0), min(bb[1].stop + margin, PW)))
                ck = (mk[win] == k) & ~assigned[win]
                cw, pw, aw = c[win], pmask[win], assigned[win]
                med = med_k
                reg = None
                for _ in range(2):      # grow, re-estimate the median, grow again
                    tol = max(params.tol_abs, params.tol_rel * float(np.linalg.norm(med)))
                    cand = pw & ~aw & (np.linalg.norm(cw - med, axis=2) < tol)
                    lab, _ = ndimage.label(cand)
                    ids = np.unique(lab[ck & cand])
                    reg = np.isin(lab, ids[ids > 0])
                    if not reg.any():
                        break
                    med = np.median(cw[reg], axis=0)
                full_win = win[0].start == 0 and win[1].start == 0 and win[0].stop == PH and win[1].stop == PW
                touches = reg is not None and reg.any() and (
                    (reg[0].any() and win[0].start > 0) or (reg[-1].any() and win[0].stop < PH) or
                    (reg[:, 0].any() and win[1].start > 0) or (reg[:, -1].any() and win[1].stop < PW))
                if not touches or full_win:
                    break
                margin *= 3
            if reg is None or reg.sum() < params.min_region_px:
                continue
            if reg.sum() > 4 * params.max_hole_px:
                reg = ndimage.binary_fill_holes(reg) & pw & ~aw
            # keep only the region's window (a full-size mask per region needs tens of GB on a
            # large mosaic, where tape residue can connect most of the chip into one parent)
            assigned[win] |= reg
            grown.append((win, reg))
        fg_sl = fg[sl]

        ph, pw_ = pmask.shape
        for win, reg in grown:
            # work in a local window around the region (metrics need ~8 px around it)
            ys, xs = np.nonzero(reg)
            ys = ys + win[0].start; xs = xs + win[1].start
            loc = (slice(max(ys.min() - 9, 0), min(ys.max() + 10, ph)),
                   slice(max(xs.min() - 9, 0), min(xs.max() + 10, pw_)))
            m = np.zeros((loc[0].stop - loc[0].start, loc[1].stop - loc[1].start), bool)
            m[ys - loc[0].start, xs - loc[1].start] = True
            gl = (slice(sl[0].start + loc[0].start, sl[0].start + loc[0].stop),
                  slice(sl[1].start + loc[1].start, sl[1].start + loc[1].stop))
            craw, csm = contrast[gl], cs[gl]
            npx = int(m.sum())
            vals = craw[m]
            medv = np.median(vals, axis=0)
            mad = 1.4826 * np.median(np.abs(vals - medv), axis=0)
            # uniformity of the interior: the outer pixel ring mixes flake and
            # surroundings (optical blur, JPEG colour bleeding)
            inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if inner.sum() < 0.3 * npx:
                inner = m
            dev = np.linalg.norm(csm[inner] - medv, axis=1)
            tol = max(0.05, 3 * float(np.linalg.norm(noise)) * 0.6, 0.08 * float(np.linalg.norm(medv)))
            unif = float(np.mean(dev < tol))
            ew = edge_width(craw, m, medv)
            m8 = m.astype(np.uint8)
            ring = (cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=6).astype(bool)
                    & ~cv2.dilate(m8, np.ones((3, 3), np.uint8), iterations=1).astype(bool))
            thick_border = float((np.linalg.norm(csm[ring], axis=1) > 0.25).mean()) if ring.any() else 0.0
            ring_similar = _ring_similarity(csm, m, medv)
            rsl = (slice(sl[0].start + ys.min(), sl[0].start + ys.max() + 1),
                   slice(sl[1].start + xs.min(), sl[1].start + xs.max() + 1))
            rmask = m[ys.min() - loc[0].start:ys.max() + 1 - loc[0].start, xs.min() - loc[1].start:xs.max() + 1 - loc[1].start]
            labels[rsl][rmask] = next_label
            satm = saturated[gl][m]
            regions.append(Region(label=next_label, parent=pid, slices=rsl, mask=rmask,
                                  median=medv, spread=float(np.linalg.norm(mad)),
                                  uniformity=unif, saturated_frac=float(satm.any(axis=1).mean()),
                                  saturated_ch=satm.mean(axis=0),
                                  touches_chip_edge=bool(chip_edge[gl][m].any()),
                                  edge_width=ew, thick_border=thick_border,
                                  ring_similar=ring_similar))
            next_label += 1
    return regions, cs, labels
