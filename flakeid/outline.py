"""Flake and terrace outlines at the real edge, on full-resolution pixels.

The segmentation works on half-resolution pixels and keeps only the flat interior of each terrace (pixels
where the colour changes fast, i.e. the blurred edge band, are left out), so its masks sit 1-3 um inside the
flake edge.  Here every flake is redrawn on the full-resolution image: markers well inside each terrace, inside
the rest of the flake and on the bare substrate around it are grown over the contrast-gradient image
(watershed), so neighbouring markers meet on the steepest part of each edge, the middle of the step between
the two colours.  Outlines are traced half-way between inside and outside pixels (not through pixel centres).
"""
from __future__ import annotations

import cv2
import numpy as np

PAD = 14            # full-resolution pixels around the flake


def _up(img_ws, x0, y0, w, h, ws, fn=None):
    """work-scale image (labels or mask, whole image) -> full-resolution crop [y0:y0+h, x0:x0+w] (nearest);
    fn: applied to the work-scale crop first (e.g. lambda a: a == rid)."""
    f = int(round(1.0 / ws))
    a0, b0 = x0 // f, y0 // f
    a1, b1 = -(-(x0 + w) // f) + 1, -(-(y0 + h) // f) + 1
    sub = img_ws[b0:b1, a0:a1]
    if fn is not None:
        sub = fn(sub)
    up = np.repeat(np.repeat(sub, f, axis=0), f, axis=1)
    ox, oy = x0 - a0 * f, y0 - b0 * f
    out = np.zeros((h, w), up.dtype)
    hh, ww = min(h, up.shape[0] - oy), min(w, up.shape[1] - ox)
    if hh > 0 and ww > 0:
        out[:hh, :ww] = up[oy:oy + hh, ox:ox + ww]
    return out


def _erode(m, r):
    if r <= 0:
        return m.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(m.astype(np.uint8), k).astype(bool)


def _dilate(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def refine_flake(rgb, result, flake):
    # (the work scale must be 1 / an integer: 0.5 by default)
    """Full-resolution masks of one flake: (x0, y0, flake_mask, {terrace_id: mask}) in original pixels, or None."""
    from skimage.segmentation import watershed
    ws, fl = result._scale, result._flake_lab
    old = result._flake_of.get(flake.flake_id)
    if fl is None or old is None:
        return None
    H, W = rgb.shape[:2]
    if flake.bbox_w_px * flake.bbox_h_px > 4.0e6:          # (> 2000 x 2000 px: not refined, keeps memory bounded)
        return None
    x0, y0 = max(int(flake.bbox_x_px) - PAD, 0), max(int(flake.bbox_y_px) - PAD, 0)
    x1 = min(int(flake.bbox_x_px + flake.bbox_w_px) + PAD, W)
    y1 = min(int(flake.bbox_y_px + flake.bbox_h_px) + PAD, H)
    w, h = x1 - x0, y1 - y0
    flab = _up(fl, x0, y0, w, h, ws)
    fm = flab == old
    if fm.sum() < 30:
        return None
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    # local substrate: no flake within 4 px
    sub = ~_dilate(flab > 0, 4)
    if sub.sum() < 50:
        sub = ~_dilate(fm, 4)
    if sub.sum() < 50:
        return None
    s = np.median(crop[sub], axis=0)
    c = crop / np.maximum(s, 1.0) - 1.0
    c = cv2.GaussianBlur(c, (0, 0), 0.8)
    g = np.zeros((h, w), np.float32)
    for k in range(3):
        gx = cv2.Sobel(c[..., k], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(c[..., k], cv2.CV_32F, 0, 1, ksize=3)
        g += gx * gx + gy * gy
    g = np.sqrt(g)
    # markers: 1 substrate, 2 other flakes, 3 rest of this flake, 10+k terraces of this flake
    mk = np.zeros((h, w), np.int32)
    mk[sub & ~_dilate(fm, 3)] = 1
    mk[_erode((flab > 0) & ~fm, 3)] = 2
    terr = [t for t in result.terraces if t.flake_id == flake.flake_id]
    labels = result._labels
    tmask = {}
    for k, t in enumerate(terr):
        rid = result._region_of.get(t.terrace_id)
        if rid is None:
            continue
        tm = _up_region(labels, rid, x0, y0, w, h, ws)
        tmask[t.terrace_id] = tm
        core = _erode(tm, 3)
        if core.sum() < 6:
            core = _erode(tm, 1)
        mk[core & (mk == 0)] = 10 + k
    anyt = np.zeros((h, w), bool)
    for tm in tmask.values():
        anyt |= tm
    # the rest of the flake (parts that are not a reported terrace: thick or rejected areas): only compact areas
    # well away from the terraces seed it; the thin band left between a terrace and the old flake outline is
    # edge, and must go to whichever side the gradient says
    rest = _erode(fm, 3) & ~_dilate(anyt, 4)
    rest = cv2.morphologyEx(rest.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    mk[rest & (mk == 0)] = 3
    lab = watershed(g, mk)
    out_f = (lab == 3) | (lab >= 10)
    # sanity: the refined flake must stay close to the segmented one (else keep the old one)
    a_old, a_new = float(fm.sum()), float(out_f.sum())
    if not (0.8 * a_old <= a_new <= 2.5 * a_old + 200):
        out_f = fm
        out_t = tmask
    else:
        out_t = {}
        for k, t in enumerate(terr):
            if t.terrace_id in tmask:
                m = lab == 10 + k
                out_t[t.terrace_id] = m if m.sum() >= 0.8 * tmask[t.terrace_id].sum() else tmask[t.terrace_id]
    return x0, y0, out_f, out_t


def _up_region(labels, rid, x0, y0, w, h, ws):
    return _up(labels, x0, y0, w, h, ws, fn=lambda a: a == rid)


def polygons(mask, x0=0, y0=0, simplify_px=0.7, min_area=4.0):
    """Outlines of a mask, traced half-way between inside and outside pixel centres (pixel-centre coordinates
    of the original image).  Returns [[x, y, x, y, ...], ...], largest first."""
    from skimage.measure import find_contours
    m = np.pad(mask.astype(np.float32), 1)
    out = []
    for cc in find_contours(m, 0.5):
        xy = np.ascontiguousarray(cc[:, ::-1] - 1.0, np.float32)
        if abs(cv2.contourArea(xy)) < min_area:
            continue
        xy = cv2.approxPolyDP(xy.reshape(-1, 1, 2), simplify_px, True).reshape(-1, 2)
        if len(xy) < 3:
            continue
        out.append((abs(cv2.contourArea(xy)), np.round(xy + [x0, y0], 1).ravel().tolist()))
    out.sort(key=lambda e: -e[0])
    return [p for _, p in out]


def place(mask, mx0, my0, x0, y0, w, h):
    """A refined mask with origin (mx0, my0) -> the window (x0, y0, w, h), as uint8."""
    out = np.zeros((h, w), np.uint8)
    ox, oy = mx0 - x0, my0 - y0
    ys0, xs0 = max(oy, 0), max(ox, 0)
    ys1, xs1 = min(oy + mask.shape[0], h), min(ox + mask.shape[1], w)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0:ys1, xs0:xs1] = mask[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
    return out


def refine_all(rgb, result, log=None):
    """Refined outlines for every flake of a result, cached on it: result._refined = {flake_id: (x0, y0, mask, {tid: mask})}."""
    if getattr(result, "_refined", None) is not None:
        return result._refined
    ref = {}
    for f in result.flakes:
        try:
            r = refine_flake(rgb, result, f)
        except Exception as exc:                      # one odd flake must not stop the viewer
            if log:
                log(f"outline of flake {f.flake_id} not refined: {exc}")
            r = None
        if r is not None:
            ref[f.flake_id] = r
    result._refined = ref
    return ref
