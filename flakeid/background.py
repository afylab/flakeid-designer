"""Chip detection, flat-field (background) estimation and contrast maps.

The stitched overview images have strong per-tile vignetting and darker
stitching seams.  The bare SiO2 substrate dominates the chip area, so the
background is estimated from substrate pixels only (iteratively excluding
anything that differs from the local background), first with an isotropic
smooth surface and then with thin, line-shaped kernels that remove the
horizontal and vertical stitching seams.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ContrastMap:
    contrast: np.ndarray        # (H, W, 3) float32, I / I_substrate - 1 (RGB)
    chip_mask: np.ndarray       # (H, W) bool, usable chip interior
    substrate_mask: np.ndarray  # (H, W) bool, pixels classified as bare substrate
    saturated: np.ndarray       # (H, W, 3) bool, channel clipped in the raw image
    substrate_rgb: np.ndarray   # (3,) median raw substrate value
    noise: np.ndarray           # (3,) robust std of substrate contrast
    scale: float                # working pixels per original pixel (<= 1)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    ff = mask.astype(np.uint8).copy()
    h, w = ff.shape
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 1)
    return mask.astype(bool) | (ff == 0)


def _norm_blur(img, weight, ksize_xy, sigma_xy):
    """Weighted (normalised) Gaussian blur; returns (blurred, weight_sum)."""
    sx, sy = sigma_xy
    num = cv2.GaussianBlur(img * weight[..., None], ksize_xy, sigmaX=sx, sigmaY=sy)
    den = cv2.GaussianBlur(weight, ksize_xy, sigmaX=sx, sigmaY=sy)
    return num / np.maximum(den, 1e-6)[..., None], den


def _chip_by_colour(rgb_small, valid, tol=0.08):
    """Connected region of the most common colour (the bare substrate), flakes on it included; or None."""
    v = np.maximum(rgb_small.astype(np.float32), 1.0)
    gr, bg = np.log(v[..., 1] / v[..., 0]), np.log(v[..., 2] / v[..., 1])
    lum = np.log(v.mean(axis=2))
    sel = valid & np.isfinite(gr)
    h, xe, ye = np.histogram2d(gr[sel], bg[sel], bins=80)
    h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 1.0)
    i, j = np.unravel_index(np.argmax(h), h.shape)
    g0, b0 = 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])
    near = sel & (np.abs(gr - g0) < tol) & (np.abs(bg - b0) < tol)
    if near.sum() < 100:
        return None
    l0 = np.median(lum[near])
    near &= np.abs(lum - l0) < 0.4
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(near, connectivity=8)
    if n <= 1:
        return None
    chip = lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return chip if chip.sum() >= 0.2 * valid.sum() else None


def detect_chip(rgb_small: np.ndarray, erode_px: int, dark_below: float | None = None) -> np.ndarray:
    """Chip = largest bright connected region (the stage/background is dark).

    dark_below: brightness (mean of RGB) below which a pixel is off the chip, when the substrate level
    is known (live analysis of part of a scan: with a strip of surround next to bright residue, a
    threshold from the image alone can split substrate from residue instead of chip from surround)."""
    g = rgb_small.mean(axis=2)
    gray = np.clip(g * (255.0 / max(float(np.percentile(g, 99.5)), 1.0)), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    valid = g > 0                               # exactly 0: outside a tile mosaic (no data)
    # (compare with a bright level, not the median: the median is the background itself when
    # the chip covers less than half of the image)
    if (g[valid] < 0.5 * np.percentile(g[valid], 90)).mean() < 0.01:
        # no dark surround (e.g. a tile mosaic inside the chip): every pixel with data is chip
        return cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    mask = ((g > dark_below) if dark_below is not None else (gray > thr)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.ones(gray.shape, bool) if dark_below is None else mask.astype(bool)
    chip = lab == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if dark_below is None and chip.sum() < 0.2 * valid.sum():
        # the brightness split did not separate chip from surround (a surround as bright as the substrate,
        # e.g. the holder next to a 90 nm oxide chip; bright flakes then form the "bright" class): the chip is
        # the connected region of the substrate colour instead
        by = _chip_by_colour(rgb_small, valid)
        if by is not None:
            chip = by
    chip = _fill_holes(chip)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        chip = cv2.erode(chip.astype(np.uint8), k).astype(bool)
    return chip


def compute_contrast(rgb: np.ndarray, pixel_um: float, work_scale: float = 0.5, max_value: int = 255,
                     edge_margin_um: float = 150.0, bg_sigma_um: float = 120.0,
                     seam_sigma_um: float = (600.0, 12.0), substrate_tol: float = 0.035,
                     n_iter: int = 4, substrate_hint=None) -> ContrastMap:
    """Background-normalised RGB contrast of a stitched overview image.

    rgb:         (H, W, 3) uint8 or uint16 RGB image (linear)
    max_value:   full-scale (clipping) value of ``rgb``
    pixel_um:    original pixel size in micrometres
    work_scale:  resampling factor for the returned maps (0.5 => half size)
    substrate_hint: known substrate RGB level in ``rgb`` units (e.g. from other parts of the same scan);
                 the chip is then everything brighter than half of it
    """
    if work_scale != 1.0:
        work = cv2.resize(rgb, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    else:
        work = rgb
    saturated = work >= 0.98 * max_value
    work = work.astype(np.float32)
    px = pixel_um / work_scale                      # um per working pixel

    # Coarse grid (~8 um/px) for the background surface estimate
    bs = min(1.0, px / 8.0)
    small = cv2.resize(work, None, fx=bs, fy=bs, interpolation=cv2.INTER_AREA)
    spx = px / bs
    chip_small = detect_chip(small, int(round(edge_margin_um / spx)),
                             dark_below=None if substrate_hint is None else 0.5 * float(np.mean(substrate_hint)))

    # no image data (mosaic gaps, garbage tiles left out of a Micro-Manager mosaic) is never chip
    nodata = small.max(axis=2) <= 0
    if nodata.any():
        chip_small &= ~cv2.dilate(nodata.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    mask = chip_small.astype(np.float32)
    sig = bg_sigma_um / spx
    tols = np.linspace(0.15, substrate_tol, n_iter)
    for tol in tols:
        bg, den = _norm_blur(small, mask, (0, 0), (sig, sig))
        # fill large holes (e.g. under big flakes) from a much wider kernel
        bg_wide, _ = _norm_blur(small, mask, (0, 0), (4 * sig, 4 * sig))
        a = np.clip(den / 0.3, 0, 1)[..., None]
        bg = a * bg + (1 - a) * bg_wide
        c = small / np.maximum(bg, 1.0) - 1.0
        mask = ((np.abs(c).max(axis=2) < tol) & chip_small).astype(np.float32)

    bg_full = cv2.resize(bg, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_LINEAR)
    chip = cv2.resize(chip_small.astype(np.uint8), (work.shape[1], work.shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    # in place: for a large mosaic each full-size float copy is several GB; the raw values are
    # needed once more (substrate level), at every 2nd pixel
    work_s = work[::2, ::2].copy()
    np.maximum(bg_full, 1.0, out=bg_full)
    contrast = work
    np.divide(contrast, bg_full, out=contrast)
    contrast -= 1.0
    del bg_full, work

    # Seam correction at working resolution: residual along thin lines
    sub = (np.abs(contrast).max(axis=2) < substrate_tol) & chip
    long_s, short_s = (s / px for s in seam_sigma_um)
    for sx, sy in ((long_s, short_s), (short_s, long_s)):
        w = sub.astype(np.float32)
        # anisotropic down-sampling keeps the long kernels affordable
        fx, fy = min(1.0, 6.0 / sx), min(1.0, 3.0 / sy)
        cs = cv2.resize(contrast * w[..., None], None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        ws = cv2.resize(w, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        cs /= np.maximum(ws, 1e-6)[..., None]
        res, den = _norm_blur(cs, ws, (0, 0), (sx * fx, sy * fy))
        res *= np.clip(den / 0.2, 0, 1)[..., None]
        res = cv2.resize(res, (contrast.shape[1], contrast.shape[0]), interpolation=cv2.INTER_LINEAR)
        contrast += 1.0
        res += 1.0
        contrast /= res
        contrast -= 1.0
        del res
        sub = (np.abs(contrast).max(axis=2) < substrate_tol) & chip

    sub_vals = contrast[sub]
    med = np.median(sub_vals, axis=0)
    contrast -= med                                   # re-centre the substrate at 0
    noise = 1.4826 * np.median(np.abs(sub_vals - med), axis=0)
    substrate_rgb = np.median(work_s[sub[::2, ::2]], axis=0)
    del work_s

    return ContrastMap(contrast=contrast.astype(np.float32, copy=False), chip_mask=chip,
                       substrate_mask=sub, saturated=saturated & chip[..., None],
                       substrate_rgb=substrate_rgb, noise=noise, scale=work_scale)
