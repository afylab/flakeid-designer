"""Flake-level grouping: a flake is one connected region of material; the
uniform-thickness terraces found by the segmentation are regions of interest
inside it.

The flake mask is stricter than the segmentation foreground so that residue
haze (hBN) and faint film do not join neighbouring flakes into one object:
significant contrast, and for transparent materials also |C| above a haze
level; a morphological opening cuts thin bridges; small holes are filled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

HAZE_LEVEL = {"hbn": 0.15}       # min |C| of flake material (transparent materials only)


@dataclass
class FlakeObj:
    fid: int                      # flake label in the flake label image
    slices: tuple                 # bbox in working pixels
    mask: np.ndarray              # bool mask in the bbox
    area_um2: float
    inscribed_d_um: float
    length_um: float
    width_um: float
    centroid_px: tuple            # original pixels
    regions: list = field(default_factory=list)   # segment.Region labels inside
    metrics: dict = field(default_factory=dict)


def material_mask(cs: np.ndarray, noise: np.ndarray, chip: np.ndarray, material: str) -> np.ndarray:
    thr = np.maximum(4 * noise * 0.6, 0.035)
    fg = (np.abs(cs) > thr[None, None, :]).any(axis=2) & chip
    lvl = HAZE_LEVEL.get(material)
    if lvl:
        fg &= np.linalg.norm(cs, axis=2) > lvl
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    from skimage.morphology import remove_small_holes
    return remove_small_holes(fg, area_threshold=60) & chip


MAX_FLAKE_UM = 3000.0       # larger connected "flakes" are dropped (chip-edge bands, artefacts)


def build(regions, region_labels: np.ndarray, cs: np.ndarray, noise, chip, material: str,
          px_um: float, work_scale: float, min_area_um2: float = 50.0, exclude=()):
    """Label flakes and attach regions.  exclude: labels of regions that are not material (tape residue): kept out
    of the flake masks.  Returns (flakes, flake_label_image, region -> flake)."""
    fm = material_mask(cs, noise, chip, material)
    # every other segmented region belongs to the material: add it (the stricter mask
    # can lose faint graphene), then label
    if exclude:
        res = np.isin(region_labels, np.fromiter(exclude, np.int32))
        fm = (fm & ~res) | ((region_labels > 0) & ~res)
    else:
        fm |= region_labels > 0
    lab, n = ndimage.label(fm, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    min_px = min_area_um2 / px_um ** 2
    flakes = {}
    for fid, sl in enumerate(objs, start=1):
        if sl is None or sizes[fid] < min_px:
            continue
        # no exfoliated flake is millimetres across: a band along the chip edge or a mosaic artefact
        if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * px_um > MAX_FLAKE_UM:
            continue
        m = lab[sl] == fid
        dist = cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, 5)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.vstack(cnts).reshape(-1, 2).astype(np.float32)
        (_, _), (w_, h_), _ = cv2.minAreaRect(pts)
        ys, xs = np.nonzero(m)
        flakes[fid] = FlakeObj(
            fid=fid, slices=sl, mask=m, area_um2=float(m.sum()) * px_um ** 2,
            inscribed_d_um=2 * float(dist.max()) * px_um,
            length_um=(max(w_, h_) + 1) * px_um, width_um=(min(w_, h_) + 1) * px_um,
            centroid_px=((xs.mean() + sl[1].start + 0.5) / work_scale - 0.5,
                         (ys.mean() + sl[0].start + 0.5) / work_scale - 0.5))
    region_flake = {}
    for r in regions:
        ys, xs = np.nonzero(r.mask)
        yy, xx = ys + r.slices[0].start, xs + r.slices[1].start
        ids = lab[yy, xx]
        ids = ids[ids > 0]
        if not len(ids):
            continue
        fid = int(np.bincount(ids).argmax())
        if fid in flakes:
            flakes[fid].regions.append(r.label)
            region_flake[r.label] = fid
    return flakes, lab, region_flake


def focus_width_um(rgb: np.ndarray, fl: FlakeObj, work_scale: float, pixel_um: float,
                   substrate_rgb: np.ndarray) -> float:
    """Edge width of the flake outline measured on full-resolution pixels:
    contrast step across the boundary divided by the steepest gradient there
    (robust 90th percentile).  ~1-2 um for a focused 6.3x image; defocused
    edges are several um wide."""
    s = 1.0 / work_scale
    y0, x0 = int(fl.slices[0].start * s), int(fl.slices[1].start * s)
    h, w = int(fl.mask.shape[0] * s), int(fl.mask.shape[1] * s)
    pad = 12
    ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
    yb, xb = min(y0 + h + pad, rgb.shape[0]), min(x0 + w + pad, rgb.shape[1])
    crop = rgb[ya:yb, xa:xb].astype(np.float32) / np.maximum(np.asarray(substrate_rgb, np.float32), 1)
    M = np.zeros(crop.shape[:2], np.uint8)
    mm = cv2.resize(fl.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    M[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] = mm[:M.shape[0] - (y0 - ya), :M.shape[1] - (x0 - xa)]
    k = np.ones((3, 3), np.uint8)
    band = cv2.dilate(M, k, iterations=3).astype(bool) & ~cv2.erode(M, k, iterations=3).astype(bool)
    inner = cv2.erode(M, k, iterations=4).astype(bool)
    outer = cv2.dilate(M, k, iterations=8).astype(bool) & ~cv2.dilate(M, k, iterations=5).astype(bool)
    if band.sum() < 20 or inner.sum() < 10 or outer.sum() < 10:
        return float("nan")
    g = np.zeros(crop.shape[:2], np.float32)
    for c in range(3):
        gx = cv2.Sobel(crop[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(crop[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8
        g += gx * gx + gy * gy
    g = np.sqrt(g)
    step = float(np.linalg.norm(np.median(crop[inner], axis=0) - np.median(crop[outer], axis=0)))
    if step < 0.03:
        return float("nan")
    gp = float(np.percentile(g[band], 90))
    return step / max(gp, 1e-6) * pixel_um


def crowding(fl: FlakeObj, fm_lab: np.ndarray, px_um: float, r0_um=5.0, r1_um=30.0) -> float:
    """Fraction of the ring r0..r1 around the flake covered by other material."""
    H, W = fm_lab.shape
    r1 = int(np.ceil(r1_um / px_um))
    r0 = max(1, int(round(r0_um / px_um)))
    sl = (slice(max(fl.slices[0].start - r1, 0), min(fl.slices[0].stop + r1, H)),
          slice(max(fl.slices[1].start - r1, 0), min(fl.slices[1].stop + r1, W)))
    m = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), np.uint8)
    oy, ox = fl.slices[0].start - sl[0].start, fl.slices[1].start - sl[1].start
    m[oy:oy + fl.mask.shape[0], ox:ox + fl.mask.shape[1]] = fl.mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outer = cv2.dilate(m, k, iterations=r1).astype(bool)
    inner = cv2.dilate(m, k, iterations=r0).astype(bool)
    ring = outer & ~inner
    if not ring.any():
        return 0.0
    other = (fm_lab[sl] > 0) & (fm_lab[sl] != fl.fid)
    return float(other[ring].mean())
