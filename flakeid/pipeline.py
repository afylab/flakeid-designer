"""High-level flake identification pipeline.

    from flakeid import find_flakes
    result = find_flakes("g31.jpg", material="graphite", oxide_nm=90)
    result.to_csv("g31_flakes.csv")

The result carries everything later stages need (e.g. to drive the
microscope to each flake for high-magnification imaging): positions in
image pixels, micrometres, and — when a stage origin is known — stage
coordinates, plus thickness estimates with uncertainties and quality flags.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .background import compute_contrast
from .io import ImageData, load_image
from scipy import ndimage

from .optics import resolve_material
from .flakes import build as build_flakes, crowding, focus_width_um
from .scorer import Scorer, scorer_path
from .thickness import DEFAULT_PROFILE
from .segment import SegmentParams, segment
from .thickness import ThicknessEstimator, load_profile, refine_oxide

# Default thickness window of interest (nm) per material
DEFAULT_RANGE_NM = {"graphite": (0.3, 25.0), "hbn": (0.0, 100.0)}

# Minimum |contrast| for a reliable identification: graphene monolayers on
# ~90 nm SiO2 give ~0.06-0.09 (gamma-encoded); hBN on 285 nm SiO2 reaches
# 0.15 at ~3.5 nm.  Thin hBN is separated from residue by the full-resolution
# flatness (fill) and shape (fullres_solidity) checks.
DEFAULT_MIN_CONTRAST = {"graphite": 0.06, "hbn": 0.15}


@dataclass
class Terrace:
    """A region of uniform thickness (region of interest) inside a flake."""
    terrace_id: int
    flake_id: int                   # the flake (connected region of material) it belongs to
    x_px: float                     # centroid, original image pixels
    y_px: float
    target_x_px: float              # point deepest inside the region (best for imaging)
    target_y_px: float
    x_um: float                     # centroid in um from the image origin (top-left)
    y_um: float
    target_x_um: float
    target_y_um: float
    stage_x_um: float | None
    stage_y_um: float | None
    thickness_nm: float
    thickness_min_nm: float
    thickness_max_nm: float
    layers: float                   # thickness / interlayer spacing
    alt_thickness_nm: str           # other thickness solutions that fit the colour
    length_um: float                # long side of the minimum-area bounding rectangle
    width_um: float                 # short side
    orientation_deg: float          # angle of the long side, degrees from +x (image)
    area_um2: float
    inscribed_diameter_um: float    # largest circle fully inside the region
    solidity: float                 # area / convex hull area (ragged debris is low)
    fill: float                     # full-res fraction of the interior matching the flake colour
    fullres_solidity: float         # solidity of the matching full-res pixels (dendrites are low)
    ring_similar: float             # fraction of surroundings with the same colour (residue haze fields)
    bbox_x_px: int
    bbox_y_px: int
    bbox_w_px: int
    bbox_h_px: int
    contrast_r: float
    contrast_g: float
    contrast_b: float
    fit_residual: float
    uniformity: float
    edge_width_um: float            # 20-80 % width of the flake edge (sharp ~2-3 um; smudges/defocus larger)
    saturated_frac: float
    # --- the flake this terrace belongs to (whole connected region of material)
    flake_area_um2: float
    flake_inscribed_d_um: float
    flake_focus_um: float           # edge width of the flake outline at full resolution (focus; ~1-2 um sharp)
    flake_crowding: float           # fraction of the 5-30 um surroundings covered by other material
    flake_frac_window: float        # fraction of the flake whose thickness is inside the window
    flake_frac_thick: float         # fraction thicker than the window / bulk / unidentifiable dark
    score: float                    # learned P(worth imaging) from user labels (profiles/scorer.json)
    quality: str                    # "good", or ';'-separated warning flags
    # --- context for thin / thick hBN (transparent materials; 0 otherwise): colours only thick films show
    edge_thick_um2: float = 0.0     # edge pixels (+-2 um of the outline, not in another region) of a thick-only colour
    thick_nb_um2: float = 0.0       # thick-only colour nearby: touching regions + pixels 2-8 um out (> ~60 nm)


Flake = Terrace                      # backwards compatibility


@dataclass
class FlakeSummary:
    """One flake: a whole connected region of material, with its terraces."""
    flake_id: int                   # rank by score: 1 = best candidate
    score: float                    # best terrace score
    material: str
    n_terraces: int
    terrace_ids: str                # ';'-separated, best first
    thicknesses_nm: str             # ';'-separated, same order
    best_terrace_id: int
    best_thickness_nm: float
    best_terrace_d_um: float
    x_px: float                     # flake centroid
    y_px: float
    target_x_px: float              # centre of the best terrace (aim here)
    target_y_px: float
    x_um: float
    y_um: float
    target_x_um: float
    target_y_um: float
    stage_x_um: float | None        # of the target point
    stage_y_um: float | None
    area_um2: float
    inscribed_diameter_um: float
    length_um: float
    width_um: float
    bbox_x_px: int
    bbox_y_px: int
    bbox_w_px: int
    bbox_h_px: int
    focus_um: float
    crowding: float
    frac_window: float
    frac_thick: float
    quality: str                    # flags of the best terrace


@dataclass
class FlakeResult:
    image: str
    material: str
    oxide_nm: float
    oxide_nm_used: float
    pixel_um: float
    pixel_um_source: str
    flakes: list                              # FlakeSummary, ranked
    terraces: list = field(default_factory=list)   # Terrace
    params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    # working arrays (not serialised) for annotation / further analysis
    _labels: np.ndarray | None = None
    _scale: float = 1.0
    _region_of: dict = field(default_factory=dict)   # terrace_id -> region label
    _flake_lab: np.ndarray | None = None              # flake label image (working res)
    _flake_of: dict = field(default_factory=dict)     # flake_id -> label in _flake_lab
    _contrast: np.ndarray | None = None      # working-resolution contrast map (for the viewer)
    _substrate_rgb: np.ndarray | None = None
    _oxide_samples: np.ndarray | None = None  # contrast samples used to refine the oxide (live: pooled over blocks)
    _clip_max: np.ndarray | None = None
    _clean: list = field(default_factory=list)  # clean areas (find_flakes clean_areas=...; flakeid.clean), else []

    @staticmethod
    def _write(path, rows, cls):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cls.__dataclass_fields__))
            w.writeheader()
            for f in rows:
                row = asdict(f)
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = round(v, 4)
                w.writerow(row)

    def to_csv(self, path, terraces_path=None):
        """Flake table to `path`; terrace table to `terraces_path` (default: *_terraces.csv next to it)."""
        path = Path(path)
        self._write(path, self.flakes, FlakeSummary)
        tp = Path(terraces_path) if terraces_path else path.with_name(path.stem.replace("_flakes", "") + "_terraces.csv")
        self._write(tp, self.terraces, Terrace)
        return tp

    def to_json(self, path):
        d = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        d["flakes"] = [asdict(fl) for fl in self.flakes]
        d["terraces"] = [asdict(t) for t in self.terraces]
        Path(path).write_text(json.dumps(d, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _collect_uniform_samples(contrast, chip, saturated, n=20000, seed=0):
    """Contrast samples from flat, non-substrate, unsaturated areas (for oxide refinement)."""
    cb = cv2.blur(contrast, (5, 5))
    c2 = cv2.blur(contrast * contrast, (5, 5))
    sd = np.sqrt(np.maximum(c2 - cb * cb, 0)).max(axis=2)
    satd = cv2.dilate(saturated.any(axis=2).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m = chip & ~satd & (sd < 0.02) & (np.abs(cb).max(axis=2) > 0.05)
    s = cb[m]
    if len(s) > n:
        s = s[np.random.default_rng(seed).choice(len(s), n, replace=False)]
    return s


def fullres_check(rgb, r, work_scale, tol_abs=0.07, tol_rel=0.12):
    """Verify a region on the original full-resolution pixels.

    Returns (fill, solidity):
      fill      fraction of the region interior whose contrast matches the
                region's median (flat flakes ~0.9; dendritic residue or clusters
                of tiny fragments, which only look uniform after down-sampling,
                are much lower)
      solidity  area / convex hull of the matching pixels (simple flake
                geometry ~0.8-1; thin dendrites low)
    """
    k3 = np.ones((3, 3), np.uint8)
    y0, x0 = int(r.slices[0].start / work_scale), int(r.slices[1].start / work_scale)
    h, w = int(round(r.mask.shape[0] / work_scale)), int(round(r.mask.shape[1] / work_scale))
    pad = 16
    ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
    yb, xb = min(y0 + h + pad, rgb.shape[0]), min(x0 + w + pad, rgb.shape[1])
    crop = rgb[ya:yb, xa:xb].astype(np.float32)
    M = np.zeros(crop.shape[:2], np.uint8)
    mm = cv2.resize(r.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    M[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] = mm[:M.shape[0] - (y0 - ya), :M.shape[1] - (x0 - xa)]
    ring = cv2.dilate(M, k3, iterations=12).astype(bool) & ~cv2.dilate(M, k3, iterations=5).astype(bool)
    inner = cv2.erode(M, k3, iterations=2).astype(bool)
    if ring.sum() < 20 or inner.sum() < 12:
        return float("nan"), float("nan")
    ref = np.median(crop[ring], axis=0)
    ok = np.all(np.abs(crop[ring] / np.maximum(ref, 1) - 1) < 0.05, axis=1)   # substrate-like ring pixels
    if ok.sum() >= 20:
        ref = np.median(crop[ring][ok], axis=0)
    c = crop / np.maximum(ref, 1.0) - 1.0
    med = np.median(c[inner], axis=0)
    tol = max(tol_abs, tol_rel * float(np.linalg.norm(med)))
    on = np.linalg.norm(c - med, axis=2) < tol
    fill = float(on[inner].mean())
    cand = on & cv2.dilate(M, k3, iterations=2).astype(bool)
    lab, n = ndimage.label(cand)
    if n == 0:
        return fill, 0.0
    big = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
    pts = np.argwhere(big)[:, ::-1].astype(np.float32)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True) + 1.0
    return fill, float(min(1.0, big.sum() / hull_area))


def _solidity(mask):
    """Area / convex-hull area of a region mask."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.vstack(cnts).reshape(-1, 2).astype(np.float32))
    # polygon through boundary-pixel centres misses ~half a pixel all round
    hull_area = cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True) + 1.0
    return float(min(1.0, mask.sum() / max(hull_area, 1.0)))


def thick_context(regions, labels, contrast, est, px_um, thin_hi=60.0, band_um=2.0, near_um=8.0, min_c=0.15):
    """region label -> (edge_thick_um2, thick_nb_um2): thick-only colours at the region's edge and in touching regions.

    A colour is thick-only if it is within 0.10 of the model curve beyond thin_hi and more than 0.25 from every
    colour of films up to thin_hi (curve points closer than 0.30 to a thin colour are left out: shared colours)."""
    from scipy.spatial import cKDTree
    d, cv = est.d, est.curve
    thin = cKDTree(cv[d <= thin_hi])
    thick = cv[d > thin_hi]
    only = thick[thin.query(thick)[0] > 0.30]
    if len(only) == 0:
        return {}
    tonly = cKDTree(only)

    def is_thick(v):
        return (tonly.query(v)[0] < 0.10) & (thin.query(v)[0] > 0.25)
    rmed = {r.label: r for r in regions}
    # thick-only colour, or clipped in a channel no thin film's colour clips (the camera saturates on thick hBN)
    thin_max = cv[d <= thin_hi].max(axis=0)
    def thick_region(r):
        if r.saturated_frac < 0.5:
            return bool(is_thick(r.median[None, :])[0])
        clip_ch = r.saturated_ch > 0.5
        return bool(np.any(clip_ch & (thin_max < (est.clip_max if est.clip_max is not None else np.inf) - 0.05)))
    thick_reg = {r.label for r in regions if thick_region(r)}
    H, W = labels.shape
    b = max(1, int(round(band_um / px_um)))
    bn = max(b + 1, int(round(near_um / px_um)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b + 1, 2 * b + 1))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bn + 1, 2 * bn + 1))
    # the bands and rings of all regions first, then is_thick once on the pixels any of them needs (neighbouring
    # regions' rings overlap: the per-pixel test was a tenth of a 10x block's analysis), then the sums
    todo = []
    need = np.zeros((H, W), bool)
    for r in regions:
        if float(np.linalg.norm(r.median)) < min_c:
            continue
        y0, y1 = max(r.slices[0].start - bn - 1, 0), min(r.slices[0].stop + bn + 1, H)
        x0, x1 = max(r.slices[1].start - bn - 1, 0), min(r.slices[1].stop + bn + 1, W)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        m[r.slices[0].start - y0:r.slices[0].stop - y0, r.slices[1].start - x0:r.slices[1].stop - x0][r.mask] = 1
        outer = cv2.dilate(m, ker).astype(bool)
        inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
        lab = labels[y0:y1, x0:x1]
        band = outer & ~inner & ((lab == 0) | (lab == r.label))
        nb = set(np.unique(lab[outer & ~m.astype(bool)])) - {0, r.label}
        a = sum(float(rmed[k].mask.sum()) for k in nb if k in thick_reg) * px_um ** 2
        # thick material nearby that is no uniform region (colour gradients of thick crystals stay unlabelled):
        # thick-only colour pixels 2-8 um out, not in this region's own band
        ring = cv2.dilate(m, kern).astype(bool) & ~outer & (lab != r.label)
        need[y0:y1, x0:x1] |= band | ring
        todo.append((r.label, y0, y1, x0, x1, band, ring, a))
    T = np.zeros((H, W), bool)
    if need.any():
        T[need] = is_thick(contrast[need])
    out = {}
    for lbl, y0, y1, x0, x1, band, ring, a in todo:
        Tw = T[y0:y1, x0:x1]
        e = float(Tw[band].sum()) * px_um ** 2 if band.any() else 0.0
        if ring.any():
            a += float(Tw[ring].sum()) * px_um ** 2
        out[lbl] = (e, a)
    return out


def find_flakes(image, material: str, oxide_nm: float, *,
                thickness_range_nm: tuple | None = None,
                min_area_um2: float = 100.0,
                pixel_um: float | None = None,
                work_scale: float = 0.5,
                profile: str | dict | None = None,
                refine_oxide_thickness: bool = True,
                max_residual: float = 0.20,
                min_uniformity: float = 0.6,
                max_edge_width_um: float = 5.0,
                indistinct_ratio: float = 1.5,
                min_solidity: float = 0.6,
                min_contrast: float | None = None,
                max_blue_green: float = 0.8,
                ambiguous_residual: float = 0.25,
                prefer_thin_ambiguous: bool = True,
                thick_margin: float = 0.08,
                bulk_margin: float = 0.03,
                min_fill: float = 0.75,
                min_fullres_solidity: float = 0.6,
                faint_contrast: float = 0.4,
                min_fill_faint: float = 0.9,
                min_fullres_solidity_faint: float = 0.75,
                max_ring_similar: float = 0.3,
                haze_contrast: float = 0.6,
                residue_ring_similar: float = 0.15,
                residue_solidity: float = 0.75,
                residue_edge_um: float = 3.0,
                low_confidence_contrast: float = 0.25,
                thick_edge_um2: float = 12.0,
                thick_near_um2: float = 50.0,
                max_saturated_frac: float = 0.2,
                stage_origin_um: tuple | None = None,
                stage_axis_sign: tuple = (1.0, 1.0),
                include_rejected: bool = False,
                clean_areas: tuple | None = None,
                keep_contrast: bool = False,
                segment_params: SegmentParams | None = None,
                substrate_hint=None,
                oxide_only: bool = False,
                verbose: bool = False) -> FlakeResult:
    """Identify uniform-contrast flakes of a given material in an overview image.

    image:              path or ImageData
    material:           'graphite' or 'hbn' (see optics.MATERIALS)
    oxide_nm:           nominal SiO2 thickness
    thickness_range_nm: (min, max) thickness of interest; default per material
    min_area_um2:       smallest region reported
    work_scale:         processing resolution relative to the original image
    stage_origin_um:    stage (x, y) of the image's top-left pixel; defaults to
                        the origin found in the .vsi metadata, if available
    stage_axis_sign:    (+1/-1, +1/-1) mapping image +x/+y to stage +x/+y
    clean_areas:        (min, max) nm: also the clean areas of that thickness, whatever the flakes' outlines
                        (flakeid.clean; FlakeResult._clean: their middle in image pixels and stage um, area, thickness,
                        and the identified flake there, if any)
    prefer_thin_ambiguous: inclusive about thickness (2026-10-25): a region whose best colour fit is thicker than the
                        window but whose colour an in-window film also matches acceptably (ambiguous_thickness) is
                        kept and reported at the in-window (thinner) thickness, the thick one as its alternative, and
                        is not counted as thick around the other regions; False: as before 2026-10-25
    include_rejected:   also report regions outside the thickness range or
                        failing quality cuts (flagged in the 'quality' column)
    substrate_hint:     known substrate RGB level (image units) for chip detection (live analysis)
    oxide_only:         stop after the oxide refinement: no flakes, only the oxide, its samples and the substrate level
                        (live analysis: the oxide is fixed from the first blocks before any block is analysed in full)
    """
    t0 = time.time()
    log = (lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a)) if verbose else (lambda *a: None)

    material = resolve_material(material)
    img = image if isinstance(image, ImageData) else load_image(image, pixel_um=pixel_um)
    log(f"loaded {img.path.name} {img.rgb.shape[1]}x{img.rgb.shape[0]} px, "
        f"{img.pixel_um:.4f} um/px ({img.pixel_um_source})")
    if profile is None and img.metadata.get("source") == "micromanager" \
            and img.metadata.get("data_format", "").upper().startswith("RAW"):
        profile = "bx51_xc10_mm_raw16"          # linear raw sensor data (Stream profiles include its processing)
        if material == "graphite":              # (its own fit: the hBN-calibrated one put 2-12 nm graphite at 8-11 nm)
            profile = "bx51_xc10_mm_raw16_graphite"
            log("graphite: profile fitted on graphite_20261019_run_1 (1-4 layer anchors + the run's colours); above "
                "1.3 nm not yet checked against AFM")
    prof = profile if isinstance(profile, dict) else load_profile(profile)
    log(f"profile: {Path(prof['_path']).stem if '_path' in prof else 'custom'}")
    tmin, tmax = thickness_range_nm or DEFAULT_RANGE_NM.get(material, (0.0, 100.0))
    seg_p = segment_params or SegmentParams()
    if min_contrast is None:
        min_contrast = DEFAULT_MIN_CONTRAST.get(material, 0.08)

    cm = compute_contrast(img.rgb, img.pixel_um, work_scale=work_scale, max_value=img.max_value,
                          substrate_hint=substrate_hint)
    px_um = img.pixel_um / work_scale
    log(f"background flattened; substrate noise (RGB) = {np.round(cm.noise, 4)}")

    # largest contrast each channel can record before the camera clips
    clip_level = img.clip_value if img.clip_value is not None else img.max_value - 0.5
    clip_max = clip_level / np.maximum(cm.substrate_rgb, 1.0) - 1.0
    oxide_used, oxide_gain = float(oxide_nm), 0.0
    samples = None
    if refine_oxide_thickness:
        samples = _collect_uniform_samples(cm.contrast, cm.chip_mask, cm.saturated)
        if len(samples) > 500:
            ox, gain = refine_oxide(samples, material, oxide_nm, prof, clip_max=clip_max)
            if gain > 0.05:
                oxide_used, oxide_gain = ox, gain
        log(f"oxide thickness used: {oxide_used:.1f} nm (nominal {oxide_nm}, gain {oxide_gain:.2f})")
    if oxide_only:
        return FlakeResult(image=str(img.path), material=material, oxide_nm=float(oxide_nm), oxide_nm_used=oxide_used,
                           pixel_um=img.pixel_um, pixel_um_source=img.pixel_um_source, flakes=[],
                           params=dict(oxide_refine_gain=oxide_gain, profile=prof.get("_path")), metadata=dict(img.metadata),
                           _substrate_rgb=cm.substrate_rgb, _oxide_samples=samples, _clip_max=clip_max)

    est = ThicknessEstimator(material, oxide_used, prof, clip_max=clip_max)
    mono = est.monolayer_contrast()

    min_px = max(seg_p.min_region_px, int(min_area_um2 / px_um ** 2))
    regions, cs, labels = segment(cm.contrast, cm.chip_mask, cm.saturated, cm.noise,
                                  seg_p, min_parent_px=min_px)
    log(f"segmented {len(regions)} uniform regions")


    meta = dict(img.metadata)
    if stage_origin_um is None and "origin_x_um" in meta:
        stage_origin_um = (meta["origin_x_um"], meta["origin_y_um"])
    sx, sy = stage_axis_sign

    def to_orig(v):  # working pixel coordinate -> original pixel coordinate
        return (v + 0.5) / work_scale - 0.5

    # pass 1: colour fit of every region, and per-parent share of material
    # that is thicker than the window or not identifiable (bulk, shadows, ...)
    fits = {}
    par_area, par_thick = {}, {}
    absorbing = est.model._n_fl.imag.mean() > 0.3        # graphite-like: colour saturates to bulk
    cmin = est.curve.min(axis=0)
    from .branch import load as load_branch
    branch = None if absorbing else load_branch(Path(prof.get("_path", "")).stem, material, oxide_used, log)
    branch_p = {}                                       # region label -> P(thick) where two branches fit

    def bulk_like(c):
        # thick (bulk or lifted) graphite: neutral grey and darker than any thin film can be on this oxide;
        # the nearest-curve fit would otherwise put it at the darkest point of the curve (~5 nm)
        return bool(absorbing and c[1] < cmin[1] - bulk_margin and c[2] / c[1] > max_blue_green)

    for r in regions:
        # channels clipped in most pixels only give a lower bound on the contrast
        fit = est.fit(r.median, censored=r.saturated_ch > 0.5)
        dec = branch.decide(est, r.median, censored=r.saturated_ch > 0.5) if branch else None
        if dec is not None:
            # colour fits both a thin and a thick branch: reported thin with the thick alternative; thin or thick is
            # decided over the whole chip once it is analysed (branch.resolve, at the target list), since the
            # boundary between the two colour groups shifts slightly from chip to chip
            p_k, tn, dn, tk, dk = dec
            branch_p[r.label] = p_k
            fit.thickness_nm, fit.residual = tn, dn
            fit.range_nm = (tn, tn)
            fit.alternatives_nm = [round(tk, 1)]
        fits[r.label] = fit
        a = float(r.mask.sum())
        par_area[r.parent] = par_area.get(r.parent, 0.0) + a
        thick_ = (fit.thickness_nm > tmax or r.saturated_frac > 0.5
                  or fit.residual > max(max_residual, 0.25 * float(np.linalg.norm(r.median)))
                  or bulk_like(r.median))
        if thick_ and prefer_thin_ambiguous and not absorbing:
            # inclusive (2026-10-25): a colour that a film inside the window also has (an acceptable match) is not
            # counted as thick material around the other regions (enclosed_in_thick, the flake's thick fraction)
            t_in, r_in = est.best_in_range(r.median, tmin, tmax, censored=r.saturated_ch > 0.5)
            thick_ = not r_in <= max(max_residual, 0.25 * float(np.linalg.norm(r.median)))
        if thick_:
            par_thick[r.parent] = par_thick.get(r.parent, 0.0) + a

    # context for thin / thick (transparent materials): a ~150 nm hBN flake has the 10x colour of a ~20 nm one, but its
    # tapered edges and thinner steps pass through colours no film under ~60 nm shows (orange, red, purple on ~300 nm
    # oxide), and it often touches thick material whose colour is unambiguous
    ctx = {}
    if not absorbing:
        ctx = thick_context(regions, labels, cm.contrast, est, px_um, thin_hi=(branch.cal["thin_range_nm"][1] if branch else 60.0))

    # tape residue (transparent materials): a faint region (|C| < haze_contrast) in a field of similar haze, irregular
    # or with diffuse edges is residue film, not a thin flake (10x residue labels, multitest chip_1, 2026-10-02: caught
    # 30 of 31 residue terraces, lost 2 of 9 faint flake terraces, both ~2 nm with score ~0.02).  It is flagged, and
    # kept out of the flake masks: the film otherwise joins a real flake next to it and the flake outline spreads
    # over the residue field (a flake mask 6x its terraces on mixed detections)
    residue = set()
    if not absorbing:
        for r in regions:
            if float(np.linalg.norm(r.median)) < haze_contrast and (
                    r.ring_similar > residue_ring_similar or _solidity(r.mask) < residue_solidity
                    or r.edge_width * px_um > residue_edge_um):
                residue.add(r.label)

    # flakes: whole connected regions of material; terraces are regions of interest inside
    FLK, flake_lab, region_flake = build_flakes(regions, labels, cs, cm.noise, cm.chip_mask, material,
                                                px_um, work_scale, exclude=residue)
    comp = {fid: [0.0, 0.0] for fid in FLK}
    for r in regions:
        fid = region_flake.get(r.label)
        if fid is None:
            continue
        fit = fits[r.label]
        a = float(r.mask.sum()) * px_um ** 2
        ts = [fit.thickness_nm] + list(fit.alternatives_nm)
        bad = fit.residual > max(max_residual, 0.25 * float(np.linalg.norm(r.median)))
        if not bad and not bulk_like(r.median) and any(tmin <= t <= tmax for t in (ts if not absorbing else ts[:1])):
            comp[fid][0] += a
        elif bulk_like(r.median) or fit.thickness_nm > tmax or r.saturated_frac > 0.5 or (bad and float(np.mean(r.median)) < -0.3):
            comp[fid][1] += a
    fmetric = {}

    def flake_metrics(fid):
        if fid not in fmetric:
            fl = FLK[fid]
            fmetric[fid] = dict(area=fl.area_um2, d=fl.inscribed_d_um,
                                focus=focus_width_um(img.rgb, fl, work_scale, img.pixel_um, cm.substrate_rgb),
                                crowd=crowding(fl, flake_lab, px_um),
                                win=min(1.0, comp[fid][0] / fl.area_um2), thick=min(1.0, comp[fid][1] / fl.area_um2))
        return fmetric[fid]

    flakes, region_of = [], {}
    for r in regions:
        npx = int(r.mask.sum())
        area = npx * px_um ** 2
        if area < min_area_um2:
            continue
        fit = fits[r.label]
        flags = []
        bp = branch_p.get(r.label)
        if bp is not None:
            flags.append("two_branch")
        elif not (tmin <= fit.thickness_nm <= tmax):
            # interference colours repeat; keep regions that also have an
            # acceptable solution inside the window, flagged as ambiguous
            t_in, r_in = est.best_in_range(r.median, tmin, tmax, censored=r.saturated_ch > 0.5)
            cn = float(np.linalg.norm(r.median))
            near = r_in <= max(ambiguous_residual, 1.5 * fit.residual) and cn >= 0.25
            if prefer_thin_ambiguous:           # (inclusive: any acceptable colour match inside the window)
                near = r_in <= max(ambiguous_residual, 1.5 * fit.residual, max_residual, 0.25 * cn)
            if not absorbing and near:
                flags.append("ambiguous_thickness")
                # (not for a colour just past tmax: its in-window match is the window's end, so it would be reported at
                # tmax exactly; it is a little thick, not the repeated thin colour; multitest chip_1, 2026-10-25: ~30
                # terraces fitted 100-106 nm would all have read 100.0)
                if prefer_thin_ambiguous and fit.thickness_nm > tmax and t_in < min(fit.thickness_nm, tmax - 5.0):
                    # the thinner, in-window thickness fits nearly as well: reported as the thickness (exfoliated flakes
                    # of that size are more often the thinner one; 2026-10-25), the thick one as the alternative
                    t_out = round(float(fit.thickness_nm), 1)
                    fit.thickness_nm, fit.residual, fit.range_nm = float(t_in), float(r_in), (float(t_in), float(t_in))
                    fit.alternatives_nm = sorted((set(fit.alternatives_nm) | {t_out}) - {round(t_in, 1)})
                else:
                    fit.alternatives_nm = sorted(set(fit.alternatives_nm) | {round(t_in, 1)})
            else:
                flags.append("out_of_range")
        elif not absorbing and tmax < est.d[-1]:
            # interference colours repeat: on ~300 nm oxide a ~150 nm hBN flake has nearly the colour of a ~20 nm
            # one at 10x.  A thicker-than-window branch that fits almost as well -> the flake may well be thick
            t_k, r_k = est.best_in_range(r.median, tmax, float(est.d[-1]), censored=r.saturated_ch > 0.5)
            if r_k <= fit.residual + thick_margin:
                flags.append("possibly_thick")
                fit.alternatives_nm = sorted(set(fit.alternatives_nm) | {round(t_k, 1)})
        if float(np.max(r.median)) < -0.97:
            flags.append("no_signal")          # black: outside the imaged area or the chip (edge slivers)
        elif bulk_like(r.median):
            flags.append("bulk")
        # model errors grow with contrast, so the tolerance scales with |C|
        if fit.residual > max(max_residual, 0.25 * float(np.linalg.norm(r.median))):
            flags.append("poor_colour_match")
        if getattr(fit, "off_model", False):
            flags.append("off_model")          # (soft: the thickness is a wide range, flakeid.thickness.OFF_MODEL_RESIDUAL)
        # colour barely distinguishable from bare substrate relative to the fit
        # quality (e.g. faint residue that happens to match a near-invisible
        # thickness): not a reliable identification
        if np.linalg.norm(r.median) < indistinct_ratio * fit.residual:
            flags.append("indistinct")
        if np.linalg.norm(r.median) < min_contrast:
            flags.append("low_contrast")
        # Few-layer graphite (absorbing films) on ~90 nm SiO2 is green-dominated
        # (observed B/G ~0.35-0.65).  Faint neutral-grey regions (B/G ~0.85-1)
        # are facets of thick crystals, dirt or shadows, not thin flakes.
        if absorbing and fit.thickness_nm < 5 and r.median[1] < 0:
            bg_ratio = float(r.median[2] / r.median[1])
            if bg_ratio > max_blue_green:
                flags.append("colour_mismatch")
        if r.label in residue:
            flags.append("residue_like")
        # thick by context: a colour that may be thin or thick, with clear thick-only colours at its edge or around it
        # (user thin/thick labels, multitest chip_1, 2026-10-03: 34 of 83 thick flakes, 0 of 55 thin)
        e_k, n_k = ctx.get(r.label, (0.0, 0.0))
        if (e_k >= thick_edge_um2 or n_k >= thick_near_um2) and any(
                f in flags for f in ("two_branch", "possibly_thick", "ambiguous_thickness")):
            flags.append("thick_context")
        if r.uniformity < min_uniformity:
            flags.append("non_uniform")
        if r.edge_width * px_um > max_edge_width_um:        # nan (no free edge) passes
            flags.append("blurry_edges")
        if r.saturated_frac > max_saturated_frac:
            flags.append("saturated")
        if r.touches_chip_edge:
            flags.append("chip_edge")
        if (any(tmin <= a <= tmax for a in fit.alternatives_nm) and "out_of_range" not in flags
                and "ambiguous_thickness" not in flags):
            flags.append("ambiguous_thickness")
        # context: faint facets inside thick crystals resemble few-layer films
        thick_parent = par_thick.get(r.parent, 0.0) / par_area[r.parent] > 0.2
        if (absorbing and thick_parent and "out_of_range" not in flags
                and fit.thickness_nm < 0.6 * tmax and r.thick_border > 0.5):
            flags.append("enclosed_in_thick")
        hard = {"out_of_range", "poor_colour_match", "non_uniform", "blurry_edges",
                "indistinct", "low_contrast", "enclosed_in_thick", "colour_mismatch", "bulk", "no_signal",
                "residue_like"}
        # user labels: strongly coloured regions with an imperfect colour fit are
        # as often worth imaging as the rest -> only a soft flag (scored)
        if float(np.linalg.norm(r.median)) >= 0.4:
            hard = hard - {"poor_colour_match"}
        if not include_rejected and hard & set(flags):
            continue

        # geometry
        m8 = r.mask.astype(np.uint8)
        ys, xs = np.nonzero(r.mask)
        y0, x0 = r.slices[0].start, r.slices[1].start
        cy, cx = ys.mean() + y0, xs.mean() + x0
        dist = cv2.distanceTransform(np.pad(m8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        ty, tx = np.unravel_index(np.argmax(dist), dist.shape)
        inscribed = 2 * float(dist.max()) * px_um
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.vstack(cnts).reshape(-1, 2).astype(np.float32)
        (_, _), (w_, h_), ang = cv2.minAreaRect(pts)
        solidity = _solidity(r.mask)
        fill, fr_sol = fullres_check(img.rgb, r, work_scale)
        # faint regions are where residue (dendritic, patchy) is confused with
        # thin flakes: demand flatter contrast and simpler geometry there
        faint = float(np.linalg.norm(r.median)) < faint_contrast
        if faint and not (fill == fill):      # could not be verified at full resolution (nan)
            fill, fr_sol = 0.0, 0.0
        if solidity < min_solidity or fr_sol < (min_fullres_solidity_faint if faint else min_fullres_solidity):
            flags.append("irregular_shape")
        if fill < (min_fill_faint if faint else min_fill):
            flags.append("not_flat")
        cmag = float(np.linalg.norm(r.median))
        if cmag < haze_contrast and r.ring_similar > max_ring_similar:
            flags.append("in_haze")
        if cmag < low_confidence_contrast:
            flags.append("low_confidence")      # faint: identification less certain
        if not include_rejected and {"not_flat", "in_haze"} & set(flags):     # irregular_shape: soft (labels)
            continue
        if w_ < h_:
            w_, h_, ang = h_, w_, ang + 90.0
        length, width = (w_ + 1) * px_um, (h_ + 1) * px_um

        x_px, y_px = to_orig(cx), to_orig(cy)
        tx_px, ty_px = to_orig(tx + x0), to_orig(ty + y0)
        pxo = img.pixel_um
        stage_x = stage_y = None
        if img.stage_map is not None:           # Micro-Manager mosaic: exact per-tile mapping
            stage_x, stage_y = img.stage_map.to_stage(tx_px, ty_px)
        elif stage_origin_um is not None:
            stage_x = stage_origin_um[0] + sx * tx_px * pxo
            stage_y = stage_origin_um[1] + sy * ty_px * pxo

        fid = region_flake.get(r.label)
        fmx = flake_metrics(fid) if fid is not None else dict(area=area, d=inscribed, focus=float("nan"),
                                                               crowd=0.0, win=1.0, thick=0.0)
        flakes.append(Terrace(
            terrace_id=0, flake_id=fid if fid is not None else -r.label,
            x_px=x_px, y_px=y_px, target_x_px=tx_px, target_y_px=ty_px,
            x_um=x_px * pxo, y_um=y_px * pxo, target_x_um=tx_px * pxo, target_y_um=ty_px * pxo,
            stage_x_um=stage_x, stage_y_um=stage_y,
            thickness_nm=round(fit.thickness_nm, 1),
            thickness_min_nm=round(fit.range_nm[0], 1), thickness_max_nm=round(fit.range_nm[1], 1),
            layers=round(fit.thickness_nm / est.layer_nm, 1),
            alt_thickness_nm=";".join(f"{a:g}" for a in fit.alternatives_nm),
            length_um=length, width_um=width, orientation_deg=(-ang) % 180.0,
            area_um2=area, inscribed_diameter_um=inscribed, solidity=solidity,
            fill=fill, fullres_solidity=fr_sol, ring_similar=r.ring_similar,
            bbox_x_px=int(to_orig(x0 - 0.5) + 0.5), bbox_y_px=int(to_orig(y0 - 0.5) + 0.5),
            bbox_w_px=int(round(r.mask.shape[1] / work_scale)),
            bbox_h_px=int(round(r.mask.shape[0] / work_scale)),
            contrast_r=float(r.median[0]), contrast_g=float(r.median[1]), contrast_b=float(r.median[2]),
            fit_residual=fit.residual, uniformity=r.uniformity,
            edge_width_um=r.edge_width * px_um, saturated_frac=r.saturated_frac,
            flake_area_um2=fmx["area"], flake_inscribed_d_um=fmx["d"], flake_focus_um=fmx["focus"],
            flake_crowding=fmx["crowd"], flake_frac_window=fmx["win"], flake_frac_thick=fmx["thick"],
            score=float("nan"), quality=";".join(flags) if flags else "good",
            edge_thick_um2=round(ctx.get(r.label, (0.0, 0.0))[0], 1), thick_nb_um2=round(ctx.get(r.label, (0.0, 0.0))[1], 1)))
        region_of[id(flakes[-1])] = r.label

    # score terraces; a flake scores as its best terrace.  Flakes are ranked by
    # score (flake 1 = best candidate), terraces grouped by flake, best first.
    scorer = Scorer(scorer_path(Path(prof.get("_path", "")).stem, material))
    terr = flakes
    # the default scorer was trained on Stream images, whose contrasts are compressed by
    # its processing; with another camera profile its contrast feature is taken
    # from the Stream model at the fitted thickness instead of the measured value.
    # A scorer fitted on this camera's own labels (profiles/scorer_<profile>_<material>.json) uses the measured contrast.
    ref_est = None
    if scorer.ok and not scorer.native and Path(prof.get("_path", "")).stem != DEFAULT_PROFILE:
        ref_est = ThicknessEstimator(material, oxide_nm, load_profile(None))
    for t in terr:
        cmag = None
        if ref_est is not None and t.thickness_nm == t.thickness_nm:
            cmag = float(np.linalg.norm(ref_est.model.contrast([t.thickness_nm])[0]))
        t.score = round(scorer(t, material, tmin, tmax, contrast_mag=cmag), 4) if scorer.ok else float("nan")
    key = (lambda t: -t.score) if scorer.ok else (lambda t: -t.area_um2)
    groups = {}
    for t in terr:
        groups.setdefault(t.flake_id, []).append(t)
    for g in groups.values():
        g.sort(key=key)
    order = sorted(groups, key=lambda k: key(groups[k][0]))
    terraces, summaries, reg_map, flake_of = [], [], {}, {}
    for rank, old in enumerate(order, start=1):
        g = groups[old]
        for t in g:
            reg_map[len(terraces) + 1] = region_of[id(t)]
            t.terrace_id = len(terraces) + 1
            t.flake_id = rank
            terraces.append(t)
        best = g[0]
        fl = FLK.get(old)
        if fl is not None:
            flake_of[rank] = old
            sl = fl.slices
            fx, fy = fl.centroid_px
            area, dd, L_, W_ = fl.area_um2, fl.inscribed_d_um, fl.length_um, fl.width_um
            bx, by = int(to_orig(sl[1].start - 0.5) + 0.5), int(to_orig(sl[0].start - 0.5) + 0.5)
            bw, bh = int(round(fl.mask.shape[1] / work_scale)), int(round(fl.mask.shape[0] / work_scale))
        else:
            fx, fy, area, dd, L_, W_ = best.x_px, best.y_px, best.area_um2, best.inscribed_diameter_um, best.length_um, best.width_um
            bx, by, bw, bh = best.bbox_x_px, best.bbox_y_px, best.bbox_w_px, best.bbox_h_px
        pxo = img.pixel_um
        summaries.append(FlakeSummary(
            flake_id=rank, score=best.score, material=material, n_terraces=len(g),
            terrace_ids=";".join(str(t.terrace_id) for t in g),
            thicknesses_nm=";".join(f"{t.thickness_nm:g}" for t in g),
            best_terrace_id=best.terrace_id, best_thickness_nm=best.thickness_nm,
            best_terrace_d_um=best.inscribed_diameter_um,
            x_px=fx, y_px=fy, target_x_px=best.target_x_px, target_y_px=best.target_y_px,
            x_um=fx * pxo, y_um=fy * pxo, target_x_um=best.target_x_um, target_y_um=best.target_y_um,
            stage_x_um=best.stage_x_um, stage_y_um=best.stage_y_um,
            area_um2=area, inscribed_diameter_um=dd, length_um=L_, width_um=W_,
            bbox_x_px=bx, bbox_y_px=by, bbox_w_px=bw, bbox_h_px=bh,
            focus_um=best.flake_focus_um, crowding=best.flake_crowding,
            frac_window=best.flake_frac_window, frac_thick=best.flake_frac_thick, quality=best.quality))
    flakes = summaries
    log(f"{len(flakes)} flakes with {len(terraces)} terraces reported")
    clean = []
    if clean_areas is not None:
        from .clean import clean_areas as find_clean
        rank_of = {lab_: rank for rank, lab_ in flake_of.items()}
        fsum = {f.flake_id: f for f in flakes}
        for c in find_clean(cm.contrast, cm.chip_mask, est, px_um, clean_areas, min_contrast):
            tx_px, ty_px = to_orig(c["tx"]), to_orig(c["ty"])
            st = (None, None)
            if img.stage_map is not None:
                st = img.stage_map.to_stage(tx_px, ty_px)
            elif stage_origin_um is not None:
                st = (stage_origin_um[0] + sx * tx_px * img.pixel_um, stage_origin_um[1] + sy * ty_px * img.pixel_um)
            fid = rank_of.get(int(flake_lab[c["ty"], c["tx"]])) if flake_lab is not None else None
            f_ = fsum.get(fid)
            clean.append(dict(c, target_x_px=float(tx_px), target_y_px=float(ty_px), stage_x_um=st[0], stage_y_um=st[1],
                              flake_id=fid, flake_stage=[f_.stage_x_um, f_.stage_y_um] if f_ is not None else None))
        log(f"{len(clean)} clean areas of {clean_areas[0]:g}-{clean_areas[1]:g} nm" + (f" (largest {clean[0]['area_um2']:.0f} um2)" if clean else ""))

    params = dict(material=material, thickness_range_nm=[tmin, tmax], min_area_um2=min_area_um2,
                  work_scale=work_scale, max_residual=max_residual, min_uniformity=min_uniformity,
                  max_edge_width_um=max_edge_width_um, min_solidity=min_solidity,
                  indistinct_ratio=indistinct_ratio, min_contrast=min_contrast, max_blue_green=max_blue_green, ambiguous_residual=ambiguous_residual,
                  prefer_thin_ambiguous=prefer_thin_ambiguous,
                  thick_margin=thick_margin, bulk_margin=bulk_margin,
                  min_fill=min_fill, min_fullres_solidity=min_fullres_solidity,
                  faint_contrast=faint_contrast, min_fill_faint=min_fill_faint,
                  min_fullres_solidity_faint=min_fullres_solidity_faint,
                  max_ring_similar=max_ring_similar, haze_contrast=haze_contrast,
                  residue_ring_similar=residue_ring_similar, residue_solidity=residue_solidity, residue_edge_um=residue_edge_um,
                  low_confidence_contrast=low_confidence_contrast, thick_edge_um2=thick_edge_um2, thick_near_um2=thick_near_um2,
                  refine_oxide=refine_oxide_thickness, oxide_refine_gain=oxide_gain,
                  profile=prof.get("_path"), scorer=scorer.path.name if scorer.ok else None, segment=asdict(seg_p),
                  substrate_noise=cm.noise.tolist(), substrate_rgb=cm.substrate_rgb.tolist(),
                  monolayer_contrast=mono.tolist(), stage_origin_um=stage_origin_um,
                  stage_axis_sign=list(stage_axis_sign), version=__version__,
                  runtime_s=round(time.time() - t0, 1))
    return FlakeResult(image=str(img.path), material=material, oxide_nm=float(oxide_nm),
                       oxide_nm_used=oxide_used, pixel_um=img.pixel_um,
                       pixel_um_source=img.pixel_um_source, flakes=flakes, terraces=terraces, params=params,
                       metadata=meta, _labels=labels, _scale=work_scale, _region_of=reg_map,
                       _flake_lab=flake_lab, _flake_of=flake_of,
                       _contrast=cm.contrast if keep_contrast else None,
                       _substrate_rgb=cm.substrate_rgb, _oxide_samples=samples, _clip_max=clip_max, _clean=clean)
