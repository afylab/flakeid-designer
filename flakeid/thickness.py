"""Mapping measured RGB contrast to flake thickness.

A ContrastModel (see optics.py) is tabulated on a fine thickness grid.  A
measured contrast vector is assigned the thickness of the closest point on
that curve; the curve distance is reported as a fit residual, and other
thickness branches that fit comparably well (thin-film interference makes
the curve loop back on itself) are reported as ambiguities.

Camera/illumination parameters are stored in JSON "profiles" so that a
calibration made on one microscope set-up can be reused.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .optics import MATERIALS, CameraModel, ContrastModel, resolve_material

PROFILE_DIR = Path(__file__).parent / "profiles"
DEFAULT_PROFILE = "bx51_xc10_6.3x"

# Search range of the thickness model per material (nm).  Wider than the
# range of interest so that thick flakes are recognised as thick instead of
# being forced onto a thin solution.
SEARCH_RANGE_NM = {"graphite": (0.0, 100.0), "hbn": (0.0, 400.0), "mos2": (0.0, 150.0)}

# Graphite thicker than the measured 1-4 layer table: the raw16 profile (calibrated on hBN) cannot reach the real
# colours of ~5-12 nm graphite (dark red with weak blue): they all land on its nearest point near 9-10 nm with residuals
# 0.15-0.20 (graphite_20261019_run_1).  Such a fit gets the range of every thickness within twice its residual instead
# of the narrow best-branch range, until the model is calibrated on AFM heights (TODO.md).
OFF_MODEL_RESIDUAL = {"graphite": 0.12}


def load_profile(name_or_path: str | None = None) -> dict:
    p = Path(name_or_path) if name_or_path else PROFILE_DIR / f"{DEFAULT_PROFILE}.json"
    if not p.exists():
        p = PROFILE_DIR / f"{name_or_path}.json"
    with open(p) as fh:
        prof = json.load(fh)
    prof["_path"] = str(p)
    return prof


def camera_from_profile(profile: dict) -> CameraModel:
    c = profile.get("camera", {})
    return CameraModel(
        lamp_temp_k=c.get("lamp_temp_k", 3100.0),
        centers_nm=tuple(c.get("centers_nm", (600.0, 535.0, 465.0))),
        fwhm_nm=tuple(c.get("fwhm_nm", (80.0, 85.0, 70.0))),
        ir_cut_nm=c.get("ir_cut_nm", 680.0),
        gamma=c.get("gamma", 2.2),
        ccm=tuple(map(tuple, c["ccm"])) if c.get("ccm") is not None else None,
        flare=tuple(c.get("flare", (0.0, 0.0, 0.0))),
    )


def save_profile(path, camera: CameraModel, na: float, notes: str = "", extra: dict | None = None):
    prof = {"camera": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(camera).items()},
            "na": na, "notes": notes}
    if extra:
        prof.update(extra)
    Path(path).write_text(json.dumps(prof, indent=2))


@dataclass
class ThicknessFit:
    thickness_nm: float
    residual: float            # distance in contrast space to the model curve
    range_nm: tuple            # (low, high) of the best branch within tolerance
    alternatives_nm: list      # thicknesses of other acceptable branches
    off_model: bool = False    # the colour is far from every point of the model curve (OFF_MODEL_RESIDUAL): range widened


class ThicknessEstimator:
    def __init__(self, material: str, oxide_nm: float, profile: dict | None = None,
                 step_nm: float = 0.1, search_range_nm: tuple | None = None,
                 clip_max: np.ndarray | None = None):
        """clip_max: (3,) largest contrast each channel can record (255/substrate - 1);
        the model curve is clipped the same way the camera clips."""
        self.material = resolve_material(material)
        self.oxide_nm = float(oxide_nm)
        self.profile = profile or load_profile()
        self.camera = camera_from_profile(self.profile)
        self.na = float(self.profile.get("na", 0.30))
        lo, hi = search_range_nm or SEARCH_RANGE_NM.get(self.material, (0.0, 200.0))
        self.d = np.arange(lo, hi + step_nm / 2, step_nm)
        table = self.profile.get("materials", {}).get(self.material, {}).get("index_table")
        self.model = ContrastModel(self.material, self.oxide_nm, na=self.na, camera=self.camera,
                                   index_table=table)
        self.curve = self.model.contrast(self.d)           # (N, 3)
        self.clip_max = None if clip_max is None else np.asarray(clip_max, float)
        if self.clip_max is not None:
            self.curve = np.minimum(self.curve, self.clip_max[None, :])
        self._tree = cKDTree(self.curve)

    @property
    def layer_nm(self):
        return self.model.layer_nm

    def model_index(self, wl):
        """Complex refractive index of the flake used by this model at wavelengths ``wl``."""
        t = self.model.index_table
        if t is not None:
            t = np.asarray(t, float)
            return np.interp(wl, t[:, 0], t[:, 1]) + 1j * np.interp(wl, t[:, 0], t[:, 2])
        return MATERIALS[self.material]["index"](np.asarray(wl, float))

    def best_in_range(self, c, lo, hi, censored=None):
        """Best thickness within [lo, hi] and its curve distance."""
        sel = (self.d >= lo) & (self.d <= hi)
        diff = self.curve[sel] - np.asarray(c)[None, :]
        if censored is not None and np.any(censored):
            cen = np.asarray(censored, bool)
            diff[:, cen] = np.minimum(diff[:, cen], 0.0)
        dist = np.linalg.norm(diff, axis=1)
        k = int(np.argmin(dist))
        return float(self.d[sel][k]), float(dist[k])

    def monolayer_contrast(self):
        return self.model.contrast([self.layer_nm])[0]

    def per_pixel(self, contrast: np.ndarray):
        """Nearest-curve thickness and residual for an (..., 3) array."""
        flat = contrast.reshape(-1, 3)
        dist, idx = self._tree.query(flat)
        return self.d[idx].reshape(contrast.shape[:-1]), dist.reshape(contrast.shape[:-1])

    def fit(self, c: np.ndarray, tol: float = 0.05, censored=None, prefer_thin: float = 0.02) -> ThicknessFit:
        """Thickness of one (median) contrast vector with uncertainty.

        tol:      contrast-space tolerance used to define the acceptable range
                  and competing solutions (systematic model error dominates).
        censored: (3,) bool, channels that are clipped (saturated) in the
                  image; for these the measured value is only a lower bound.
        """
        diff = self.curve - np.asarray(c)[None, :]
        if censored is not None and np.any(censored):
            cen = np.asarray(censored, bool)
            diff[:, cen] = np.minimum(diff[:, cen], 0.0)    # model above the clip level is fine
        dist = np.linalg.norm(diff, axis=1)
        best = float(dist.min())
        ok = dist <= np.hypot(best, tol)
        # contiguous branches of acceptable thickness, each with its best point
        lab, n = ndimage.label(ok)
        branches = []
        for k in range(1, n + 1):
            idx = np.flatnonzero(lab == k)
            j = idx[0] + int(np.argmin(dist[idx[0]:idx[-1] + 1]))
            branches.append((idx[0], idx[-1], j))
        # interference colours repeat: when a thinner branch fits almost as well
        # (within prefer_thin), take the thinner one, which is the likelier for
        # exfoliated flakes; the others are reported as alternatives
        good = [b for b in branches if dist[b[2]] <= best + prefer_thin]
        s0, e0, i = min(good, key=lambda b: self.d[b[2]])
        rng = (float(self.d[s0]), float(self.d[e0]))
        alts = [round(float(self.d[j]), 1) for s_, e_, j in branches if j != i]
        best = float(dist[i])
        lim = OFF_MODEL_RESIDUAL.get(self.material)
        if lim is not None and best > lim and self.d[i] > 1.5:        # (1-4 layers: the measured table, graphite.py)
            near = dist <= 2.0 * best
            lo = hi = i
            while lo > 0 and near[lo - 1]:
                lo -= 1
            while hi < len(near) - 1 and near[hi + 1]:
                hi += 1
            return ThicknessFit(float(self.d[i]), best, (float(self.d[lo]), float(self.d[hi])), alts, True)
        return ThicknessFit(float(self.d[i]), float(best), rng, alts)


LOCAL_CAL = Path.home() / ".flakeid"                # calibrations made on this computer (kept across program updates)


def model_oxide_nm(entered: float, log=None) -> float:
    """The oxide thickness the analysis models for an entered (nominal) one: a measured value (Filmetrics, recorded with a
    flat field in ~/.flakeid/oxide_measured.json) first, else profiles/oxide_nominal.json (e.g. "90 nm" -> 82 nm); other
    values are returned unchanged."""
    try:
        meas = json.loads((LOCAL_CAL / "oxide_measured.json").read_text(encoding="utf-8"))
        for k, v in meas.items():
            if abs(float(k) - float(entered)) <= 0.5:
                if log:
                    log(f"oxide: {float(entered):g} nm (nominal) modelled as {float(v['measured_nm']):g} nm (measured {v.get('date', '')}, "
                        f"{LOCAL_CAL / 'oxide_measured.json'})")
                return float(v["measured_nm"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        table = json.loads((PROFILE_DIR / "oxide_nominal.json").read_text())
    except (OSError, ValueError):
        return float(entered)
    for k, v in table.items():
        try:
            if abs(float(k) - float(entered)) <= 0.5:
                if log:
                    log(f"oxide: {float(entered):g} nm (nominal) modelled as {float(v):g} nm (profiles/oxide_nominal.json)")
                return float(v)
        except ValueError:
            continue
    return float(entered)


def refine_oxide(samples: np.ndarray, material: str, oxide_nm: float, profile: dict,
                 rel_range: float = 0.10, n: int = 21, trim: float = 0.6, clip_max=None, max_rel: float = 0.20) -> tuple:
    """Find the oxide thickness (within +/- rel_range) whose model curve best
    explains the contrast of uniform flake regions.  Returns (oxide_nm, gain)
    where gain is the relative improvement of the trimmed loss."""
    def loss(ox):
        est = ThicknessEstimator(material, ox, profile, step_nm=0.5, clip_max=clip_max)
        d, _ = est._tree.query(samples)
        d = np.sort(d)[: max(1, int(trim * len(d)))]
        return float(np.mean(d ** 2))

    grid = oxide_nm * np.linspace(1 - rel_range, 1 + rel_range, n)
    losses = np.array([loss(o) for o in grid])
    k = int(np.argmin(losses))
    # best fit at an end of the range: the entered oxide is off by more than rel_range (a "90 nm" graphite chip that
    # measures ~80 nm, microscope PC 2026-10-03: every block stopped at 81.0 = -10 %, the oxide was never fixed and
    # the layer calibration was not used).  Extend the grid on that side, up to max_rel
    step = grid[1] - grid[0]
    while (k == 0 or k == len(grid) - 1) and abs(grid[k] / oxide_nm - 1) < max_rel - 1e-9:
        new = grid[0] - step * np.arange(1, 4)[::-1] if k == 0 else grid[-1] + step * np.arange(1, 4)
        new = new[np.abs(new / oxide_nm - 1) <= max_rel + 1e-9]
        if not len(new):
            break
        nl = np.array([loss(o) for o in new])
        grid, losses = (np.concatenate([new, grid]), np.concatenate([nl, losses])) if k == 0 else \
            (np.concatenate([grid, new]), np.concatenate([losses, nl]))
        k = int(np.argmin(losses))
    n = len(grid)
    # parabolic refinement around the grid minimum
    best = grid[k]
    if 0 < k < n - 1:
        y0, y1, y2 = losses[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        if den > 0:
            best = grid[k] + 0.5 * (y0 - y2) / den * (grid[1] - grid[0])
    base = loss(oxide_nm)
    return float(best), (base - losses[k]) / base if base > 0 else 0.0
