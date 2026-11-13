"""'Worth imaging' score learned from user labels (tools/fit_scorer.py).

A logistic model over terrace and flake features.  The score of a flake is
the best score of its terraces.  Coefficients live in profiles/scorer.json;
without that file the score is not computed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES = [
    "log_terrace_d", "log_terrace_area", "log_flake_area", "log_flake_d", "flake_focus_um",
    "edge_width_um", "haze_ring", "crowding", "flake_frac_window", "flake_frac_thick",
    "fullres_solidity", "solidity", "fill", "contrast_mag", "fit_residual", "t_in_window",
    "saturated", "ambiguous", "low_confidence",
]
HARD = {"out_of_range", "poor_colour_match", "non_uniform", "blurry_edges", "indistinct", "low_contrast",
        "enclosed_in_thick", "colour_mismatch", "irregular_shape", "not_flat", "in_haze", "bulk", "no_signal", "residue_like"}


def features(f, material, tmin, tmax):
    q = set(f.quality.split(";"))
    t = f.thickness_nm
    if "ambiguous_thickness" in q and f.alt_thickness_nm:
        ins = [float(a) for a in f.alt_thickness_nm.split(";") if tmin <= float(a) <= tmax]
        if ins and not (tmin <= t <= tmax):
            t = ins[0]
    tw = float(np.clip((t - tmin) / max(tmax - tmin, 1e-6), -0.5, 1.5))
    cm = float(np.linalg.norm([f.contrast_r, f.contrast_g, f.contrast_b]))
    ew = f.edge_width_um if f.edge_width_um == f.edge_width_um else np.nan
    return [np.log(max(f.inscribed_diameter_um, 1.0)), np.log(max(f.area_um2, 1.0)),
            np.log(max(f.flake_area_um2, 1.0)), np.log(max(f.flake_inscribed_d_um, 1.0)),
            min(f.flake_focus_um, 15.0) if f.flake_focus_um == f.flake_focus_um else np.nan,
            ew, f.ring_similar, f.flake_crowding, f.flake_frac_window, f.flake_frac_thick,
            f.fullres_solidity, f.solidity, f.fill, cm, f.fit_residual, tw,
            float(f.saturated_frac > 0.5), float("ambiguous_thickness" in q), float("low_confidence" in q)]


def scorer_path(profile_stem=None, material=None):
    """profiles/scorer_*.json fitted for this camera profile and material (tools/fit_scorer_mm.py), else
    profiles/scorer.json (fitted on Stream exports)."""
    d = Path(__file__).parent / "profiles"
    if profile_stem and material:
        for p in sorted(d.glob("scorer_*.json")):
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            if j.get("material") == material and profile_stem in j.get("profiles", []):
                return p
    return d / "scorer.json"


class Scorer:
    def __init__(self, path=None):
        p = Path(path) if path else Path(__file__).parent / "profiles" / "scorer.json"
        self.ok = p.exists()
        self.path, self.native = p, False
        if self.ok:
            d = json.loads(p.read_text())
            self.mu, self.sd = np.array(d["mean"]), np.array(d["std"])
            self.sd = np.where(self.sd > 0, self.sd, 1.0)
            self.fillv, self.w, self.b = np.array(d["fill_value"]), np.array(d["coef"]), d["intercept"]
            self.native = bool(d.get("native_contrast", False))   # fitted on this camera: measured contrast as is

    def __call__(self, f, material, tmin, tmax, contrast_mag=None):
        if not self.ok:
            return float("nan")
        x = np.array(features(f, material, tmin, tmax), float)
        if contrast_mag is not None:
            x[FEATURES.index("contrast_mag")] = contrast_mag
        x = np.where(np.isfinite(x), x, self.fillv)
        z = (x - self.mu) / self.sd
        return float(1 / (1 + np.exp(-(z @ self.w + self.b))))
