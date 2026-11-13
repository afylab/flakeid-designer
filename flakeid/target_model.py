"""'Kept at 100x' probability of a 10x flake (tools/fit_target_model.py), for ranking the 100x targets.

The 10x score (scorer.py) was fitted on 10x user labels: worth imaging, judged from the 10x picture.  This model is
fitted on what happened at 100x instead: whether a targeted flake was kept in the designer review (keep / used) or
discarded / dropped by the 100x screening.  Its inputs are flake-level numbers from the live flake and terrace tables:
the 10x score itself, the flake's and its best terrace's size, shape, contrast and edge measures, a few summaries over
all of its terraces and the quality flags.  A logistic model per material; coefficients in
profiles/target_<material>.json; without that file the probability is not computed (nan) and nothing changes.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROFILES = Path(__file__).parent / "profiles"
FLAGS = ["thick_context", "two_branch", "low_confidence", "ambiguous_thickness", "possibly_thick", "irregular_shape",
         "poor_colour_match"]
BT_COLS = ["solidity", "fill", "fullres_solidity", "ring_similar", "fit_residual", "uniformity", "edge_width_um",
           "saturated_frac"]


def _f(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _flags(s):
    return set(filter(None, (s or "").split(";")))


def flake_features(r, ts):
    """Feature dict of one flake: r its flake-table row, ts its terrace-table rows (strings or numbers)."""
    o = {}
    o["score"] = _f(r.get("score"))
    o["log_best_d"] = math.log(max(_f(r.get("best_terrace_d_um"), 0.0), 1.0))
    o["best_t"] = _f(r.get("best_thickness_nm"))
    fa, fd = _f(r.get("area_um2"), 0.0), _f(r.get("inscribed_diameter_um"), 0.0)
    o["log_flake_area"] = math.log(max(fa, 1.0))
    o["log_flake_d"] = math.log(max(fd, 1.0))
    o["aspect"] = _f(r.get("length_um")) / max(_f(r.get("width_um")), 1e-6)
    o["n_terraces"] = _f(r.get("n_terraces"))
    fo = _f(r.get("focus_um"))
    o["focus_um"] = min(fo, 15.0) if fo == fo else fo
    o["crowding"] = _f(r.get("crowding"))
    o["frac_window"] = _f(r.get("frac_window"))
    o["frac_thick"] = _f(r.get("frac_thick"))
    q = _flags(r.get("quality"))
    for fl in FLAGS:
        o["q_" + fl] = float(fl in q)
    bt = next((t for t in ts if str(t.get("terrace_id")) == str(r.get("best_terrace_id"))), None)
    g = (lambda c: _f(bt.get(c))) if bt is not None else (lambda c: float("nan"))
    for c in BT_COLS:
        o["bt_" + c] = g(c)
    o["bt_contrast_mag"] = math.sqrt(g("contrast_r") ** 2 + g("contrast_g") ** 2 + g("contrast_b") ** 2)
    o["bt_log_edge_thick"] = math.log1p(g("edge_thick_um2"))
    o["bt_log_thick_nb"] = math.log1p(g("thick_nb_um2"))
    o["bt_t_range"] = g("thickness_max_nm") - g("thickness_min_nm")
    o["bt_ext_vs_inscr"] = g("length_um") / max(g("inscribed_diameter_um"), 1.0)
    o["bt_alt"] = float(bool(bt.get("alt_thickness_nm"))) if bt is not None else float("nan")
    nan = float("nan")
    o.update(n_ter_big=nan, best_area_frac=nan, ter_t_spread=nan, min_uniformity=nan, max_ring=nan)
    if ts:
        ar = np.array([_f(t.get("area_um2")) for t in ts])
        th = np.array([_f(t.get("thickness_nm")) for t in ts])
        ba = g("area_um2")
        o["best_area_frac"] = ba / fa if fa > 0 else nan
        o["ter_t_spread"] = float(np.nanmax(th) - np.nanmin(th)) if len(ts) > 1 else 0.0
        un = [_f(t.get("uniformity")) for t in ts]
        rg = [_f(t.get("ring_similar")) for t in ts]
        o["min_uniformity"] = float(np.nanmin(un)) if np.isfinite(un).any() else nan
        o["max_ring"] = float(np.nanmax(rg)) if np.isfinite(rg).any() else nan
        o["n_ter_big"] = float(np.sum(ar > 0.1 * ba)) if ba == ba else nan
    return o


def read_terraces(path):
    """flake_id (str) -> its terrace rows."""
    by = defaultdict(list)
    p = Path(str(path))
    if p.is_file():
        with open(str(p), newline="", encoding="utf-8") as fh:
            for t in csv.DictReader(fh):
                by[str(t["flake_id"])].append(t)
    return by


def model_path(material):
    return PROFILES / f"target_{material}.json"


class TargetModel:
    def __init__(self, material=None, path=None):
        p = Path(path) if path else model_path(material or "")
        self.ok = bool(material or path) and p.is_file()
        self.path = p
        if self.ok:
            d = json.loads(p.read_text(encoding="utf-8"))
            self.meta = d
            self.features = d["features"]
            self.mu, self.sd = np.array(d["mean"], float), np.array(d["std"], float)
            self.sd = np.where(self.sd > 0, self.sd, 1.0)
            self.fillv, self.w, self.b = np.array(d["fill_value"], float), np.array(d["coef"], float), float(d["intercept"])

    def __call__(self, feats):
        if not self.ok:
            return float("nan")
        x = np.array([_f(feats.get(c)) for c in self.features], float)
        x = np.where(np.isfinite(x), x, self.fillv)
        z = float(((x - self.mu) / self.sd) @ self.w + self.b)
        return 1.0 / (1.0 + math.exp(-min(max(z, -50.0), 50.0)))


def annotate(rows, flakes, terraces_csv, material, log=print):
    """Set r['p_keep100'] on each target row (any dict with flake_id) from the flake table rows `flakes` and the
    terrace table; returns whether the model was available.  Never raises (a missing model or table: nan)."""
    try:
        m = TargetModel(material)
        if not m.ok:
            for r in rows:
                r["p_keep100"] = float("nan")
            return False
        ter = read_terraces(terraces_csv)
        fl = {str(r["flake_id"]): r for r in flakes}
        for r in rows:
            f = fl.get(str(r["flake_id"]))
            r["p_keep100"] = m(flake_features(f, ter.get(str(r["flake_id"]), []))) if f is not None else float("nan")
        return True
    except Exception as e:                      # (never costs the target list)
        log(f"(kept-at-100x probability not computed: {type(e).__name__}: {e})")
        for r in rows:
            r["p_keep100"] = float("nan")
        return False
