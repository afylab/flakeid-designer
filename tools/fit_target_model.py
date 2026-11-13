#!/usr/bin/env python3
"""Fit the 'kept at 100x' target model (flakeid/target_model.py) on past 100x outcomes.

    python tools/fit_target_model.py --live LIVE_DIR --labels target_labels_10x.csv [--write]

LIVE_DIR holds <run>/<piece>/ copies of the live scan tables (*_tiles_live_flakes.csv, *_tiles_live_terraces.csv);
the labels CSV has key, material, run, piece, flake_id, outcome (keep | discard | dropped: a targeted flake kept in
the designer review, discarded there, or dropped by the 100x screening).  Features are computed with the scan's own
code (target_model.flake_features), so training and use cannot drift apart.

Per material: an L2 logistic model (class-balanced, intercept then corrected to the real keep rate), C chosen by
leave-one-run-out AUC; reported against the 10x score + best terrace size refit the same way, and as keeps captured
per piece when only the top K targets are imaged.  --write saves flakeid/profiles/target_<material>.json (needs
sklearn: run with the analysis / ML environment, not on the microscope PC).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid import target_model as TM  # noqa: E402

# feature sets as tested (2026-10-23, ml/README.md): hBN without the quality flags (rare on the hBN targets), graphite all
COMMON = ["score", "log_best_d", "best_t", "log_flake_area", "log_flake_d", "aspect", "n_terraces", "n_ter_big",
          "best_area_frac", "ter_t_spread", "focus_um", "crowding", "frac_window", "frac_thick", "bt_solidity", "bt_fill",
          "bt_fullres_solidity", "bt_ring_similar", "bt_fit_residual", "bt_uniformity", "bt_edge_width_um",
          "bt_contrast_mag", "bt_log_edge_thick", "bt_log_thick_nb", "min_uniformity", "max_ring"]
FEATURES = {"hbn": COMMON,
            "graphite": COMMON + ["bt_saturated_frac", "bt_t_range", "bt_ext_vs_inscr", "bt_alt"]
            + ["q_" + f for f in TM.FLAGS]}
BASE = ["score", "log_best_d"]
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def load(live, labels):
    tabs = {}
    out = defaultdict(list)
    with open(labels, newline="", encoding="utf-8") as fh:
        L = list(csv.DictReader(fh))
    for r in L:
        k = (r["run"], r["piece"])
        if k not in tabs:
            d = Path(live) / r["run"] / r["piece"]
            fl = glob.glob(str(d / "*_tiles_live_flakes.csv"))
            if not fl:
                tabs[k] = None
                continue
            with open(fl[0], newline="", encoding="utf-8") as fh:
                rows = {str(x["flake_id"]): x for x in csv.DictReader(fh)}
            tabs[k] = (rows, TM.read_terraces(fl[0].replace("_flakes.csv", "_terraces.csv")))
        if tabs[k] is None or str(r["flake_id"]) not in tabs[k][0]:
            continue
        rows, ter = tabs[k]
        f = TM.flake_features(rows[str(r["flake_id"])], ter.get(str(r["flake_id"]), []))
        out[r["material"]].append(dict(key=r["key"], run=r["run"], piece=r["piece"], y=int(r["outcome"] == "keep"), f=f))
    missing = sum(1 for v in tabs.values() if v is None)
    return out, missing


class Fit:
    def __init__(self, X, y, C):
        fill = np.nanmedian(X, 0)
        self.fill = np.where(np.isfinite(fill), fill, 0.0)
        Xf = np.where(np.isfinite(X), X, self.fill)
        self.mu, self.sd = Xf.mean(0), Xf.std(0)
        self.sd = np.where(self.sd > 0, self.sd, 1.0)
        m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((Xf - self.mu) / self.sd, y)
        self.w = m.coef_[0]
        n1, n0 = y.sum(), len(y) - y.sum()
        self.b = float(m.intercept_[0] - np.log(n0 / n1))       # balanced weights shift the intercept by ln(n0/n1)

    def decision(self, X):
        Xf = np.where(np.isfinite(X), X, self.fill)
        return ((Xf - self.mu) / self.sd) @ self.w + self.b


def loro(X, y, runs, Cs):
    """held-out decision values, C chosen per held-out run by leave-one-run-out on the other runs."""
    out = np.zeros(len(y))
    for r in np.unique(runs):
        te, tr = runs == r, runs != r
        best = None
        for C in Cs:
            z = np.zeros(tr.sum())
            Xt, yt, rt = X[tr], y[tr], runs[tr]
            for r2 in np.unique(rt):
                a = rt != r2
                z[~a] = Fit(Xt[a], yt[a], C).decision(Xt[~a])
            auc = roc_auc_score(yt, z)
            if best is None or auc > best[0]:
                best = (auc, C)
        out[te] = Fit(X[tr], y[tr], best[1]).decision(X[te])
    return out


def choose_c(X, y, runs, Cs):
    best = None
    for C in Cs:
        z = np.zeros(len(y))
        for r in np.unique(runs):
            a = runs != r
            z[~a] = Fit(X[a], y[a], C).decision(X[~a])
        auc = roc_auc_score(y, z)
        if best is None or auc > best[0]:
            best = (auc, C)
    return best[1]


def per_run_auc(z, y, runs):
    v = [roc_auc_score(y[runs == r], z[runs == r]) for r in np.unique(runs) if 0 < y[runs == r].sum() < (runs == r).sum()]
    return float(np.mean(v))


def topk(z, y, pieces, K):
    k = n = 0
    for p in np.unique(pieces):
        i = np.where(pieces == p)[0]
        o = i[np.argsort(-z[i], kind="stable")][:K]
        k += int(y[o].sum()); n += len(o)
    return k, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--material", nargs="*", default=["hbn", "graphite"])
    ap.add_argument("--write", action="store_true", help="save flakeid/profiles/target_<material>.json")
    a = ap.parse_args(argv)
    data, missing = load(a.live, a.labels)
    if missing:
        print(f"({missing} pieces without live tables in {a.live}: their labels skipped)")
    for mat in a.material:
        D = data.get(mat, [])
        if not D:
            print(mat, "no labelled flakes"); continue
        feats = FEATURES[mat]
        X = np.array([[d["f"][c] for c in feats] for d in D], float)
        XB = np.array([[d["f"][c] for c in BASE] for d in D], float)
        y = np.array([d["y"] for d in D]); runs = np.array([d["run"] for d in D])
        pieces = np.array([d["run"] + "/" + d["piece"] for d in D])
        print(f"\n{mat}: {len(D)} targets, {y.sum()} kept, {len(set(runs))} runs, {len(set(pieces))} pieces "
              f"(leave one run out, C chosen on the other runs)")
        zb = loro(XB, y, runs, CS)
        zm = loro(X, y, runs, CS)
        size = XB[:, 1]
        print(f"  AUC pooled / mean per run:  score + size {roc_auc_score(y, zb):.3f} / {per_run_auc(zb, y, runs):.3f}   "
              f"target model ({len(feats)} numbers) {roc_auc_score(y, zm):.3f} / {per_run_auc(zm, y, runs):.3f}")
        print("  keeps when only the top K per piece are imaged (by: largest best terrace | 10x score | target model):")
        for K in (10, 20, 30, 40, 60):
            ks, n = topk(size, y, pieces, K)
            print(f"    K {K:2d}: {n:5d} shots ({100 * n / len(y):3.0f} %)  {ks:4d} | {topk(XB[:, 0], y, pieces, K)[0]:4d} | "
                  f"{topk(zm, y, pieces, K)[0]:4d}   of {y.sum()}")
        C = choose_c(X, y, runs, CS)
        m = Fit(X, y, C)
        order = np.argsort(-np.abs(m.w))
        print(f"  final fit on all runs, C {C:g}; largest weights (per standard deviation): "
              + ", ".join(f"{feats[i]} {m.w[i]:+.2f}" for i in order[:8]))
        if a.write:
            p = TM.model_path(mat)
            j = dict(material=mat, features=feats, mean=m.mu.tolist(), std=m.sd.tolist(), fill_value=m.fill.tolist(),
                     coef=m.w.tolist(), intercept=m.b, C=C,
                     fitted=datetime.date.today().isoformat(), runs=sorted(set(runs.tolist())), n=int(len(y)),
                     kept=int(y.sum()), auc_loro=round(float(roc_auc_score(y, zm)), 3),
                     auc_loro_score_size=round(float(roc_auc_score(y, zb)), 3),
                     note="P(kept at 100x | targeted), from tools/fit_target_model.py")
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(json.dumps(j, indent=1), encoding="utf-8")
            os.replace(str(tmp), str(p))
            print(f"  wrote {p}")


if __name__ == "__main__":
    main()
