#!/usr/bin/env python3
"""Fit the 'worth imaging' scorer to the user's labels.

    python tools/fit_scorer.py --labels label_set/labels_joined.json -o flakeid/profiles/scorer.json

For every scan the pipeline is run with rejected regions included; each
labelled crop is matched to its terrace (target point within 3 um), terrace
and flake features are computed, and a regularised logistic regression is
fitted for P(useful or maybe).  Evaluation is leave-one-scan-out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid import find_flakes  # noqa: E402
from flakeid.scorer import FEATURES, features  # noqa: E402

SCANS = {"g31": ("graphite", 90), "g32": ("graphite", 90), "g33": ("graphite", 90), "g34": ("graphite", 90),
         "g35": ("graphite", 90), "bn24": ("hbn", 285), "bn25": ("hbn", 285), "bn26": ("hbn", 285),
         "fullchipbn23": ("hbn", 285)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="label_set/labels_joined.json")
    ap.add_argument("--images", default="example images")
    ap.add_argument("-o", "--out", default="flakeid/profiles/scorer.json")
    ap.add_argument("--cache", help="npz with extracted features (skip the pipeline runs)")
    a = ap.parse_args()
    R = json.load(open(a.labels))

    if a.cache and Path(a.cache).exists():
        z = np.load(a.cache, allow_pickle=True)
        X, y_good, y_useful, scans, mats, tags, focus_tag = (z[k] for k in
                                                              ["X", "yg", "yu", "scans", "mats", "tags", "ft"])
        X = X[:, :len(FEATURES)]           # older caches carried extra trailing columns
    else:
        rows = []
        for stem, (mat, ox) in SCANS.items():
            lab = [r for r in R if r["item"]["scan"] == stem]
            if not lab:
                continue
            res = find_flakes(Path(a.images) / f"{stem}.jpg", mat, ox, include_rejected=True)
            tmin, tmax = res.params["thickness_range_nm"]
            T = np.array([[f.target_x_um, f.target_y_um] for f in res.terraces])
            n_miss = 0
            for r in lab:
                it = r["item"]
                d = np.hypot(T[:, 0] - it["x"], T[:, 1] - it["y"])
                k = int(np.argmin(d))
                if d[k] > 3:
                    n_miss += 1
                    continue
                f = res.terraces[k]
                v = r["label"].get("verdict")
                tags_ = r["label"].get("tags") or []
                rows.append((features(f, mat, tmin, tmax), v in ("useful", "maybe"), v == "useful", stem, mat,
                             ";".join(tags_), "focus" in tags_))
            print(f"{stem}: {len(lab) - n_miss}/{len(lab)} labelled crops matched")
        X = np.array([r[0] for r in rows], float)
        y_good, y_useful = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
        scans, mats = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
        tags, focus_tag = np.array([r[5] for r in rows]), np.array([r[6] for r in rows])
        if a.cache:
            np.savez(a.cache, X=X, yg=y_good, yu=y_useful, scans=scans, mats=mats, tags=tags, ft=focus_tag)

    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    print(f"\n{len(X)} labelled terraces; useful+maybe = {y_good.mean():.2f}, useful = {y_useful.mean():.2f}")
    print("\nsingle-feature AUC (useful+maybe vs not)   |  AUC vs 'out of focus' tag")
    for j, name in enumerate(FEATURES):
        auc = roc_auc_score(y_good, X[:, j])
        aucf = roc_auc_score(focus_tag, X[:, j]) if focus_tag.any() else float("nan")
        print(f"  {name:22s} {auc:5.2f}   |   {aucf:5.2f}")

    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd

    def fit(Zt, yt):
        return LogisticRegression(C=0.5, max_iter=2000).fit(Zt, yt)

    res = {}
    for target, y in (("useful_or_maybe", y_good), ("useful", y_useful)):
        p = np.zeros(len(y))
        for s in np.unique(scans):
            m = scans == s
            p[m] = fit(Z[~m], y[~m]).predict_proba(Z[m])[:, 1]
        auc = roc_auc_score(y, p)
        order = np.argsort(-p)
        prec = {k: float(y[order[:k]].mean()) for k in (25, 50, 100)}
        by_mat = {mt: float(roc_auc_score(y[mats == mt], p[mats == mt])) for mt in np.unique(mats)}
        res[target] = dict(auc_loso=float(auc), precision_top=prec, auc_by_material=by_mat)
        print(f"\n{target}: leave-one-scan-out AUC {auc:.3f}  by material {by_mat}  precision@25/50/100 {prec}")
    model = fit(Z, y_good)
    coef = dict(zip(FEATURES, model.coef_[0].round(4).tolist()))
    print("\ncoefficients (standardised):")
    for k, v in sorted(coef.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:22s} {v:+.3f}")
    out = dict(features=FEATURES, mean=mu.round(6).tolist(), std=sd.round(6).tolist(),
               fill_value=med.round(6).tolist(), coef=model.coef_[0].round(6).tolist(),
               intercept=float(model.intercept_[0]), target="P(useful or maybe) from user labels, 391 crops",
               evaluation=res)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
