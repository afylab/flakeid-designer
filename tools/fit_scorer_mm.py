#!/usr/bin/env python3
"""Fit a camera-specific 'worth imaging' scorer from labels on an analysed scan (no pipeline rerun).

    python tools/fit_scorer_mm.py --flakes "testing images/bigbntest_2_results/bigbntest_2_tiles_flakes.json" \
        --labels LABEL_DIR_OR_JSON -o flakeid/profiles/scorer_bx51_xc10_mm_raw16_hbn.json

Labels: the label page's documents ({csv_flake_id, verdict: useful | maybe | not, tags}), either a folder of
one JSON file per document or one JSON list.  Each label is matched to its flake's best terrace (the outlined
region on the label page); the terrace features are those of flakeid/scorer.py, with the measured contrast
(the scorer is fitted on this camera's own data, so no mapping to the Stream model).  Target: P(useful or
maybe).  Evaluation: 5-fold cross-validation repeated 10 times.

The pipeline uses this scorer for scans analysed with any of the profiles listed in "profiles" and the
given material; other scans keep profiles/scorer.json (fitted on Stream exports).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.scorer import FEATURES, features  # noqa: E402


def load_labels(src):
    p = Path(src)
    docs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(p / "*.json")))] if p.is_dir() \
        else json.loads(p.read_text())
    return {int(d["csv_flake_id"]): d for d in docs if d.get("verdict") in ("useful", "maybe", "not")}


def terrace_ns(t):
    ns = SimpleNamespace(**{k: (float("nan") if v is None else v) for k, v in t.items()})
    ns.quality = t.get("quality") or "good"
    ns.alt_thickness_nm = t.get("alt_thickness_nm") or ""
    return ns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flakes", required=True, help="<scan>_flakes.json of the labelled scan")
    ap.add_argument("--labels", required=True, help="folder of label documents, or a JSON list of them")
    ap.add_argument("--profiles", nargs="+", default=["bx51_xc10_mm_raw16", "bx51_xc10_mm_raw16_afm"],
                    help="camera profiles this scorer is for")
    ap.add_argument("--C", type=float, default=0.3, help="inverse L2 regularisation of the logistic model")
    ap.add_argument("--old-threshold", type=float, default=0.75,
                    help="flakeid.live --min-score used so far: the new threshold giving the same number of targets is printed")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)

    F = json.loads(Path(a.flakes).read_text())
    mat = F["material"]
    tmin, tmax = F["params"]["thickness_range_nm"]
    T = {t["terrace_id"]: t for t in F["terraces"]}
    fl = {f["flake_id"]: f for f in F["flakes"]}
    L = load_labels(a.labels)
    rows = [(fid, d["verdict"], fl[fid]["score"], features(terrace_ns(T[fl[fid]["best_terrace_id"]]), mat, tmin, tmax))
            for fid, d in L.items() if fid in fl]
    v = np.array([r[1] for r in rows])
    old = np.array([r[2] for r in rows], float)
    X = np.array([r[3] for r in rows], float)
    y = (v != "not").astype(int)
    yu = (v == "useful").astype(int)
    print(f"{len(rows)} of {len(L)} labels matched to flakes of {Path(a.flakes).name} ({mat}): "
          f"useful {yu.sum()}, maybe {(v == 'maybe').sum()}, not {(v == 'not').sum()}")

    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)

    def fit(Xt, yt):
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)              # a feature constant in the labels (e.g. no saturated flake) gets no weight
        return LogisticRegression(C=a.C, max_iter=5000).fit((Xt - mu) / sd, yt), mu, sd

    # repeated 5-fold CV: out-of-fold probabilities, averaged over repeats
    P = np.zeros((10, len(y)))
    for r in range(10):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m, mu, sd = fit(X[tr], y[tr])
            P[r, te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    auc = [roc_auc_score(y, p) for p in P]
    aucu = [roc_auc_score(yu, p) for p in P]
    p = P.mean(0)
    ev = dict(n=len(y), auc_cv=float(np.mean(auc)), auc_cv_sd=float(np.std(auc)), auc_useful_cv=float(np.mean(aucu)),
              auc_old_scorer=float(roc_auc_score(y, old)), auc_useful_old_scorer=float(roc_auc_score(yu, old)))
    print(f"useful+maybe vs not: AUC old scorer {ev['auc_old_scorer']:.3f} -> refit (cross-validated) "
          f"{ev['auc_cv']:.3f} ± {ev['auc_cv_sd']:.3f};  useful vs rest {ev['auc_useful_old_scorer']:.3f} -> {ev['auc_useful_cv']:.3f}")
    for k in (25, 50, 100):
        o, n = np.argsort(-old)[:k], np.argsort(-p)[:k]
        print(f"  top {k:3d} of the labelled flakes: useful+maybe {y[o].mean():.2f} -> {y[n].mean():.2f}, "
              f"useful {yu[o].mean():.2f} -> {yu[n].mean():.2f}")
        ev[f"top{k}"] = dict(old=[float(y[o].mean()), float(yu[o].mean())], new=[float(y[n].mean()), float(yu[n].mean())])

    model, mu, sd = fit(X, y)
    coef = model.coef_[0]
    print("coefficients (standardised), largest first:")
    for j in np.argsort(-np.abs(coef)):
        print(f"  {FEATURES[j]:20s} {coef[j]:+.3f}")
    out = dict(features=FEATURES, mean=mu.round(6).tolist(), std=sd.round(6).tolist(), fill_value=med.round(6).tolist(),
               coef=coef.round(6).tolist(), intercept=float(model.intercept_[0]), material=mat, profiles=a.profiles,
               native_contrast=True, target=f"P(useful or maybe) from {len(y)} labels on {Path(a.flakes).stem}",
               evaluation=ev)

    # rescore the whole scan: a flake scores as its best terrace (as in the pipeline)
    best = {}
    for t in F["terraces"]:
        x = np.array(features(terrace_ns(t), mat, tmin, tmax), float)
        x = np.where(np.isfinite(x), x, med)
        s = float(1 / (1 + np.exp(-(((x - mu) / sd) @ coef + model.intercept_[0]))))
        best[t["flake_id"]] = max(best.get(t["flake_id"], 0.0), s)
    news = np.array(sorted(best.values(), reverse=True))
    n_old = sum(1 for f in F["flakes"] if f["score"] == f["score"] and f["score"] >= a.old_threshold)
    thr = float(news[n_old - 1]) if 0 < n_old <= len(news) else float("nan")
    print(f"whole scan: {n_old} flakes had score >= {a.old_threshold} (old scorer); the same number has new score >= {thr:.3f}")
    for t_ in (0.5, 0.6, 0.7, 0.8, 0.9):
        sel = p >= t_
        print(f"  new score >= {t_:.1f}: {int((news >= t_).sum()):4d} flakes on the scan; labelled flakes above it: "
              f"{int(sel.sum())}, useful+maybe {y[sel].mean() if sel.any() else float('nan'):.2f}, "
              f"recall of useful {yu[sel].sum() / max(yu.sum(), 1):.2f}")
    out["equivalent_threshold"] = dict(old=a.old_threshold, new=round(thr, 3), n_targets=n_old)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
