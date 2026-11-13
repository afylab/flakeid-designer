#!/usr/bin/env python3
"""What separates the 100x images scored useful from the rest, and what the 10x analysis could have known.

    python tools/analyze_100x_scores.py --images "testing images/bigbntest_2_100x" \
        --scores label_set/bigbntest_2_100x_scores --flakes "testing images/bigbntest_2_results/bigbntest_2_tiles_flakes.json" \
        --flat "testing images/bigbntest_100x_score/flat100x.npy" --substrate 946 1221.5 815 -o testing_afm/../results/bigbntest_2_100x_analysis.json

For every scored 100x image: the 100x terrace features of tools/rank100x.py, the 10x features of its flake's
best terrace (flakeid/scorer.py) and both 10x scores (the Stream scorer stored in the flake table and the
RAW16 scorer refitted on the 10x labels).  Prints, for the verdict and for each tag, how well every feature
separates (AUC) and the medians of the two groups, and writes everything to a JSON file.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from flakeid.scorer import FEATURES, Scorer, features, scorer_path  # noqa: E402
from rank100x import load_contrast, terrace_features  # noqa: E402

F100 = ["area_um2", "d_um", "hole_frac", "n_holes", "texture", "texture_rel", "step_frac", "flake_ring", "edge_width_um", "sharp", "sat_frac"]


def auc(y, x):
    m = np.isfinite(x)
    if m.sum() < 5 or len(set(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], x[m]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--scores", required=True, help="folder of the scoring page's documents")
    ap.add_argument("--flakes", required=True, help="<scan>_flakes.json the 100x targets came from")
    ap.add_argument("--flat", required=True, help="100x flat field (.npz) or optics pattern (.npy)")
    ap.add_argument("--substrate", nargs=3, type=float, required=True, help="100x substrate counts R G B (dark-subtracted)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)

    flat = np.load(a.flat) if a.flat.endswith(".npy") else __import__("flakeid.mm", fromlist=["FlatField"]).FlatField.load(a.flat)
    F = json.loads(Path(a.flakes).read_text())
    T = {t["terrace_id"]: t for t in F["terraces"]}
    fl = {f["flake_id"]: f for f in F["flakes"]}
    mat, (tmin, tmax) = F["material"], F["params"]["thickness_range_nm"]
    raw16 = Scorer(scorer_path(Path(F["params"]["profile"]).stem, mat))
    rows = []
    for f in sorted(glob.glob(str(Path(a.scores) / "*.json"))):
        d = json.loads(Path(f).read_text())
        if d.get("verdict") not in ("useful", "maybe", "not"):
            continue
        sid = Path(f).stem
        fid = int(sid.split("_F")[1].split("_")[0])
        C, sub, subfrac, sat = load_contrast(Path(a.images) / f"{sid}.tif", flat, a.substrate)
        out = terrace_features(C, sat)
        r = dict(id=sid, flake=fid, verdict=d["verdict"], tags=d.get("tags") or [], sub_frac=subfrac)
        if out is not None:
            r.update({k: float(v) for k, v in out[0].items() if k in F100})
        fk = fl.get(fid)
        if fk:
            t = T[fk["best_terrace_id"]]
            ns = SimpleNamespace(**{k: (float("nan") if v is None else v) for k, v in t.items()})
            ns.quality, ns.alt_thickness_nm = t.get("quality") or "good", t.get("alt_thickness_nm") or ""
            x = features(ns, mat, tmin, tmax)
            r.update({f"10x_{k}": float(v) for k, v in zip(FEATURES, x)})
            r.update({"10x_thickness_nm": t["thickness_nm"], "10x_n_terraces": fk["n_terraces"], "10x_score_stream": fk["score"],
                      "10x_score_raw16": raw16(ns, mat, tmin, tmax) if raw16.ok else float("nan"),
                      "10x_quality": fk["quality"]})
        rows.append(r)
        print(f"  {sid}: {d['verdict']:6s} {','.join(r['tags'])}")

    y = np.array([r["verdict"] != "not" for r in rows], int)
    keys = [k for k in F100] + [k for k in rows[0] if k.startswith("10x_") and k != "10x_quality"]
    X = {k: np.array([r.get(k, np.nan) for r in rows], float) for k in keys}
    res = dict(n=len(rows), n_useful=int(sum(r["verdict"] == "useful" for r in rows)), n_maybe=int(sum(r["verdict"] == "maybe" for r in rows)),
               n_not=int((y == 0).sum()), rows=rows, separation={})

    def table(name, yy):
        out = []
        for k in keys:
            x = X[k]
            out.append(dict(feature=k, auc=auc(yy, x), median_pos=float(np.nanmedian(x[yy == 1])) if (yy == 1).any() else None,
                            median_neg=float(np.nanmedian(x[yy == 0])) if (yy == 0).any() else None))
        out.sort(key=lambda e: -abs((e["auc"] if e["auc"] == e["auc"] else 0.5) - 0.5))
        res["separation"][name] = out
        print(f"\n{name}  ({int(yy.sum())} vs {int((yy == 0).sum())}):  feature, AUC, median in group / rest")
        for e in out[:8]:
            print(f"  {e['feature']:24s} {e['auc']:.2f}   {e['median_pos']:.3g} / {e['median_neg']:.3g}")

    print(f"\n{len(rows)} scored: useful {res['n_useful']}, maybe {res['n_maybe']}, not {res['n_not']}")
    table("useful or maybe vs not", y)
    tags = sorted({t for r in rows for t in r["tags"]})
    for tg in tags:
        yy = np.array([tg in r["tags"] for r in rows], int)
        if yy.sum() >= 5:
            table(f"tag {tg}", yy)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
