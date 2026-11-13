"""10x: does a network on crops around a target predict a kept flake (vs discarded or dropped) better than the 10x
numbers?  Leave one run out, nested C (as evaluate.py).  python evaluate10.py hbn|graphite NET ...
Writes z10_<material>.npz (held-out scores) for boot.py-style checks and the summary page."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
import evaluate as E      # (loro, report: imported with argv patched below)
from paths import SETS, OUT, dpath

HERE = Path(__file__).parent
mat, nets = sys.argv[1], sys.argv[2:]
S = [s for s in json.load(open(SETS / "set10.json")) if s["material"] == mat]
keys = [s["key"] for s in S]
y = np.array([s["y"] for s in S]); runs = np.array([s["run"] for s in S])
SIZE = ["score", "log_best_d"]
HAND = sorted({k for s in S for k in s["hand"]})
H = lambda cols: np.array([[s["hand"].get(c, np.nan) for c in cols] for s in S])
print(f"{mat} 10x: {len(S)} targets, {y.sum()} kept, {len(set(runs))} runs (keep vs discard + dropped)")
out = dict(y=y, runs=runs, keys=np.array(keys))
out["size"] = E.report("10x score + best terrace size", E.loro(H(SIZE), y, runs, None, [0.01, 0.1, 1, 10]), y, runs)
out["hand"] = E.report(f"10x numbers, all {len(HAND)}", E.loro(H(HAND), y, runs, None, [0.003, 0.01, 0.1, 1]), y, runs)
def sp(z):
    o = z.copy()
    for r in np.unique(runs):
        o[runs == r] = (z[runs == r] - z[runs == r].mean()) / (z[runs == r].std() + 1e-9)
    return o
for net in nets:
    Z = np.load(OUT / f"emb10_{net}.npz")
    idx = {k: i for i, k in enumerate(Z["keys"])}
    for crop in ("X150", "X400", "both"):
        X = np.hstack([Z["X150"], Z["X400"]]) if crop == "both" else Z[crop]
        X = X[[idx[k] for k in keys]]
        nz = E.report(f"{net} {crop} (PCA 64)", E.loro(X, y, runs, 64), y, runs)
        out[f"{net}_{crop}"] = nz
        out[f"{net}_{crop}_plus_hand"] = E.report("  + 10x numbers (mean of z)", sp(nz) + sp(out["hand"]), y, runs)
out["hand_s"] = sp(out["hand"])
np.savez(OUT / f"z10_{mat}.npz", **out)
