"""Paired bootstrap of the AUC gain (combined - hand), resampling runs and flakes within runs."""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
from paths import SETS, OUT, dpath
Z = np.load(OUT / f"z10_{sys.argv[1]}.npz"); y, runs = Z["y"], Z["runs"]
rng = np.random.default_rng(0); R = np.unique(runs)
for k in sys.argv[2:]:
    g = []
    for _ in range(2000):
        idx = np.concatenate([rng.choice(np.where(runs == r)[0], (runs == r).sum()) for r in R])
        if y[idx].min() == y[idx].max():
            continue
        g.append(roc_auc_score(y[idx], Z[k][idx]) - roc_auc_score(y[idx], Z["hand_s"][idx]))
    g = np.array(g)
    print(f"{sys.argv[1]} {k}: gain over hand {roc_auc_score(y, Z[k]) - roc_auc_score(y, Z['hand_s']):+.3f}, 95 % CI {np.percentile(g, 2.5):+.3f} .. {np.percentile(g, 97.5):+.3f}, P(gain <= 0) {np.mean(g <= 0):.3f}")
