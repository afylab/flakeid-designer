"""Leave-one-run-out AUC of keep vs discard: hand-feature model, network features, and both.
python evaluate.py hbn|graphite NET [NET ...]

Network: standardise -> PCA (fit on the training runs only) -> L2 logistic; C chosen per held-out run on the other
runs (nested), so the held-out run never influences anything.  Hand model: hBN refit here on the defects v2 columns
(nested the same way); graphite: the deployed v3 fit's held-out scores (focus_refit/scores_omit5b.csv).
Both: mean of the two held-out z-scores (rank-free, nothing refit on the held-out run)."""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from paths import SETS, OUT, dpath

HERE = Path(__file__).parent
which, nets = sys.argv[1], sys.argv[2:]
CS = [0.001, 0.003, 0.01, 0.03, 0.1]
HAND_HBN = ["p_quality", "clean_d", "edge_um_per_1000", "off_frac", "tex_frac", "defect_frac", "tex_p90", "a_d"]

if which == "hbn":
    R = list(csv.DictReader(open(SETS / "hbn100.csv", encoding="utf-8")))
    meta = {r["key"]: dict(run=r["run"], y=int(r["decision"] != "discard"),
                           hand=[float(r[c]) if r[c] not in ("", "nan") else np.nan for c in HAND_HBN]) for r in R}
else:
    meta = {s["key"]: dict(run=s["run"], y=s["y"], hand_z=s["hand_z"]) for s in json.load(open(SETS / "graphite100.json")) if s["ok"]}


def loro(X, y, runs, pca=None, Cs=CS):
    """held-out decision values (nested choice of C)."""
    out = np.zeros(len(y))
    for r in np.unique(runs):
        te, tr = runs == r, runs != r
        best = None
        for C in Cs:                                    # inner: leave one of the training runs out
            z = np.zeros(tr.sum()); yt, rt, Xt = y[tr], runs[tr], X[tr]
            for r2 in np.unique(rt):
                a, b = rt != r2, rt == r2
                if yt[a].min() == yt[a].max():
                    continue
                z[b] = fit(Xt[a], yt[a], C, pca).decision_function(prep_(Xt[b]))
            auc = roc_auc_score(yt, z)
            if best is None or auc > best[0]:
                best = (auc, C)
        out[te] = fit(X[tr], y[tr], best[1], pca).decision_function(prep_(X[te]))
    return out


_state = {}
def prep_(X):
    sc, p = _state["sc"], _state["pca"]
    Z = sc.transform(np.where(np.isfinite(X), X, _state["fill"]))
    return p.transform(Z) if p is not None else Z


def fit(X, y, C, pca):
    fill = np.nanmedian(X, 0); fill = np.where(np.isfinite(fill), fill, 0)
    Xf = np.where(np.isfinite(X), X, fill)
    sc = StandardScaler().fit(Xf)
    Z = sc.transform(Xf)
    p = PCA(n_components=min(pca, len(Z) - 1), random_state=0).fit(Z) if pca else None
    _state.update(sc=sc, pca=p, fill=fill)
    m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(p.transform(Z) if p is not None else Z, y)
    sc_, p_, f_ = sc, p, fill
    class M:
        def decision_function(self, Zt):
            return m.decision_function(Zt)
    return M()


def report(name, z, y, runs):
    per = [roc_auc_score(y[runs == r], z[runs == r]) for r in np.unique(runs) if 0 < y[runs == r].sum() < (runs == r).sum()]
    # flakes removed when 95 % of the keeps are kept (pooled, on held-out z standardised per run)
    zs = np.concatenate([(z[runs == r] - z[runs == r].mean()) / (z[runs == r].std() + 1e-9) for r in np.unique(runs)])
    ys = np.concatenate([y[runs == r] for r in np.unique(runs)])
    cut = np.quantile(zs[ys == 1], 0.05)
    print(f"  {name:34s} AUC pooled {roc_auc_score(y, z):.3f}  mean per run {np.mean(per):.3f}   "
          f"discards removed at 95 % keeps: {int(((zs < cut) & (ys == 0)).sum())} of {int((ys == 0).sum())}")
    return z


def zs(z, runs):
    return np.concatenate([[0]])  # (unused)


keys = sorted(meta)
y = np.array([meta[k]["y"] for k in keys]); runs = np.array([meta[k]["run"] for k in keys])
print(f"{which}: {len(keys)} flakes, {y.sum()} kept, {len(set(runs))} runs; leave one run out, C chosen on the other runs")
if which == "hbn":
    H = np.array([meta[k]["hand"] for k in keys])
    hz = report("hand features (refit here)", loro(H, y, runs, None, [0.01, 0.1, 1, 10]), y, runs)
else:
    hz = report("hand features (deployed v3, held out)", np.array([meta[k]["hand_z"] for k in keys]), y, runs)


def std_per_run(z):
    o = z.copy()
    for r in np.unique(runs):
        o[runs == r] = (z[runs == r] - z[runs == r].mean()) / (z[runs == r].std() + 1e-9)
    return o


SAVE = {"y": y, "runs": runs, "hand": hz}
PCAS = [int(v) for v in __import__("os").environ.get("PCAS", "32,128").split(",")]
for net in nets:
    E = np.load(OUT / f"emb_{which}_{net}.npz")
    idx = {k: i for i, k in enumerate(E["keys"])}
    X = E["X"][[idx[k] for k in keys]]
    for pca in PCAS:
        nz = report(f"{net} features, PCA {pca}", loro(X, y, runs, pca), y, runs)
        report(f"  + hand (mean of z)", std_per_run(nz) + std_per_run(hz), y, runs)
        SAVE[f"{net}_{pca}"] = nz
        SAVE[f"{net}_{pca}_both"] = std_per_run(nz) + std_per_run(hz)
SAVE["hand_s"] = std_per_run(hz)
np.savez(OUT / f"z_{which}.npz", **SAVE)
