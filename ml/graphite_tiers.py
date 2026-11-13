"""Graphite 100x screening with the setup C network: the tiers it would give against the current pick model (v3).
python graphite_tiers.py -> results/graphite_combo.json (the combination to deploy) + the comparison

Scores, all held out by run: the pick model's (graphite100.json hand_z, its logit) and the network's
(z_graphite_bundle.npz ft_bundle, the mean logit of 3 seeds).  New P(keep) = logistic on the two logits (plain, so that
0.5 means even odds as in the pick model), fitted leave one run out.  Tiers as now: screened out below reject_below
(chosen so that keep_share of the keeps on the training runs score above it), keep from keep_above 0.5, else review.
Every cutoff is chosen without the run it is applied to."""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from paths import SETS, OUT

KEEP_SHARE, KEEP_ABOVE = 0.95, 0.5
G = sorted((s for s in json.load(open(SETS / "graphite100.json")) if s["ok"]), key=lambda s: s["key"])
Z = np.load(OUT / "z_graphite_bundle.npz")
y, runs = Z["y"], Z["runs"]
assert (np.array([s["y"] for s in G]) == y).all() and (np.array([s["run"] for s in G]) == runs).all()
hand = np.array([s["hand_z"] for s in G])
net = Z["ft_bundle"]
X = {"pick v3 (now)": hand[:, None], "network": net[:, None], "pick + network": np.c_[hand, net]}


def probs(Xm):
    """held-out P(keep): the pick model's own logit as it is; the others refit on the other runs."""
    if Xm.shape[1] == 1 and Xm is X["pick v3 (now)"]:
        return 1 / (1 + np.exp(-Xm[:, 0]))
    p = np.zeros(len(y))
    for r in np.unique(runs):
        tr = runs != r
        p[~tr] = LogisticRegression(C=1.0).fit(Xm[tr], y[tr]).predict_proba(Xm[~tr])[:, 1]
    return p


print(f"{len(y)} decided graphite flakes, {y.sum()} kept, {len(np.unique(runs))} runs (all scores and cutoffs held out by run)\n")
print(f"  {'':16s} {'AUC':>6s} {'cutoff':>7s}  {'screened out':>23s}  {'review':>16s}  {'keep tier':>18s}")
res = {}
for name, Xm in X.items():
    p = probs(Xm)
    tier = np.empty(len(y), object)
    cuts = []
    for r in np.unique(runs):                          # the screened-out cutoff from the other runs' keeps
        s, tr = runs == r, runs != r
        cut = float(np.quantile(p[tr & (y == 1)], 1 - KEEP_SHARE))
        cuts.append(cut)
        tier[s] = np.where(p[s] < cut, "out", np.where(p[s] >= KEEP_ABOVE, "keep", "review"))
    t = {k: dict(keeps=int(((tier == k) & (y == 1)).sum()), discards=int(((tier == k) & (y == 0)).sum())) for k in ("out", "review", "keep")}
    res[name] = dict(auc=round(roc_auc_score(y, p), 3), tiers=t)
    print(f"  {name:16s} {roc_auc_score(y, p):6.3f} {np.median(cuts):7.3f}  "
          f"{t['out']['discards']:4d} discards, {t['out']['keeps']:2d} keeps  "
          f"{t['review']['discards'] + t['review']['keeps']:4d} ({t['review']['keeps']:3d} keeps)  "
          f"{t['keep']['keeps']:3d} keeps, {t['keep']['discards']:3d} disc.")
# the combination to deploy: fitted on all runs; its screened-out cutoff from the held-out probabilities of all runs
m = LogisticRegression(C=1.0).fit(X["pick + network"], y)
p_all = probs(X["pick + network"])
combo = dict(inputs=["pick_logit", "net_logit"], coef=m.coef_[0].round(4).tolist(), intercept=round(float(m.intercept_[0]), 4),
             reject_below=round(float(np.quantile(p_all[y == 1], 1 - KEEP_SHARE)), 4), keep_above=KEEP_ABOVE, keep_share=KEEP_SHARE,
             evaluation=res, n=int(len(y)), n_keep=int(y.sum()))
json.dump(combo, open(OUT / "graphite_combo.json", "w"), indent=1)
print(f"\ndeploy: P(keep) = sigmoid({combo['coef'][0]:+.3f} pick_logit {combo['coef'][1]:+.3f} net_logit {combo['intercept']:+.3f}), "
      f"screened out below {combo['reject_below']}, keep from {KEEP_ABOVE}")
