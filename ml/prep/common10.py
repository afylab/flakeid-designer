"""Shared loading / feature / stats helpers for the 10x-predictor analysis."""
import csv, glob, json, os, re, sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

SCR = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(SCR, "live")
# the repository (two folders up) and the local data copy (FLAKEID_FLAKEDATA, else flakedata next to the repository)
REPO = os.path.dirname(os.path.dirname(SCR))
FLAKEDATA = os.environ.get("FLAKEID_FLAKEDATA", os.path.join(os.path.dirname(REPO), "flakedata"))
SF = os.path.join(FLAKEDATA, "screen_features.csv")
EXCL = ("saturated", "chip_edge", "thick_branch", "possibly_thick", "bulk", "no_signal", "thick_context")


def rcsv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def thick_branch_ids(run, piece):
    """terrace ids decided 'thick' by branch.resolve, as the live target list did."""
    sys.path.insert(0, REPO)
    from flakeid import branch as B
    d = os.path.join(LIVE, run, piece)
    log = open(glob.glob(os.path.join(d, "*_live_log.txt"))[0], encoding="utf-8", errors="replace").read()
    m = re.findall(r"oxide fixed at ([\d.]+) nm", log)
    if m:
        ox = float(m[-1])
    else:
        st = json.load(open(glob.glob(os.path.join(d, "*_live_status.json"))[0]))
        ox = float(st["oxide_nm"])
    bm = B.load("bx51_xc10_mm_raw16", "hbn", ox)
    if bm is None:
        return set(), ox
    te = rcsv(glob.glob(os.path.join(d, "*_tiles_live_terraces.csv"))[0])
    dec = B.resolve(te, bm.cal, log=lambda m: None)
    return {k for k, v in dec.items() if v == "thick"}, ox


def derive(r):
    """numeric feature dict from a joined/pool row (strings)."""
    g = {k: fl(v) for k, v in r.items()}
    o = {}
    o["score"] = g["score"]
    o["log_best_d"] = np.log(max(g["best_d"], 1))
    o["best_t"] = g["best_t"]
    o["log_flake_area"] = np.log(max(g["flake_area"], 1))
    o["log_flake_d"] = np.log(max(g["flake_d"], 1))
    o["aspect"] = g["flake_L"] / max(g["flake_W"], 1e-6)
    o["n_terraces"] = g["n_terraces"]
    o["n_ter_big"] = g.get("n_ter_big", np.nan)
    o["best_area_frac"] = g.get("best_area_frac", np.nan)
    o["ter_t_spread"] = g.get("ter_t_spread", np.nan)
    o["focus_um"] = min(g["focus_um"], 15) if g["focus_um"] == g["focus_um"] else np.nan
    o["crowding"] = g["crowding"]
    o["frac_window"] = g["frac_window"]
    o["frac_thick"] = g["frac_thick"]
    for c in ["solidity", "fill", "fullres_solidity", "ring_similar", "fit_residual", "uniformity", "edge_width_um",
              "saturated_frac"]:
        o["bt_" + c] = g.get("bt_" + c, np.nan)
    o["bt_contrast_mag"] = float(np.sqrt(g.get("bt_contrast_r", np.nan) ** 2 + g.get("bt_contrast_g", np.nan) ** 2 +
                                         g.get("bt_contrast_b", np.nan) ** 2))
    o["bt_log_edge_thick"] = np.log1p(g.get("bt_edge_thick_um2", np.nan))
    o["bt_log_thick_nb"] = np.log1p(g.get("bt_thick_nb_um2", np.nan))
    o["bt_t_range"] = g.get("bt_thickness_max_nm", np.nan) - g.get("bt_thickness_min_nm", np.nan)
    o["bt_ext_vs_inscr"] = g.get("bt_length_um", np.nan) / max(g.get("bt_inscribed_diameter_um", np.nan), 1)
    o["min_uniformity"] = g.get("min_uniformity", np.nan)
    o["max_ring"] = g.get("max_ring", np.nan)
    for q in ["thick_context", "two_branch", "low_confidence", "ambiguous_thickness", "possibly_thick",
              "irregular_shape", "poor_colour_match"]:
        o["q_" + q] = g.get("q_" + q, np.nan)
    o["bt_alt"] = g.get("bt_alt", np.nan)
    return o


def auc(x, y):
    """P(x_pos > x_neg) with ties 1/2; nan-safe."""
    x, y = np.asarray(x, float), np.asarray(y, int)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    p, n = x[y == 1], x[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan, 0
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([p, n]))
    a = (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
    return a, len(p) * len(n)


def strat_auc(x, y, g):
    """concordance over pairs within the same group (run): sum of per-group AUC x pairs / pairs."""
    x, y, g = np.asarray(x, float), np.asarray(y, int), np.asarray(g)
    num = den = 0
    for k in np.unique(g):
        m = g == k
        a, w = auc(x[m], y[m])
        if w:
            num += a * w
            den += w
    return num / den if den else np.nan


def boot_strat_auc(x, y, g, n=500, seed=0):
    rng = np.random.default_rng(seed)
    x, y, g = np.asarray(x, float), np.asarray(y, int), np.asarray(g)
    idx = {k: np.where(g == k)[0] for k in np.unique(g)}
    out = []
    for _ in range(n):
        s = np.concatenate([rng.choice(v, len(v)) for v in idx.values()])
        out.append(strat_auc(x[s], y[s], g[s]))
    return np.nanpercentile(out, [2.5, 97.5])


class Logit:
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y, w=None):
        X = np.asarray(X, float)
        self.med = np.nanmedian(X, 0)
        X = np.where(np.isfinite(X), X, self.med)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        Z = (X - self.mu) / self.sd
        w = np.ones(len(y)) if w is None else w
        y = np.asarray(y, float)

        def nll(b):
            z = Z @ b[1:] + b[0]
            l = np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * self.lam * np.sum(b[1:] ** 2)
            p = 1 / (1 + np.exp(-z))
            gr = np.concatenate([[np.sum(w * (p - y))], Z.T @ (w * (p - y)) + self.lam * b[1:]])
            return l, gr
        b0 = np.zeros(Z.shape[1] + 1)
        b0[0] = np.log(max(y.mean(), 1e-3) / max(1 - y.mean(), 1e-3))
        self.b = minimize(nll, b0, jac=True, method="L-BFGS-B").x
        return self

    def decision(self, X):
        X = np.asarray(X, float)
        X = np.where(np.isfinite(X), X, self.med)
        return ((X - self.mu) / self.sd) @ self.b[1:] + self.b[0]

    def proba(self, X):
        return 1 / (1 + np.exp(-self.decision(X)))
