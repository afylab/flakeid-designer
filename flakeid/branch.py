"""Thin / thick decision for hBN whose 10x colour fits two interference branches (e.g. ~20 nm and ~150 nm on
~300 nm SiO2), learned from the user's labels (profiles/hbn_branch_*.json) instead of taken from the thin-film model.

A calibration applies to the camera profile and oxide (+-oxide_tol_nm) it was labelled on.  It is used only where
both readings fit (best thin- and thick-range curve distances within `gap` of each other, the better <= max_fit):
the pipeline flags such terraces 'two_branch' (reported thin, thick alternative), and resolve() decides each one over
the whole chip: the labelled logistic score gives the direction (lower blue contrast = thick), the boundary is the gap
between the chip's two colour groups (it shifts by a few hundredths in contrast between chips), else score 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROFILE_DIR = Path(__file__).parent / "profiles"


@dataclass
class BranchModel:
    cal: dict

    @property
    def p_thick(self):
        return float(self.cal.get("p_thick", 0.5))

    @property
    def p_uncertain(self):
        return float(self.cal.get("p_uncertain", 0.25))

    def decide(self, est, c, censored=None):
        """(p_thick, thin_nm, thin_dist, thick_nm, thick_dist) if both branches fit, else None."""
        lo, hi = self.cal["thin_range_nm"]
        tn, dn = est.best_in_range(c, lo, hi, censored=censored)
        lo, hi = self.cal["thick_range_nm"]
        tk, dk = est.best_in_range(c, lo, hi, censored=censored)
        if min(dn, dk) > self.cal["max_fit"] or abs(dn - dk) > self.cal["gap"]:
            return None
        z = float(np.dot(self.cal["coef"], np.asarray(c, float)[:3]) + self.cal["intercept"])
        return 1.0 / (1.0 + np.exp(-z)), tn, dn, tk, dk


def load(profile_stem, material, oxide_nm, log=None):
    """The branch calibration for this profile, material and oxide, or None."""
    for p in sorted(PROFILE_DIR.glob(f"{material}_branch_*.json")):
        try:
            cal = json.loads(p.read_text())
        except Exception:
            continue
        if cal.get("profile") == profile_stem and abs(float(oxide_nm) - cal["oxide_nm"]) <= cal.get("oxide_tol_nm", 10.0):
            if log:
                log(f"thin/thick branch decision from labels: {p.name}")
            return BranchModel(cal)
    return None


def score(cal, c):
    """Linear score (log-odds of thick under the labelled calibration) of a contrast vector."""
    return float(np.dot(cal["coef"], np.asarray(c, float)[:3]) + cal["intercept"])


def chip_threshold(z, min_n=30, min_sep=2.5):
    """Boundary between the thin and thick colour groups of one chip: the equal-posterior point of a 2-component
    Gaussian mixture of the scores, if the chip has enough two-branch terraces and two well-separated groups;
    else None (use the labelled boundary, score 0).  On chip_1 (labels) the mixture boundary and the labelled one
    agree (0.93 accuracy each); on bigbntest_2 the groups sit ~3.5 higher and only the chip boundary gets both
    AFM sites right (146.5 nm thick, 17.3 nm thin)."""
    z = np.asarray(z, float)
    if len(z) < min_n:
        return None
    m, v, w = gmm2(z)
    s = np.sqrt(v)
    o = np.argsort(m)
    if (m[o[1]] - m[o[0]]) / s.mean() < min_sep or w.min() < 0.1:
        return None
    xs = np.linspace(m[o[0]], m[o[1]], 1000)
    post = _post(xs, m, v, w)[:, o[1]]
    return float(xs[np.argmin(np.abs(post - 0.5))])


def _post(x, m, v, w):
    """Posterior of each component at x (n,) -> (n, 2)."""
    x = np.asarray(x, float)[:, None]
    lp = np.log(w) - 0.5 * np.log(2 * np.pi * v) - 0.5 * (x - m) ** 2 / v
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)


def gmm2(z, max_iter=100, tol=1e-3, reg=1e-6):
    """A 2-component 1-D Gaussian mixture by EM: (means, variances, weights).  As scikit-learn's GaussianMixture(2)
    (k-means start, reg_covar, tol on the mean log-likelihood), in numpy: the designer apps no longer need
    scikit-learn (whose compiled modules a newer version broke in the Windows app, 2026-10-20)."""
    z = np.asarray(z, float)
    c = np.percentile(z, [25, 75]).astype(float)                       # k-means start
    for _ in range(100):
        lab = np.abs(z[:, None] - c[None, :]).argmin(axis=1)
        new = np.array([z[lab == k].mean() if np.any(lab == k) else c[k] for k in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    r = np.eye(2)[lab]
    ll_old = -np.inf
    for _ in range(max_iter):
        nk = r.sum(axis=0) + 10 * np.finfo(float).eps
        m = (r * z[:, None]).sum(axis=0) / nk
        v = (r * (z[:, None] - m) ** 2).sum(axis=0) / nk + reg
        w = nk / len(z)
        lp = np.log(w) - 0.5 * np.log(2 * np.pi * v) - 0.5 * (z[:, None] - m) ** 2 / v
        mx = lp.max(axis=1, keepdims=True)
        ll = float(np.mean(mx.ravel() + np.log(np.exp(lp - mx).sum(axis=1))))
        r = np.exp(lp - mx)
        r /= r.sum(axis=1, keepdims=True)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return m, v, w


def resolve(terraces, cal, min_d=15.0, log=print):
    """terrace_id -> 'thin' | 'thick' for the 'two_branch' terraces of one chip (terrace-table rows)."""
    tb = [t for t in terraces if "two_branch" in (t.get("quality") or "")]
    if not tb:
        return {}
    zf = lambda t: score(cal, [float(t["contrast_r"]), float(t["contrast_g"]), float(t["contrast_b"])])  # noqa: E731
    thr = chip_threshold([zf(t) for t in tb if float(t.get("inscribed_diameter_um") or 0) >= min_d])
    use = thr if thr is not None else 0.0
    # 'thick_context' (thick-only colours at the edge or around it, pipeline.thick_context) overrides the colour score
    out = {t["terrace_id"]: ("thick" if zf(t) > use or "thick_context" in (t.get("quality") or "") else "thin") for t in tb}
    nk = sum(v == "thick" for v in out.values())
    log(f"thin/thick (colour fits ~20 and ~150 nm): {nk} thick, {len(out) - nk} thin of {len(out)} terraces; boundary "
        + (f"from this chip's two colour groups (score {thr:.2f})" if thr is not None else
           "from the labels (too few two-branch terraces or no clear groups on this chip)"))
    return out
