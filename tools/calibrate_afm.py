#!/usr/bin/env python3
"""Refine the camera / optical model with AFM-measured flakes (supervised calibration).

    python tools/calibrate_afm.py \
        --flakes "testing images/bigbntest_2_results/bigbntest_2_tiles_flakes.json" \
        --sites "testing images/bigbntest_2_afm_set/data.json" \
        --steps testing_afm/bigbn2_all_results/steps.csv \
        --exclude 14 -o flakeid/profiles/bx51_xc10_mm_raw16_afm.json --report testing_afm/calibration

Loss = (trimmed curve distance of every flat terrace on the chip, as in flakeid.calibrate, normalised to the
starting profile) + (AFM term: mean squared thickness residual / sigma, capped).  The thickness of each AFM
site is estimated exactly as the pipeline does it (ThicknessEstimator.fit at the oxide that refine_oxide finds
on the same terraces), so the AFM term is what the pipeline would report.  Free parameters: channel centres,
widths, IR cut, flare, oxide, and an offset to the hBN index (the unsupervised term fixes the curve's shape but
hardly its thickness scale; the AFM sets the scale).

AFM sites: 'substrate' sites (and 'step' sites whose scan contains the substrate, i.e. the measured height
matches the target's full thickness) give absolute thicknesses.  Other 'step' sites measure the step to a
neighbouring terrace of the same flake; they are used as difference constraints, with the neighbour chosen
from the starting model (the one whose predicted difference is closest), and flagged ambiguous when two
neighbours fit within 3 nm (validation only).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.calibrate import camera_from_params  # noqa: E402
from flakeid.optics import ContrastModel, n_hbn  # noqa: E402
from flakeid.thickness import ThicknessEstimator, load_profile, save_profile  # noqa: E402

WL_TABLE = np.arange(380.0, 781.0, 20.0)
FREE = [("c_r", 595, 650), ("c_g", 520, 570), ("c_b", 455, 500), ("w_r", 60, 120), ("w_g", 60, 120),
        ("w_b", 60, 120), ("ir_cut", 650, 720), ("f_r", 0.0, 0.08), ("f_g", 0.0, 0.08), ("f_b", 0.0, 0.08),
        ("dn", -0.15, 0.15)]


def hbn_table(dn):
    return np.stack([WL_TABLE, n_hbn(WL_TABLE).real + dn, np.zeros_like(WL_TABLE)], axis=1)


def params_from_profile(prof):
    c = prof["camera"]
    p = dict(zip(["c_r", "c_g", "c_b"], c["centers_nm"])) | dict(zip(["w_r", "w_g", "w_b"], c["fwhm_nm"]))
    p |= dict(zip(["f_r", "f_g", "f_b"], c.get("flare", (0, 0, 0))), ir_cut=c["ir_cut_nm"], dn=0.0)
    t = prof.get("materials", {}).get("hbn", {}).get("index_table")
    if t is not None:   # an earlier AFM profile: recover its offset
        t = np.asarray(t)
        p["dn"] = float(np.mean(t[:, 1] - n_hbn(t[:, 0]).real))
    return p


def profile_from_params(p, na=0.30):
    prof = {"camera": {"lamp_temp_k": 3100.0, "centers_nm": [p["c_r"], p["c_g"], p["c_b"]],
                       "fwhm_nm": [p["w_r"], p["w_g"], p["w_b"]], "ir_cut_nm": p["ir_cut"], "gamma": 1.0,
                       "ccm": None, "flare": [p["f_r"], p["f_g"], p["f_b"]]}, "na": na}
    if abs(p.get("dn", 0.0)) > 1e-9:
        prof["materials"] = {"hbn": {"index_table": hbn_table(p["dn"]).round(5).tolist()}}
    return prof


def curve_loss(prof, oxide, samples, trim=0.7):
    est = ThicknessEstimator("hbn", oxide, prof, step_nm=0.5)
    d, _ = est._tree.query(samples)
    d = np.sort(d)[: int(trim * len(d))]
    return float(np.mean(d ** 2))


def best_oxide(prof, samples, guess, rel=0.06, n=13):
    """refine_oxide's search (grid + parabola), on the same samples."""
    grid = guess * np.linspace(1 - rel, 1 + rel, n)
    L = np.array([curve_loss(prof, o, samples) for o in grid])
    k = int(np.argmin(L))
    best = grid[k]
    if 0 < k < n - 1:
        y0, y1, y2 = L[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        if den > 0:
            best = grid[k] + 0.5 * (y0 - y2) / den * (grid[1] - grid[0])
    return float(best)


def thickness(prof, oxide, c):
    est = ThicknessEstimator("hbn", oxide, prof, step_nm=0.1)
    return [est.fit(np.asarray(x)).thickness_nm for x in c]


def load_points(sites_json, steps_csv, exclude, prof0, oxide0):
    D = json.loads(Path(sites_json).read_text())
    by_site, flakes = {}, {}
    for F in D["flakes"]:
        flakes[F["flake"]] = F
        for t in F["terraces"]:
            if t.get("site"):
                by_site[(F["flake"], t["id"])] = (F, t)
    rows = list(csv.DictReader(open(steps_csv)))
    best = {}
    for r in rows:   # one result per file: largest level in the main block
        if r["main_block"] != "True" or not r["site"]:
            continue
        if r["file"] not in best or float(r["area_frac"]) > float(best[r["file"]]["area_frac"]):
            best[r["file"]] = r
    pts = []
    for r in sorted(best.values(), key=lambda r: int(r["site"])):
        n = int(r["site"])
        F, t = by_site[(int(r["flake"]), int(r["terrace"]))]
        h, err = float(r["step_nm"]), float(r["err_nm"])
        P = dict(site=n, flake=F["flake"], terrace=t["id"], c=t["c"], h=h, err=err, pred_10x=t["t"],
                 kind=t["site"]["kind"], file=r["file"], use=n not in exclude, why="excluded by hand" if n in exclude else "")
        if P["kind"] == "step":
            t0 = thickness(prof0, oxide0, [t["c"]])[0]
            sib = [(u, thickness(prof0, oxide0, [u["c"]])[0]) for u in F["terraces"] if u["id"] != t["id"]]
            # candidates: absolute (scan reaches the substrate) or a step to any neighbour, either direction
            cands = [(abs(t0 - h), None, +1)] + [(abs(abs(t0 - ts) - h), u, 1 if t0 > ts else -1) for u, ts in sib]
            cands.sort(key=lambda x: x[0])
            dev, u, sgn = cands[0]
            if u is not None:
                P.update(kind="difference", other=u["id"], other_c=u["c"], sign=sgn)
                P["why"] = P["why"] or f"step to T{u['id']} ({'below' if sgn > 0 else 'above'})"
            else:
                P["kind"] = "absolute (step scan reached substrate)"
            # ambiguous only when another candidate fits as well and implies a different neighbour thickness
            implied = lambda c: 0.0 if c[1] is None else t0 - c[2] * h
            rivals = [c for c in cands[1:] if c[0] - dev < 3.0 and abs(implied(c) - implied(cands[0])) > 3.0]
            if rivals:
                cands = [cands[0]] + rivals
                P["use"], P["why"] = False, f"ambiguous: steps to {('substrate' if u is None else 'T%d' % u['id'])} " \
                                            f"and {('substrate' if cands[1][1] is None else 'T%d' % cands[1][1]['id'])} both fit"
        pts.append(P)
    return pts


def predict(prof, oxide, pts):
    cs = [p["c"] for p in pts] + [p["other_c"] for p in pts if p["kind"] == "difference"]
    t = thickness(prof, oxide, cs)
    out, k = [], len(pts)
    for i, p in enumerate(pts):
        if p["kind"] == "difference":
            out.append(p["sign"] * (t[i] - t[k]))
            k += 1
        else:
            out.append(t[i])
    return np.array(out)


def sigma(p):
    """AFM error plus 3 % / 0.7 nm for the 10x contrast (terrace inhomogeneity, model)."""
    return float(np.sqrt(p["err"] ** 2 + (0.03 * p["h"]) ** 2 + 0.7 ** 2))


class Loss:
    def __init__(self, names, samples, pts, u0, ox0, w_afm=1.0, cap=4.0):
        self.names, self.samples, self.pts, self.u0, self.ox0 = names, samples, pts, u0, ox0
        self.w, self.cap = w_afm, cap
        self.h = np.array([p["h"] for p in pts])
        self.s = np.array([sigma(p) for p in pts])

    def parts(self, x):
        p = dict(zip(self.names, x))
        ox = self.ox0 * p.pop("ox")
        prof = profile_from_params(p)
        u = curve_loss(prof, ox, self.samples) / self.u0
        r = (predict(prof, ox, self.pts) - self.h) / self.s if self.pts else np.zeros(0)
        return u, r

    def __call__(self, x):
        u, r = self.parts(x)
        a = float(np.mean(np.minimum(r ** 2, self.cap ** 2))) if len(r) else 0.0
        return u + self.w * a


def fit(p0, ox0, samples, pts, u0, maxiter, seed=1, w_afm=1.0):
    names = [k for k, _, _ in FREE] + ["ox"]
    bounds = [(lo, hi) for _, lo, hi in FREE] + [(0.95, 1.05)]
    x0 = np.clip([p0[k] for k, _, _ in FREE] + [1.0], [b[0] for b in bounds], [b[1] for b in bounds])
    loss = Loss(names, samples, pts, u0, ox0, w_afm)
    res = differential_evolution(loss, bounds, x0=x0, popsize=8, maxiter=maxiter, tol=1e-6, seed=seed,
                                 polish=False, workers=-1, updating="deferred")
    p = dict(zip(names, res.x))
    return p, ox0 * p.pop("ox"), res.fun


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flakes", required=True, help="the chip's <name>_flakes.json (terrace contrasts)")
    ap.add_argument("--sites", required=True, help="AFM calibration page data.json")
    ap.add_argument("--steps", required=True, help="tools/afm_steps.py steps.csv")
    ap.add_argument("--profile", default="bx51_xc10_mm_raw16")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], help="site numbers to leave out")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--report", help="folder for report.json and figures")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--kfold", type=int, default=0, help="k-fold cross-validation of the AFM term (e.g. 5)")
    ap.add_argument("--cv-cold", action="store_true", help="k-fold fits start from the starting profile (a true cross-validation; slow)")
    ap.add_argument("--cv-from", help="profile written by an earlier run: reuse its fitted parameters instead of fitting again")
    a = ap.parse_args(argv)

    prof0 = load_profile(a.profile)
    p0 = params_from_profile(prof0)
    F = json.loads(Path(a.flakes).read_text())
    T = [t for t in F["terraces"] if t["quality"] == "good" and t["saturated_frac"] == 0 and t["fit_residual"] < 0.2]
    samples = np.array([[t["contrast_r"], t["contrast_g"], t["contrast_b"]] for t in T])
    ox_nom = float(F["oxide_nm_used"])
    print(f"{len(samples)} flat terraces from {Path(a.flakes).name}; oxide used by the analysis {ox_nom:.1f} nm")

    ox0 = best_oxide(prof0, samples, ox_nom)
    u0 = curve_loss(prof0, ox0, samples)
    pts = load_points(a.sites, a.steps, set(a.exclude), prof0, ox0)
    use = [p for p in pts if p["use"]]
    before = predict(prof0, ox0, pts)
    print(f"starting profile {Path(prof0['_path']).stem}: oxide {ox0:.1f} nm, curve loss {u0:.5f}")
    for p, b in zip(pts, before):
        print(f"  site {p['site']:2d} F{p['flake']} T{p['terrace']} {p['kind']:<12s} AFM {p['h']:7.2f} ± {p['err']:.2f}"
              f"  model {b:7.2f}  {'' if p['use'] else 'NOT USED: '}{p['why']}")

    t0 = time.time()
    if a.cv_from:
        afm = json.loads(Path(a.cv_from).read_text())["calibration"]["afm"]
        p1, ox1_fit = dict(afm["params"]), float(afm["oxide_nm"])
    else:
        p1, ox1_fit, f = fit(p0, ox0, samples, use, u0, a.maxiter)
    prof1 = profile_from_params(p1)
    ox1 = best_oxide(prof1, samples, ox1_fit)          # what the pipeline will find with the new profile
    after = predict(prof1, ox1, pts)
    print(f"fit {time.time() - t0:.0f}s: oxide {ox1:.1f} nm (fit {ox1_fit:.1f}), curve loss "
          f"{curve_loss(prof1, ox1, samples):.5f}, hBN index offset {p1['dn']:+.4f}")

    loo = None
    if a.kfold:
        # k-fold cross-validation: every site predicted by a fit that did not see it (--cv-cold: each fold fitted from the
        # starting profile; otherwise a quick warm start from the full fit, which tends to stay at the full-fit optimum)
        loo = {}
        order = np.random.default_rng(0).permutation(len(use))
        for k in range(a.kfold):
            held = [use[i] for i in order[k::a.kfold]]
            rest = [q for q in use if q not in held]
            if a.cv_cold:
                pl, oxl, _ = fit(p0, ox0, samples, rest, u0, a.maxiter, seed=2 + k)
            else:
                pl, oxl, _ = fit(p1, ox1, samples, rest, u0, max(10, a.maxiter // 3), seed=2 + k)
            prl = profile_from_params(pl)
            oxl = best_oxide(prl, samples, oxl)
            for q, v in zip(held, predict(prl, oxl, held)):
                loo[q["site"]] = float(v)
                print(f"  fold {k + 1}: site {q['site']} AFM {q['h']:.2f}  held-out model {v:.2f}")

    def stats(pred, which):
        r = np.array([pred[i] - p["h"] for i, p in enumerate(pts) if which(p)])
        rel = np.array([(pred[i] - p["h"]) / p["h"] for i, p in enumerate(pts) if which(p)])
        return dict(n=len(r), rms_nm=float(np.sqrt(np.mean(r ** 2))), mean_nm=float(np.mean(r)),
                    rms_rel=float(np.sqrt(np.mean(rel ** 2))), max_abs_nm=float(np.max(np.abs(r))))
    used = lambda p: p["use"]
    summ = {"before": stats(before, used), "after": stats(after, used)}
    if loo:
        lp = np.array([loo.get(p["site"], np.nan) for p in pts])
        summ["cross_validated"] = stats(lp, lambda p: p["use"] and p["site"] in loo)
    for k, v in summ.items():
        print(f"{k:>14s}: n {v['n']}, rms {v['rms_nm']:.2f} nm ({100 * v['rms_rel']:.1f} %), mean {v['mean_nm']:+.2f} nm,"
              f" max |err| {v['max_abs_nm']:.2f} nm")

    notes = (prof0.get("notes", "") + f" AFM-refined ({len(use)} AFM-measured hBN terraces on bigbntest_2, "
             f"tools/calibrate_afm.py): thickness rms {summ['before']['rms_nm']:.2f} -> {summ['after']['rms_nm']:.2f} nm; "
             f"hBN index offset {p1['dn']:+.3f}; fitted oxide {ox1:.1f} nm.").replace(" Not validated against AFM.", "")
    extra = {"materials": prof1.get("materials", {}),
             "calibration": dict(prof0.get("calibration", {}), afm=dict(
                 sites=a.sites, steps=a.steps, excluded=a.exclude, oxide_nm=round(ox1, 1),
                 params={k: round(float(v), 5) for k, v in p1.items()}, summary=summ))}
    save_profile(a.output, camera_from_params(dict(p1, gamma=1.0)), na=0.30, notes=notes, extra=extra)
    print(f"wrote {a.output}")

    if a.report:
        out = Path(a.report)
        out.mkdir(parents=True, exist_ok=True)
        rep = dict(start_profile=Path(prof0["_path"]).stem, new_profile=a.output, oxide_before=ox0, oxide_after=ox1,
                   params_before=p0, params_after=p1, summary=summ,
                   points=[dict({k: v for k, v in p.items() if k not in ("c", "other_c")}, c=p["c"], sigma=sigma(p),
                                before=float(b), after=float(af), loo=(loo or {}).get(p["site"]))
                           for p, b, af in zip(pts, before, after)])
        # model curves for the figures
        d = np.arange(0, 200.1, 0.5)
        rep["curves"] = {"d": d.tolist(),
                         "before": ContrastModel("hbn", ox0, camera=ThicknessEstimator("hbn", ox0, prof0).camera).contrast(d).round(4).tolist(),
                         "after": ThicknessEstimator("hbn", ox1, prof1, step_nm=0.5, search_range_nm=(0, 200)).curve.round(4).tolist()}
        rep["samples"] = samples[np.random.default_rng(0).choice(len(samples), min(1500, len(samples)), replace=False)].round(4).tolist()
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        print(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
