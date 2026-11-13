"""Unsupervised calibration of the camera / optical model from example images.

The thin-film model has a few parameters that depend on the microscope set-up
(camera tone curve, colour channel spectral responses) and on imperfectly
known optical constants.  Because flakes of every thickness trace a single
curve in RGB-contrast space, these parameters can be fitted without knowing
any flake thicknesses: the model curve is adjusted until it passes through
the observed contrasts of flat flake regions (robust, trimmed loss).  Several
materials/oxides are fitted jointly with shared camera parameters and a
separate oxide thickness per image.

Usage:
    python -m flakeid.calibrate --image g31.jpg graphite 90 --image bn24.jpg hbn 285 \\
        -o flakeid/profiles/my_setup.json

A calibration against AFM-measured flakes (supervised) is the natural next
step; `reference` rows can be added to the loss once such data exist.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

from .background import compute_contrast
from .io import load_image
from .optics import MATERIALS, CameraModel, ContrastModel, resolve_material
from .pipeline import _collect_uniform_samples
from .thickness import SEARCH_RANGE_NM, load_profile, save_profile


def stratified(samples, n=12000, edges=(0.0, 0.15, 0.4, 0.8, 10.0), seed=0):
    """Equal numbers of samples per contrast-magnitude bin so that the (rare)
    thin flakes weigh as much as the (abundant) thick ones."""
    rng = np.random.default_rng(seed)
    mag = np.linalg.norm(samples, axis=1)
    out = []
    per = n // (len(edges) - 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = samples[(mag >= lo) & (mag < hi)]
        if len(s):
            out.append(s[rng.choice(len(s), min(per, len(s)), replace=False)])
    return np.vstack(out)


def graphite_table(n550, k550, k_slope, n_slope=0.10):
    wl = np.array([380.0, 400, 450, 500, 550, 600, 650, 700, 780])
    x = (wl - 550.0) / 100.0
    return np.stack([wl, n550 + n_slope * x, k550 + k_slope * x], axis=1)


PARAMS_GAMMA = [("gamma", 1.6, 2.6)]
PARAMS_FLARE = [("f_r", 0.0, 0.4), ("f_g", 0.0, 0.4), ("f_b", 0.0, 0.4)]
PARAMS = [  # name, low, high  (bounds keep the camera physically plausible)
    ("c_r", 585, 630), ("c_g", 510, 555), ("c_b", 440, 480),
    ("w_r", 60, 120), ("w_g", 70, 130), ("w_b", 70, 130),
    ("ir_cut", 640, 700),
    # colour-correction matrix off-diagonals (rows are renormalised to sum to 1)
    ("m_rg", -0.6, 0.2), ("m_rb", -0.4, 0.2), ("m_gr", -0.5, 0.2),
    ("m_gb", -0.5, 0.2), ("m_br", -0.4, 0.2), ("m_bg", -0.6, 0.2),
    ("gr_n550", 2.4, 3.0), ("gr_k550", 1.1, 1.6), ("gr_kslope", -0.05, 0.25),
]


def camera_from_params(p):
    m = lambda k: p.get(k, 0.0)
    off = np.array([[0, m("m_rg"), m("m_rb")], [m("m_gr"), 0, m("m_gb")], [m("m_br"), m("m_bg"), 0]])
    ccm = np.eye(3) + off
    ccm = ccm / ccm.sum(axis=1, keepdims=True)
    return CameraModel(gamma=p.get("gamma", 1.0), centers_nm=(p["c_r"], p["c_g"], p["c_b"]),
                       fwhm_nm=(p["w_r"], p["w_g"], p["w_b"]), ir_cut_nm=p["ir_cut"],
                       ccm=tuple(map(tuple, ccm.round(5))),
                       flare=(p.get("f_r", 0.0), p.get("f_g", 0.0), p.get("f_b", 0.0)))


def _model(p, material, oxide, **kw):
    return ContrastModel(material, oxide, camera=camera_from_params(p), n_angles=3,
                         wl=np.arange(400.0, 721.0, 10.0),
                         index_table=graphite_table(p["gr_n550"], p["gr_k550"], p["gr_kslope"])
                         if material == "graphite" else None, **kw)


class _Loss:
    """Picklable objective (differential_evolution runs it in worker processes)."""

    def __init__(self, names, datasets, substrate_pairs, trim, w_sub):
        self.names, self.datasets, self.pairs = names, datasets, list(substrate_pairs)
        self.trim, self.w_sub = trim, w_sub

    def __call__(self, x):
        p = dict(zip(self.names, x))
        tot, models = 0.0, []
        for i, (s, mat, ox, _, _) in enumerate(self.datasets):
            lo, hi = SEARCH_RANGE_NM.get(mat, (0, 200))
            m = _model(p, mat, ox * p[f"oxide_{i}"])
            models.append(m)
            curve = np.minimum(m.contrast(np.linspace(lo, hi, 700)),
                               self.datasets[i][4] / self.datasets[i][3][None, :] - 1.0)   # camera clipping
            if not np.all(np.isfinite(curve)):
                return 1e6
            d, _ = cKDTree(curve).query(s)
            d = np.sort(d)[: int(self.trim * len(d))]
            tot += float(np.mean(d ** 2))
        for i, j in self.pairs:
            pred = (models[j].substrate_reflectance() / models[i].substrate_reflectance()) ** (1 / p.get("gamma", 1.0))
            meas = (self.datasets[j][3] / self.datasets[j][4]) / (self.datasets[i][3] / self.datasets[i][4])
            tot += self.w_sub * float(np.mean(np.log(pred / meas) ** 2))
        return tot


def fit(datasets, substrate_pairs=(), trim=0.7, maxiter=80, seed=1, w_sub=20.0, verbose=True, tone="flare",
        ccm=True):
    """datasets: list of (samples (N,3), material, nominal oxide nm, substrate_rgb (3,)).
    substrate_pairs: (i, j) dataset indices imaged with identical camera settings;
    the ratio of their substrate colours constrains the camera model."""
    extra = PARAMS_FLARE if tone == "flare" else PARAMS_GAMMA
    base = [q for q in PARAMS if ccm or not q[0].startswith("m_")]
    names = [q[0] for q in extra + base] + [f"oxide_{i}" for i in range(len(datasets))]
    bounds = [(lo, hi) for _, lo, hi in extra + base] + [(0.9, 1.1)] * len(datasets)
    loss = _Loss(names, datasets, substrate_pairs, trim, w_sub)
    t0 = time.time()
    res = differential_evolution(loss, bounds, popsize=10, maxiter=maxiter, tol=1e-7, seed=seed,
                                 polish=True, workers=-1, updating="deferred")
    if verbose:
        print(f"calibration loss {res.fun:.5f} after {time.time() - t0:.0f}s")
    return dict(zip(names, res.x)), res.fun


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs=3, action="append", metavar=("PATH", "MATERIAL", "OXIDE_NM"), required=True)
    ap.add_argument("-o", "--output", required=True, help="profile JSON to write")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-ccm", action="store_true",
                    help="no colour-correction matrix (raw sensor data: crosstalk is in the channel responses)")
    ap.add_argument("--flat", help="flat field .npz for Micro-Manager tile folders")
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--tone", choices=["flare", "gamma"], default="flare",
                    help="contrast-compression model: stray light (physical, linear data) or a tone curve")
    ap.add_argument("--same-settings", action="store_true",
                    help="all images were taken with identical exposure / white balance / lamp setting")
    args = ap.parse_args(argv)

    datasets = []
    for path, mat, ox in args.image:
        img = load_image(path, flat=args.flat)
        cm = compute_contrast(img.rgb, img.pixel_um, max_value=img.max_value)
        s = stratified(_collect_uniform_samples(cm.contrast, cm.chip_mask, cm.saturated, n=200000))
        datasets.append((s, resolve_material(mat), float(ox), cm.substrate_rgb.astype(float),
                         img.clip_value if img.clip_value is not None else img.max_value - 0.5))
        print(f"{path}: {len(s)} samples, substrate RGB {cm.substrate_rgb}")
    # images of different oxide thickness taken with the same settings constrain the camera
    pairs = [(i, j) for i in range(len(datasets)) for j in range(len(datasets))
             if datasets[i][2] < datasets[j][2]] if args.same_settings else []
    p, _ = fit(datasets, substrate_pairs=pairs, maxiter=args.maxiter, tone=args.tone, ccm=not args.no_ccm)
    cam = camera_from_params(p)
    oxides = {path: round(float(ox) * p[f"oxide_{i}"], 1) for i, (path, _, ox) in enumerate(args.image)}
    # graphite optical constants are only constrained by graphite images
    mats = {"graphite": {"index_table": graphite_table(p["gr_n550"], p["gr_k550"], p["gr_kslope"]).round(4).tolist()}} \
        if any(resolve_material(m) == "graphite" for _, m, _ in args.image) else {}
    extra = {"materials": mats,
        "calibration": {"images": args.image, "same_settings": args.same_settings, "fitted_oxide_nm": oxides,
                        "params": {k: round(float(v), 4) for k, v in p.items()}}}
    save_profile(args.output, cam, na=0.30, notes=args.notes, extra=extra)
    print(json.dumps(extra["calibration"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
