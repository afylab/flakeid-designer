"""Each 100x target's thickness from its 100x colour, next to its 10x thickness, and the glue on its terrace (Avram,
2026-10-30: "for all the targets, both thicknesses at 100x and 10x should always be compared, not just if the 10x is
ambiguous"; flakeid.hundred.branch100 only chose between two 10x readings).

Thickness: the target terrace's median colour against the bare substrate of the same frame (contrast = terrace /
substrate - 1, R G B), fitted to the thin-film model at 100x (NA 0.9, the run's oxide, the camera profile) over
T_RANGE.  At 100x the colours repeat (hBN on ~290 nm oxide: ~125 nm apart, e.g. 87 and 215 nm both fit BN16 F006's
green terrace), so every local minimum that fits nearly as well as the best (ALT_RATIO, ALT_ABS) is a reading too.
The 100x thickness t is the best fit, found without the 10x one (picking the reading nearest the 10x would hide what
this is for: Avram's F006, "too green to be 85 nm", fits 216 nm best and 87.5 nm nearly as well); alts are the other
readings, and agree compares the 10x thickness (within AGREE_NM or AGREE_REL of it) with them: "yes" (the best
fit), "alt" (only another reading: ambiguous at 100x too), "no" (none: e.g. BN16, ~25 targets read 38 - 51 nm at 10x
fit ~175 - 190 nm at 100x with no reading near 45).  A fit worse than FIT_MAX (misfit: distance in contrast) is
"poor": the colour fits no thickness well (residue, a step across the terrace, a light problem).

Glue: the share of the target terrace covered by tape residue (0: none, 1: all of it), from flakeid.defects100.residue
(dark specks and streaks, yellow at edges, small off-level islands) on the natural picture as the designer bundle stores
it, at half resolution (where the detector was fitted), with the terrace as the flake.

    python -m flakeid.thick100 <viewer folder> [...]
    python -m flakeid.thick100 --oxide <viewer folder> [...]    (only the run's own 100x oxide, see recalibrate)

measures every 100x picture of an existing viewer (its 100x/<stem>.jpg and _regions.json) and writes the values into
the regions files and the viewer page (the page's template is brought up to date: the table's 10x / 100x / glue columns).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

T_RANGE = {"hbn": (1.0, 300.0, 0.5), "graphite": (0.34, 60.0, 0.05)}   # (from, to, step) nm
NA_100X = 0.9
ALT_RATIO, ALT_ABS = 1.25, 0.03     # a minimum within best x 1.25 + 0.03 of the best misfit is a reading too
AGREE = {"hbn": (5.0, 0.20)}        # hBN: 100x and 10x agree within max(5 nm, 20 % of the 10x thickness): the same colour
                                    # order (the 100x colours repeat every ~125 nm); was 3 nm / 10 % until 2026-11-02, when
                                    # the run's own 100x oxide (calibrate_oxide) put 20261101_1's 30 - 75 nm flakes 7 - 12 nm
                                    # under a 10x whose substrate colour was off (R/G +28 %): flagged red, all of them
AGREE_NM, AGREE_REL = 3.0, 0.10     # other materials (graphite: no repeat in its range): max(3 nm, 10 %)
FIT_MAX = 0.45                      # misfit above this: the colour fits no thickness well (BN16: median 0.28, 90 % below 0.45)
EDGE_UM = 1.5                       # the terrace's colour without its outermost band (edge halo, steps)
MIN_PX = 400                        # fewer terrace pixels than this (after the edge band): no reading
SUB_MIN_FRAC = 0.15                 # the frame's commonest colour off the terrace is its substrate when it covers this
SUB_MAX_DEV = 0.10                  # share of the rest and is within 10 % of the run's substrate (else the run's is used:
                                    # a flake filling the frame made its own colour the "substrate", BN16 F006: 1 nm)
LAYER_NM = 0.335                    # graphite: nm per layer (labels of rhombohedral runs are in layers)
GLUE_YELLOW_K = 8.0                 # the residue detector's yellow seeds above 8 x their noise too (defects100.residue; BN16: clean
                                    # 4 - 18 nm terraces read 0.25 - 0.78 without, <= 0.08 with but one; glue spots stay)
GLUE_K = 0.5                        # the picture's scale for the residue detector (the designer's cut-out scale)
OXIDE_SHIFTS = np.arange(-12.0, 24.5, 1.0)  # the run's 100x oxide: shifts of the model's oxide tried (nm), see calibrate_oxide
OXIDE_SHIFT_DEFAULT = {"hbn": 9.0}  # the 100x model's oxide shift without a run's own (hBN only: graphite, not checked, keeps
                                    # its model and is not calibrated) (BN15 20261030_run_2, BN16 run 1 and
                                    # BN15 20261101_1 each fit best at +9 nm, median misfit 0.176 / 0.279 / 0.302 -> 0.092 /
                                    # 0.067 / 0.083)
OXIDE_CAL_MIN = 12                  # pictures needed for the run's own 100x oxide
OXIDE_CAL_GAIN = 0.85               # ... and the run's median misfit must drop below 0.85 x the one at the default shift
# Few-layer graphite on thin oxide: whole layers (Avram, 2026-11-05: 3L flakes read 0.79 nm "2 layers" with their step
# edges in plain view).  (2026-11-12) The count, like every thickness here, from all three channels together (Avram: "the
# thickness should be given by some requirement for the values of the three colors, not only 1"): the colour is thin when
# its best fit on the whole model is at most LAYERS_MAX layers (+ half a layer), and then the whole layer number whose
# model colour is nearest in R, G and B.  (Until 2026-11-12 the count came from green alone, and the 100x model was the
# hBN-calibrated camera's: 4 - 6 nm purple flakes read ~10 nm, the colour of 9 - 10 nm blue ones; see PROFILE_100X.)
LAYERS_MAX = 10                     # few-layer readings up to this many layers
LAYER_OXIDE_MAX = 150.0             # checked at ~84 nm oxide; thick oxide (rhombohedral screening: flakeid.rhombo) is not this
# The camera and optics of the 100x model per material (profiles/<name>.json; its "na" and its materials' index tables).
# Graphite (2026-11-12): bx51_xc10_mm_raw16_graphite (camera and graphite's optical constants fitted at 10x) with the 100x
# stray light per channel and NA fitted to 1141 terrace colours of GR11' - GR15' (1 - 4 layer flakes as anchors): median
# distance to the curve 0.008, and the thickness agrees with the 10x label for 97 % (the hBN camera: 0.097 and 49 %).
# Other materials: bx51_xc10_mm_raw16 (hBN-calibrated) at NA_100X, as before.
PROFILE_100X = {"graphite": "bx51_xc10_mm_raw16_graphite_100x"}
PROFILE_DEFAULT = "bx51_xc10_mm_raw16"
_MODELS = {}
_LAYER_MODELS = {}
_CM = {}


def model100(material, oxide_nm):
    """The thin-film model (flakeid.optics.ContrastModel) of a material at 100x on oxide_nm (cached): the camera, NA and
    optical constants of PROFILE_100X's profile for it (else PROFILE_DEFAULT at NA_100X).  Every 100x thickness of the
    material is read against this one model (thick100, the designer's Thickness cursor, its terrace maps, imported flakes)."""
    material = str(material).lower()
    key = (material, round(float(oxide_nm), 1))
    if key not in _CM:
        from .optics import ContrastModel
        from .thickness import camera_from_profile, load_profile
        name = PROFILE_100X.get(material)
        if name:
            p = load_profile(name)
            it = ((p.get("materials") or {}).get(material) or {}).get("index_table")
            _CM[key] = ContrastModel(material, float(oxide_nm), na=float(p.get("na", NA_100X)),
                                     camera=camera_from_profile(p), index_table=it)
        else:
            _CM[key] = ContrastModel(material, float(oxide_nm), na=NA_100X,
                                     camera=camera_from_profile(load_profile(PROFILE_DEFAULT)))
    return _CM[key]


def na100(material):
    """The 100x model's NA for a material."""
    return float(model100(material, 90.0).na)


def _model(material, oxide_nm):
    """(thicknesses, their model contrasts at 100x) for a material and oxide (cached)."""
    key = (material, round(float(oxide_nm), 1))
    if key not in _MODELS:
        a, b, s = T_RANGE.get(material, T_RANGE["hbn"])
        ts = np.arange(a, b + s / 2, s)
        _MODELS[key] = (ts, np.asarray(model100(material, oxide_nm).contrast(list(ts)), float))
    return _MODELS[key]


def layer_model(material, oxide_nm):
    """(layer numbers 1 .. LAYERS_MAX, their model contrasts at 100x), or None where the count does not apply."""
    if str(material).lower() != "graphite" or not float(oxide_nm) < LAYER_OXIDE_MAX:
        return None
    key = round(float(oxide_nm), 1)
    if key not in _LAYER_MODELS:
        ns = np.arange(1, LAYERS_MAX + 1)
        _LAYER_MODELS[key] = (ns, np.asarray(model100("graphite", oxide_nm).contrast(list(ns * LAYER_NM)), float))
    return _LAYER_MODELS[key]


def layers_tmax():
    """The thickest best fit (nm) that still counts as few-layer: LAYERS_MAX layers and half a layer."""
    return (LAYERS_MAX + 0.5) * LAYER_NM


def layers_of(contrast, material, oxide_nm):
    """The number of layers of a few-layer graphite colour (contrast R G B; see LAYERS_MAX above), or None: thicker, or
    not graphite on thin oxide.  All three channels, equally."""
    lm = layer_model(material, oxide_nm)
    if lm is None:
        return None
    c = np.asarray(contrast, float)
    ts, C = _model("graphite", oxide_nm)
    if ts[int(np.argmin(np.linalg.norm(C - c, axis=1)))] > layers_tmax():
        return None
    ns, CL = lm
    return int(ns[int(np.argmin(np.linalg.norm(CL - c, axis=1)))])


def terrace_mask(shape, polys, scale=1.0):
    m = np.zeros(shape[:2], np.uint8)
    for p in polys or []:
        q = np.round(np.asarray(p, float).reshape(-1, 2) * scale).astype(np.int32)
        if len(q) >= 3:
            cv2.fillPoly(m, [q], 1)
    return m


def colours(lin, polys, px_um, sub_run=None):
    """(terrace colour, substrate colour, terrace pixels) of a 100x frame: the terrace's median without its outer
    EDGE_UM; the substrate: the commonest colour of the frame off the terrace (and 3 um around it) when it covers
    SUB_MIN_FRAC of that and is within SUB_MAX_DEV of sub_run (the run's substrate, same units), else sub_run."""
    from .hundred import _mode_rgb
    T = terrace_mask(lin.shape, polys)
    e = int(round(EDGE_UM / px_um))
    if e > 0:
        Te = cv2.erode(T, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
        T = Te if Te.sum() >= MIN_PX else T
    n = int(T.sum())
    if n < MIN_PX:
        return None, None, n
    tc = np.median(lin[T > 0], axis=0).astype(float)
    g = int(round(3.0 / px_um))
    off = cv2.dilate(terrace_mask(lin.shape, polys), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) == 0
    sub = None if sub_run is None else np.asarray(sub_run, float)
    px = lin[::4, ::4][off[::4, ::4]]
    if len(px) >= 500:
        fm, frac = _mode_rgb(px.reshape(-1, 1, 3), step=3)
        fm = np.asarray(fm, float)
        if frac >= SUB_MIN_FRAC and (sub is None or float(np.abs(fm / np.maximum(sub, 1e-6) - 1.0).max()) <= SUB_MAX_DEV):
            sub = fm
    if sub is None:
        return None, None, n
    return tc, sub, n


def fit(contrast, material, oxide_nm, t10=None):
    """The thickness readings of a 100x contrast (R G B): dict(t (the best fit), misfit, alts (other readings), t10,
    agree ("yes" / "alt" / "no", None without t10), poor, contrast) or None."""
    c = np.asarray(contrast, float)
    if c.shape != (3,) or not np.all(np.isfinite(c)):
        return None
    ts, C = _model(material, oxide_nm)
    n = layers_of(c, material, oxide_nm)
    if n is not None:                       # (few-layer graphite: whole layers, from the green contrast)
        t = n * LAYER_NM
        agree = None
        if t10 is not None and np.isfinite(t10):
            an, ar = AGREE.get(material, (AGREE_NM, AGREE_REL))
            agree = "yes" if abs(t - t10) <= max(an, ar * t10) else "no"
        mis = float(np.linalg.norm(layer_model(material, oxide_nm)[1][n - 1] - c))
        return dict(t=round(float(t), 1), misfit=round(mis, 3), alts=[], alt_misfits=[],
                    t10=None if t10 is None else round(float(t10), 1), agree=agree, poor=bool(mis > FIT_MAX),
                    contrast=[round(float(v), 3) for v in c], na=na100(material), oxide_nm=round(float(oxide_nm), 1))
    d = np.linalg.norm(C - c, axis=1)
    mins = [i for i in range(len(ts)) if (i == 0 or d[i] <= d[i - 1]) and (i == len(ts) - 1 or d[i] <= d[i + 1])]
    ib = int(np.argmin(d))
    lim = d[ib] * ALT_RATIO + ALT_ABS
    reads = sorted({i for i in mins if d[i] <= lim} | {ib}, key=lambda i: d[i])
    agree = None
    if t10 is not None and np.isfinite(t10):
        an, ar = AGREE.get(material, (AGREE_NM, AGREE_REL))
        near = lambda i: abs(ts[i] - t10) <= max(an, ar * t10)     # noqa: E731
        agree = "yes" if near(ib) else "alt" if any(near(i) for i in reads) else "no"
    r1 = lambda v: round(float(v), 1)                       # noqa: E731
    return dict(t=r1(ts[ib]), misfit=round(float(d[ib]), 3), alts=[r1(ts[i]) for i in reads if i != ib],
                alt_misfits=[round(float(d[i]), 3) for i in reads if i != ib], t10=None if t10 is None else r1(t10),
                agree=agree, poor=bool(d[ib] > FIT_MAX), contrast=[round(float(v), 3) for v in c], na=na100(material),
                oxide_nm=r1(oxide_nm))


def measure(lin, polys, material, oxide_nm, t10=None, px_um=None, sub_run=None):
    """The 100x thickness of the terrace polys (frame px) in a 100x frame lin (linear RGB, any scale): fit() + n_px, or
    None (no terrace, too small)."""
    from .hundred import _M100
    px_um = px_um or float(np.sqrt(abs(np.linalg.det(_M100()))))
    tc, sub, n = colours(np.asarray(lin, np.float32), polys, px_um, sub_run)
    if tc is None:
        return None
    d0 = OXIDE_SHIFT_DEFAULT.get(material, 0.0)
    r = fit(tc / np.maximum(sub, 1e-6) - 1.0, material, oxide_nm + d0, t10)
    if r is not None:                       # (the default shift until the run's own: recalibrate, when attached)
        r["n_px"] = n
        r["oxide_shift"] = d0
    return r


def glue(rgb8, polys, material, oxide_nm, t_a, px_um=None):
    """Share of the terrace (polys, frame px) covered by tape residue, 0..1 (flakeid.defects100.residue on the
    natural picture as the designer stores it, at GLUE_K scale), or None."""
    from . import defects100 as D
    from .hundred import _M100
    px_um = px_um or float(np.sqrt(abs(np.linalg.det(_M100()))))
    bgr = D.view_bgr(cv2.cvtColor(np.ascontiguousarray(rgb8), cv2.COLOR_RGB2BGR))
    h = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None, fx=GLUE_K, fy=GLUE_K, interpolation=cv2.INTER_AREA)
    A = terrace_mask(h.shape, polys, GLUE_K)
    if A.sum() < 50:
        return None
    px = px_um / GLUE_K
    bx, by, bw, bh = cv2.boundingRect(A)
    m = int(np.ceil(3.0 / px)) + 8
    y0, y1, x0, x1 = max(0, by - m), min(h.shape[0], by + bh + m), max(0, bx - m), min(h.shape[1], bx + bw + m)
    lin = h.astype(np.float32) + 0.5
    try:
        d, rel = D.thickness_map(lin, A, A, material, oxide_nm, float(t_a), (y0, y1, x0, x1), rel=True)
        f, _ = D.residue(d, A[y0:y1, x0:x1], A[y0:y1, x0:x1], px, rel, material, oxide_nm, float(t_a), yellow_k=GLUE_YELLOW_K)
    except Exception:
        return None
    return round(float(np.clip(f["res_a_frac"], 0.0, 1.0)), 3)


def target_t10(label, flake_terraces, fallback=None):
    """The 10x thickness of a target's terrace (nm): the flake's terrace (viewer terraces [dict(t, ...)]) nearest the
    thickness in the target's label (<n>_F<id>_<t>nm, or <n>_F<id>_<layers>L: rhombohedral graphite runs), else that
    label thickness, else fallback."""
    m = re.search(r"_(\d+(?:\.\d+)?)(nm|L)(?:_|$)", label or "")
    lt = (float(m.group(1)) * (LAYER_NM if m.group(2) == "L" else 1.0)) if m else None
    ts = [float(t["t"]) for t in flake_terraces or [] if t.get("t") is not None]
    if lt is not None and ts:
        best = min(ts, key=lambda t: abs(t - lt))
        return best if abs(best - lt) <= max(1.0, 0.05 * lt) else lt
    return lt if lt is not None else fallback


def calibrate_oxide(contrasts, material, oxide_nm):
    """The run's own 100x oxide from all its 100x colours (Avram, 2026-11-02: "the more yellow ones are true 90nm,
    this one and the other green ones are way thicker"; 20261101_1 F032 read 91 nm at 100x): the shift of the model's
    oxide (OXIDE_SHIFTS) at which the run's colours fit the model best (median misfit of the best reading).  The 100x
    pictures are white-balanced on the substrate, but the 100x objective, the light and the camera move the colours of a
    film by a few nm of oxide: BN15 20261030_run_2 and BN16 run 1 both fit best ~8 nm thicker (median misfit 0.139 ->
    0.091 and 0.229 -> 0.062), and at the shifted oxide the green ~90 nm readings (BN15 F094: 91 nm, misfit 0.31
    against 0.38 at 220 nm) read ~210 nm.  Returns dict(shift, oxide_nm (shifted), n, misfit0 (at the default shift),
    misfit) with shift OXIDE_SHIFT_DEFAULT when fewer than OXIDE_CAL_MIN colours, or when the best shift does not lower
    the median misfit below OXIDE_CAL_GAIN x the one at the default shift."""
    cs = np.asarray([c for c in contrasts if c is not None and len(c) == 3], float).reshape(-1, 3)
    cs = cs[np.all(np.isfinite(cs), axis=1)]
    d0 = OXIDE_SHIFT_DEFAULT.get(material)
    if d0 is None:                          # (not calibrated: graphite)
        return dict(shift=0.0, oxide_nm=round(float(oxide_nm), 1), n=int(len(cs)), misfit0=None, misfit=None)
    out = dict(shift=d0, oxide_nm=round(float(oxide_nm) + d0, 1), n=int(len(cs)), misfit0=None, misfit=None)
    if len(cs) == 0:
        return out
    med = {}
    for dz in OXIDE_SHIFTS:
        ts, C = _model(material, oxide_nm + float(dz))
        d = np.linalg.norm(C[None, :, :] - cs[:, None, :], axis=2).min(axis=1)
        med[float(dz)] = float(np.median(d))
    best = min(med, key=lambda k: (med[k], abs(k)))
    out.update(misfit0=round(med[d0], 3), misfit=round(med[d0], 3))
    if len(cs) >= OXIDE_CAL_MIN and best != d0 and med[best] <= OXIDE_CAL_GAIN * med[d0]:
        out.update(shift=best, oxide_nm=round(float(oxide_nm) + best, 1), misfit=round(med[best], 3))
    return out


def recalibrate(vdir, H=None, log=print):
    """A viewer's 100x thicknesses refitted at the run's own 100x oxide (calibrate_oxide over all its pictures'
    colours, kept in the regions files' t100.contrast): every regions file's t100 refitted (t100.oxide_nm the oxide
    used, t100.oxide_shift the run's shift) and the H100 entries (H: {flake id: [entries]}, changed in place) given the
    new readings.  The 10x thicknesses are not changed.  Returns calibrate_oxide's dict (+ changed: pictures whose
    100x thickness changed), or None (no pictures)."""
    vdir = Path(vdir)
    regs = []
    for rj in sorted((vdir / "100x").glob("*_regions.json")):
        try:
            reg = json.loads(rj.read_text(encoding="utf-8"))
        except ValueError:
            continue
        t = reg.get("t100") or {}
        if t.get("contrast") is not None:
            regs.append((rj, reg))
    if not regs:
        return None
    material = None
    for _, reg in regs:
        material = material or reg.get("material")
    material = material or _material_of(vdir)
    # the model oxide of the run before any shift (t100.oxide_nm minus a shift applied before; regions oxide_nm)
    ox0 = [float(reg.get("oxide_nm") or (reg["t100"].get("oxide_nm", 300.0) - reg["t100"].get("oxide_shift", 0.0)))
           for _, reg in regs]
    ox = float(np.median(ox0))
    cal = calibrate_oxide([reg["t100"]["contrast"] for _, reg in regs], material, ox)
    by_file = {}
    for k, es in (H or {}).items():
        for e in es:
            by_file[e.get("f")] = e
    n_ch = 0
    for (rj, reg), o in zip(regs, ox0):
        t = reg["t100"]
        r = fit(t["contrast"], material, o + cal["shift"], t.get("t10"))
        if r is None:
            continue
        r["oxide_shift"] = cal["shift"]
        for k in ("n_px",):
            if k in t:
                r[k] = t[k]
        n_ch += r["t"] != t.get("t")
        if r != t:
            reg["t100"] = r
            rj.write_text(json.dumps(reg, separators=(",", ":")), encoding="utf-8")
        e = by_file.get(f"100x/{rj.name[:-len('_regions.json')]}.jpg")
        if e is not None:
            e.pop("t100", None)
            e.update(summary(reg))
    cal["changed"] = n_ch
    d0 = OXIDE_SHIFT_DEFAULT.get(material)
    if d0 is None:
        return cal
    own = cal["n"] >= OXIDE_CAL_MIN
    log(f"  100x thickness: the run's 100x oxide {cal['oxide_nm']} nm (model oxide {ox:.1f} nm {cal['shift']:+.0f} nm, "
        + (f"from {cal['n']} pictures; median colour misfit {cal['misfit0']} at {d0:+.0f} nm -> {cal['misfit']}"
           if own else f"the default: {cal['n']} pictures, too few for the run's own") + f"); {n_ch} readings changed")
    return cal


def _material_of(vdir):
    try:
        html = (Path(vdir) / "index.html").read_text(encoding="utf-8")
        return json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1)).get("material") or "hbn"
    except Exception:
        return "hbn"


def summary(reg):
    """What the viewer keeps of a picture's readings (H100 entry): dict(t100=..., glue=...) without the long fields."""
    out = {}
    t = reg.get("t100")
    if t:
        out["t100"] = {k: t[k] for k in ("t", "t10", "agree", "poor", "misfit", "alts", "oxide_shift") if k in t}
    if reg.get("glue") is not None:
        out["glue"] = reg["glue"]
    return out


def flake_terraces(D):
    """{flake id: [its viewer terraces]} of a viewer's data."""
    T = D.get("terraces") or {}
    return {int(f["id"]): [T[str(k)] for k in (f.get("terr") or []) if str(k) in T] for f in D.get("flakes") or []}


def remeasure_viewer(vdir, log=print):
    """Every 100x picture of an existing viewer: its readings written into its regions file and the page (H100), the
    page's template brought up to date.  Returns the number of pictures measured."""
    from .hundred import DEMO_SUBSTRATE, WHITE, WHITE_BY_MATERIAL, _flake_label_id
    vdir = Path(vdir)
    page = vdir / "index.html"
    html = page.read_text(encoding="utf-8")
    D = json.loads(re.search(r"const D=(\{.*?\});\n", html, re.S).group(1))
    mh = re.search(r"const H100=(.*?);\n", html, re.S)
    H = json.loads(mh.group(1)) if mh and mh.group(1).strip() not in ("null", "") else {}
    material = D.get("material") or "hbn"
    FT = flake_terraces(D)
    n, n_dis, n_poor = 0, 0, 0
    for rj in sorted((vdir / "100x").glob("*_regions.json")):
        stem = rj.name[:-len("_regions.json")]
        img = vdir / "100x" / f"{stem}.jpg"
        reg = json.loads(rj.read_text(encoding="utf-8"))
        a = cv2.imread(str(img))
        if a is None or not reg.get("terrace") or (reg.get("seed") or {}).get("found") is False:
            continue
        rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) + 0.5    # (<stem>.jpg: linear, white-balanced)
        ox = float(reg.get("oxide_nm") or D.get("oxide") or 300.0)
        fid = _flake_label_id(reg.get("label"))
        t10 = target_t10(reg.get("label"), FT.get(fid))
        sub = np.asarray(DEMO_SUBSTRATE.get(material, DEMO_SUBSTRATE["hbn"]), float) * 255.0 / WHITE_BY_MATERIAL.get(material, WHITE)
        reg["t100"] = measure(rgb, reg["terrace"], material, ox, t10, reg.get("px_um"), sub)
        reg["glue"] = glue(a[..., ::-1], reg["terrace"], material, ox, t10 or 30.0, reg.get("px_um"))
        rj.write_text(json.dumps(reg, separators=(",", ":")), encoding="utf-8")
        for e in H.get(str(fid), []):
            if e.get("f") == f"100x/{stem}.jpg":
                e.pop("t100", None)
                e.pop("glue", None)
                e.update(summary(reg))
        n += 1
        t = reg["t100"] or {}
        n_dis += t.get("agree") == "no"
        n_poor += bool(t.get("poor"))
        log(f"  {stem}: 10x {t.get('t10')} nm, 100x {t.get('t')} nm (misfit {t.get('misfit')}"
            f"{', also ' + ', '.join(str(v) for v in t.get('alts') or []) if t.get('alts') else ''})"
            f"{' DISAGREE' if t.get('agree') == 'no' else ' (10x = another reading)' if t.get('agree') == 'alt' else ''}"
            f"{' poor fit' if t.get('poor') else ''} · glue {reg['glue']}")
    if n:
        recalibrate(vdir, H, log)                   # (the run's own 100x oxide)
    if mh:
        line = "const H100=" + json.dumps(H, separators=(",", ":")) + ";\n"
        html = html[:mh.start()] + line + html[mh.end():]
        # (a new file swapped in, with retries: on the share the page was "temporarily unavailable", errno 35, while
        # open elsewhere, BN16 2026-10-30; written in place it could also be left half-written)
        from .zoomview import _write_swap
        _write_swap(page, html)
    try:
        from .zoomview import upgrade_page
        upgrade_page(page, backup=False)
    except Exception as e:                              # (the data are written; the template stays as it was)
        log(f"  the viewer page's template was not updated ({type(e).__name__}: {e})")
    log(f"{vdir.name}: {n} 100x pictures measured; the 10x thickness is none of the 100x readings on {n_dis}, poor colour fit on {n_poor}")
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m flakeid.thick100", description=__doc__.split("\n\n")[0])
    ap.add_argument("viewers", nargs="+", help="viewer folders (index.html and 100x/)")
    ap.add_argument("--oxide", action="store_true",
                    help="only refit the 100x thicknesses at the run's own 100x oxide from the colours already measured "
                         "(seconds), and bring the page up to date (the 100x score 0-1)")
    a = ap.parse_args(argv)
    for v in a.viewers:
        if a.oxide:
            recalibrate_page(v)
        else:
            remeasure_viewer(v)


def recalibrate_page(vdir, log=print):
    """recalibrate on an existing viewer: its regions files and its page's H100 refitted, the page's template
    brought up to date.  Returns recalibrate's dict."""
    from .zoomview import _write_swap, upgrade_page
    page = Path(vdir) / "index.html"
    html = page.read_text(encoding="utf-8")
    mh = re.search(r"const H100=(.*?);\n", html, re.S)
    H = json.loads(mh.group(1)) if mh and mh.group(1).strip() not in ("null", "") else {}
    cal = recalibrate(vdir, H, log)
    if mh:
        line = "const H100=" + json.dumps(H, separators=(",", ":")) + ";\n"
        _write_swap(page, html[:mh.start()] + line + html[mh.end():])
    upgrade_page(page, backup=False)
    log(f"{Path(vdir).name}: page written")
    return cal


if __name__ == "__main__":
    sys.exit(main())
