"""Device photos (scan program: start screen "Device photos"): pictures of a finished device or heterostructure at
100x, 50x, 10x and 5x (and any extra ones), made into colour photos, high-contrast frames of its terraces at 100x and
a page to look through them.

    python -m flakeid.devicephoto <device folder> [--flat10 flat_10x.npz] [--flat100 flat_100x_285nm.npz] [--oxide 285]

The scan program saves, per photo, the frame-averaged raw picture (<n>_<objective>_<label>.tif, 16-bit Bayer) and at
100x / 50x the same averaged at shifted stage positions (_s1.tif, ...), listed in <device>_index.csv.  Here:

* each picture is demosaiced, dark / flat corrected (the 10x and 100x flat fields of the scan; 50x and 5x have none:
  a smooth fit of the bare substrate takes the illumination fall-off out instead), and at 100x / 50x combined with
  its shifted pictures aligned on the device (flakeid.hundred.shift_combine): dust in the light path and camera specks
  land in another place in each and are averaged away;
* white balance: the bare substrate (the most common colour of the low-magnification photos) gets the colour of the
  scan's 100x pictures of that oxide, so photos of different devices look alike;
* 100x: the device's terraces (areas of one colour = one stack of layers) are found, and for each one larger than
  min_um2 a high-contrast frame is written: colour +-8 % around that terrace's colour (as the scan's "colour +-8 %"
  view of a flake), its illumination flattened;
* <device>.html: all photos with scale bars, the terraces (map, area, colour contrast to the substrate) and their
  high-contrast frames; <device>_photos.json: the same as data.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .hundred import (DEMO_SUBSTRATE, WHITE_BY_MATERIAL, _flatten, _mode_rgb, _u8, jpeg_params, shift_combine)

MAG = {"100x": 100.0, "50x": 50.0, "10x": 10.0, "5x": 5.0}
ORDER = ("100x", "50x", "10x", "5x")
UM_PER_PX_10X = 1.0911            # (flakeid.mm.DEFAULT_M: 10x, BX51 + XC10); other objectives scaled by magnification
RATIO = {"100x": 9.54}            # measured 100x / 10x pixel ratio; the others nominal
TERRACE_TOL = 0.035               # a terrace: colour within 3.5 % of its centre colour (smoothed 2 px)
MIN_UM2 = 12.0                    # terraces smaller than this are not listed (~2500 px at 100x)
MAX_TERRACES = 8
HC_HALF = 0.08                    # high-contrast frames: the terrace's colour +- 8 %
SUBSTRATE_CHROMA_TOL = 0.12       # a colour within 12 % of the substrate's chroma (R/G, B/G) at the low magnifications
                                  # counts as bare substrate (the objectives' and flat fields' colour differs by a few %)
PIECE_UM2 = 2.0                   # a terrace is made of connected pieces of at least this area
MERGE_TOL = 0.05                  # two colour classes closer than this are one terrace (illumination left over)


def um_per_px(objective):
    return UM_PER_PX_10X / RATIO.get(objective, MAG.get(objective, 10.0) / 10.0)


def _demo(oxide_nm):
    """(display colour of the bare substrate, white level): the scan's 100x pictures of hBN on ~285 nm or graphite on
    ~90 nm oxide."""
    key = "hbn" if (oxide_nm or 285) >= 180 else "graphite"
    return np.asarray(DEMO_SUBSTRATE[key], np.float64), WHITE_BY_MATERIAL[key]


def load_flat(path):
    if not path:
        return None
    try:
        from .mm import FlatField
        return FlatField.load(path)
    except Exception:
        return None


def linear(raw, ff, dark):
    """Raw Bayer (or RGB) picture -> linear RGB float32, dark / flat corrected."""
    from .mm import BAYER_CODE
    rgb = cv2.cvtColor(raw, BAYER_CODE).astype(np.float32) if raw.ndim == 2 else raw[..., ::-1].astype(np.float32)
    if ff is not None and ff.flat.shape[:2] == rgb.shape[:2]:
        return ff.apply(rgb)
    return rgb - np.float32(dark)


def chroma(c):
    c = np.asarray(c, np.float64)
    return np.array([c[0] / max(c[1], 1e-6), c[2] / max(c[1], 1e-6)])


def substrate_of(lin, ref_chroma=None):
    """(substrate colour, fraction of the picture) or (None, 0): the most common colour, if it has the substrate's
    chroma (ref_chroma; any chroma without one) and covers at least 5 % of the picture.  With a reference, the most
    common colour among the pixels of that chroma (the device itself may fill most of a 100x picture)."""
    sm = cv2.resize(lin, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    flat = sm.reshape(-1, 3)
    step = max(2.0, float(np.median(flat[:, 1])) / 160.0)
    if ref_chroma is not None:
        g = np.maximum(flat[:, 1], 1e-6)
        near = (np.abs(flat[:, 0] / g - ref_chroma[0]) < SUBSTRATE_CHROMA_TOL * ref_chroma[0]) & \
               (np.abs(flat[:, 2] / g - ref_chroma[1]) < SUBSTRATE_CHROMA_TOL * ref_chroma[1])
        if near.mean() < 0.05:
            return None, 0.0
        m, _ = _mode_rgb(flat[near].reshape(-1, 1, 3), step=step)
    else:
        m, _ = _mode_rgb(sm, step=step)
    frac = float((np.abs(flat / np.maximum(m, 1e-6) - 1).max(axis=1) < 0.04).mean())
    return (m if frac >= 0.05 else None), frac


def flatten_by_substrate(lin, sub):
    """Illumination fall-off of a picture without a flat field: a smooth fit of its bare-substrate pixels (colour within
    6 % of sub), divided out (brightness only).  Unchanged if the substrate covers less than 15 % of the picture."""
    r = lin / np.maximum(np.asarray(sub, np.float32), 1e-6)
    m = (np.abs(r - 1.0).max(axis=2) < 0.06)
    if m.mean() < 0.15:
        return lin, False
    k = 8
    h, w = m.shape
    L = r.mean(axis=2)
    Ls = cv2.resize((L * m).astype(np.float32), (w // k, h // k), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m.astype(np.float32), (w // k, h // k), interpolation=cv2.INTER_AREA)
    sig = max(w, h) / k / 6.0
    bg = cv2.GaussianBlur(Ls, (0, 0), sig) / np.maximum(cv2.GaussianBlur(ms, (0, 0), sig), 1e-3)
    wide = float(np.median(L[m]))
    bg = np.where(cv2.GaussianBlur(ms, (0, 0), sig) > 0.05, bg, wide)
    bg = cv2.resize(bg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) / wide
    return lin / np.clip(bg, 0.3, 3.0)[..., None], True


def terraces(lin, px_um, sub=None, min_um2=MIN_UM2, tol=TERRACE_TOL, max_n=MAX_TERRACES):
    """The picture's areas of one colour, largest first: [dict(colour, mask, area_um2, frac, substrate, contrast)].
    Edges between them (a colour gradient) are left out; each colour class is the most common colour of what is left,
    within tol, opened by 2 px so thin edges and specks do not count."""
    sm = cv2.GaussianBlur(lin, (0, 0), 2.0)
    # edges: brightness gradient against the picture's bright level (on a log scale the noise of dark terraces, e.g.
    # thick graphite, read as edges everywhere)
    ref = float(sub[1]) if sub is not None else float(np.percentile(sm[..., 1], 90))
    L = (sm.mean(axis=2) / max(ref, 1e-6)).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)) / 8.0
    edge = cv2.dilate((grad > 0.006).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    left = ~edge
    h, w = L.shape
    min_px = min_um2 / (px_um * px_um)
    out = []
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(3 * max_n):
        if len(out) >= max_n or left.mean() < 0.01:
            break
        pts = sm[left][::7]
        step = max(2.0, float(np.median(pts[:, 1])) / 200.0)
        c, _ = _mode_rgb(pts.reshape(-1, 1, 3), step=step)
        near = (np.abs(sm / np.maximum(c, 1e-6) - 1.0).max(axis=2) < tol)
        cls = cv2.morphologyEx((near & ~edge).astype(np.uint8), cv2.MORPH_OPEN, k5)
        # only pieces of at least PIECE_UM2: noise on a dark terrace scatters specks of a slightly other colour
        n_, lab, st, _ = cv2.connectedComponentsWithStats(cls, connectivity=8)
        keep = np.zeros(n_, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= PIECE_UM2 / (px_um * px_um)
        cls = keep[lab]
        left &= ~near
        if cls.sum() < min_px:
            if near.sum() < min_px:          # (the most common colour left is small: the rest is smaller still)
                break
            continue
        col = np.median(sm[cls], axis=0)
        same = next((t for t in out if np.abs(col / t["colour"] - 1).max() < MERGE_TOL), None)
        if same is not None:                 # (the same terrace a little brighter or darker elsewhere in the picture)
            same["mask"] |= cls
            same["area_um2"] = float(same["mask"].sum()) * px_um * px_um
            same["frac"] = float(same["mask"].mean())
            continue
        out.append(dict(colour=col, mask=cls, area_um2=float(cls.sum()) * px_um * px_um, frac=float(cls.mean())))
    for t in out:
        t["substrate"] = bool(sub is not None and np.abs(t["colour"] / np.asarray(sub) - 1).max() < MERGE_TOL)
        t["contrast"] = (t["colour"] / np.asarray(sub) - 1.0).tolist() if sub is not None else None
    out.sort(key=lambda t: (t["substrate"], -t["area_um2"]))
    return out


PALETTE = [(255, 196, 0), (0, 200, 255), (255, 90, 160), (120, 255, 120), (255, 140, 40), (180, 140, 255), (255, 255, 120),
           (90, 255, 230)]


def terrace_map(d8, ts):
    """The natural picture dimmed, each terrace outlined in its colour and numbered (BGR)."""
    bgr = (cv2.cvtColor(d8, cv2.COLOR_RGB2BGR) * 0.55).astype(np.uint8)
    n = 0
    for t in ts:
        if t["substrate"]:
            continue
        n += 1
        col = PALETTE[(n - 1) % len(PALETTE)][::-1]
        m = t["mask"].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, col, 2)
        dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        cv2.putText(bgr, f"T{n}", (int(x) - 14, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(bgr, f"T{n}", (int(x) - 14, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    return bgr


def high_contrast(lin, colour, half=HC_HALF):
    """Colour +-half around a terrace's colour, the illumination flattened on that terrace (RGB uint8)."""
    r = _flatten(cv2.GaussianBlur(lin, (0, 0), 0.8) / np.maximum(np.asarray(colour, np.float32), 1e-6))
    return _u8((r - (1.0 - half)) / (2 * half))


def load_combined(folder, r, ff):
    """One photo of the index (row r) linear, combined with its shifted pictures: dict(row, obj, lin, flat, shifts,
    shifts_used, align)."""
    folder = Path(folder)
    dark = float(r.get("dark") or 0.0)

    def load(fn):
        a = cv2.imread(str(folder / fn), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise OSError(f"{folder / fn} not readable")
        return linear(a, ff, dark)
    lin0 = load(r["file"])
    sh = []
    for f_, d_ in zip((r.get("shift_files") or "").split(";"), (r.get("shift_um") or "").split(";")):
        if f_ and (folder / f_).is_file() and ":" in d_:
            sh.append((load(f_), tuple(float(v) for v in d_.split(":"))))
    n0, ns = float(r.get("frames") or 1), float(r.get("shift_frames") or 1)
    lin, info = shift_combine(lin0, sh, "mean", weights=[n0] + [ns] * len(sh)) if sh else (lin0, [])
    used = bool(sh) and len(info) == len(sh) and all(q[2] >= 0.9 for q in info)
    return dict(row=r, obj=r["objective"], lin=lin, flat=ff is not None, shifts=len(sh), shifts_used=used, align=info)


def sharpness(lin):
    """Focus measure of a picture: the 99th percentile of the brightness gradient over the median brightness."""
    g = cv2.GaussianBlur(lin.mean(axis=2), (0, 0), 1.0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return float(np.percentile(m, 99)) / max(float(np.median(g)), 1e-6)


def preview(folder, r, ff, oxide_nm, path, width=1100):
    """A quick look at one photo just taken (the scan program shows it before going on): combined with its shifted
    pictures, white-balanced on its own substrate if it shows some, written to path (JPEG).  Returns notes for the
    user: [str], and whether something looks wrong."""
    p = load_combined(folder, r, ff)
    lin = p["lin"]
    demo, white = _demo(oxide_nm)
    sub, _ = substrate_of(lin)
    if sub is not None and not p["flat"]:
        lin, _ = flatten_by_substrate(lin, sub)
    if sub is not None:
        g = demo / np.maximum(sub, 1e-6)
    else:
        g = np.full(3, 0.8 * white / max(float(np.percentile(lin[..., 1], 99.5)), 1e-6))
    d8 = np.clip(lin * g * (255.0 / white) + 0.5, 0, 255).astype(np.uint8)
    h, w = d8.shape[:2]
    cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(d8, (width, int(round(width * h / w))), interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_RGB2BGR), jpeg_params(92))
    notes, bad = [], False
    if p["shifts"]:
        if p["shifts_used"]:
            notes.append(f"{p['shifts']} shifted pictures aligned and combined")
        else:
            bad = True
            sc_ = ", ".join(f"{q[2]:.2f}" for q in p["align"])
            notes.append(f"the shifted pictures did NOT align (match {sc_}; 0.9 needed): something moved or the focus "
                         "drifted, or there is too little structure; the averaged picture alone is used")
    sat = float(r.get("saturated_frac") or 0)
    if sat > 0.001:
        bad = True
        notes.append(f"{sat * 100:.2f} % of the pixels are saturated: retake (the exposure is set again)")
    notes.append(f"sharpness {sharpness(lin):.3f} (compare retakes: higher is sharper)")
    return notes, bad


def read_index(folder):
    folder = Path(folder)
    idx = next(iter(sorted(folder.glob("*_index.csv"))), None)
    if idx is None:
        raise FileNotFoundError(f"{folder}: no *_index.csv (the scan program's device-photo list)")
    with open(idx, newline="") as fh:
        return idx, list(csv.DictReader(fh))


def process(folder, flat10=None, flat100=None, oxide_nm=285.0, log=print, quality=100):
    """Make the photos, terrace frames and page of a device folder.  Returns the path of the page."""
    folder = Path(folder)
    idx, rows = read_index(folder)
    name = idx.name[:-len("_index.csv")]
    out = folder / "photos"
    out.mkdir(exist_ok=True)
    ffs = {"10x": load_flat(flat10), "100x": load_flat(flat100)}
    demo, white = _demo(oxide_nm)
    # 1) every picture linear, combined with its shifted ones; the substrate's chroma from the low magnifications
    lins = []
    for r in rows:
        p = load_combined(folder, r, ffs.get(r["objective"]))
        lins.append(p)
        log(f"  {r['file']}: {p['obj']}" + (f", {p['shifts']} shifted pictures " + ("combined" if p["shifts_used"] else "NOT combined (did not "
            "align: the averaged picture alone)") if p["shifts"] else "") + ("" if p["flat"] else ", no flat field"))
    low = [p for p in lins if p["obj"] in ("5x", "10x")] or lins
    subs = [substrate_of(p["lin"])[0] for p in low]
    subs = [s_ for s_ in subs if s_ is not None]
    ref = np.median(np.array([chroma(s_) for s_ in subs]), axis=0) if subs else None
    if ref is None:
        log("  (no bare substrate found in the low-magnification photos: no white balance from it)")
    gain_c = (chroma(demo) / ref) if ref is not None else np.ones(2)
    gains = np.array([gain_c[0], 1.0, gain_c[1]])
    # 2) per photo: brightness from its own substrate (else its bright end), photo, terraces, high-contrast frames
    rec = dict(device=name, oxide_nm=oxide_nm, substrate_chroma=None if ref is None else ref.tolist(), photos=[])
    for k, p in enumerate(lins):
        r, obj, lin = p["row"], p["obj"], p["lin"]
        sub, frac = substrate_of(lin, ref)
        if not p["flat"] and sub is not None:
            lin, flattened = flatten_by_substrate(lin, sub)
            if flattened:
                sub, frac = substrate_of(lin, ref)
        if sub is not None:
            scale = demo[1] / max(float(sub[1]), 1e-6)
        else:
            scale = 0.8 * white / max(float(np.percentile(lin[..., 1], 99.5)), 1e-6)
        g = gains * scale
        d8 = np.clip(lin * g * (255.0 / white) + 0.5, 0, 255).astype(np.uint8)
        stem = Path(r["file"]).stem
        cv2.imwrite(str(out / f"{stem}.jpg"), cv2.cvtColor(d8, cv2.COLOR_RGB2BGR), jpeg_params(quality))
        h, w = d8.shape[:2]
        th = cv2.resize(d8, (480, int(round(480 * h / w))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{stem}_thumb.jpg"), cv2.cvtColor(th, cv2.COLOR_RGB2BGR), jpeg_params(92))
        px = float(r.get("um_per_px") or 0) or um_per_px(obj)   # (measured by the objective offsets tool, else nominal)
        ph = dict(file=r["file"], label=r.get("label", ""), objective=obj, jpg=f"photos/{stem}.jpg", thumb=f"photos/{stem}_thumb.jpg",
                  um_per_px=round(px, 5), w=w, h=h, stage_um=[float(r.get("stage_x_um") or 0), float(r.get("stage_y_um") or 0)],
                  z_um=float(r.get("z_um") or 0), frames=int(float(r.get("frames") or 1)), shifts=p["shifts"],
                  shifts_combined=p["shifts_used"], flat_field=p["flat"], substrate_found=sub is not None, terraces=[])
        if obj == "100x":
            ts = terraces(lin, px, sub)
            if ts:
                cv2.imwrite(str(out / f"{stem}_terraces.jpg"), terrace_map(d8, ts), jpeg_params(95))
                ph["terrace_map"] = f"photos/{stem}_terraces.jpg"
            n = 0
            for t in ts:
                if t["substrate"]:
                    continue
                n += 1
                hc = high_contrast(lin, t["colour"])
                fn = f"{stem}_T{n}_c8.jpg"
                cv2.imwrite(str(out / fn), cv2.cvtColor(hc, cv2.COLOR_RGB2BGR), jpeg_params(quality))
                ph["terraces"].append(dict(n=n, area_um2=round(t["area_um2"], 1), frac=round(t["frac"], 4),
                                           colour=[round(float(v) * float(gg) * 255.0 / white, 1) for v, gg in zip(t["colour"], g)],
                                           contrast=None if t["contrast"] is None else [round(v, 4) for v in t["contrast"]],
                                           hc=f"photos/{fn}"))
            log(f"  {stem}: {n} terraces of at least {MIN_UM2:g} um^2" + (" (bare substrate found)" if sub is not None else ""))
        rec["photos"].append(ph)
    (folder / f"{name}_photos.json").write_text(json.dumps(rec, indent=1))
    page = folder / f"{name}.html"
    page.write_text(make_page(rec), encoding="utf-8")
    log(f"device photos: {len(rec['photos'])} photos -> {out}; page {page}")
    return page


def _bar(px_um, w):
    """A round scale-bar length (um) about a fifth of the picture's width, and its width in % of the picture."""
    want = w * px_um / 5.0
    L = max(v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if v <= max(want, 1))
    return L, 100.0 * L / (w * px_um)


def make_page(rec):
    """A standalone page: the photos by objective (scale bars), the 100x terraces and their high-contrast frames.  With
    rec["step"] (a fabrication step of a stack folder, flakeid.scope.devicephotos): its name, date, who and what was
    done on top; the picture paths in rec are used as they are (relative to the page)."""
    e = html.escape
    step = rec.get("step")
    if step:
        import datetime
        when = datetime.datetime.fromtimestamp(rec["at"] / 1000.0).strftime("%Y-%m-%d %H:%M") if rec.get("at") else ""
        title = f"{rec['device']} · {step}"
        sub = " · ".join(x for x in (f"Fabrication step {e(step)}", e(when), f"by {e(rec['by'])}" if rec.get("by") else "",
                                     f"SiO₂ {rec['oxide_nm']:g} nm", "click a picture for full resolution") if x)
        desc = (f'<p class="desc">{e(rec["description"])}</p>' if rec.get("description") else "")
    else:
        title = f"{rec['device']} · device photos"
        sub = f"Device photos · SiO₂ {rec['oxide_nm']:g} nm · click a picture for full resolution"
        desc = ""
    rank = {o: i for i, o in enumerate(ORDER)}
    secs = []
    for ph in sorted(rec["photos"], key=lambda p: (rank.get(p["objective"], 9), p["file"])):
        L, pct = _bar(ph["um_per_px"], ph["w"])
        fw, fh = ph["w"] * ph["um_per_px"], ph["h"] * ph["um_per_px"]
        notes = [f"{fw:.0f} × {fh:.0f} µm", f"{ph['frames']} frames averaged"]
        if ph["shifts"]:
            notes.append(f"{ph['shifts']} shifted positions " + ("combined" if ph["shifts_combined"] else "not combined (did not align)"))
        if not ph["flat_field"]:
            notes.append("no flat field" + (" (illumination flattened on the substrate)" if ph["substrate_found"] else ""))
        fig = (f'<figure><a href="{e(ph["jpg"])}" target="_blank"><div class="pic"><img src="{e(ph["jpg"])}" alt="{e(ph["objective"])} photo '
               f'{e(ph["label"])}" loading="lazy"><div class="bar" style="width:{pct:.2f}%"><span>{L:g} µm</span></div></div></a>'
               f'<figcaption><b>{e(ph["objective"])}</b> · {e(ph["label"])} · {" · ".join(e(n) for n in notes)}</figcaption></figure>')
        ts = ""
        if ph.get("terraces"):
            rows = "".join(
                f'<tr><td><b>T{t["n"]}</b></td><td>{t["area_um2"]:.0f}</td>'
                f'<td><span class="sw" style="background:rgb({",".join(str(int(min(255, v))) for v in t["colour"])})"></span></td>'
                f'<td>{"" if t["contrast"] is None else " / ".join(f"{v * 100:+.1f}" for v in t["contrast"])}</td>'
                f'<td><a href="{e(t["hc"])}" target="_blank"><img class="hc" src="{e(t["hc"])}" alt="T{t["n"]} high contrast" loading="lazy"></a></td></tr>'
                for t in ph["terraces"])
            ts = (f'<div class="terr"><a href="{e(ph["terrace_map"])}" target="_blank"><img class="map" src="{e(ph["terrace_map"])}" '
                  f'alt="terrace map" loading="lazy"></a><table><thead><tr><th>terrace</th><th>area µm²</th><th>colour</th>'
                  f'<th>contrast to the substrate R / G / B %</th><th>colour ±{HC_HALF * 100:g} % around it</th></tr></thead>'
                  f'<tbody>{rows}</tbody></table></div>')
        elif ph["objective"] == "100x":
            ts = '<p class="note">No terraces of at least {:g} µm² found.</p>'.format(MIN_UM2)
        secs.append(f'<section>{fig}{ts}</section>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><style>
:root{{--bg:#16161c;--panel:#20202a;--ink:#ecebf5;--muted:#a3a1b8;--rule:#34343f;--accent:#4fc3b8}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:0 auto;padding:16px}}h1{{margin:0 0 4px;font-size:22px}}.sub{{color:var(--muted);margin-bottom:18px}}
section{{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:12px;margin-bottom:16px}}
figure{{margin:0}}.pic{{position:relative;line-height:0}}.pic img{{width:100%;height:auto;border-radius:6px}}
.bar{{position:absolute;right:2%;bottom:4%;height:6px;background:#fff;box-shadow:0 0 0 1px #000}}
.bar span{{position:absolute;right:0;bottom:10px;line-height:1;font:600 14px -apple-system,"Segoe UI",sans-serif;color:#fff;text-shadow:0 0 3px #000,0 0 3px #000;white-space:nowrap}}
figcaption{{color:var(--muted);margin-top:6px}}figcaption b{{color:var(--ink)}}
.terr{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:12px;margin-top:12px;align-items:start}}
.map{{width:100%;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule);vertical-align:middle}}th{{color:var(--muted);font-weight:500}}
.hc{{width:220px;max-width:100%;border-radius:4px;display:block}}.sw{{display:inline-block;width:28px;height:16px;border-radius:3px;border:1px solid #0006}}
.note{{color:var(--muted)}}a{{color:var(--accent)}}.desc{{white-space:pre-wrap;margin:-8px 0 18px;max-width:900px}}
@media (max-width:800px){{.terr{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>{e(title if step else rec['device'])}</h1>
<div class="sub">{sub}</div>{desc}
{''.join(secs)}</main></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flakeid.devicephoto", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("folder", help="the device folder written by the scan program (<device>_index.csv + TIFFs)")
    ap.add_argument("--flat10", help="10x flat field .npz")
    ap.add_argument("--flat100", help="100x flat field .npz (of the chip's oxide)")
    ap.add_argument("--oxide", type=float, default=285.0, help="oxide (SiO2) thickness, nm: the substrate colour of the photos")
    a = ap.parse_args(argv)
    process(a.folder, a.flat10, a.flat100, a.oxide)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
