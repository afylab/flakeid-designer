#!/usr/bin/env python3
"""Designer catalog from high-magnification images of individual flakes.

    python tools/make_demo_catalog.py --src "example images/demo_high_magnification" -o designer_demo

Expected files per flake: JP_<chip>_<letter>_100x.jpg (colour, 100x objective, 0.63x adapter)
and JP_<chip>_<letter>_5x.jpg (context).  Flakes with any file whose name ends in " X" are
already used and are skipped.  Chips named BN* are hBN, g* graphite.

Per flake:
* the 100x image is flat-fielded on the bare substrate; the flake is the connected region of
  material covering the image centre (the imaged flake); terraces are its uniform-contrast
  regions (flakeid.segment);
* thickness per terrace from the thin-film model with the 100x objective's NA (0.9).  This is
  the 6.3x camera calibration applied at another objective: indicative only;
* the 100x field is located in the 5x image (template matching) so the designer's
  "Show on chip" view shows the flake in its surroundings.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.io import DEFAULT_PIXEL_UM  # noqa: E402
from flakeid.segment import SegmentParams, segment  # noqa: E402
from flakeid.thickness import ThicknessEstimator, load_profile  # noqa: E402

CAL_MAG = 6.3                                    # magnification at which DEFAULT_PIXEL_UM holds
PX = {100: DEFAULT_PIXEL_UM * CAL_MAG / 63.0, 5: DEFAULT_PIXEL_UM * CAL_MAG / 3.15}
NA = {100: 0.9, 5: 0.15}
OXIDE = {"hbn": 285.0, "graphite": 90.0}
# search ranges: the flakes chosen for high-magnification imaging are the thin ones (hBN < ~100 nm,
# graphite < ~25 nm), so higher interference orders are excluded (the pipeline's alt values)
RANGE = {"hbn": (0.5, 150.0), "graphite": (0.3, 60.0)}
WS = 0.25                                        # working scale for segmentation (0.4 um/px)
ATLAS = 4096


def substrate_reference(paths):
    """Substrate colour shared by a set of images (same chips, same settings): the most common
    chromaticity cluster, counted once per image, so a flake filling one frame cannot win."""
    feats = []
    for p in paths:
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        sm = cv2.resize(im, (86, 65), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
        sm = sm[sm.mean(1) > 40]
        feats.append(sm)
    allp = np.vstack(feats)
    chrom = allp / allp.sum(1, keepdims=True)
    K = 8
    _, lab, cen = cv2.kmeans(chrom.astype(np.float32), K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4), 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    off, pres = 0, np.zeros(K)
    for f in feats:
        l = lab[off:off + len(f)]
        off += len(f)
        pres += np.bincount(l, minlength=K) / len(f) > 0.08      # cluster present in this image
    k = int(np.argmax(pres + 1e-3 * np.bincount(lab, minlength=K)))
    return allp[lab == k].mean(0)


def flatten(rgb, ref=None):
    """Contrast map C = I/I_substrate - 1 with a smooth substrate estimate (vignetting).
    ref: expected substrate colour (seeds the substrate mask)."""
    f = rgb.astype(np.float32)
    small = cv2.resize(f, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    g = small.mean(2)
    mask = np.ones(g.shape, np.float32)
    if ref is not None:
        ref = np.asarray(ref, np.float32)
        ch = small / np.maximum(small.sum(2, keepdims=True), 1)
        cr = ref / ref.sum()
        near = np.linalg.norm(ch - cr, axis=2) < 0.02
        br = small.mean(2) / ref.mean()
        near &= (br > 0.75) & (br < 1.3)
        if near.mean() > 0.02:
            mask = near.astype(np.float32)
    # start from the dominant colour: pixels close to the median of the brightest-common hue
    for tol in ((0.12, 0.07, 0.05) if ref is not None else (0.25, 0.12, 0.07, 0.05)):
        w = cv2.GaussianBlur(mask, (0, 0), 25) + 1e-6
        bg = cv2.GaussianBlur(small * mask[..., None], (0, 0), 25) / w[..., None]
        wide = (small * mask[..., None]).reshape(-1, 3).sum(0) / max(mask.sum(), 1)
        a = np.clip(w / 0.3, 0, 1)[..., None]
        bg = a * bg + (1 - a) * wide[None, None, :]
        c = small / np.maximum(bg, 1) - 1
        mask = (np.abs(c).max(2) < tol).astype(np.float32)
    bgf = cv2.resize(bg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    C = f / np.maximum(bgf, 1) - 1
    sub = np.abs(C).max(2) < 0.05
    med = np.median(C[sub], axis=0)
    C = (C + 1) / (1 + med) - 1
    noise = 1.4826 * np.median(np.abs(C[sub] - 0), axis=0)
    dark = rgb.mean(2) < 0.35 * np.median(rgb[sub].mean(1))      # chip edge / stage
    return C.astype(np.float32), noise, np.median(rgb[sub], axis=0), dark


def poly_of(mask, simplify=1.5):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, simplify, True).reshape(-1, 2) + 0.5
    return np.round(c, 1).ravel().tolist()


def thickness_5x(files, loc, S, terr_masks_full, material, prof, ref5):
    """Median 5x contrast inside each terrace (mapped from the 100x frame) -> thickness."""
    im5 = cv2.cvtColor(cv2.imread(files["col5"]), cv2.COLOR_BGR2RGB)
    C5, _, _, _ = flatten(im5, ref5)
    p = copy.deepcopy(prof)
    p["na"] = NA[5]
    est = ThicknessEstimator(material, OXIDE[material], p, search_range_nm=RANGE[material])
    (lx, ly) = loc
    out = []
    for m in terr_masks_full:
        ms = cv2.resize(m, None, fx=S, fy=S, interpolation=cv2.INTER_AREA) > 0.9    # fully inside
        ms = cv2.erode(ms.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ys, xs = np.nonzero(ms)
        if len(ys) < 6:
            out.append(None)
            continue
        yy, xx = ys + ly, xs + lx
        ok = (yy >= 0) & (yy < C5.shape[0]) & (xx >= 0) & (xx < C5.shape[1])
        if ok.sum() < 6:
            out.append(None)
            continue
        c = np.median(C5[yy[ok], xx[ok]], axis=0)
        f = est.fit(c)
        out.append(dict(t=round(f.thickness_nm, 1), tlo=round(f.range_nm[0], 1), thi=round(f.range_nm[1], 1),
                        alt=";".join(f"{x:g}" for x in f.alternatives_nm[:2]), res=round(f.residual, 3), n=int(ok.sum())))
    return out


def process(name, files, material, prof, ref100=None):
    rgb = cv2.cvtColor(cv2.imread(files["col100"]), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    C, noise, sub_rgb, dark = flatten(rgb, ref100)
    px = PX[100]
    work = cv2.resize(C, None, fx=WS, fy=WS, interpolation=cv2.INTER_AREA)
    wpx = px / WS
    chip = ~cv2.resize(dark.astype(np.uint8), (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    sat = cv2.resize((rgb >= 252).astype(np.uint8), (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    # flake = connected material covering the centre (or the largest central component)
    cs = cv2.bilateralFilter(work, 5, 0.08, 1.5)
    thr = max(0.035, 4 * float(np.linalg.norm(noise)))
    fg = (np.linalg.norm(cs, axis=2) > thr) & chip
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = ndimage.binary_fill_holes(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))).astype(bool)
    lab, n = ndimage.label(fg)
    if n == 0:
        return None
    h, w = lab.shape
    yy, xx = np.mgrid[:h, :w]
    centre = ((xx - w / 2) / (w * 0.3)) ** 2 + ((yy - h / 2) / (h * 0.3)) ** 2 < 1
    counts = np.bincount(lab[centre].ravel(), minlength=n + 1)
    counts[0] = 0
    fid = int(counts.argmax()) if counts.max() > 0 else int(np.bincount(lab.ravel())[1:].argmax() + 1)
    fmask = lab == fid
    touches = bool(fmask[0].any() or fmask[-1].any() or fmask[:, 0].any() or fmask[:, -1].any())
    # terraces: segmentation restricted to the flake
    regions, _, rlab = segment(np.where(fmask[..., None], work, 0).astype(np.float32), fmask, np.repeat(sat[..., None], 3, 2),
                               noise, SegmentParams(tol_abs=0.035, tol_rel=0.06, min_region_px=40), min_parent_px=40)
    p = copy.deepcopy(prof)
    p["na"] = NA[100]
    est = ThicknessEstimator(material, OXIDE[material], p, search_range_nm=RANGE[material])
    terr = []
    for r in sorted(regions, key=lambda r: -r.mask.sum()):
        m = np.zeros(fmask.shape, np.uint8)
        m[r.slices][r.mask] = 1
        area = float(m.sum()) * wpx ** 2
        if area < 30 or float(np.linalg.norm(r.median)) < 0.04:   # um^2; faint = residue / substrate
            continue
        fit = est.fit(r.median)
        dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        ty, tx = np.unravel_index(np.argmax(dist), dist.shape)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (a_, b_), _ = cv2.minAreaRect(np.vstack(cnts).reshape(-1, 2).astype(np.float32))
        terr.append(dict(t=round(fit.thickness_nm, 1), tlo=round(fit.range_nm[0], 1), thi=round(fit.range_nm[1], 1),
                         alt=";".join(f"{x:g}" for x in fit.alternatives_nm[:2]),
                         layers=round(fit.thickness_nm / est.layer_nm, 1), area=round(area, 1),
                         d=round(2 * float(dist.max()) * wpx, 1), L=round((max(a_, b_) + 1) * wpx, 1),
                         W=round((min(a_, b_) + 1) * wpx, 1), m=m, c=(tx, ty), res=round(fit.residual, 3),
                         cvec=[round(float(v), 3) for v in r.median],
                         mfull=cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST),
                         quality="" + ("touches_image_edge" if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() else "")))
        if len(terr) >= 8:
            break
    if not terr:
        return None
    # full-resolution cut-out of the flake
    full = cv2.resize(fmask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero(full)
    x0, y0, x1, y1 = max(xs.min() - 4, 0), max(ys.min() - 4, 0), min(xs.max() + 5, W), min(ys.max() + 5, H)
    mask = full[y0:y1, x0:x1]
    alpha = cv2.GaussianBlur(mask.astype(np.float32) * 255, (0, 0), 1.2)
    rgba = np.dstack([rgb[y0:y1, x0:x1], np.clip(alpha, 0, 255).astype(np.uint8)])
    k = 1 / WS

    def to_cut(poly):
        a = np.array(poly, float).reshape(-1, 2) * k - [x0, y0]
        return np.round(a, 1).ravel().tolist()
    return dict(name=name, rgba=rgba, origin_px=(x0, y0), touches=touches, sub_rgb=sub_rgb.tolist(),
                outline=poly_of(mask, 2.0),
                tmasks=[t["mfull"] for t in terr],
                terraces=[dict({kk: vv for kk, vv in t.items() if kk not in ("m", "c", "mfull")},
                               poly=to_cut(poly_of(t["m"], 0.8)),
                               c=[int(round(t["c"][0] * k - x0)), int(round(t["c"][1] * k - y0))]) for t in terr])


def locate_in_5x(files, rgb100):
    """Top-left of the 100x field in 5x pixels (template match of the down-scaled 100x image)."""
    if "col5" not in files:
        return None, None
    im5 = cv2.imread(files["col5"])
    s = PX[100] / PX[5]
    t = cv2.resize(cv2.cvtColor(rgb100, cv2.COLOR_RGB2BGR), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gt = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = cv2.matchTemplate(g5, gt, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    if mx < 0.4:          # fall back: the 100x image was taken at the 5x image centre
        loc = ((im5.shape[1] - t.shape[1]) // 2, (im5.shape[0] - t.shape[0]) // 2)
    return im5, (loc, float(mx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="example images/demo_high_magnification")
    ap.add_argument("-o", "--out", default="designer_demo")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof = load_profile()
    sets = {}
    for f in sorted(glob.glob(str(Path(a.src) / "*" / "*.jpg"))):
        m = re.match(r"JP_(\w+?)_([a-z])_(\d+)x(_bw)?( X)?\.jpg$", os.path.basename(f))
        if not m:
            continue
        chip, let, mag, bw, used = m.groups()
        d = sets.setdefault(f"{chip}_{let}", {"used": False})
        d["used"] |= bool(used)
        d[("bw" if bw else "col") + mag] = f
    flakes, cut, scans = [], [], {}
    mat_of = lambda n: "hbn" if n.lower().startswith("bn") else "graphite"
    refs = {}
    for mat in ("hbn", "graphite"):
        refs[(mat, 100)] = substrate_reference([v["col100"] for k, v in sets.items() if mat_of(k) == mat and "col100" in v])
        refs[(mat, 5)] = substrate_reference([v["col5"] for k, v in sets.items() if mat_of(k) == mat and "col5" in v])
        print(mat, "substrate reference 100x", refs[(mat, 100)].round(0), "5x", refs[(mat, 5)].round(0))
    for name, files in sets.items():
        if files["used"] or "col100" not in files:
            continue
        material = mat_of(name)
        r = process(name, files, material, prof, refs[(material, 100)])
        if r is None:
            print(name, "no flake found")
            continue
        rgb100 = cv2.cvtColor(cv2.imread(files["col100"]), cv2.COLOR_BGR2RGB)
        im5, loc = locate_in_5x(files, rgb100)
        S = PX[100] / PX[5]
        t5 = thickness_5x(files, loc[0], S, r["tmasks"], material, prof, refs[(material, 5)]) if im5 is not None else [None] * len(r["terraces"])
        if loc is None:
            loc = (None, 0.0)
        for t, a5 in zip(r["terraces"], t5):
            t["t100"], t["alt100"] = t["t"], t["alt"]
            good5 = (a5 is not None and loc[0] is not None and loc[1] >= 0.6
                     and a5["t"] > RANGE[material][0] + 0.5 * (material == "hbn"))
            if a5 is not None and not good5:
                a5 = None
            if a5 is not None:                 # 5x is closer to the calibrated optics: primary value
                t.update(t=a5["t"], tlo=a5["tlo"], thi=a5["thi"], alt=a5["alt"], res=a5["res"])
                t["layers"] = round(a5["t"] / (0.335 if material == "graphite" else 0.333), 1)
                t["quality"] = ("thickness_from_5x;" + t["quality"]).strip(";")
                if abs(a5["t"] - t["t100"]) > 0.3 * max(a5["t"], t["t100"], 1.0):
                    t["quality"] += ";5x_100x_disagree"
            else:
                t["quality"] = ("thickness_from_100x;" + t["quality"]).strip(";")
        if im5 is not None:
            cv2.imwrite(str(out / f"chip_{name}.jpg"), im5, [cv2.IMWRITE_JPEG_QUALITY, 85])
            (lx, ly), score = loc
            # origin of the cut-out in "100x-pixel units" of the 5x image frame
            origin = [lx / S + r["origin_px"][0], ly / S + r["origin_px"][1]]
            scans[name] = dict(file=f"chip_{name}.jpg", w=im5.shape[1], h=im5.shape[0], scale=S, px_um=PX[100],
                               material=material, oxide_nm=OXIDE[material], stage_origin_um=None, stage_sign=[1, 1],
                               match=round(score, 3), viewer=None)
        else:
            origin = list(r["origin_px"])
        entry = dict(id=f"{name}", scan=name, chip=name.split("_")[0], material=material, oxide_nm=OXIDE[material],
                     rank=None, score=None, px_um=PX[100], w=int(r["rgba"].shape[1]), h=int(r["rgba"].shape[0]),
                     origin_px=[round(origin[0], 1), round(origin[1], 1)],
                     origin_um=[round(origin[0] * PX[100], 1), round(origin[1] * PX[100], 1)], stage_origin_um=None,
                     area=round(float((r["rgba"][..., 3] > 127).sum()) * PX[100] ** 2, 1),
                     d=max(t["d"] for t in r["terraces"]), cropped=r["touches"],
                     L=round(max(r["rgba"].shape[:2]) * PX[100], 1), W=round(min(r["rgba"].shape[:2]) * PX[100], 1),
                     source=os.path.relpath(files["col100"], a.src),
                     outline=r["outline"], terraces=[])
        r["terraces"] = [t for t in r["terraces"] if not (material == "hbn" and t["t"] < 2)] or r["terraces"][:1]
        for k, t in enumerate(r["terraces"]):
            t["quality"] = ";".join(q for q in t["quality"].split(";") if q)
            entry["terraces"].append(dict(t, id=f"{name}.{chr(97 + k)}", usable=True, terrace_id=None, score=None))
        flakes.append(entry)
        cut.append(r["rgba"])
        print(f"{name}: {material}, {len(entry['terraces'])} terraces "
              f"(5x/100x: {', '.join(str(t['t']) + '/' + str(t['t100']) for t in entry['terraces'])} nm), match {scans.get(name, {}).get('match')}")
    # pack: simple shelf packing on ATLAS-wide sheets
    pos, sheet, x, y, shelf = [], 0, 0, 0, 0
    for c in cut:
        h, w = c.shape[:2]
        if x + w > ATLAS:
            x, y, shelf = 0, y + shelf + 2, 0
        if y + h > ATLAS:
            sheet, x, y, shelf = sheet + 1, 0, 0, 0
        pos.append((sheet, x, y))
        x += w + 2
        shelf = max(shelf, h)
    n = max(p[0] for p in pos) + 1
    sheets = [np.zeros((ATLAS, ATLAS, 4), np.uint8) for _ in range(n)]
    used_h = [0] * n
    for f, c, (s, x, y) in zip(flakes, cut, pos):
        sheets[s][y:y + c.shape[0], x:x + c.shape[1]] = c
        used_h[s] = max(used_h[s], y + c.shape[0])
        f["atlas"], f["rect"] = s, [x, y, c.shape[1], c.shape[0]]
    for s, sh in enumerate(sheets):
        cv2.imwrite(str(out / f"flakes_{s:02d}.webp"), cv2.cvtColor(sh[:used_h[s]], cv2.COLOR_RGBA2BGRA),
                    [cv2.IMWRITE_WEBP_QUALITY, 86])
    cat = dict(version=2, demo=True,
               note="Demo: 100x images of individual flakes (unused flakes of chips BN19-22, g26-30). "
                    "Thicknesses estimated from 100x colour with the 6.3x camera calibration (not AFM-calibrated).",
               atlas_size=ATLAS, atlases=[f"flakes_{s:02d}.webp" for s in range(n)], scans=scans, flakes=flakes)
    (out / "catalog.json").write_text(json.dumps(cat, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"{len(flakes)} flakes, {sum(len(f['terraces']) for f in flakes)} terraces, {n} atlases -> {out}")


if __name__ == "__main__":
    main()
