"""Features of the target terrace in 100x images (mm_chip_scan.bsh targets mode), for ranking flakes.

Each RAW16 frame is dark-subtracted, divided by the fixed optical pattern (tools/optics_flat.py) and
normalised to the bare substrate, giving the contrast C = I / I_substrate - 1 per channel.  The frame
is centred on the best terrace found at 10x, so the terrace is the region of the centre's colour that
contains the centre.  Measured on it:

  area_um2, d_um         area and inscribed diameter (in the frame; border = cut by the frame edge)
  hole_frac, n_holes     area / number of spots inside the terrace outline that differ from its colour
                         (bubbles, residue specks, debris)
  texture                fine-scale roughness of the terrace colour (residue films, wrinkles), / noise
  step_frac              flake material just outside the terrace whose colour is close to it
                         (terraces a few layers thicker or thinner: steps)
  edge_width_um          10-90 % width of the terrace edge (focus)
  sharp                  normalised gradient energy of the whole frame (focus)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flakeid.mm import BAYER_CODE  # noqa: E402

UM_PX = 1.0894 / 9.54


def load_contrast(path, flat, sub_ref, dark=128.0):
    """Contrast image (float32 HxWx3) and the substrate level used."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if hasattr(flat, "apply"):           # a measured flat field (flakeid.mm.FlatField: dark + optics pattern)
        rgb = flat.apply(cv2.cvtColor(raw, BAYER_CODE).astype(np.float32))
    else:                                # an estimated pattern (tools/optics_flat.py) and a fixed black level
        rgb = (cv2.cvtColor(raw, BAYER_CODE).astype(np.float32) - dark) / flat
    sat = raw >= 4094
    # substrate: pixels whose chromaticity is within 5 % of the reference substrate
    s = rgb[::4, ::4].reshape(-1, 3)
    chrom = s / np.maximum(s.sum(1, keepdims=True), 1)
    ref = np.asarray(sub_ref, np.float32)
    rc = ref / ref.sum()
    near = (np.abs(chrom - rc) / rc).max(1) < 0.05
    b = s[near].sum(1) / ref.sum()
    near2 = near.copy()
    if near.mean() > 0.03:
        med = np.median(b)
        near2[near] = np.abs(b - med) < 0.05
    sub = np.median(s[near2], 0) if near2.mean() > 0.03 else ref
    return rgb / sub - 1.0, sub, float(near2.mean()), float(sat.mean())


def terrace_features(C, sat_frac=0.0):
    H, W = C.shape[:2]
    Cs = cv2.GaussianBlur(C, (0, 0), 2.0)
    # terrace colour: most common colour in the central 20 um
    r = int(10 / UM_PX)
    win = Cs[H // 2 - r:H // 2 + r, W // 2 - r:W // 2 + r].reshape(-1, 3)
    q = np.round(win / 0.01).astype(np.int32)
    keys, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    ct = win[inv == cnt.argmax()].mean(0)
    # refine: median of pixels near that colour
    d = np.abs(win - ct).max(1)
    ct = np.median(win[d < 0.03 + 0.02 * np.abs(ct).max()], 0)
    tol = 0.025 + 0.015 * float(np.abs(ct).max())
    dist = np.abs(Cs - ct).max(2)
    m = (dist < tol).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    # component containing the centre, else the largest within 15 um of it
    c0 = lab[H // 2, W // 2]
    if c0 == 0:
        yy, xx = np.nonzero(lab[H // 2 - r:H // 2 + r, W // 2 - r:W // 2 + r])
        if len(yy) == 0:
            return None
        cc = lab[H // 2 - r:H // 2 + r, W // 2 - r:W // 2 + r][yy, xx]
        c0 = np.bincount(cc).argmax()
    terr = lab == c0
    # outline without holes; holes = spots inside that differ from the terrace colour
    ff = terr.astype(np.uint8).copy()
    cnts, _ = cv2.findContours(ff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(ff)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    filled = filled.astype(bool)
    holes = filled & ~terr
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
    big = hst[1:, cv2.CC_STAT_AREA] * UM_PX ** 2 >= 0.5 if nh > 1 else np.array([])
    area_px = int(terr.sum())
    dt = cv2.distanceTransform(np.pad(filled.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    dt_t = cv2.distanceTransform(np.pad(terr.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    border = bool(terr[0].any() or terr[-1].any() or terr[:, 0].any() or terr[:, -1].any())
    # texture: fine-scale deviations inside the terrace (away from its edge), relative to substrate noise
    inner = (dt_t > 6)
    hp = cv2.GaussianBlur(C, (0, 0), 1.0) - cv2.GaussianBlur(C, (0, 0), 6.0)
    tex = float(np.median(np.abs(hp[inner]).max(1))) if inner.sum() > 200 else np.nan
    subm = np.abs(Cs).max(2) < 0.03
    subm = cv2.erode(subm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    noise = float(np.median(np.abs(hp[subm]).max(1))) if subm.sum() > 2000 else np.nan
    # steps: flake (not substrate) pixels in a 2-8 um ring around the terrace with colour near it
    ring = cv2.dilate(filled.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (141, 141))).astype(bool)
    ring &= ~cv2.dilate(filled.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))).astype(bool)
    flake = np.abs(Cs).max(2) > 0.06
    ringf = ring & flake
    close = ringf & (dist < 6 * tol) & (dist >= tol)
    step_frac = float(close.sum() / max(ring.sum(), 1))
    flake_ring = float(ringf.sum() / max(ring.sum(), 1))
    # edge width: profile of the colour distance across the terrace outline
    g = cv2.Sobel(dist, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(dist, cv2.CV_32F, 0, 1) ** 2
    edge = cv2.morphologyEx(terr.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool) & ~holes
    step = np.median(np.abs(Cs[ring & flake] - ct).max(1)) if (ring & flake).sum() > 100 else \
        np.median(np.abs(Cs[ring]).max(1) + np.abs(ct).max()) if ring.sum() else np.nan
    gmax = float(np.percentile(np.sqrt(g[edge]), 75)) / 8.0 if edge.sum() > 50 else np.nan     # Sobel -> per px
    edge_w = float(0.8 * max(float(np.max(np.abs(ct))), 0.05) / max(gmax, 1e-6) * UM_PX) if gmax == gmax else np.nan
    gray = C.mean(2)
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F)
    sharp = float(np.sqrt(np.mean(lap[flake] ** 2)) / max(np.abs(gray[flake]).mean(), 1e-3)) if flake.sum() > 500 else np.nan
    return dict(
        tc_r=float(ct[0]), tc_g=float(ct[1]), tc_b=float(ct[2]), cmag=float(np.linalg.norm(ct)),
        area_um2=area_px * UM_PX ** 2, d_um=2 * float(dt_t.max()) * UM_PX, d_filled_um=2 * float(dt.max()) * UM_PX,
        border=border, hole_frac=float(holes.sum() / max(filled.sum(), 1)), n_holes=int(big.sum()),
        texture=tex, noise=noise, texture_rel=tex / noise if noise == noise and noise > 0 else np.nan,
        step_frac=step_frac, flake_ring=flake_ring, edge_width_um=edge_w, sharp=sharp, sat_frac=sat_frac,
    ), dict(terr=terr, holes=holes, close=close, ct=ct, tol=tol)


def display(C, gain=(220, 151, 192)):
    """8-bit RGB with the substrate at the designer_demo substrate colour."""
    return np.clip((C + 1.0) * np.asarray(gain, np.float32), 0, 255).astype(np.uint8)


def overlay(C, masks, scale=0.5):
    img = display(C)
    out = img.copy()
    t = masks["terr"].astype(np.uint8)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)
    out[masks["holes"]] = (255, 0, 255)
    out[masks["close"]] = (0.5 * out[masks["close"]] + [0, 90, 127]).astype(np.uint8)
    return cv2.resize(np.hstack([img, out]), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def extract(folder, flat, sub_ref, flakes_csv=None):
    folder = Path(folder)
    idx = list(csv.DictReader(open(next(folder.glob("*_100x_index.csv")))))
    fl = {int(r["flake_id"]): r for r in csv.DictReader(open(flakes_csv))} if flakes_csv else {}
    rows, masks = [], {}
    for r in idx:
        C, sub, subfrac, sat = load_contrast(folder / r["file"], flat, sub_ref)
        out = terrace_features(C, sat)
        fid = int(r["label"].split("_F")[1].split("_")[0])
        f10 = fl.get(fid, {})
        base = dict(id=Path(r["file"]).stem, label=r["label"], flake=fid, focus=r.get("focus", ""),
                    af=float(r["af_ratio"]) if r.get("af_ratio") not in (None, "", "nan") else np.nan,
                    sub_frac=subfrac, thickness=float(f10.get("best_thickness_nm", "nan") or "nan"),
                    d10=float(f10.get("best_terrace_d_um", "nan") or "nan"), score10=float(f10.get("score", "nan") or "nan"),
                    area10=float(f10.get("area_um2", "nan") or "nan"), quality10=f10.get("quality", ""),
                    stage_x=float(r["stage_x_um"]), stage_y=float(r["stage_y_um"]))
        if out is None:
            base["failed"] = True
            rows.append(base)
            continue
        feats, mk = out
        base.update(feats)
        rows.append(base)
        masks[base["id"]] = (C, mk)
    return rows, masks
