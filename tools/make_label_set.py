#!/usr/bin/env python3
"""Build the image set for the flake-labelling page.

    python tools/make_label_set.py --images "example images" -o label_set

Runs the pipeline with rejected regions included, samples a stratified set of
regions per scan (mostly accepted ones, plus regions rejected by exactly one
quality cut, so that both false positives and false negatives are measured),
and writes:

    label_set/atlas_NN.jpg   crops, 256x256 each, 8x8 per atlas
    label_set/items.json     one record per crop (metadata, outline polygon)

tools/label_page.html is the labelling UI; tools/build_label_page.py merges it
with items.json into a publishable page.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid import find_flakes  # noqa: E402
from flakeid.io import load_image  # noqa: E402

CELL, GRID = 256, 8
SOFT_OK = {"good", "saturated", "ambiguous_thickness", "low_confidence", "chip_edge"}
ONE_FLAG = {"blurry_edges", "not_flat", "irregular_shape", "in_haze", "colour_mismatch", "non_uniform",
            "poor_colour_match", "low_contrast", "indistinct", "enclosed_in_thick"}
SCANS = [("g31", "graphite", 90), ("g32", "graphite", 90), ("g33", "graphite", 90), ("g34", "graphite", 90),
         ("g35", "graphite", 90), ("bn24", "hbn", 285), ("bn25", "hbn", 285), ("bn26", "hbn", 285),
         ("fullchipbn23", "hbn", 285)]


def outline(result, f, x0, y0, s):
    """Polygon of region f in crop coordinates (crop origin x0,y0, scale s px_crop/px_orig)."""
    labels, ws = result._labels, result._scale
    rid = result._region_of[f.terrace_id]
    bx, by = int(f.bbox_x_px * ws) - 2, int(f.bbox_y_px * ws) - 2
    ex, ey = bx + int(f.bbox_w_px * ws) + 4, by + int(f.bbox_h_px * ws) + 4
    bx, by = max(bx, 0), max(by, 0)
    sub = (labels[by:ey, bx:ex] == rid).astype(np.uint8)
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        c = cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2).astype(np.float32)
        c = ((c + [bx + 0.5, by + 0.5]) / ws - 0.5 - [x0, y0]) * s
        polys.append(np.round(c, 1).ravel().tolist())
    return polys


def bin_of(f, mat):
    t = f.thickness_nm
    edges = [0, 1.2, 5, 10, 15, 25.1] if mat == "graphite" else [0, 10, 20, 40, 60, 80, 100.1]
    return int(np.digitize(t, edges))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="example images")
    ap.add_argument("-o", "--out", default="label_set")
    ap.add_argument("--per-scan", type=int, default=44)
    ap.add_argument("--rejected-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    items, crops = [], []
    for stem, mat, ox in SCANS:
        img = load_image(Path(a.images) / f"{stem}.jpg")
        res = find_flakes(img, mat, ox, include_rejected=True)
        tmin, tmax = res.params["thickness_range_nm"]
        # flake ids of the default (accepted-only) CSV, matched by target position
        ref = list(csv.DictReader(open(Path(a.images) / f"{stem}_terraces.csv")))
        ref_xy = np.array([[float(r["target_x_px"]), float(r["target_y_px"])] for r in ref]) if ref else np.zeros((0, 2))
        acc, rej = [], []
        for f in res.terraces:
            q = set(f.quality.split(";"))
            if not (q - SOFT_OK):
                acc.append(f)
            elif len(q & ONE_FLAG) == 1 and not (q - SOFT_OK - ONE_FLAG) and f.area_um2 >= 150 \
                    and tmin <= f.thickness_nm <= tmax:
                rej.append(f)
        n_rej = int(round(a.per_scan * a.rejected_frac))
        n_acc = a.per_scan - n_rej

        def sample(pool, n):
            # stratify by thickness bin, weight larger flakes a little (sqrt area)
            if len(pool) <= n:
                return list(pool)
            bins = {}
            for f in pool:
                bins.setdefault(bin_of(f, mat), []).append(f)
            picked = []
            keys = sorted(bins)
            quota = {k: max(1, n * len(bins[k]) // len(pool)) for k in keys}
            # at least a few from every non-empty bin
            for k in keys:
                quota[k] = max(quota[k], min(4, len(bins[k])))
            while sum(quota.values()) > n:
                k = max(quota, key=lambda k: quota[k])
                quota[k] -= 1
            for k in keys:
                pl = bins[k]
                w = np.sqrt([f.area_um2 for f in pl])
                idx = rng.choice(len(pl), size=min(quota[k], len(pl)), replace=False, p=w / w.sum())
                picked += [pl[i] for i in idx]
            return picked

        chosen = [(f, True) for f in sample(acc, n_acc)] + [(f, False) for f in sample(rej, n_rej)]
        rgb = img.to_uint8()
        H, W = rgb.shape[:2]
        for f, is_acc in chosen:
            side = max(f.bbox_w_px, f.bbox_h_px) * 1.8 + 50
            side = int(np.clip(side, 90, 700))
            cx, cy = f.x_px, f.y_px
            x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
            y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
            c = rgb[y0:y0 + side, x0:x0 + side]
            if c.shape[0] != side or c.shape[1] != side:
                pad = np.zeros((side, side, 3), np.uint8)
                pad[:c.shape[0], :c.shape[1]] = c
                c = pad
            s = CELL / side
            c = cv2.resize(c, (CELL, CELL), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            crops.append(c)
            csv_id = None
            if len(ref_xy):
                d = np.hypot(*(ref_xy - [f.target_x_px, f.target_y_px]).T)
                k = int(np.argmin(d))
                if d[k] < 3:
                    csv_id = int(ref[k]["terrace_id"])
            items.append(dict(
                id=f"{stem}-{len(items):03d}", scan=stem, material=mat, csv_flake_id=csv_id,
                accepted=is_acc, quality=f.quality, t=f.thickness_nm, tlo=f.thickness_min_nm, thi=f.thickness_max_nm,
                alt=f.alt_thickness_nm, layers=f.layers, area=round(f.area_um2, 1), d=round(f.inscribed_diameter_um, 1),
                L=round(f.length_um, 1), Wd=round(f.width_um, 1),
                x=round(f.target_x_um, 1), y=round(f.target_y_um, 1),
                sx=None if f.stage_x_um is None else round(f.stage_x_um, 1),
                sy=None if f.stage_y_um is None else round(f.stage_y_um, 1),
                c=[round(f.contrast_r, 3), round(f.contrast_g, 3), round(f.contrast_b, 3)],
                metrics=dict(fill=round(f.fill, 3), fr_sol=round(f.fullres_solidity, 3), sol=round(f.solidity, 3),
                             ew=round(f.edge_width_um, 2) if f.edge_width_um == f.edge_width_um else None,
                             unif=round(f.uniformity, 3), res=round(f.fit_residual, 3),
                             haze=round(f.ring_similar, 3), sat=round(f.saturated_frac, 3)),
                field_um=round(side * img.pixel_um, 1), poly=outline(res, f, x0, y0, s)))
        print(f"{stem}: {sum(1 for _, a_ in chosen if a_)} accepted + {sum(1 for _, a_ in chosen if not a_)} rejected")

    # shuffle so that scans / accepted-rejected are interleaved
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    crops = [crops[i] for i in order]
    per = GRID * GRID
    for k in range(0, len(crops), per):
        sheet = np.zeros((GRID * CELL, GRID * CELL, 3), np.uint8)
        for j, c in enumerate(crops[k:k + per]):
            r_, c_ = divmod(j, GRID)
            sheet[r_ * CELL:(r_ + 1) * CELL, c_ * CELL:(c_ + 1) * CELL] = c
        rows = (len(crops[k:k + per]) + GRID - 1) // GRID
        cv2.imwrite(str(out / f"atlas_{k // per:02d}.jpg"), cv2.cvtColor(sheet[:rows * CELL], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    for i, it in enumerate(items):
        it["atlas"], it["cell"] = divmod(i, per)
    (out / "items.json").write_text(json.dumps(dict(cell=CELL, grid=GRID, items=items), separators=(",", ":")))
    print(f"{len(items)} items, {(len(items) + per - 1) // per} atlases -> {out}")


if __name__ == "__main__":
    main()
