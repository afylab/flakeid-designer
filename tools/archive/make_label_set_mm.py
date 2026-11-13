#!/usr/bin/env python3
"""Flake-labelling set (tools/label_page.html) from a Micro-Manager scan analysed by find_flakes.

    python tools/make_label_set_mm.py "testing images/bigbntest_2_tiles" "testing images/bigbntest_2_results" \
        --flat flat.npz --img100 "testing images/bigbntest_2_100x" -o label_b2 [--n 400]
    python tools/build_label_page.py label_b2

One item per flake (its best terrace outlined), sampled to refine the hBN score: weighted towards the
scores where the choice of what to image is made (0.5-0.85), with some low and very high scores, spread
over thickness, plus every flake already imaged at 100x.  Colours: the designer_demo white balance.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_afm_set import load_viewer_data, to_display  # noqa: E402

CELL, GRID = 256, 8
# score bin -> share of the sample
SCORE_BINS = [(0.0, 0.3, 0.08), (0.3, 0.5, 0.14), (0.5, 0.65, 0.22), (0.65, 0.75, 0.22), (0.75, 0.85, 0.22), (0.85, 1.01, 0.12)]
T_EDGES = [0, 5, 15, 30, 50, 80, 120, 1000]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles")
    ap.add_argument("results")
    ap.add_argument("--flat")
    ap.add_argument("--img100")
    ap.add_argument("--cache", help="mosaic .npy from make_afm_set.py --cache")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    res = Path(a.results)
    D = load_viewer_data(res)
    fj = json.loads(next(res.glob("*_flakes.json")).read_text())
    terr_rows = {int(r["terrace_id"]): r for r in csv.DictReader(open(next(res.glob("*_terraces.csv"))))}
    flakes_csv = {int(r["flake_id"]): r for r in csv.DictReader(open(next(res.glob("*_flakes.csv"))))}
    scan = Path(a.tiles).name.replace("_tiles", "")
    imaged = set()
    if a.img100:
        for r in csv.DictReader(open(next(Path(a.img100).glob("*_100x_index.csv")))):
            imaged.add(int(r["label"].split("_F")[1].split("_")[0]))

    F = [f for f in D["flakes"] if f["score"] is not None]
    pick = {f["id"] for f in F if f["id"] in imaged}
    n_rest = max(0, a.n - len(pick))
    for lo, hi, share in SCORE_BINS:
        pool = [f for f in F if lo <= f["score"] < hi and f["id"] not in pick]
        k = int(round(share * n_rest))
        if len(pool) <= k:
            pick |= {f["id"] for f in pool}
            continue
        # spread over thickness: equal share per thickness bin present, then fill at random
        by_t = {}
        for f in pool:
            by_t.setdefault(int(np.searchsorted(T_EDGES, f["t"], side="right")), []).append(f)
        per = max(1, k // max(len(by_t), 1))
        chosen = []
        for b, fs in by_t.items():
            idx = rng.choice(len(fs), size=min(per, len(fs)), replace=False)
            chosen += [fs[i]["id"] for i in idx]
        left = [f["id"] for f in pool if f["id"] not in set(chosen)]
        if len(chosen) < k and left:
            chosen += list(rng.choice(left, size=min(k - len(chosen), len(left)), replace=False))
        pick |= set(chosen[:k])
    print(f"{len(pick)} flakes ({len(imaged & pick)} imaged at 100x)")

    if a.cache and Path(a.cache).exists():
        rgb = np.load(a.cache, mmap_mode="r")
    else:
        from flakeid.mm import load_dataset
        img, _ = load_dataset(a.tiles, flat=a.flat, verbose=False)
        rgb = img.rgb
    H, W = rgb.shape[:2]
    sub_ref = np.array(fj["params"]["substrate_rgb"], np.float32)
    T = {int(k): v for k, v in D["terraces"].items()}
    px_um = D["pixel_um"]
    items, crops = [], []
    FW, FH = (int(v) for v in fj["metadata"]["stage_map"]["frame_wh"])      # one 10x camera frame
    (out / "frames").mkdir(exist_ok=True)
    for f in sorted((f for f in F if f["id"] in pick), key=lambda f: f["id"]):
        t = T[int(f["terr"][0])]                   # best terrace first
        tr = terr_rows[t["id"]]
        side = int(np.clip(max(f["bw"], f["bh"]) * 1.8 + 50, 90, 700))
        cx, cy = f["x"], f["y"]
        x0 = int(np.clip(cx - side / 2, 0, max(W - side, 0)))
        y0 = int(np.clip(cy - side / 2, 0, max(H - side, 0)))
        c = np.asarray(rgb[y0:y0 + side, x0:x0 + side]).astype(np.float32)
        v = c.reshape(-1, 3)
        near = np.abs(v / sub_ref - 1).max(1) < 0.08
        sub = np.median(v[near], 0) if near.sum() > 200 else sub_ref
        d = to_display(c, sub)
        if d.shape[0] != side or d.shape[1] != side:
            pad = np.zeros((side, side, 3), np.uint8)
            pad[:d.shape[0], :d.shape[1]] = d
            d = pad
        s = CELL / side
        crops.append(cv2.resize(d, (CELL, CELL), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
        polys = [[round((p[i] - x0) * s if i % 2 == 0 else (p[i] - y0) * s, 1) for i in range(len(p))] for p in t["poly"]]
        # the main view: one full camera frame centred on the terrace, at the native scale (same for every flake)
        fx0, fy0 = int(round(t["tx"] - FW / 2)), int(round(t["ty"] - FH / 2))
        cx0, cy0 = max(fx0, 0), max(fy0, 0)
        fr = np.asarray(rgb[cy0:min(fy0 + FH, H), cx0:min(fx0 + FW, W)]).astype(np.float32)
        frame = np.zeros((FH, FW, 3), np.uint8)
        frame[cy0 - fy0:cy0 - fy0 + fr.shape[0], cx0 - fx0:cx0 - fx0 + fr.shape[1]] = to_display(fr, sub)
        fname = f"frames/F{f['id']:04d}.jpg"
        cv2.imwrite(str(out / fname), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        fpolys = [[round(p[i] - fx0 if i % 2 == 0 else p[i] - fy0, 1) for i in range(len(p))] for p in t["poly"]]
        num = lambda k, n=3: (round(float(tr[k]), n) if tr.get(k) not in (None, "", "nan") else None)   # noqa: E731
        items.append(dict(
            id=f"{scan}-F{f['id']:04d}", scan=scan, material=D["material"], csv_flake_id=f["id"], accepted=True,
            quality=t["q"], t=t["t"], tlo=t["tlo"], thi=t["thi"], alt=t["alt"], layers=t["layers"], area=t["area"], d=t["d"],
            L=t["L"], Wd=t["W"], x=round(t["tx"] * px_um, 1), y=round(t["ty"] * px_um, 1), sx=t["sx"], sy=t["sy"], c=t["c"],
            score=f["score"], n_terraces=f["n"], flake_area=f["area"], imaged100=f["id"] in imaged,
            metrics=dict(fill=num("fill"), fr_sol=num("fullres_solidity"), sol=num("solidity"), ew=num("edge_width_um", 2),
                         unif=num("uniformity"), res=num("fit_residual"), haze=num("ring_similar"), sat=num("saturated_frac")),
            field_um=round(side * px_um, 1), poly=polys,
            frame=dict(file=fname, w=FW, h=FH, field_um=round(FW * px_um, 1), poly=fpolys)))
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    crops = [crops[i] for i in order]
    per = GRID * GRID
    for k in range(0, len(crops), per):
        chunk = crops[k:k + per]
        rows = (len(chunk) + GRID - 1) // GRID
        sheet = np.zeros((rows * CELL, GRID * CELL, 3), np.uint8)
        for j, c in enumerate(chunk):
            r, q = divmod(j, GRID)
            sheet[r * CELL:(r + 1) * CELL, q * CELL:(q + 1) * CELL] = c
        cv2.imwrite(str(out / f"atlas_{k // per:02d}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        for j in range(len(chunk)):
            items[k + j]["atlas"] = k // per
            items[k + j]["cell"] = j
    sub = (f"{len(items)} flakes from {scan} · BX51 10x + XC10 (RAW16) · hBN on 285 nm SiO₂ · "
           f"weighted towards scores 0.5–0.85, plus every flake imaged at 100x")
    (out / "items.json").write_text(json.dumps(dict(cell=CELL, grid=GRID, items=items, subtitle=sub, title=f"{scan} Flake Labels"),
                                               separators=(",", ":")))
    print(f"wrote {len(items)} items, {(len(crops) + per - 1) // per} atlases to {out}")


if __name__ == "__main__":
    main()
