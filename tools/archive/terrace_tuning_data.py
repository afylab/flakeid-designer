#!/usr/bin/env python3
"""Data for the terrace-threshold judging page: the step-line terraces (tools/terraces100.py) of a set of hBN flakes for
every combination of the thresholds, with the flakes' pictures.

    python tools/terrace_tuning_data.py <run folder> -o <out> [--flakes 3-f0013,3-f0081,...] [--workers 6]

Writes <out>/data.json (per flake: terrace a, the crystal direction, and per threshold combination the step lines and
terraces in the picture's pixels) and <out>/img/<flake>_{nat,c8,dt}.jpg (half resolution: natural colour, high
contrast, thickness change from terrace a, +-1.5 nm).
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terraces100 as T  # noqa: E402

GRID = dict(step_min=[0.08, 0.12, 0.18, 0.25], join_nm=[0.10, 0.15, 0.25], rim_um=[2.0, 4.0], win_um=[4.0, 6.0, 10.0])
DEFAULT = dict(step_min=T.STEP_MIN_NM, join_nm=T.LEVEL_JOIN_NM, rim_um=T.RIM_UM, win_um=T.STEP_WIN_UM)


def key(c):
    return f"s{c['step_min']:g}_j{c['join_nm']:g}_r{c['rim_um']:g}_w{c['win_um']:g}"


def picture(run, f):
    scan, num = f["id"].split(":")[-1].rsplit("-f", 1)
    c = [q for q in glob.glob(str(Path(run) / scan / "100x" / f"*_F{int(num):03d}_*.jpg")) if not re.search(r"_(c8|sg|dt|avg|debris)\.jpg$", q)]
    return Path(c[0]) if c else None


def one(job):
    run, f, out = job
    img = picture(run, f)
    if img is None:
        return None
    main = f["terraces"][0]
    P = T.prepare(img, f, main)
    if P is None:
        return None
    name = f["id"].split(":")[-1]
    (Path(out) / "img").mkdir(parents=True, exist_ok=True)
    q = [cv2.IMWRITE_JPEG_QUALITY, 86]
    cv2.imwrite(str(Path(out) / "img" / f"{name}_nat.jpg"), P["h"], q)
    c8p = Path(str(img).replace(".jpg", "_c8.jpg"))
    c8 = cv2.imread(str(c8p)) if c8p.is_file() else None
    if c8 is not None:
        cv2.imwrite(str(Path(out) / "img" / f"{name}_c8.jpg"), cv2.resize(c8, (P["h"].shape[1], P["h"].shape[0]), interpolation=cv2.INTER_AREA), q)
    dv = cv2.applyColorMap(np.clip(128 + (P["d"] - np.median(P["d"][P["A"] > 0])) / 1.5 * 127, 0, 255).astype(np.uint8),
                           cv2.COLORMAP_TWILIGHT_SHIFTED)
    dv[P["F"] == 0] = (dv[P["F"] == 0] * 0.25).astype(np.uint8)
    cv2.imwrite(str(Path(out) / "img" / f"{name}_dt.jpg"), dv, q)
    off = P["off"]

    def fr(poly):                                                    # cut-out px -> picture px, 1 decimal
        a = np.array(poly, float).reshape(-1, 2) + off
        return [round(v, 1) for v in a.ravel().tolist()]
    combos = {}
    for win in GRID["win_um"]:
        cands = T.step_candidates(P["d"], P["inner"], P["px"], P["F"], win_um=win, floor=min(GRID["step_min"]))
        for sm, rim in itertools.product(GRID["step_min"], GRID["rim_um"]):
            lines = T.select_steps(cands, P["d"].shape, P["px"], step_min=sm, rim_um=rim)
            ln = [[round(float(p0[0]), 1), round(float(p0[1]), 1), round(float(p1[0]), 1), round(float(p1[1]), 1), round(float(s_), 2)]
                  for s_, a_, p0, p1, L in lines]
            for jn in GRID["join_nm"]:
                ter = T.terraces_from(P, lines, join_nm=jn, rim_um=rim, rough_nm=1e9)     # residue filter: on the page
                c = dict(step_min=sm, join_nm=jn, rim_um=rim, win_um=win)
                combos[key(c)] = dict(lines=ln, ter=[[t["t"], t["area"], t["step_nm"], fr(t["poly"]), t["features"]["rough"]] for t in ter])
    return dict(key=name, id=f["id"], label=f"{name.replace('-f', ' · flake ')}", t_a=main["t"], w=int(P["h"].shape[1]),
                h=int(P["h"].shape[0]), px_um=P["px"], crystal_deg=None if P["crystal_deg"] is None else round(P["crystal_deg"], 1),
                crystal_frac=round(P["crystal_edge_frac"], 2), a_poly=fr(main["poly"]), outline=fr(f["outline"]),
                c8=c8 is not None, combos=combos)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--flakes", help="comma-separated flake ids (the end of the designer id, e.g. 3-f0013)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    b = json.loads((Path(a.run) / "designer" / "bundle.json").read_text())
    want = a.flakes.split(",") if a.flakes else None
    fl = [f for f in b["flakes"] if f["material"] == "hbn" and (want is None or any(f["id"].endswith(w) for w in want))]
    if want:
        fl.sort(key=lambda f: next(i for i, w in enumerate(want) if f["id"].endswith(w)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        res = [r for r in pool.map(one, [(a.run, f, str(out)) for f in fl]) if r]
    data = dict(version="v3", run=Path(a.run).name, grid=GRID, default=DEFAULT, flakes=res)
    (out / "data_v3.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(res)} flakes, {len(next(iter(res))['combos']) if res else 0} combinations each -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
