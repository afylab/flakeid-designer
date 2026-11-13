#!/usr/bin/env python3
"""Designer catalog (tools/designer_page.html) from live-analysed Micro-Manager scans.

    python tools/make_catalog_live.py ~/FlakeResults/multitest/chip_1 ~/FlakeResults/multitest/chip_2 -o designer_multitest
    cd designer_multitest && python3 -m http.server 8000         # then open http://localhost:8000

Needs only the results folders (no tiles, no network share): flake / terrace outlines and thicknesses from the
viewer data, pixels from the viewer's full-resolution image tiles (the chip viewer's colours: counts / substrate x
0.31), the chip overview from its 1/4-scale level.  Thicknesses follow the measured calibrations used for the
target lists:
  graphite  1-4 layers from the measured green-contrast peaks (profiles/graphite_layers.json); bulk excluded
  hBN       terraces whose colour fits both ~20 and ~150 nm are decided thin / thick per chip (flakeid.branch);
            thick ones are kept in the catalog for context but marked unusable
Usable terraces: inscribed diameter >= --min-d, hBN <= --hbn-max-nm, no hard flag.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_afm_set import load_viewer_data  # noqa: E402
from make_catalog import ATLAS, MAX_SIDE, pack  # noqa: E402
from flakeid import branch as B  # noqa: E402
from flakeid import graphite as GR  # noqa: E402

TILE = 256                                  # (viewers before 2026-10-27; each viewer's own: zoomview.tile_size)
OV_LEVELS_DOWN = 2                   # overview: 1/4 of full resolution
UNUSABLE = {"thick_branch", "bulk", "no_signal", "out_of_range", "saturated"}


class Pyramid:
    """Read rectangles of one deep-zoom level of a chip viewer (tiles/image/<level>/<tx>_<ty>.jpg)."""

    def __init__(self, viewer: Path):
        d = viewer / "tiles" / "image"
        self.top = max(int(p.name) for p in d.iterdir() if p.name.isdigit())
        self.d = d
        self.cache = {}
        from flakeid.zoomview import tile_size
        self.T = tile_size(d / str(self.top), TILE)              # (the viewer's own: 256 before 2026-10-27, then 1024)

    def tile(self, level, tx, ty):
        k = (level, tx, ty)
        if k not in self.cache:
            a = cv2.imread(str(self.d / str(level) / f"{tx}_{ty}.jpg"))
            self.cache[k] = None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            if len(self.cache) > 600:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[k]

    def read(self, x0, y0, w, h, level=None):
        level = self.top if level is None else level
        out = np.zeros((h, w, 3), np.uint8)
        T = self.T
        for ty in range(max(y0, 0) // T, (y0 + h - 1) // T + 1):
            for tx in range(max(x0, 0) // T, (x0 + w - 1) // T + 1):
                t = self.tile(level, tx, ty)
                if t is None:
                    continue
                ax, ay = tx * T, ty * T
                sx0, sy0 = max(x0, ax), max(y0, ay)
                sx1, sy1 = min(x0 + w, ax + t.shape[1]), min(y0 + h, ay + t.shape[0])
                if sx1 > sx0 and sy1 > sy0:
                    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[sy0 - ay:sy1 - ay, sx0 - ax:sx1 - ax]
        return out

    def level_image(self, level):
        d = self.d / str(level)
        nx = 1 + max(int(p.stem.split("_")[0]) for p in d.glob("*.jpg"))
        ny = 1 + max(int(p.stem.split("_")[1]) for p in d.glob("*.jpg"))
        last = self.tile(level, nx - 1, ny - 1)
        return self.read(0, 0, (nx - 1) * self.T + last.shape[1], (ny - 1) * self.T + last.shape[0], level)


def fill(polys, x0, y0, w, h):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        a = np.round(np.array(p, float).reshape(-1, 2) - [x0, y0]).astype(np.int32)
        cv2.fillPoly(m, [a], 255)
    return m


def shift(polys, x0, y0):
    return [[round(p[i] - (x0 if i % 2 == 0 else y0), 1) for i in range(len(p))] for p in polys]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="live results folders (<name>_tiles_viewer, *_live_terraces.csv)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--min-d", type=float, default=8.0, help="min inscribed diameter of a usable terrace (um)")
    ap.add_argument("--hbn-max-nm", type=float, default=100.0)
    ap.add_argument("--max-per-scan", type=int, default=400, help="flakes per chip (graphene first, then by score)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flakes, cutouts, scans = [], [], {}
    for res in (Path(r).expanduser() for r in a.results):
        D = load_viewer_data(res)
        viewer = next(res.glob("*_viewer"))
        stem = D["name"].replace("_tiles", "")
        mat, ox, px = D["material"], float(D["oxide"]), float(D["pixel_um"])
        pyr = Pyramid(viewer)
        rows = {r["terrace_id"]: r for r in csv.DictReader(open(next(res.glob("*_live_terraces.csv"))))}
        T = {int(k): v for k, v in D["terraces"].items()}

        # measured-calibration thickness per terrace
        layers, thick_dec = {}, {}
        hard = {}                                       # flags added since older tables were written
        if mat == "graphite":
            from flakeid.thickness import ThicknessEstimator, load_profile
            cmin = ThicknessEstimator("graphite", ox, load_profile("bx51_xc10_mm_raw16")).curve.min(axis=0)
            for tid, r in rows.items():
                c = [float(r["contrast_r"]), float(r["contrast_g"]), float(r["contrast_b"])]
                if max(c) < -0.97:
                    hard[int(tid)] = "no_signal"
                elif c[1] < cmin[1] - 0.03 and c[2] / c[1] > 0.8:
                    hard[int(tid)] = "bulk"
            cal = GR.load_calibration()
            if cal and abs(ox - cal["oxide_nm"]) <= cal.get("oxide_tol_nm", 6.0):
                for tid, r in rows.items():
                    n = GR.layers_from_contrast([float(r["contrast_r"]), float(r["contrast_g"]), float(r["contrast_b"])], cal)
                    if n:
                        layers[int(tid)] = n
        else:
            bm = B.load("bx51_xc10_mm_raw16", "hbn", ox)
            if bm is not None:
                from flakeid.thickness import ThicknessEstimator, load_profile
                est = ThicknessEstimator("hbn", ox, load_profile("bx51_xc10_mm_raw16"))
                tb = []
                for tid, r in rows.items():
                    c = np.array([float(r["contrast_r"]), float(r["contrast_g"]), float(r["contrast_b"])])
                    dec = bm.decide(est, c)
                    if dec is not None:
                        tb.append(dict(r, quality=(r["quality"] + ";two_branch"), _thin=dec[1], _thick=dec[3]))
                res_dec = B.resolve(tb, bm.cal, log=lambda s: print(f"{stem}: {s}"))
                for r in tb:
                    thick = res_dec.get(r["terrace_id"]) == "thick"
                    thick_dec[int(r["terrace_id"])] = (thick, round(r["_thin"], 1), round(r["_thick"], 1))

        def terrace_entry(fid, k, t, x0, y0):
            q = set(filter(None, (t["q"] or "").split(";"))) - {"good"}
            if t["id"] in hard:
                q.add(hard[t["id"]])
            tt, alt, nl = t["t"], t["alt"], layers.get(t["id"])
            if nl:
                tt, alt = round(nl * GR.LAYER_NM, 2), ""
            if t["id"] in thick_dec:
                thick, tn, tk = thick_dec[t["id"]]
                tt, alt = (tk, f"{tn:g}") if thick else (tn, f"{tk:g}")
                q.add("thick_branch" if thick else "thin_branch")
            ok = (t["d"] >= a.min_d and not (q & UNUSABLE)
                  and not (mat == "hbn" and tt > a.hbn_max_nm))
            return dict(id=f"{fid}.{chr(97 + k) if k < 26 else k}", t=tt, tlo=tt if (nl or t["id"] in thick_dec) else t["tlo"],
                        thi=tt if (nl or t["id"] in thick_dec) else t["thi"], alt=alt,
                        layers=nl if nl else t["layers"], area=t["area"], d=t["d"], L=t["L"], W=t["W"],
                        c=[int(round(t["tx"] - x0)), int(round(t["ty"] - y0))], poly=shift(t["poly"], x0, y0),
                        quality=";".join(sorted(q)) or "good", score=t["score"], usable=bool(ok), terrace_id=t["id"])

        # flake order: graphene (1-4 layer terraces) first for graphite, then score
        def prio(f):
            nl = min((layers.get(int(i), 99) for i in f["terr"]), default=99)
            return (0 if nl < 99 else 1, -(f["score"] or 0))
        n_scan = 0
        for f in sorted(D["flakes"], key=prio):
            if n_scan >= a.max_per_scan:
                break
            pad = 6
            x0, y0 = int(f["bx"]) - pad, int(f["by"]) - pad
            w, h = int(f["bw"]) + 2 * pad, int(f["bh"]) + 2 * pad
            cropped = False
            if max(w, h) > MAX_SIDE:                        # window very large flakes around the best terrace
                bt = T[int(f["terr"][0])]
                w, h = min(w, MAX_SIDE), min(h, MAX_SIDE)
                x0, y0 = int(bt["tx"] - w / 2), int(bt["ty"] - h / 2)
                cropped = True
            fid = f"{stem}-f{f['id']:04d}"
            terr = [terrace_entry(fid, k, T[int(i)], x0, y0) for k, i in enumerate(f["terr"])]
            if not any(t["usable"] for t in terr):
                continue
            fpoly = [p for p in f["poly"] if len(p) >= 6] or [p for i in f["terr"] for p in T[int(i)]["poly"] if len(p) >= 6]
            if not fpoly:
                continue
            rgb = pyr.read(x0, y0, w, h)
            alpha = cv2.dilate(fill(fpoly, x0, y0, w, h), np.ones((3, 3), np.uint8))
            cutouts.append(np.dstack([rgb, alpha]))
            flakes.append(dict(id=fid, scan=stem, material=mat, oxide_nm=round(ox, 1), rank=n_scan + 1, score=f["score"],
                               px_um=px, w=w, h=h, origin_px=[x0, y0], origin_um=[round(x0 * px, 1), round(y0 * px, 1)],
                               stage_origin_um=None, stage_um=[f["sx"], f["sy"]], area=f["area"], d=f["d"], L=f["L"],
                               W=f["W"], focus_um=f.get("focus"), cropped=cropped, outline=max(shift(fpoly, x0, y0), key=len),
                               terraces=terr))
            n_scan += 1
        ov = pyr.level_image(pyr.top - OV_LEVELS_DOWN)
        cv2.imwrite(str(out / f"chip_{stem}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        scans[stem] = dict(file=f"chip_{stem}.jpg", w=ov.shape[1], h=ov.shape[0], scale=0.5 ** OV_LEVELS_DOWN, px_um=px,
                           material=mat, oxide_nm=round(ox, 1), stage_origin_um=None, stage_sign=[1.0, 1.0])
        nu = sum(t["usable"] for fl in flakes if fl["scan"] == stem for t in fl["terraces"])
        print(f"{stem} ({mat}, {ox:.1f} nm): {n_scan} flakes, {nu} usable terraces"
              + (f", {len(layers)} graphene terraces (1-4 layers)" if mat == "graphite" else ""))

    pos = pack([(c.shape[1], c.shape[0]) for c in cutouts])
    n_sheets = max(p[0] for p in pos) + 1
    sheets = [np.zeros((ATLAS, ATLAS, 4), np.uint8) for _ in range(n_sheets)]
    used_h = [0] * n_sheets
    for f, c, (s, x, y) in zip(flakes, cutouts, pos):
        sheets[s][y:y + c.shape[0], x:x + c.shape[1]] = c
        used_h[s] = max(used_h[s], y + c.shape[0])
        f["atlas"], f["rect"] = s, [x, y, c.shape[1], c.shape[0]]
    for s, sh in enumerate(sheets):
        cv2.imwrite(str(out / f"flakes_{s:02d}.webp"), cv2.cvtColor(sh[:used_h[s]], cv2.COLOR_RGBA2BGRA),
                    [cv2.IMWRITE_WEBP_QUALITY, 88])
    cat = dict(version=2, note="whole flakes cut out of 10x Micro-Manager tile scans (live analysis); graphene layer "
                               "numbers and hBN thin/thick decisions from measured calibrations.",
               atlas_size=ATLAS, atlases=[f"flakes_{s:02d}.webp" for s in range(n_sheets)], scans=scans, flakes=flakes)
    (out / "catalog.json").write_text(json.dumps(cat, separators=(",", ":")))
    if (ROOT / "tools" / "designer_page.html").is_file():   # (the old standalone page; not in the packaged designer app)
        shutil.copy(ROOT / "tools" / "designer_page.html", out / "index.html")
    print(f"{len(flakes)} flakes, {sum(len(f['terraces']) for f in flakes)} terraces, {n_sheets} atlases -> {out}")


if __name__ == "__main__":
    main()
