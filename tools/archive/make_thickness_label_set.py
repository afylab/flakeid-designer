#!/usr/bin/env python3
"""hBN thin / thick labelling set (tools/thickness_page.html) from a live-analysed Micro-Manager scan.

    python tools/make_thickness_label_set.py "<run>/chip_1_tiles" ~/FlakeResults/multitest/chip_1 \
        --flat "example images/micromanager_test/flat_10x_3ms.npz" -o thick_labels_chip1
    python tools/make_thickness_label_set.py --page thick_labels_chip1        # (re)build index.html only

At 10x on ~300 nm SiO2, ~20 nm and ~150 nm hBN have nearly the same median colour, so the colour fit cannot
tell them apart.  This set collects the user's thin / thick judgement (made routinely by eye at the microscope)
on the flakes where the two readings compete, plus thin flakes of similar colour as references, so that the
features that separate them (edge contrast, colour saturation, texture) can be learned from labels instead of
the thin-film model.  The analysis's thickness is NOT shown on the page (blind labelling).

Images come straight from the raw 10x tiles (flat-field corrected, never the stitched mosaic), rendered as the
chip viewer renders them: counts / substrate x 0.31 (substrate neutral grey, white at ~3.2x the substrate).
Each item has a close-up (the flake at ~2.4x its size) and a 600 um context view, both packed into atlases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_afm_set import load_viewer_data  # noqa: E402
from flakeid.hundred import _mode_rgb  # noqa: E402
from flakeid.mm import full_scale, prepare_flat, read_dataset, read_frame  # noqa: E402
from flakeid.thickness import ThicknessEstimator, load_profile  # noqa: E402

CLOSE, CTX = 480, 360           # atlas cell sizes (px)
GRID = 4                        # cells per atlas row / column
CTX_UM = 600.0
DISPLAY = 0.31                  # chip viewer: substrate at 0.31 of full scale


def render(lin, sub):
    return np.clip(lin / sub[None, None, :] * DISPLAY * 255.0 + 0.5, 0, 255).astype(np.uint8)


def pick(D, est, min_d, gap, n_ref, rng):
    """Ambiguous flakes (thin and thick readings within `gap` of each other) + thin references of similar colour."""
    T = {int(k): v for k, v in D["terraces"].items()}
    amb, ref = [], []
    for f in D["flakes"]:
        t = T[int(f["terr"][0])]
        if t["d"] < min_d:
            continue
        c = np.array(t["c"], float)
        tn, dn = est.best_in_range(c, 0, 60)
        tk, dk = est.best_in_range(c, 110, 250)
        rec = dict(f=f, t=t, thin=round(tn, 1), d_thin=round(dn, 3), thick=round(tk, 1), d_thick=round(dk, 3))
        if min(dn, dk) <= 0.15 and abs(dn - dk) <= gap:
            amb.append(rec)
        elif dk - dn > 0.3 and dn <= 0.12 and 10 <= tn <= 40:
            ref.append(rec)
    if amb:
        g = np.array([r["t"]["c"][1] for r in amb])
        ref = [r for r in ref if g.min() - 0.3 <= r["t"]["c"][1] <= g.max() + 0.3]
    if len(ref) > n_ref:
        ref = [ref[i] for i in rng.choice(len(ref), n_ref, replace=False)]
    for r in amb:
        r["group"] = "ambiguous"
    for r in ref:
        r["group"] = "reference"
    return amb + ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiles", nargs="?")
    ap.add_argument("results", nargs="?", help="live results folder (<name>_tiles_viewer/index.html, *_flakes.json)")
    ap.add_argument("--flat")
    ap.add_argument("--profile", default="bx51_xc10_mm_raw16")
    ap.add_argument("--min-d", type=float, default=15.0, help="smallest best-terrace width (um)")
    ap.add_argument("--gap", type=float, default=0.2, help="thin and thick readings within this contrast distance")
    ap.add_argument("--n-ref", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--page", help="only (re)build <folder>/index.html from its items.json")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    if a.page:
        build_page(Path(a.page))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    res = Path(a.results).expanduser()
    D = load_viewer_data(res)
    est = ThicknessEstimator("hbn", D["oxide"], load_profile(a.profile))
    sel = pick(D, est, a.min_d, a.gap, a.n_ref, rng)
    print(f"{sum(r['group'] == 'ambiguous' for r in sel)} ambiguous + {sum(r['group'] == 'reference' for r in sel)} reference flakes")

    sm = D["stage_map"]
    q = np.array(sm["q"], float)
    W, H = (int(v) for v in sm["wh"])
    P = np.array(sm["P"], float)
    tiles = read_dataset(a.tiles)
    st = np.array([t.stage_xy for t in tiles], float)
    tile_of = [int(np.argmin(((st - p) ** 2).sum(1))) for p in P]        # stage map row -> tile
    flat, _ = prepare_flat(a.flat, tiles[0], full_scale(tiles[0])) if a.flat else (None, 1.0)
    px_um = D["pixel_um"]
    cache = {}

    def frame(k):
        if k not in cache:
            tl = tiles[tile_of[k]]
            lin = read_frame(tl)
            if getattr(tl, "glitch", None):
                cache[k] = None
            else:
                lin = flat.apply(lin) if flat is not None else lin
                cache[k] = (lin, _mode_rgb(lin[::4, ::4], step=8)[0])
            if len(cache) > 24:
                cache.pop(next(iter(cache)))
        return cache[k]

    def crop(cx, cy, side):
        """side x side px around mosaic (cx, cy) from the one tile that covers most of it."""
        x0, y0 = cx - side / 2, cy - side / 2
        ov = (np.clip(np.minimum(q[:, 0] + W, x0 + side) - np.maximum(q[:, 0], x0), 0, None)
              * np.clip(np.minimum(q[:, 1] + H, y0 + side) - np.maximum(q[:, 1], y0), 0, None))
        for k in np.argsort(-ov)[:3]:
            fr = frame(int(k))
            if fr is None:
                continue
            lin, sub = fr
            ix0, iy0 = int(round(x0 - q[k, 0])), int(round(y0 - q[k, 1]))
            img = np.zeros((side, side, 3), np.uint8)
            sx0, sy0 = max(ix0, 0), max(iy0, 0)
            sx1, sy1 = min(ix0 + side, W), min(iy0 + side, H)
            if sx1 > sx0 and sy1 > sy0:
                img[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = render(lin[sy0:sy1, sx0:sx1], sub)
            return img, (x0, y0), float(ov[k]) / side ** 2
        return None, (x0, y0), 0.0

    items, close, ctx = [], [], []
    order = rng.permutation(len(sel))
    for n, i in enumerate(order):
        r = sel[i]
        f, t = r["f"], r["t"]
        # close-up framed on the terrace in question (outlined), the rest of the flake outlined faintly
        tp = np.concatenate([np.array(p, float).reshape(-1, 2) for p in t["poly"]])
        tx0, ty0 = tp.min(0)
        tx1, ty1 = tp.max(0)
        tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
        side = int(np.clip(max(tx1 - tx0, ty1 - ty0) * 3.0 + 30, 110, 460))
        img, (x0, y0), cov = crop(tcx, tcy, side)
        if img is None:
            continue
        s = CLOSE / side
        close.append(cv2.resize(img, (CLOSE, CLOSE), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
        sc = lambda polys, ox, oy, k: [[round((p[j] - ox) * k if j % 2 == 0 else (p[j] - oy) * k, 1)   # noqa: E731
                                         for j in range(len(p))] for p in polys]
        polys, fpolys = sc(t["poly"], x0, y0, s), sc(f["poly"], x0, y0, s)
        cs = int(round(CTX_UM / px_um))
        cimg, (cx0, cy0), _ = crop(tcx, tcy, cs)
        ctx.append(cv2.resize(cimg, (CTX, CTX), interpolation=cv2.INTER_AREA))
        cpolys = sc(t["poly"], cx0, cy0, CTX / cs)
        items.append(dict(id=f"{D['name']}-F{f['id']:04d}", flake=f["id"], group=r["group"], d=t["d"],
                          area=f["area"], field_um=round(side * px_um), ctx_um=CTX_UM, coverage=round(cov, 2),
                          c=t["c"], thin_nm=r["thin"], thick_nm=r["thick"], d_thin=r["d_thin"], d_thick=r["d_thick"],
                          sx=f["sx"], sy=f["sy"], poly=polys, fpoly=fpolys, cpoly=cpolys))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(sel)}")
    per = GRID * GRID
    for name, imgs, cell in (("close", close, CLOSE), ("ctx", ctx, CTX)):
        for k in range(0, len(imgs), per):
            chunk = imgs[k:k + per]
            rows = (len(chunk) + GRID - 1) // GRID
            sheet = np.zeros((rows * cell, GRID * cell, 3), np.uint8)
            for j, c in enumerate(chunk):
                rr, qq = divmod(j, GRID)
                sheet[rr * cell:(rr + 1) * cell, qq * cell:(qq + 1) * cell] = c
            cv2.imwrite(str(out / f"{name}_{k // per:02d}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    for j, it in enumerate(items):
        it["atlas"], it["cell"] = divmod(j, per)
    scan = D["name"].replace("_tiles", "")
    meta = dict(title=f"{scan} hBN Thickness Labels", scan=scan, oxide_nm=round(D["oxide"], 1),
                close=CLOSE, ctx=CTX, grid=GRID, px_um=px_um, items=items)
    (out / "items.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote {len(items)} items, {(len(items) + per - 1) // per} atlases of each view to {out}")
    build_page(out)


def build_page(d: Path):
    meta = json.loads((d / "items.json").read_text())
    # the page shows no analysis thickness: strip what could bias the label
    pub = dict(meta, items=[{k: v for k, v in it.items() if k in ("id", "flake", "d", "field_um", "ctx_um", "atlas", "cell",
                                                                    "poly", "fpoly", "cpoly", "coverage")} for it in meta["items"]])
    tpl = (Path(__file__).parent / "thickness_page.html").read_text()
    html = tpl.replace("__TITLE__", meta["title"]).replace("__DATA__", json.dumps(pub, separators=(",", ":")))
    (d / "index.html").write_text(html)
    print("wrote", d / "index.html")


if __name__ == "__main__":
    main()
