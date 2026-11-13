#!/usr/bin/env python3
"""Flake vs tape-residue labelling set (tools/residue_page.html) from live-analysed chips.

    python tools/make_residue_label_set.py ~/FlakeResults/multitest/chip_1 ~/FlakeResults/multitest/chip_2 -o OUT [--n 160]

Detections (flakes of the live table) are sampled per chip, weighted towards the faint and low-score ones where tape
residue is mistaken for flakes, plus a share of confident ones as references.  Crops come from the chip viewer's own
image tiles (as the user sees the chip), with the flake outlined: a close-up and a 600 um context view, packed into
atlases.  The page asks: flake / tape residue / other junk (dust, debris, haze, edge) / can't tell.
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
from make_catalog_live import Pyramid  # noqa: E402

CLOSE, CTX, GRID = 480, 360, 4
CTX_UM = 600.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n", type=int, default=160, help="flakes per chip")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    items, close, ctx = [], [], []
    for res in (Path(r).expanduser() for r in a.results):
        D = load_viewer_data(res)
        viewer = next(res.glob("*_viewer"))
        pyr = Pyramid(viewer)
        px = float(D["pixel_um"])
        chip = D["name"].replace("_tiles", "")
        T = {int(k): v for k, v in D["terraces"].items()}
        F = [f for f in D["flakes"] if f["terr"] and f["poly"]]
        cmag = np.array([float(np.linalg.norm(T[int(f["terr"][0])]["c"])) for f in F])
        score = np.array([f["score"] or 0 for f in F])
        # weights: faint (low |C|) and middling scores, where residue gets in; some confident ones as references
        faint = 1 - cmag.argsort().argsort() / max(len(cmag) - 1, 1)          # 1 = faintest on this chip
        w = 2.0 * faint ** 2 + 1.0 * ((score > 0.3) & (score < 0.9)) + 0.4
        w = w / w.sum()
        pick = rng.choice(len(F), size=min(a.n, len(F)), replace=False, p=w)
        for i in pick:
            f = F[int(i)]
            t = T[int(f["terr"][0])]
            side = int(np.clip(max(f["bw"], f["bh"]) * 2.2 + 40, 110, 560))
            x0, y0 = int(f["x"] - side / 2), int(f["y"] - side / 2)
            img = pyr.read(x0, y0, side, side)
            s = CLOSE / side
            close.append(cv2.resize(img, (CLOSE, CLOSE), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC))
            sc = lambda polys, ox, oy, k: [[round((p[j] - (ox if j % 2 == 0 else oy)) * k, 1) for j in range(len(p))]   # noqa: E731
                                           for p in polys]
            cs = int(round(CTX_UM / px))
            cx0, cy0 = int(f["x"] - cs / 2), int(f["y"] - cs / 2)
            ctx.append(cv2.resize(pyr.read(cx0, cy0, cs, cs), (CTX, CTX), interpolation=cv2.INTER_AREA))
            items.append(dict(id=f"{chip}-F{f['id']:04d}", chip=chip, flake=f["id"], material=D["material"],
                              field_um=round(side * px), score=f["score"], t=t["t"], c=t["c"], d=t["d"], area=f["area"],
                              poly=sc(t["poly"], x0, y0, s), fpoly=sc(f["poly"], x0, y0, s),
                              cpoly=sc(f["poly"], cx0, cy0, CTX / cs)))
        print(f"{chip}: {len(pick)} flakes")
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    close = [close[i] for i in order]
    ctx = [ctx[i] for i in order]
    per = GRID * GRID
    for name, imgs, cell in (("close", close, CLOSE), ("ctx", ctx, CTX)):
        for k in range(0, len(imgs), per):
            chunk = imgs[k:k + per]
            rows = (len(chunk) + GRID - 1) // GRID
            sheet = np.zeros((rows * cell, GRID * cell, 3), np.uint8)
            for j, c in enumerate(chunk):
                r, q = divmod(j, GRID)
                sheet[r * cell:(r + 1) * cell, q * cell:(q + 1) * cell] = c
            cv2.imwrite(str(out / f"{name}_{k // per:02d}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for j, it in enumerate(items):
        it["atlas"], it["cell"] = divmod(j, per)
    meta = dict(title="Tape Residue Labels", close=CLOSE, ctx=CTX, grid=GRID, items=items)
    (out / "items.json").write_text(json.dumps(meta, separators=(",", ":")))
    pub = dict(meta, items=[{k: v for k, v in it.items() if k in ("id", "chip", "flake", "material", "field_um", "d", "atlas",
                                                                   "cell", "poly", "fpoly", "cpoly")} for it in items])
    tpl = (Path(__file__).parent / "residue_page.html").read_text()
    (out / "index.html").write_text(tpl.replace("__DATA__", json.dumps(pub, separators=(",", ":"))))
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
