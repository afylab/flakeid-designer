#!/usr/bin/env python3
"""A designer run from a folder of 100x flake images (flakes that were never scanned), e.g. to record a stack made
before the autoscanner as built.

    python tools/make_stack_bundle.py "testing images/stack_alignment/DQD3" [-o <folder>/designer]

The folder holds stack_order.txt (one flake name per line, top to bottom; a header line is ignored) and flakes/ with
a 100x image of each (<name>_100x*.jpg; "_bw" high-contrast versions are skipped).  Each image's flake is the region
of non-substrate colour at (or nearest to) the image centre; the BX51 + XC10 scale at 100x (0.1143 um/px, Olympus
Stream's "63x") is assumed unless --um-px is given.  Material from the name: "BN" -> hBN, "g" -> graphite.
Thicknesses are unknown here (placeholders: hBN 30 nm, graphite 5 nm, flagged thickness_unknown).  Writes
bundle.json, run.json and the images for the designer (add the folder that holds it as a scan-results folder).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

UM_PX_100X = 0.1143
K = 0.5                           # cut-outs at half resolution (as the scan's bundles: 0.229 um/px)


def flake_region(img, overlay_box=(0, 0.89, 0.35, 1.0), tol=0.06):
    """Mask of the flake at (or nearest to) the image centre: colour differs from the substrate's (the commonest
    colour) by more than tol, cleaned; the Olympus Stream text overlay (bottom left) is ignored.  Also the mask of
    the image border (outline points there are the frame, not the flake's edge)."""
    h, w = img.shape[:2]
    a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    q = (a // 6).astype(np.int32)
    key = q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]
    vals, cnt = np.unique(key[::4, ::4], return_counts=True)
    sub = np.median(a[key == vals[cnt.argmax()]], axis=0)
    rel = a / np.maximum(sub, 1.0)
    m = (np.abs(rel - 1).max(axis=2) > tol).astype(np.uint8)
    x0, y0, x1, y1 = (int(overlay_box[0] * w), int(overlay_box[1] * h), int(overlay_box[2] * w), int(overlay_box[3] * h))
    m[y0:y1, x0:x1] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    c = lab[h // 2, w // 2]
    if c == 0:                                                     # nearest sizeable component to the centre
        big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 2000]
        c = min(big or range(1, n), key=lambda i: np.hypot(cen[i][0] - w / 2, cen[i][1] - h / 2))
    f = (lab == c).astype(np.uint8)
    # fill holes (residue spots inside the flake)
    inv = (1 - f).astype(np.uint8)
    nn, ll = cv2.connectedComponents(inv)
    border = set(np.unique(np.concatenate([ll[0], ll[-1], ll[:, 0], ll[:, -1]])))
    for i in range(1, nn):
        if i not in border:
            f[ll == i] = 1
    return f


def polygon(mask, eps=0.8):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float) + 0.5
    return [round(v, 1) for v in c.ravel().tolist()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("-o", "--out")
    ap.add_argument("--um-px", type=float, default=UM_PX_100X, help="scale of the flake images (um per pixel)")
    a = ap.parse_args(argv)
    folder = Path(a.folder)
    run = re.sub(r"[^A-Za-z0-9_.-]", "_", folder.name)
    out = Path(a.out) if a.out else folder / "designer"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = [ln.strip() for ln in (folder / "stack_order.txt").read_text().splitlines()
             if ln.strip() and not re.match(r"(?i)^(top|order|bottom)", ln.strip())]
    flakes, atlas, nat = [], [], []
    ax, ay, rowh, AW = 0, 0, 0, 4096
    atlas_img = np.zeros((AW, AW, 3), np.uint8)
    px = a.um_px / K
    for i, nm in enumerate(names):
        cand = sorted(p for p in (folder / "flakes").glob(f"{nm}_100x*") if "_bw" not in p.name.lower())
        if not cand:
            print(f"  {nm}: no 100x image, left out")
            continue
        full = cv2.imread(str(cand[0]), cv2.IMREAD_COLOR)
        img = cv2.resize(full, None, fx=K, fy=K, interpolation=cv2.INTER_AREA)
        m = flake_region(img)
        if m is None:
            print(f"  {nm}: no flake found in {cand[0].name}, left out")
            continue
        ys, xs = np.nonzero(m)
        mg = 6
        x0, y0 = max(xs.min() - mg, 0), max(ys.min() - mg, 0)
        x1, y1 = min(xs.max() + mg + 1, img.shape[1]), min(ys.max() + mg + 1, img.shape[0])
        cut, cm = img[y0:y1, x0:x1], m[y0:y1, x0:x1]
        cw, ch = cut.shape[1], cut.shape[0]
        if ax + cw > AW:
            ax, ay, rowh = 0, ay + rowh, 0
        atlas_img[ay:ay + ch, ax:ax + cw] = cut
        rect = [ax, ay, cw, ch]
        ax, rowh = ax + cw, max(rowh, ch)
        dt = cv2.distanceTransform(np.pad(cm, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]       # (padded: the image edge is not inside)
        cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs - x0, ys - y0]).astype(np.float32))
        poly = polygon(cm)
        cut_by_frame = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= img.shape[1] - 2 or ys.max() >= img.shape[0] - 2)
        mat = "hbn" if re.search(r"(?i)bn", nm) else "graphite"
        t = 30.0 if mat == "hbn" else 5.0
        fid = f"{run}:{nm}"
        nat.append((fid, full))
        area = float(cm.sum()) * px * px
        flakes.append(dict(id=fid, scan=run, material=mat, oxide_nm=285.0 if mat == "hbn" else 90.0, rank=i + 1, score=None, px_um=px,
                           w=cw, h=ch, origin_um=[0.0, 0.0], stage_origin_um=None, area=round(area, 1), d=round(2 * float(dt.max()) * px, 1),
                           L=round(max(rw, rh) * px, 1), W=round(min(rw, rh) * px, 1), cropped=cut_by_frame, outline=poly, atlas=0, rect=rect,
                           source=cand[0].name, crop_px=[int(x0 / K), int(y0 / K)],
                           terraces=[dict(id=fid + ".a", t=t, tlo=t, thi=t, alt="", layers=None, area=round(area, 1), d=round(2 * float(dt.max()) * px, 1),
                                          L=round(max(rw, rh) * px, 1), W=round(min(rw, rh) * px, 1), c=[int(cx), int(cy)], poly=poly,
                                          quality="thickness_unknown", score=None, cut_by_frame=cut_by_frame)]))
        print(f"  {nm}: {mat}, {area:.0f} um^2, {max(rw, rh) * px:.0f} x {min(rw, rh) * px:.0f} um" + (" (cut by the image edge)" if cut_by_frame else ""))
    atlas_img = atlas_img[:max(ay + rowh, 8)]
    cv2.imwrite(str(out / "flakes_00.webp"), atlas_img, [cv2.IMWRITE_WEBP_QUALITY, 90])
    files = ["flakes_00.webp"]
    # the whole 100x frames (the designer's "100x image" button), one sheet
    if nat:
        hh, ww = nat[0][1].shape[:2]
        cols = min(4, len(nat))
        rows = (len(nat) + cols - 1) // cols
        sheet = np.zeros((rows * hh, cols * ww, 3), np.uint8)
        for j, (fid, im) in enumerate(nat):
            x, y = (j % cols) * ww, (j // cols) * hh
            sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
            f = next(f for f in flakes if f["id"] == fid)
            f["img"] = dict(s="x100_nat_00.webp", x=x, y=y, w=int(im.shape[1]), h=int(im.shape[0]))
            f["frame"] = dict(w=int(im.shape[1]), h=int(im.shape[0]), x0=f["crop_px"][0], y0=f["crop_px"][1], k=K)
        cv2.imwrite(str(out / "x100_nat_00.webp"), sheet, [cv2.IMWRITE_WEBP_QUALITY, 86])
        files.append("x100_nat_00.webp")
    mats = sorted({f["material"] for f in flakes})
    bundle = dict(kind="flakeid-designer-bundle", version=1, run=run, title=f"{run} (flakes from 100x images)", created=time.strftime("%Y-%m-%d %H:%M"),
                  note="flakes from single 100x images (no scan); thicknesses unknown (placeholders)", atlas_size=AW, atlases=["flakes_00.webp"],
                  thumbs=[], scans={}, flakes=flakes, materials=mats, files=files)
    (out / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(dict(kind="flakeid-designer-run", run=run, title=bundle["title"], created=bundle["created"],
                                                  n_flakes=len(flakes), materials=mats, scans=[], files=files), indent=1), encoding="utf-8")
    print(f"designer run: {out}  ({len(flakes)} flakes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
