#!/usr/bin/env python3
"""Build the flake catalog used by the heterostructure designer.

    python tools/make_catalog.py --images "example images" -o designer

For every scan the identification pipeline is run; accepted terraces that are
large enough to build with are grouped into flakes (terraces of one physical
flake share one image), and for each flake a cut-out is made:

* RGBA image at the scan's full resolution (1 px = pixel_um), background
  transparent outside the flake (mask from the flattened contrast map);
* outline polygon of the flake and of every catalogued terrace (crop pixels);
* thickness, size, inscribed circle (centre and diameter) per terrace.

Cut-outs are packed into WebP atlases (<= 2048 px) and described in
catalog.json.  When high-resolution images of individual flakes exist they
can replace the cut-outs one by one: only `atlas`, `rect` and `px_um` change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid import find_flakes  # noqa: E402
from flakeid.io import load_image  # noqa: E402

SOFT_OK = {"good", "saturated", "ambiguous_thickness", "low_confidence", "chip_edge"}
SCANS = [("g31", "graphite", 90), ("g32", "graphite", 90), ("g33", "graphite", 90), ("g34", "graphite", 90),
         ("g35", "graphite", 90), ("bn24", "hbn", 285), ("bn25", "hbn", 285), ("bn26", "hbn", 285),
         ("fullchipbn23", "hbn", 285)]
ATLAS = 2048
OV_SCALE = 0.25        # chip overview images for the designer
MAX_SIDE = 1400        # largest cut-out side (px); bigger flakes are windowed around the best terrace


def region_mask_full(res, f, x0, y0, w, h):
    """Full-resolution mask of region f inside the window (x0, y0, w, h)."""
    lab, ws = res._labels, res._scale
    rid = res._region_of[f.terrace_id]
    a0, b0 = int(np.floor(x0 * ws)), int(np.floor(y0 * ws))
    a1, b1 = int(np.ceil((x0 + w) * ws)) + 1, int(np.ceil((y0 + h) * ws)) + 1
    sub = (lab[max(b0, 0):b1, max(a0, 0):a1] == rid).astype(np.uint8)
    full = cv2.resize(sub, (sub.shape[1] * int(round(1 / ws)), sub.shape[0] * int(round(1 / ws))),
                      interpolation=cv2.INTER_NEAREST)
    ox, oy = int(round(max(a0, 0) / ws)) - x0, int(round(max(b0, 0) / ws)) - y0
    out = np.zeros((h, w), np.uint8)
    ys, xs = slice(max(oy, 0), min(oy + full.shape[0], h)), slice(max(ox, 0), min(ox + full.shape[1], w))
    out[ys, xs] = full[ys.start - oy:ys.stop - oy, xs.start - ox:xs.stop - ox]
    return out


from flakeid.outline import place, refine_all  # noqa: E402


def poly_of(mask, simplify=1.0):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, simplify, True).reshape(-1, 2) + 0.5
    return np.round(c, 1).ravel().tolist()


def pack(sizes):
    """Shelf packing into ATLAS x ATLAS sheets. Returns [(sheet, x, y)]."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    pos = [None] * len(sizes)
    sheet, x, y, shelf = 0, 0, 0, 0
    for i in order:
        w, h = sizes[i]
        if x + w > ATLAS:
            x, y, shelf = 0, y + shelf + 2, 0
        if y + h > ATLAS:
            sheet, x, y, shelf = sheet + 1, 0, 0, 0
        pos[i] = (sheet, x, y)
        x += w + 2
        shelf = max(shelf, h)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="example images")
    ap.add_argument("-o", "--out", default="designer")
    ap.add_argument("--min-diameter", type=float, default=8.0, help="min inscribed diameter of a terrace (um)")
    ap.add_argument("--max-per-scan", type=int, default=220, help="flakes per scan (best score first)")
    ap.add_argument("--hbn-max-nm", type=float, default=100.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    flakes, cutouts, overviews = [], [], {}
    for stem, mat, ox in SCANS:
        img = load_image(Path(a.images) / f"{stem}.jpg")
        res = find_flakes(img, mat, ox)
        rgb = img.to_uint8()
        H, W = rgb.shape[:2]
        tmin, tmax = res.params["thickness_range_nm"]
        ws = res._scale
        T = {t.terrace_id: t for t in res.terraces}
        # chip overview for the designer's "show on chip" view (1/4 scale JPEG)
        ov = cv2.resize(rgb, None, fx=OV_SCALE, fy=OV_SCALE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"chip_{stem}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        overviews[stem] = dict(file=f"chip_{stem}.jpg", w=ov.shape[1], h=ov.shape[0], scale=OV_SCALE,
                               px_um=img.pixel_um, material=mat, oxide_nm=round(res.oxide_nm_used, 1),
                               stage_origin_um=res.params.get("stage_origin_um"),
                               stage_sign=res.params.get("stage_axis_sign", [1, 1]))

        def usable(t):
            if t.inscribed_diameter_um < a.min_diameter:        # reported terraces passed the hard cuts
                return False
            if mat == "hbn":
                ts = [t.thickness_nm] + [float(x) for x in t.alt_thickness_nm.split(";") if x]
                return any(v <= a.hbn_max_nm for v in ts)
            return tmin <= t.thickness_nm <= tmax

        n_scan = 0
        for f in res.flakes:                      # ranked by score
            if n_scan >= a.max_per_scan:
                break
            terr = [T[int(x)] for x in f.terrace_ids.split(";") if x]
            if not any(usable(t) for t in terr):
                continue
            old = res._flake_of.get(f.flake_id)
            if old is None:
                continue
            # the whole flake (connected region of material), full resolution
            x0, y0, w, h = f.bbox_x_px - 6, f.bbox_y_px - 6, f.bbox_w_px + 12, f.bbox_h_px + 12
            if max(w, h) > MAX_SIDE:               # very large flakes: window around the best terrace
                bt = terr[0]
                x0, y0 = int(bt.target_x_px - MAX_SIDE / 2), int(bt.target_y_px - MAX_SIDE / 2)
                w = h = MAX_SIDE
            x0, y0 = max(x0, 0), max(y0, 0)
            w, h = min(w, W - x0), min(h, H - y0)
            a0, b0 = int(np.floor(x0 * ws)), int(np.floor(y0 * ws))
            a1, b1 = int(np.ceil((x0 + w) * ws)) + 1, int(np.ceil((y0 + h) * ws)) + 1
            sub = (res._flake_lab[b0:b1, a0:a1] == old).astype(np.uint8)
            up = cv2.resize(sub, (sub.shape[1] * 2, sub.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            ox_, oy_ = x0 - a0 * 2, y0 - b0 * 2
            mask = up[oy_:oy_ + h, ox_:ox_ + w]
            mask = np.pad(mask, ((0, h - mask.shape[0]), (0, w - mask.shape[1])))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            # the outline at the real edge (full resolution, flakeid/outline.py), when the refinement worked
            rf = refine_all(img.rgb, res).get(f.flake_id)
            if rf is not None:
                mask = place(rf[2], rf[0], rf[1], x0, y0, w, h)
            if mask.sum() < 50:
                continue
            ys, xs = np.nonzero(mask)
            cx0, cy0, cx1, cy1 = max(xs.min() - 2, 0), max(ys.min() - 2, 0), min(xs.max() + 3, w), min(ys.max() + 3, h)
            mask = mask[cy0:cy1, cx0:cx1]
            X0, Y0 = x0 + cx0, y0 + cy0
            crop = rgb[Y0:Y0 + mask.shape[0], X0:X0 + mask.shape[1]]
            alpha = cv2.GaussianBlur(mask.astype(np.float32) * 255, (0, 0), 0.6)
            fid = f"{stem}-f{f.flake_id:03d}"
            entry = dict(id=fid, scan=stem, material=mat, oxide_nm=round(res.oxide_nm_used, 1), rank=f.flake_id,
                         score=None if f.score != f.score else round(f.score, 3),
                         px_um=img.pixel_um, w=int(mask.shape[1]), h=int(mask.shape[0]),
                         origin_px=[int(X0), int(Y0)],
                         origin_um=[round(X0 * img.pixel_um, 1), round(Y0 * img.pixel_um, 1)],
                         stage_origin_um=None, area=round(f.area_um2, 1), d=round(f.inscribed_diameter_um, 1),
                         L=round(f.length_um, 1), W=round(f.width_um, 1), focus_um=None if f.focus_um != f.focus_um else round(f.focus_um, 2),
                         cropped=bool(max(f.bbox_w_px, f.bbox_h_px) + 6 > MAX_SIDE),
                         outline=poly_of(mask, 1.2), terraces=[])
            if res.params.get("stage_origin_um"):
                so, sg = res.params["stage_origin_um"], res.params.get("stage_axis_sign", [1, 1])
                entry["stage_origin_um"] = [round(so[0] + sg[0] * X0 * img.pixel_um, 1),
                                            round(so[1] + sg[1] * Y0 * img.pixel_um, 1)]
            for k, t in enumerate(terr):
                if rf is not None and t.terrace_id in rf[3]:
                    tm = place(rf[3][t.terrace_id], rf[0], rf[1], X0, Y0, mask.shape[1], mask.shape[0])
                else:
                    tm = region_mask_full(res, t, X0, Y0, mask.shape[1], mask.shape[0])
                if tm.sum() < 20:
                    continue
                dist = cv2.distanceTransform(np.pad(tm, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
                ty, tx = np.unravel_index(np.argmax(dist), dist.shape)
                entry["terraces"].append(dict(
                    id=f"{fid}.{chr(97 + k) if k < 26 else k}", t=t.thickness_nm, tlo=t.thickness_min_nm,
                    thi=t.thickness_max_nm, alt=t.alt_thickness_nm, layers=t.layers, area=round(t.area_um2, 1),
                    d=round(float(dist.max()) * 2 * img.pixel_um, 1), L=round(t.length_um, 1), W=round(t.width_um, 1),
                    c=[int(tx), int(ty)], poly=poly_of(tm, 1.0), quality=t.quality,
                    score=None if t.score != t.score else round(t.score, 3), usable=usable(t),
                    terrace_id=t.terrace_id))
            if not any(tt["usable"] for tt in entry["terraces"]):
                continue
            rgba = np.dstack([crop, np.clip(alpha, 0, 255).astype(np.uint8)])
            cutouts.append(rgba)
            flakes.append(entry)
            n_scan += 1
        print(f"{stem}: {n_scan} flakes, {sum(len(f['terraces']) for f in flakes if f['scan'] == stem)} terraces")

    pos = pack([(c.shape[1], c.shape[0]) for c in cutouts])
    n_sheets = max(p[0] for p in pos) + 1
    sheets = [np.zeros((ATLAS, ATLAS, 4), np.uint8) for _ in range(n_sheets)]
    used_h = [0] * n_sheets
    for f, c, (s, x, y) in zip(flakes, cutouts, pos):
        sheets[s][y:y + c.shape[0], x:x + c.shape[1]] = c
        used_h[s] = max(used_h[s], y + c.shape[0])
        f["atlas"], f["rect"] = s, [x, y, c.shape[1], c.shape[0]]
    for s, sh in enumerate(sheets):
        bgra = cv2.cvtColor(sh[:used_h[s]], cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(out / f"flakes_{s:02d}.webp"), bgra, [cv2.IMWRITE_WEBP_QUALITY, 88])
    cat = dict(version=2, note="whole flakes (connected regions of material) cut out of 6.3x overview scans; "
                               "terraces are regions of interest. Replace cut-outs with high-resolution images later.",
               atlas_size=ATLAS, atlases=[f"flakes_{s:02d}.webp" for s in range(n_sheets)], scans=overviews,
               flakes=flakes)
    (out / "catalog.json").write_text(json.dumps(cat, separators=(",", ":")))
    print(f"{len(flakes)} flakes, {sum(len(f['terraces']) for f in flakes)} terraces, {n_sheets} atlases -> {out}")


if __name__ == "__main__":
    main()
