"""Annotated overview image highlighting identified flakes."""
from __future__ import annotations

import cv2
import numpy as np


def _colour(t, tmin, tmax):
    """Thickness -> BGR colour (perceptually uniform magma colormap)."""
    x = 0.0 if tmax <= tmin else float(np.clip((t - tmin) / (tmax - tmin), 0, 1))
    x = 0.25 + 0.75 * x          # skip magma's near-black end (invisible on SiO2)
    lut = cv2.applyColorMap(np.array([[int(x * 255)]], np.uint8), cv2.COLORMAP_MAGMA)
    return tuple(int(v) for v in lut[0, 0])


def _text(img, s, org, scale, colour, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def annotate(rgb: np.ndarray, result, out_path, scale: float = 0.5, dim: float = 0.55,
             crops_dir=None, crop_margin_um: float = 60.0):
    """Write an annotated copy of the image.

    The background is dimmed so flakes stand out; every reported flake is
    outlined in a colour encoding its thickness and labelled with its id
    and thickness.  Optionally writes a full-resolution crop per flake.
    """
    tmin, tmax = result.params["thickness_range_nm"]
    labels = result._labels
    ws = result._scale
    out = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    H, W = out.shape[:2]

    # reported flakes stay bright, everything else is dimmed
    if result._flake_lab is not None and result._flake_of:
        fl = result._flake_lab
        keep = np.zeros(int(fl.max()) + 1, bool)
        keep[list(result._flake_of.values())] = True
        highlight = keep[cv2.resize(fl, (W, H), interpolation=cv2.INTER_NEAREST)]
    else:
        keep = np.zeros(int(labels.max()) + 1, bool)
        keep[list(result._region_of.values())] = True
        highlight = keep[cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST)]
    dimmed = (out.astype(np.float32) * dim).astype(np.uint8)
    out = np.where(highlight[..., None], out, dimmed)

    fs = max(0.5, 0.00017 * max(H, W))
    th = max(1, int(round(fs * 2)))
    lw = max(1, int(round(fs * 2.5)))
    k = scale / ws
    # terraces (regions of interest): thin outline in thickness colour
    for t in result.terraces:
        rid = result._region_of[t.terrace_id]
        x0 = max(int(t.bbox_x_px * ws) - 2, 0)
        y0 = max(int(t.bbox_y_px * ws) - 2, 0)
        x1 = x0 + int(t.bbox_w_px * ws) + 6
        y1 = y0 + int(t.bbox_h_px * ws) + 6
        sub = (labels[y0:y1, x0:x1] == rid).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [((c + [x0, y0]) * k).astype(np.int32) for c in cnts]
        cv2.drawContours(out, cnts, -1, _colour(t.thickness_nm, tmin, tmax), max(1, lw - 1), cv2.LINE_AA)
    # flakes: white outline of the whole region of material, labelled with rank and best terrace
    flab = result._flake_lab
    for f in result.flakes:
        old = result._flake_of.get(f.flake_id)
        if flab is not None and old is not None:
            x0 = max(int(f.bbox_x_px * ws) - 2, 0)
            y0 = max(int(f.bbox_y_px * ws) - 2, 0)
            x1 = x0 + int(f.bbox_w_px * ws) + 6
            y1 = y0 + int(f.bbox_h_px * ws) + 6
            sub = (flab[y0:y1, x0:x1] == old).astype(np.uint8)
            cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [((c + [x0, y0]) * k).astype(np.int32) for c in cnts]
            cv2.drawContours(out, cnts, -1, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
        col = _colour(f.best_thickness_nm, tmin, tmax)
        cxy = (int(f.x_px * scale), int(f.y_px * scale))
        rad = int(0.5 * max(f.bbox_w_px, f.bbox_h_px) * scale + 25 * fs)
        lx = cxy[0] + int(rad * 0.75)
        ly = cxy[1] - int(rad * 0.75)
        more = f" +{f.n_terraces - 1}" if f.n_terraces > 1 else ""
        _text(out, f"{f.flake_id}: {f.best_thickness_nm:g}nm{more}", (min(lx, W - 1), max(ly, 12)), fs, col, th)

    # header + colour bar
    head = int(60 * fs)
    bar = np.full((head, W, 3), 30, np.uint8)
    _text(bar, f"{result.material}  SiO2 {result.oxide_nm_used:.0f} nm   {len(result.flakes)} flakes, {len(result.terraces)} terraces"
          f"   range {tmin:g}-{tmax:g} nm   min area {result.params['min_area_um2']:g} um^2",
          (10, int(40 * fs)), fs, (255, 255, 255), th)
    bw = min(W // 3, 1200)
    bx = W - bw - 20
    for i in range(bw):
        cv2.line(bar, (bx + i, int(12 * fs)), (bx + i, int(34 * fs)), _colour(tmin + (tmax - tmin) * i / bw, tmin, tmax), 1)
    _text(bar, f"{tmin:g}", (bx - int(40 * fs), int(34 * fs)), fs * 0.8, (255, 255, 255), th)
    _text(bar, f"{tmax:g} nm", (bx + bw + 5 - int(90 * fs), int(54 * fs)), fs * 0.8, (255, 255, 255), th)
    out = np.vstack([bar, out])
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    if crops_dir is not None:
        from pathlib import Path
        crops_dir = Path(crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)
        m = int(crop_margin_um / result.pixel_um)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
        for f in result.flakes:
            ya, yb = max(f.bbox_y_px - m, 0), min(f.bbox_y_px + f.bbox_h_px + m, bgr.shape[0])
            xa, xb = max(f.bbox_x_px - m, 0), min(f.bbox_x_px + f.bbox_w_px + m, bgr.shape[1])
            crop = bgr[ya:yb, xa:xb].copy()
            cx, cy = int(f.target_x_px - xa), int(f.target_y_px - ya)
            cv2.drawMarker(crop, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.imwrite(str(crops_dir / f"flake_{f.flake_id:04d}_{f.best_thickness_nm:g}nm.png"), crop)
