"""The zoomable viewer built during the live analysis, without a second pass over the tiles.

Every block worker writes its own tiles into a canvas on disk (a memory-mapped 8-bit image of the whole piece, plus
the contrast-enhanced layer at half resolution) as soon as the block is analysed; at the end the viewer pyramid is
written from the canvas in bands.  Memory stays at a few blocks whatever the size of the piece, and the tiles are
read once.

Coordinates: the global pixel frame of the live analysis (tile 0 of the scan plan at the frame centre, the
scan-wide pixel -> stage matrix M), shifted so that the canvas starts at (0, 0).  Each tile owns the cell of
canvas pixels nearer its planned centre than any other tile's (half-way to its grid neighbours); a worker copies
its core tiles' cells from its registered block mosaic, so neighbouring blocks never write the same pixel.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

GRID_RE = re.compile(r"Pos-(\d+)_(\d+)")


def grid_of(label):
    m = GRID_RE.match(label)
    return (int(m.group(1)), int(m.group(2))) if m else None


def local_dir(name: str) -> Path:
    """A folder on a local disk for the canvas (the system temp folder, or ~/.flakeid_canvas)."""
    import os
    import tempfile
    for base in (Path(tempfile.gettempdir()), Path.home() / ".flakeid_canvas"):
        try:
            d = base / "flakeid_canvas" / f"{name}_{os.getpid()}"
            d.mkdir(parents=True, exist_ok=True)
            return d
        except OSError:
            continue
    raise OSError("no local folder for the viewer canvas")


def setup(out_dir: Path, name: str, plan: dict, M, frame_wh) -> dict:
    """Canvas for a planned scan.  plan: {(col, row): (stage x, y)} in acquisition order (the first entry is tile 0)."""
    M = np.asarray(M, float)
    Minv = np.linalg.inv(M)
    P = np.array(list(plan.values()), float)
    P0 = P[0]
    wh = np.array(frame_wh, float)
    q = (Minv @ (P - P0).T).T                                  # tile top-left in the live frame (centre - wh/2)
    O = np.floor(q.min(0))
    W, H = int(math.ceil(q[:, 0].max() - O[0] + wh[0])), int(math.ceil(q[:, 1].max() - O[1] + wh[1]))
    out_dir = Path(out_dir)
    path, epath = out_dir / f"{name}_canvas.u8", out_dir / f"{name}_canvas_enh.u8"
    # (str, not Path: numpy 1.24 resolves a Path while the new file is still open, which Windows refuses under
    # Python 3.8 with "used by another process" (the microscope PC, 2026-10-03))
    np.memmap(str(path), np.uint8, "w+", shape=(H, W, 3)).flush()          # file of zeros: no data = black
    np.memmap(str(epath), np.uint8, "w+", shape=((H + 1) // 2, (W + 1) // 2, 3)).flush()
    cells = {}
    gc = {g: (Minv @ (np.array(s) - P0)) - O + wh / 2 for g, s in plan.items()}   # planned centres (canvas px)
    for g, c in gc.items():
        lo, hi = c - wh / 2, c + wh / 2
        for dg in ((-1, 0), (1, 0), (0, -1), (0, 1)):          # grid neighbours: the cell ends half-way to them
            n = gc.get((g[0] + dg[0], g[1] + dg[1]))
            if n is None:
                continue
            v = n - c
            ax = int(np.argmax(np.abs(v)))                    # (any camera rotation: the axis the neighbour lies along)
            mid = c[ax] + 0.5 * v[ax]
            if v[ax] < 0:
                lo[ax] = max(lo[ax], mid)
            else:
                hi[ax] = min(hi[ax], mid)
        cells[f"Pos-{g[0]:03d}_{g[1]:03d}"] = (int(round(lo[0])), int(round(lo[1])), int(round(hi[0])), int(round(hi[1])))
    return dict(path=str(path), epath=str(epath), shape=(H, W), O=O.tolist(), P0=P0.tolist(), Minv=Minv.tolist(),
                M=M.tolist(), wh=list(frame_wh), cells=cells,
                stage_map=dict(M=M.tolist(), P=P.tolist(), q=(q - O).tolist(), wh=list(frame_wh)))


# The canvas is written by the main process only.  Workers render their block's cells (render_block) and return them
# with the block result; the main process keeps the canvas files mapped for the whole piece and pastes them in (paste).
# On the microscope PC (Windows 7, 2026-10-03) workers mapping the 1.7 GB file in turn were refused for most of the scan
# ("another process has locked a portion of the file"): every write closed the file and the virus scanner went over it
# again.  With one long-lived mapping the file is opened once.
_MAPS = {}                                         # path -> np.memmap (main process)


def _maps(cc):
    H, W = cc["shape"]
    if cc["path"] not in _MAPS:
        _MAPS[cc["path"]] = np.memmap(cc["path"], np.uint8, "r+", shape=(H, W, 3))
        _MAPS[cc["epath"]] = np.memmap(cc["epath"], np.uint8, "r+", shape=((H + 1) // 2, (W + 1) // 2, 3))
    return _MAPS[cc["path"]], _MAPS[cc["epath"]]


def close(cc):
    for k in ("path", "epath"):
        m = _MAPS.pop(cc[k], None)
        if m is not None:
            try:
                m.flush()
            except OSError:
                pass
            del m


def paste(cc: dict, patches):
    """Main process: the cells rendered by render_block into the canvas."""
    can, enh = _maps(cc)
    for kind, y0, x0, a in patches:
        dst = can if kind == "c" else enh
        dst[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a


def flush(cc: dict):
    for k in ("path", "epath"):
        m = _MAPS.get(cc[k])
        if m is not None:
            m.flush()


def render_block(cc: dict, cmap, labels_core, rgb16, contrast, ws: float, substrate, block_origin_global=None):
    """The core tiles' cells of a block as canvas patches [(kind 'c' | 'e', y0, x0, uint8 array)] for paste.

    cmap: the block crop's StageMap (registered tile positions in crop pixels); rgb16: the crop (linear, mosaic
    units); contrast: the crop's contrast map at work scale ws (or None); substrate: RGB substrate level (mosaic
    units) for the display gain; block_origin_global: unused (cells are placed tile by tile)."""
    H, W = cc["shape"]
    EH, EW = (H + 1) // 2, (W + 1) // 2
    out = []
    sub = np.maximum(np.asarray(substrate, np.float32), 1.0)
    g8 = (0.31 * 255.0 / sub)[::-1]                             # RGB -> BGR order below
    subn = (sub / sub.max() * 170.0)[::-1]
    Minv, P0, O = np.array(cc["Minv"]), np.array(cc["P0"]), np.array(cc["O"])
    ch, cw = rgb16.shape[:2]
    for lab in labels_core:
        cell = cc["cells"].get(lab)
        if cell is None or lab not in cmap.labels:
            continue
        i = cmap.labels.index(lab)
        Gq = Minv @ (np.asarray(cmap.P[i], float) - P0) - O      # planned/stage top-left in canvas px
        d = np.round(np.asarray(cmap.q[i], float) - Gq).astype(int)   # registered crop position - canvas position
        x0, y0, x1, y1 = cell
        x0, y0 = max(x0, -d[0], 0), max(y0, -d[1], 0)
        x1, y1 = min(x1, cw - d[0], W), min(y1, ch - d[1], H)
        if x1 <= x0 or y1 <= y0:
            continue
        blk = rgb16[y0 + d[1]:y1 + d[1], x0 + d[0]:x1 + d[0], ::-1].astype(np.float32)
        out.append(("c", int(y0), int(x0), np.clip(blk * g8 + 0.5, 0, 255).astype(np.uint8)))
        if contrast is not None:
            ex0, ey0, ex1, ey1 = x0 // 2, y0 // 2, (x1 + 1) // 2, (y1 + 1) // 2
            cx0, cy0 = int((ex0 * 2 + d[0]) * ws), int((ey0 * 2 + d[1]) * ws)
            c = contrast[cy0:cy0 + (ey1 - ey0), cx0:cx0 + (ex1 - ex0), ::-1]
            c = c[:EH - ey0, :EW - ex0]
            if c.size:
                out.append(("e", int(ey0), int(ex0), np.clip(subn * (1.0 + 3.0 * c) + 0.5, 0, 255).astype(np.uint8)))
    return out


def write_block(cc: dict, cmap, labels_core, rgb16, contrast, ws: float, substrate, block_origin_global=None):
    """Render and paste in one go (single-process use)."""
    paste(cc, render_block(cc, cmap, labels_core, rgb16, contrast, ws, substrate, block_origin_global))
    flush(cc)


def summary(cc: dict, flakes, fpolys, tpolys, terr_t, path, tmin, tmax, width=2400):
    """Overview JPEG: the canvas (every k-th pixel) dimmed, flakes outlined in their thickness colour and numbered."""
    H, W = cc["shape"]
    can = np.memmap(cc["path"], np.uint8, "r", shape=(H, W, 3))
    k = max(1, int(math.ceil(W / width)))
    from .look import look_tile                         # (with the look of the pictures: flakeid.look)
    img = (look_tile(np.asarray(can[::k, ::k])).astype(np.float32) * 0.55).astype(np.uint8)
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_MAGMA).reshape(-1, 3)
    for f in flakes:
        t = f.best_thickness_nm
        col = tuple(int(v) for v in cmap[int(np.clip((t - tmin) / max(tmax - tmin, 1e-6), 0, 1) * 200 + 55)]) if t == t else (200, 200, 200)
        for p in fpolys.get(f.flake_id, []):
            q = np.round(np.array(p).reshape(-1, 2) / k).astype(np.int32)
            cv2.polylines(img, [q], True, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"{f.flake_id}", (int(f.x_px / k) + 4, int(f.y_px / k) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    del can


def build_viewer(cc: dict, result, fpolys, tpolys, out_dir, quality=85, verbose=False):
    """Pyramids from the canvas + index.html (flakeid.zoomview's page) + summary.jpg."""
    from .zoomview import _write_pyramid, write_viewer_page
    flush(cc)
    out = Path(out_dir)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    H, W = cc["shape"]
    from .look import look_tile
    info = {"image": _write_pyramid(np.memmap(cc["path"], np.uint8, "r", shape=(H, W, 3)), out / "tiles" / "image", quality, verbose=verbose,
                                    also=(out / "tiles" / "look", look_tile))}     # (tiles/look: the page's default view)
    info["look"] = dict(info["image"])
    info["enhanced"] = _write_pyramid(np.memmap(cc["epath"], np.uint8, "r", shape=((H + 1) // 2, (W + 1) // 2, 3)),
                                      out / "tiles" / "enhanced", quality, verbose=verbose)
    info["enhanced"]["scale"] = 0.5
    tmin, tmax = result.params["thickness_range_nm"]
    summary(cc, result.flakes, fpolys, tpolys, None, out / "summary.jpg", tmin, tmax)
    return write_viewer_page(out, info, result, tpolys, fpolys, stage_map=cc["stage_map"], verbose=verbose)


def cleanup(cc: dict):
    close(cc)
    for k in ("path", "epath"):
        try:
            Path(cc[k]).unlink()
        except OSError:
            pass
    try:
        Path(cc["path"]).parent.rmdir()            # the local canvas folder, if now empty
    except OSError:
        pass
