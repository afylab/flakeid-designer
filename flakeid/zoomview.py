"""Zoomable, multi-resolution viewer for a scan and its identified flakes.

Builds a tile pyramid (each level half the side length of the one below,
256 px JPEG tiles, as in the "data pyramid" of Jessen et al., Sci. Rep. 8,
6381 (2018)) and a self-contained HTML viewer based on OpenSeadragon.  Only
the tiles needed for the current view and zoom are loaded, so the full
resolution of a 100+ megapixel scan can be browsed smoothly.

Output folder layout:

    <name>_viewer/
        index.html            open this in a browser (works from disk)
        summary.jpg           reduced overview with flake outlines
        tiles/image/L/x_y.jpg      scan as recorded
        tiles/enhanced/L/x_y.jpg   flattened, contrast-amplified scan

The viewer shows flake outlines (vector overlay, colour = thickness), a
sortable / filterable flake table, click-to-zoom, scale bar, and cursor
position in um (and stage coordinates when known).
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# tile size: 1024 px since 2026-10-27 (was 256): a 1 x 7 cm piece's pyramid ~1,900 files instead of ~27,000 (two full
# layers + the enhanced one), which the copy to the share took long over (one file at a time costs, not the bytes);
# readers take the size from the viewer itself (tile_size), so older viewers (256) still work.  OpenSeadragon 4.1.0
# (BSD-3) is bundled in flakeid/vendor and copied next to index.html
TILE = 1024


def tile_size(level_dir, default=256):
    """The tile size of a viewer pyramid level (<viewer>/tiles/<layer>/<level>): its first tile's size (the last row and
    column are smaller; a level of one tile: any size fits)."""
    try:
        import cv2
        a = cv2.imread(str(Path(level_dir) / "0_0.jpg"), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if a is None:
            return default
        n = 2 * max(a.shape[:2])
        return 1 << max(0, int(round(math.log2(max(n, 1)))))    # (256, 512, 1024: a power of two)
    except Exception:
        return default


def _write_pyramid(img_bgr: np.ndarray, out: Path, quality: int = 85, threads: int = 8, verbose: bool = False,
                   also=None) -> dict:
    """Write an OpenSeadragon-compatible pyramid; level L has size ceil(W / 2^(maxL-L)).

    Written one row of tiles at a time (bounded memory for multi-GB mosaics).  also: (out dir, fn) - a second pyramid
    of the same tiles through fn (8-bit BGR tile -> 8-bit BGR tile; e.g. flakeid.look.look_tile), written alongside."""
    from .progress import Progress
    h, w = img_bgr.shape[:2]
    max_level = int(math.ceil(math.log2(max(w, h))))
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    total, lw_, lh_ = 0, w, h
    for L in range(max_level, -1, -1):
        total += ((lw_ + TILE - 1) // TILE) * ((lh_ + TILE - 1) // TILE)
        lw_, lh_ = (lw_ + 1) // 2, (lh_ + 1) // 2
    pr = Progress(f"viewer tiles ({out.name})", total, verbose)
    level_img = img_bgr
    n = 0
    with ThreadPoolExecutor(threads) as ex:
        for L in range(max_level, -1, -1):
            lh, lw = level_img.shape[:2]
            d = out / str(L)
            d.mkdir(parents=True, exist_ok=True)
            if also is not None:
                d2 = Path(also[0]) / str(L)
                d2.mkdir(parents=True, exist_ok=True)
            for ty in range(0, lh, TILE):
                row = [ex.submit(cv2.imwrite, str(d / f"{tx // TILE}_{ty // TILE}.jpg"),
                                 np.ascontiguousarray(level_img[ty:ty + TILE, tx:tx + TILE]), params)
                       for tx in range(0, lw, TILE)]
                if also is not None:
                    row += [ex.submit(lambda a, f: cv2.imwrite(f, also[1](a), params),
                                      np.ascontiguousarray(level_img[ty:ty + TILE, tx:tx + TILE]), str(d2 / f"{tx // TILE}_{ty // TILE}.jpg"))
                            for tx in range(0, lw, TILE)]
                for j in row:
                    j.result()
                n += (lw + TILE - 1) // TILE
                pr.update(n)
            if L > 0:
                level_img = _half(level_img, out)
    pr.done()
    for f in out.glob("_half_*.u8"):
        try:
            f.unlink()
        except OSError:
            pass
    return dict(width=w, height=h, maxLevel=max_level, tileSize=TILE, n_tiles=n)


def _half(img, tmp_dir, band=2048):
    """Half-size copy, made in horizontal bands (the input can be a memory-mapped file of several GB); a large result
    goes to a memory-mapped file in tmp_dir as well."""
    h, w = img.shape[:2]
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    if h2 * w2 * 3 > 256e6:
        k = len(list(Path(tmp_dir).glob("_half_*.u8")))
        out = np.memmap(str(Path(tmp_dir) / f"_half_{k}.u8"), np.uint8, "w+", shape=(h2, w2) + img.shape[2:])
    else:
        out = np.empty((h2, w2) + img.shape[2:], np.uint8)
    for y in range(0, h, band):                       # band is even: rows pair up within a band
        blk = np.ascontiguousarray(img[y:y + band])
        out[y // 2:y // 2 + (blk.shape[0] + 1) // 2] = cv2.resize(blk, (w2, (blk.shape[0] + 1) // 2), interpolation=cv2.INTER_AREA)
    return out


def _enhanced(contrast: np.ndarray, substrate_rgb, gain: float = 3.0) -> np.ndarray:
    """Flattened scan with contrast amplified: substrate colour * (1 + gain*C), 8-bit BGR."""
    sub = np.asarray(substrate_rgb, np.float32)
    sub = sub / sub.max() * 170.0                   # neutral, mid-grey-ish substrate brightness
    out = np.clip(sub[None, None, :] * (1.0 + gain * contrast), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _stage_map(result):
    sm = result.metadata.get("stage_map")
    if not sm:
        return None
    return dict(M=sm["M"], P=sm["P"], q=sm["q"], wh=sm["frame_wh"])


def _poly(mask, xa, ya, ws, simplify_px=0.8):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 2:
            continue
        c = cv2.approxPolyDP(c, simplify_px, True).reshape(-1, 2).astype(np.float32)
        c = (c + [xa + 0.5, ya + 0.5]) / ws - 0.5
        out.append(np.round(c, 1).ravel().tolist())
    return out


def _outlines(result) -> tuple:
    """Polygons (original-pixel coordinates): {terrace_id: [...]}, {flake_id: [...]}."""
    labels, ws = result._labels, result._scale
    tpolys, fpolys = {}, {}
    for t in result.terraces:
        rid = result._region_of[t.terrace_id]
        xa, ya = max(int(t.bbox_x_px * ws) - 2, 0), max(int(t.bbox_y_px * ws) - 2, 0)
        x1, y1 = xa + int(t.bbox_w_px * ws) + 6, ya + int(t.bbox_h_px * ws) + 6
        tpolys[t.terrace_id] = _poly(labels[ya:y1, xa:x1] == rid, xa, ya, ws)
    fl = result._flake_lab
    for f in result.flakes:
        old = result._flake_of.get(f.flake_id)
        if fl is None or old is None:
            continue
        xa, ya = max(int(f.bbox_x_px * ws) - 2, 0), max(int(f.bbox_y_px * ws) - 2, 0)
        x1, y1 = xa + int(f.bbox_w_px * ws) + 6, ya + int(f.bbox_h_px * ws) + 6
        fpolys[f.flake_id] = _poly(fl[ya:y1, xa:x1] == old, xa, ya, ws, 1.0)
    return tpolys, fpolys


def build_viewer(img, result, out_dir, labels_csv: str | None = None, quality: int = 85,
                 summary_scale: float = 0.35, verbose: bool = True) -> Path:
    """img: ImageData (any bit depth); result: FlakeResult (run with keep_contrast=True
    to get the enhanced layer)."""
    from .annotate import annotate

    out = Path(out_dir)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    bgr = img.to_uint8(bgr=True)                     # one 8-bit copy (the mosaic can be several GB)
    # tiles/look (the page's default view) when the substrate is balanced to the canvas grey (display_gain: mm datasets)
    from .look import look_tile
    lk = (out / "tiles" / "look", look_tile) if img.display_gain is not None else None
    info = {"image": _write_pyramid(bgr, out / "tiles" / "image", quality, verbose=verbose, also=lk)}
    if lk:
        info["look"] = dict(info["image"])
    rgb8 = bgr[..., ::-1]
    if result._contrast is not None:
        enh = _enhanced(result._contrast, result._substrate_rgb)
        info["enhanced"] = _write_pyramid(enh, out / "tiles" / "enhanced", quality, verbose=verbose)
        info["enhanced"]["scale"] = result._scale
        del enh
    annotate(rgb8, result, out / "summary.jpg", scale=summary_scale)

    tpolys, fpolys = _outlines(result)
    # outlines at the real edge (full resolution; flakeid/outline.py) where the refinement worked
    from .outline import polygons, refine_all
    for fid, (x0, y0, fm, tm) in refine_all(img.rgb, result).items():
        fpolys[fid] = polygons(fm, x0, y0, 1.0)
        for tid, m in tm.items():
            tpolys[tid] = polygons(m, x0, y0, 0.8)
    return write_viewer_page(out, info, result, tpolys, fpolys, verbose=verbose)


def write_viewer_page(out, info, result, tpolys, fpolys, stage_map=None, verbose=True) -> Path:
    """index.html of a viewer whose pyramids (info) are written: flakes, terraces and their outlines."""
    out = Path(out)
    tmin, tmax = result.params["thickness_range_nm"]
    num = lambda v, n=1: None if v is None or v != v else round(v, n)
    terr = {}
    for t in result.terraces:
        terr[t.terrace_id] = dict(id=t.terrace_id, fl=t.flake_id, t=t.thickness_nm, tlo=t.thickness_min_nm,
                                  thi=t.thickness_max_nm, alt=t.alt_thickness_nm, layers=t.layers,
                                  area=round(t.area_um2, 1), L=round(t.length_um, 1), W=round(t.width_um, 1),
                                  d=round(t.inscribed_diameter_um, 1), tx=round(t.target_x_px, 1), ty=round(t.target_y_px, 1),
                                  sx=num(t.stage_x_um), sy=num(t.stage_y_um), q=t.quality, score=num(t.score, 3),
                                  c=[round(t.contrast_r, 3), round(t.contrast_g, 3), round(t.contrast_b, 3)],
                                  poly=tpolys.get(t.terrace_id, []))
    flakes = []
    for f in result.flakes:
        flakes.append(dict(id=f.flake_id, score=num(f.score, 3), t=f.best_thickness_nm, n=f.n_terraces,
                           terr=[int(x) for x in f.terrace_ids.split(";") if x],
                           area=round(f.area_um2, 1), L=round(f.length_um, 1), W=round(f.width_um, 1),
                           d=round(f.inscribed_diameter_um, 1), bd=round(f.best_terrace_d_um, 1),
                           x=round(f.x_px, 1), y=round(f.y_px, 1), tx=round(f.target_x_px, 1), ty=round(f.target_y_px, 1),
                           bx=f.bbox_x_px, by=f.bbox_y_px, bw=f.bbox_w_px, bh=f.bbox_h_px,
                           sx=num(f.stage_x_um), sy=num(f.stage_y_um), q=f.quality,
                           focus=num(f.focus_um, 2), crowd=num(f.crowding, 2), fw=num(f.frac_window, 2),
                           fthick=num(f.frac_thick, 2), poly=fpolys.get(f.flake_id, [])))
    data = dict(name=Path(result.image).name, material=result.material, oxide=round(result.oxide_nm_used, 1),
                pixel_um=result.pixel_um, tmin=tmin, tmax=tmax, info=info, flakes=flakes, terraces=terr,
                stage_origin=result.params.get("stage_origin_um"),
                stage_sign=result.params.get("stage_axis_sign", [1, 1]),
                stage_map=stage_map or _stage_map(result))
    import shutil
    vend = Path(__file__).parent / "vendor"
    shutil.copy(vend / "openseadragon.min.js", out / "openseadragon.min.js")
    shutil.copy(vend / "OPENSEADRAGON_LICENSE.txt", out / "OPENSEADRAGON_LICENSE.txt")
    html = _HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    html = html.replace("__TITLE__", f"{data['name']} flakes")
    (out / "index.html").write_text(html)
    if verbose:
        n = sum(v["n_tiles"] for v in info.values())
        print(f"  viewer: {out / 'index.html'} ({n} tiles)")
    return out / "index.html"


def upgrade_page(index_html, backup=True) -> bool:
    """Rewrite an existing viewer's index.html with the current page (keys, large 100x view, ...), keeping its data
    (flakes, terraces, tiles) and its 100x images.  True if rewritten."""
    import re
    page = Path(index_html)
    html = page.read_text(encoding="utf-8")
    m = re.search(r"const D=(\{.*?\});\n", html, re.S)
    if not m:
        return False
    mh = re.search(r"const H100=(.*?);\n", html, re.S)
    mt = re.search(r"<title>(.*?)</title>", html, re.S)
    new = _HTML.replace("__DATA__", m.group(1)).replace("__TITLE__", mt.group(1) if mt else "flakes")
    if mh:
        new = new.replace("const H100=null;\n", "const H100=" + mh.group(1) + ";\n", 1)
    if backup and not (page.parent / "index_before_upgrade.html").exists():
        (page.parent / "index_before_upgrade.html").write_text(html, encoding="utf-8")
    page.write_text(new, encoding="utf-8")
    return True


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="openseadragon.min.js"></script>
<style>
:root{--bg:#f6f7f9;--panel:#ffffff;--ink:#1f2328;--muted:#57606a;--rule:#d0d7de;--accent:#0969da;--hi:#fff3b0}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--rule:#30363d;--accent:#58a6ff;--hi:#3b3417}}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
#app{display:grid;grid-template-columns:minmax(0,1fr) 410px;grid-template-rows:minmax(0,1fr);height:100vh;overflow:hidden}
#view{position:relative;background:#111;min-height:0;overflow:hidden}
#osd{position:absolute!important;inset:0;width:100%;height:100%}
#svg{position:absolute;inset:0;pointer-events:none;width:100%;height:100%}
#svg polygon{fill:none;stroke-width:1.6;vector-effect:non-scaling-stroke;pointer-events:visibleStroke;cursor:pointer}
#svg polygon.fl{stroke-width:1.2;stroke-dasharray:5 3;opacity:.9}
#svg polygon.sel{stroke:#fff!important;stroke-width:3}
#svg polygon.tsel{stroke:#00e5ff!important;stroke-width:3.5}
.trow.on{background:var(--bg);font-weight:600}
#g polygon{filter:drop-shadow(0 0 1px rgba(255,255,255,.95))}
#svg text{font-weight:600;font-family:sans-serif;font-size:var(--fs,12px);paint-order:stroke;stroke:#000;stroke-width:var(--sw,3px);fill:#fff;pointer-events:none}
#g.nolab text{display:none}
#svg.pick polygon{pointer-events:none}#view.pick .openseadragon-canvas{cursor:crosshair!important}
#mk line,#mk circle,#mk path{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}
#mk text{font:600 12px sans-serif;paint-order:stroke;stroke:#000;stroke-width:3px;fill:#fff}
table.noxy .xy{display:none}table:not(.noxy){font-size:12px}table:not(.noxy) th,table:not(.noxy) td{padding:4px 3px}table:not(.noxy) .nt{display:none}
#bar{display:inline-block;height:4px;background:#fff;vertical-align:middle;margin-right:6px}
#side{display:flex;flex-direction:column;border-left:1px solid var(--rule);background:var(--panel);min-width:0;min-height:0}
header{padding:10px 12px 8px;border-bottom:1px solid var(--rule)}
h1{font-size:15px;margin:0 0 2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.sub{color:var(--muted);font-size:12px}
#legend{height:8px;border-radius:3px;margin-top:6px}
.lg{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}
/* the toolbar (2026-10-27): Image options (a popup on hover), Filter and Sort (dropdowns), 100x mode, Large 100x */
#bar2{position:relative;z-index:20;padding:8px 12px;border-bottom:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button{font:inherit}
.tb,.pop button{padding:3px 9px;border:1px solid var(--rule);background:var(--bg);color:var(--ink);border-radius:5px;cursor:pointer;white-space:nowrap}
.tb.on,.pop button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tb:disabled,.pop button:disabled{opacity:.45;cursor:default}
.menu{position:relative}
.menu>.tb::after{content:" \25BE";font-size:10px;opacity:.7}
.pop{display:none;position:absolute;left:0;top:100%;margin-top:4px;min-width:250px;max-width:360px;padding:10px 12px;background:var(--panel);
 border:1px solid var(--rule);border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.18);font-size:12px}
.menu.open>.pop{display:block}
.menu.open>.pop::before{content:"";position:absolute;left:0;right:0;top:-8px;height:8px}
.pop h4{margin:8px 0 4px;font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.pop h4:first-child{margin-top:0}
.pop .row{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.pop label{display:flex;align-items:center;gap:5px;color:var(--ink);margin:3px 0}
.pop input[type=number]{width:64px;padding:2px 4px;background:var(--bg);color:var(--ink);border:1px solid var(--rule);border-radius:4px}
.pop input#fRot{width:52px}
.pop .keys{color:var(--muted);font-size:11px;margin-top:8px;line-height:1.7}
.pop kbd,#detail kbd,#bigCap kbd{font:11px ui-monospace,Consolas,monospace;padding:0 4px;border:1px solid #888;border-radius:3px}
#mImg .pop{width:340px}#mSort .pop{min-width:190px}
#mSort label{margin:2px 0;cursor:pointer}
#nf{font-size:11.5px;color:var(--muted);flex-basis:100%;display:flex;gap:6px;align-items:center}
#nf button[hidden]{display:none}
#nf button{font-size:11px;padding:0 6px;border:1px solid var(--rule);border-radius:4px;background:transparent;color:var(--muted);cursor:pointer}
#tbl{flex:1;min-height:0;overflow:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:4px 5px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{position:sticky;top:0;z-index:1;background:var(--panel);cursor:pointer;color:var(--muted);font-weight:600;font-size:12px}
th.srt{color:var(--ink)}th.srt::after{content:var(--arr);font-size:10px;margin-left:2px}
th.rn,td.rn{color:var(--muted);font-size:11px;text-align:right;padding-left:8px;cursor:default}
tr:hover td{background:var(--bg)}tr.sel td{background:var(--hi)}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
table.no100 .s1{display:none}
.h1b{font-size:9.5px;font-weight:600;padding:0 4px;border-radius:3px;background:var(--rule);color:var(--ink);margin-left:4px;vertical-align:1px}
#detail{padding:8px 12px;border-top:1px solid var(--rule);font-size:12px;color:var(--muted);max-height:34vh;overflow:auto}
#detail b{color:var(--ink)}
.trow{padding:3px 0;cursor:pointer;border-top:1px solid var(--rule)}
.trow:hover{background:var(--bg)}
.lbl2{margin-top:6px;font-weight:600;color:var(--ink)}
button.cp{font-size:11px;padding:0 6px;margin-left:4px;border:1px solid var(--rule);border-radius:4px;background:transparent;color:var(--muted);cursor:pointer}
/* over the map: the coordinates (origin, measured points) at the top right; the selected flake's 100x picture at the
   bottom right, with Hide (2026-10-27: it was under the table) */
#cst{position:absolute;top:10px;right:10px;z-index:4;max-width:min(420px,60%);padding:7px 10px;border-radius:6px;background:rgba(0,0,0,.66);
 color:#ddd;font-size:12px;font-variant-numeric:tabular-nums}
#cst[hidden]{display:none}#cst b{color:#fff}
#cst button.cp{color:#ddd;border-color:#777}
#p100{position:absolute;right:10px;bottom:10px;z-index:4;width:min(400px,36%);max-height:calc(100% - 20px);display:flex;flex-direction:column;
 background:rgba(14,14,18,.92);color:#ddd;border:1px solid #444;border-radius:8px;box-shadow:0 4px 18px rgba(0,0,0,.55);overflow:hidden}
#p100[hidden],#p100s[hidden]{display:none}
#p100 .hd{display:flex;align-items:center;gap:8px;padding:5px 6px 5px 10px;font-size:12px}
#p100 .hd span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#p100 .hd button,#p100s{font-size:12px;padding:2px 9px;border:1px solid #666;border-radius:5px;background:transparent;color:#ddd;cursor:pointer}
#p100 .bd{overflow:auto;min-height:0;padding:0 6px 6px}
#p100s{position:absolute;right:10px;bottom:10px;z-index:4;background:rgba(14,14,18,.85)}
figure.h100{margin:0 0 6px}figure.h100:last-child{margin-bottom:0}
figure.h100 img{display:block;width:100%;height:auto;max-height:46vh;object-fit:contain;border-radius:4px;background:#000}
figure.h100 figcaption{font-size:11px;color:#aaa;margin-top:3px}figure.h100 a{color:inherit}
#big{position:absolute;inset:0;background:#111;display:flex;flex-direction:column;z-index:5}
#big[hidden]{display:none}
#bigWrap{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;padding:8px}
#bigImg{max-width:100%;max-height:100%;object-fit:contain;display:block}
#bigCap{color:#ddd;font-size:12px;padding:6px 10px;background:rgba(0,0,0,.55)}
#mini{position:absolute;right:10px;bottom:52px;border:1px solid #555;border-radius:4px;background:#000;cursor:pointer;max-width:34%;box-shadow:0 2px 10px rgba(0,0,0,.6)}
@media (max-width:800px){#app{grid-template-columns:minmax(0,1fr);grid-template-rows:60vh 40vh}#side{border-left:0;border-top:1px solid var(--rule)}#p100{width:min(400px,60%)}}
/* a strip of its own under the map (2026-10-27): the scale bar and cursor position on the left, the 100x views (natural,
   linear, colour +-8 %, ...) in the middle */
#left{display:flex;flex-direction:column;min-width:0;min-height:0}
#left #view{flex:1}
#foot{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;min-height:46px;padding:6px 10px;
 background:#18181c;color:#ddd;border-top:1px solid #333}
#hud{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#vbar{display:flex;flex-wrap:wrap;justify-content:center;gap:4px}
#vbar[hidden]{display:none}
#vbar button{font:12px inherit;padding:3px 10px;border:1px solid #666;border-radius:5px;background:transparent;color:#ddd;cursor:pointer}
#vbar button[aria-pressed="true"]{background:#fff;color:#111;border-color:#fff}
#vbar kbd{font:10px ui-monospace,Consolas,monospace;opacity:.6;margin-right:4px}
</style></head><body>
<div id="app">
 <div id="left"><div id="view"><div id="osd"></div><svg id="svg"><g id="g"></g><g id="mk"></g></svg>
  <div id="cst" hidden></div>
  <div id="p100" hidden><div class="hd"><span id="p100t"></span><button id="bHide" title="Hide the 100x picture (it covers part of the map)">Hide</button></div><div class="bd" id="p100b"></div></div>
  <button id="p100s" hidden title="Show the selected flake's 100x picture">Show 100x</button>
  <div id="big" hidden><div id="bigWrap"><img id="bigImg" alt=""></div><div id="bigCap"></div><canvas id="mini" title="Click to go back to the chip map"></canvas></div></div>
  <div id="foot"><div id="hud"><span id="bar"></span><span id="barlab"></span> &nbsp; <span id="pos"></span></div><div id="vbar" hidden></div><div></div></div></div>
 <div id="side">
  <header><h1 id="title"></h1><div class="sub" id="subt"></div>
   <div id="legend"></div><div class="lg"><span id="lg0"></span><span>thickness (nm)</span><span id="lg1"></span></div></header>
  <div id="bar2">
   <div class="menu" id="mImg"><button class="tb" id="bImgOpt">Image options</button><div class="pop">
    <h4>Picture</h4><div class="row"><button id="bImg" class="on">Scan</button><button id="bLin" title="The colours the analysis measures (linear, substrate grey)">Linear</button><button id="bEnh">Enhanced ×3</button></div>
    <h4>Overlay</h4><div class="row"><button id="bOut" class="on">Outlines</button><button id="bLab" class="on">Labels</button></div>
    <h4>Orientation</h4><div class="row"><button id="bRL" title="Rotate 90° counter-clockwise (Shift+R)">↺</button><button id="bRR" title="Rotate 90° clockwise (R)">↻</button>
     <label title="Rotation of the scan image in degrees (clockwise), applied before mirroring"><input id="fRot" type="number" step="1">°</label>
     <button id="bMH" title="Mirror left-right (H)">↔</button><button id="bMV" title="Mirror top-bottom (V)">↕</button><button id="bR0" title="Original orientation">0°</button></div>
    <h4>Coordinates</h4><div class="row"><button id="bO" title="Then click a chip feature (e.g. a corner) to make it the origin (O)">Set origin</button>
     <button id="bAx" title="Then click a second point along a chip edge: it sets the +X direction, so coordinates are in the chip's own frame whatever the view (X). Click again to go back to view axes.">Align X</button>
     <button id="bMe" title="Click points on the map: their coordinates and the distance between two (M)">Measure</button><button id="bClr" title="Remove the origin, axis and measured points">Clear</button></div>
    <div class="keys" id="keys"></div>
   </div></div>
   <div class="menu" id="mFil"><button class="tb" id="bFil">Filter</button><div class="pop">
    <label id="lH" hidden><input id="fH" type="checkbox"> Only flakes with 100x images</label>
    <h4>Thickness (any terrace)</h4><div class="row"><input id="fT0" type="number" step="0.5"> – <input id="fT1" type="number" step="0.5"> nm</div>
    <h4>Size and scores</h4>
    <label>flake area ≥ <input id="fA" type="number" value="0" step="100" min="0"> µm²</label>
    <label title="10x score (0-1)">10x score ≥ <input id="fS" type="number" value="0" step="0.05" min="0" max="1"></label>
    <label id="lS1" hidden title="100x score: the largest clean area in µm²">100x score ≥ <input id="fS1" type="number" value="0" step="50" min="0"> µm²</label>
    <div class="row" style="margin-top:8px"><button id="bFR">Reset filters</button></div>
   </div></div>
   <div class="menu" id="mSort"><button class="tb" id="bSort">Sort</button><div class="pop" id="sortPop"></div></div>
   <button class="tb" id="bM100" hidden title="100x mode: only the flakes with 100x pictures, ranked by the 100x score (largest clean area), the 100x picture large; arrow keys: next / previous">100x mode</button>
   <button class="tb" id="bBig" hidden title="Show the 100x image large, with the chip map small (L)">Large 100x</button>
   <div id="nf"><span id="nft"></span><button id="bFR2" hidden>clear filters</button></div>
  </div>
  <div id="tbl"><table class="noxy"><thead><tr><th class="rn" title="Row number in the current order">#</th><th data-k="id">flake</th><th data-k="score" title="10x score: learned from your labels of which flakes were worth imaging (0-1), from the 10x scan">10x</th><th data-k="s100" class="s1" title="100x score: the largest clean area (µm²) in the flake's 100x picture: no step edges, glue / debris, bubbles or folds">100x</th><th data-k="t" title="Thickness of the flake's best terrace (nm)">t nm</th><th data-k="n" class="nt" title="Number of terraces">terr.</th><th data-k="area" title="Flake area (µm²)">area</th><th data-k="bd" title="Diameter of the best terrace (µm)">Ø µm</th><th data-k="cx" class="xy">X µm</th><th data-k="cy" class="xy">Y µm</th></tr></thead><tbody id="rows"></tbody></table></div>
  <div id="detail">Click a flake (row or outline) for its details.</div>
 </div>
</div>
<script>
const D=__DATA__;
const H100=null;
// the 100x score of each flake (flakeid.screen100.clean_score, from 2026-10-27): the best of its 100x pictures (s100:
// [largest clean area um2, its disc um])
if(H100)D.flakes.forEach(f=>{const v=(H100[f.id]||[]).filter(h=>!h.domain&&h.s100).map(h=>h.s100[0]);f.s100=v.length?Math.max(...v):null});

const $=id=>document.getElementById(id);
const MAGMA=["#50127b","#762181","#9c2e7f","#c33b74","#e65162","#f9795c","#fea671","#fdd295","#fbfcbf"];
function cmap(x){x=Math.max(0,Math.min(1,x))*(MAGMA.length-1);const i=Math.min(Math.floor(x),MAGMA.length-2),f=x-i;
 const a=MAGMA[i],b=MAGMA[i+1],h=(s,k)=>parseInt(s.substr(1+2*k,2),16);
 return `rgb(${[0,1,2].map(k=>Math.round(h(a,k)+(h(b,k)-h(a,k))*f)).join(',')})`}   // magma 0.25-1
const col=t=>cmap((t-D.tmin)/Math.max(D.tmax-D.tmin,1e-6));
$('title').textContent=D.name;
const N100=H100?D.flakes.filter(f=>H100[f.id]).length:0;
$('subt').textContent=`${D.material} · ${D.flakes.length} flakes`+(H100?` · ${N100} with 100x`:'');
$('legend').style.background=`linear-gradient(90deg,${MAGMA.join(',')})`;
$('lg0').textContent=D.tmin;$('lg1').textContent=D.tmax;$('fT0').value=D.tmin;$('fT1').value=D.tmax;
function src(i){return{height:i.height,width:i.width,tileSize:i.tileSize,tileOverlap:0,minLevel:0,maxLevel:i.maxLevel,
 getTileUrl:(L,x,y)=>`tiles/${i.dir}/${L}/${x}_${y}.jpg`}}
D.info.image.dir='image';if(D.info.enhanced)D.info.enhanced.dir='enhanced';if(D.info.look)D.info.look.dir='look';
// the scan as shown: tiles/look (flakeid.look: the reference picture's colours) when written, else tiles/image; the
// "Linear" button shows tiles/image, the colours the analysis measures
const BASE=D.info.look||D.info.image;
const V=OpenSeadragon({element:$('osd'),prefixUrl:'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
 showNavigator:true,navigatorPosition:'TOP_LEFT',maxZoomPixelRatio:4,animationTime:0.6,showRotationControl:false,
 showZoomControl:false,showHomeControl:false,showFullPageControl:false,
 tileSources:src(BASE)});
let enhItem=null,linItem=null,IMG=null;
try{V.innerTracker.keyDownHandler=null;V.innerTracker.keyHandler=null}catch(e){}     // (arrow keys: next / previous flake)
V.addHandler('open',()=>{IMG=V.world.getItemAt(0);
 if(D.info.look){V.addTiledImage({tileSource:src(D.info.image),x:0,y:0,width:1,opacity:0,success:e=>{linItem=e.item}})}
 if(D.info.enhanced){V.addTiledImage({tileSource:src(D.info.enhanced),x:0,y:0,width:1,opacity:0,success:e=>{enhItem=e.item}})}
 draw();});
const W=D.info.image.width;
const P=(x,y)=>new OpenSeadragon.Point(x,y);
// ---- vector overlay: flake outline (white) + terraces (thickness colour)
const NS='http://www.w3.org/2000/svg',G=$('g');const byId={};
function mkPoly(p,stroke,cls){const pl=document.createElementNS(NS,'polygon');const pts=[];for(let i=0;i<p.length;i+=2)pts.push(p[i]+','+p[i+1]);
 pl.setAttribute('points',pts.join(' '));pl.setAttribute('stroke',stroke);if(cls)pl.setAttribute('class',cls);return pl}
D.flakes.forEach(f=>{const g=document.createElementNS(NS,'g');g.dataset.id=f.id;
 f.poly.forEach(p=>{const pl=mkPoly(p,'#ffffff','fl');pl.addEventListener('click',()=>select(f.id,false));g.appendChild(pl)});
 f.terr.forEach(tid=>{const T=D.terraces[tid];if(!T)return;T.poly.forEach(p=>{const pl=mkPoly(p,col(T.t),'tr');pl.dataset.t=tid;pl.addEventListener('click',()=>select(f.id,false));g.appendChild(pl)})});
 const t=document.createElementNS(NS,'text');t.setAttribute('x',f.bx+f.bw+4);t.setAttribute('y',f.by+10);
 t.textContent=`${f.id}: ${f.t} nm${f.n>1?' +'+(f.n-1):''}`;g.appendChild(t);
 G.appendChild(g);byId[f.id]={f,g,t}});
// image <-> viewer-element pixels; OSD 4.1 draws a mirrored view but its coordinate conversions ignore the flip
const FL=()=>V.viewport.getFlip(),CW=()=>V.viewport.getContainerSize().x;
function img2scr(x,y){const p=IMG.imageToViewerElementCoordinates(P(x,y));if(FL())p.x=CW()-p.x;return p}
function scr2img(p){return IMG.viewerElementToImageCoordinates(FL()?P(CW()-p.x,p.y):p)}
let TXK='';
function sync(){if(!IMG)return;const p0=img2scr(0,0),p1=img2scr(W,0),p2=img2scr(0,W);
 const a=(p1.x-p0.x)/W,b=(p1.y-p0.y)/W,c=(p2.x-p0.x)/W,d=(p2.y-p0.y)/W,s=Math.hypot(a,b);
 G.setAttribute('transform',`matrix(${a},${b},${c},${d},${p0.x},${p0.y})`);
 // labels stay upright and readable in a rotated / mirrored view
 const k=[a/s,b/s,c/s,d/s].map(v=>v.toFixed(3)).join(',');
 if(k!==TXK){TXK=k;const m=`matrix(${a/s},${c/s},${b/s},${d/s},0,0)`;
  Object.values(byId).forEach(o=>{const x=o.t.getAttribute('x'),y=o.t.getAttribute('y');o.t.setAttribute('transform',`translate(${x},${y}) ${m} translate(${-x},${-y})`)})}
 drawMarks();
 // keep labels a constant screen size; hide them when zoomed far out
 G.style.setProperty('--fs',(12/s)+'px');G.style.setProperty('--sw',(3/s)+'px');
 G.classList.toggle('nolab',!(labOn&&s>0.3));
 // scale bar
 const umPerPx=D.pixel_um/s;const nice=[1,2,5,10,20,50,100,200,500,1000,2000,5000];let L=nice.find(v=>v/umPerPx>=80)||5000;
 $('bar').style.width=(L/umPerPx)+'px';$('barlab').textContent=L>=1000?(L/1000)+' mm':L+' µm'}
V.addHandler('animation',sync);V.addHandler('update-viewport',sync);V.addHandler('resize',sync);
new OpenSeadragon.MouseTracker({element:V.canvas,moveHandler:e=>{if(!IMG)return;const vp=scr2img(e.position);
 let s=XY.o?`<b>${ptTxt(vp)}</b>`:`${(vp.x*D.pixel_um).toFixed(0)}, ${(vp.y*D.pixel_um).toFixed(0)} µm`;
 const SM=D.stage_map;
 if(SM){let k=0,b=1e30;SM.q.forEach((q,i)=>{const d=(vp.x-q[0]-SM.wh[0]/2)**2+(vp.y-q[1]-SM.wh[1]/2)**2;if(d<b){b=d;k=i}});
  const dx=vp.x-SM.q[k][0]-SM.wh[0]/2,dy=vp.y-SM.q[k][1]-SM.wh[1]/2;
  s+=` · stage ${(SM.P[k][0]+SM.M[0][0]*dx+SM.M[0][1]*dy).toFixed(1)}, ${(SM.P[k][1]+SM.M[1][0]*dx+SM.M[1][1]*dy).toFixed(1)}`}
 else if(D.stage_origin){s+=` · stage ${(D.stage_origin[0]+D.stage_sign[0]*vp.x*D.pixel_um).toFixed(0)}, ${(D.stage_origin[1]+D.stage_sign[1]*vp.y*D.pixel_um).toFixed(0)}`}
 $('pos').innerHTML=s}}).setTracking(true);
// ---- view orientation (rotate / mirror) and µm coordinates relative to a chip feature; both remembered per page
const LSK='flakeid-viewer:'+location.pathname+':';
function lsGet(k){try{return JSON.parse(localStorage.getItem(LSK+k))}catch(e){return null}}
function lsSet(k,v){try{localStorage.setItem(LSK+k,JSON.stringify(v))}catch(e){}}
const OR=Object.assign({rot:180,flip:false},lsGet('view')||{});        // (default: the camera's image turned 180 deg, as the scan program's live views)
const norm360=a=>((a%360)+360)%360;
// screen direction (x right, y down) of an image direction: rotate by OR.rot clockwise, then mirror left-right
function Lm(dx,dy){const t=OR.rot*Math.PI/180,c=Math.cos(t),s=Math.sin(t);return[(OR.flip?-1:1)*(c*dx-s*dy),s*dx+c*dy]}
function Lt(sx,sy){const t=OR.rot*Math.PI/180,c=Math.cos(t),s=Math.sin(t);sx*=OR.flip?-1:1;return[c*sx+s*sy,-s*sx+c*sy]}
// the 100x pictures turned as the chip map (as the image directions: rotate, then mirror; display only)
function orientImgs(){let st=document.getElementById('orImgs');if(!st){st=document.createElement('style');st.id='orImgs';document.head.appendChild(st)}
 st.textContent=`#bigImg,#p100 figure.h100 img{transform:${OR.flip?'scaleX(-1) ':''}rotate(${OR.rot}deg)}`}
function applyOrient(now){OR.rot=norm360(+OR.rot||0);OR.flip=!!OR.flip;lsSet('view',OR);orientImgs();
 if(IMG){V.viewport.setRotation(OR.rot,!!now);V.viewport.setFlip(OR.flip)}
 $('fRot').value=+OR.rot.toFixed(2);$('bMH').classList.toggle('on',OR.flip);refreshXY();sync();drawMini()}
// R: clockwise on screen (a mirrored view turns the other way round)
function rotBy(d){OR.rot+=OR.flip?-d:d;applyOrient()}
function mirror(vert){OR.flip=!OR.flip;if(vert)OR.rot+=180;applyOrient()}
$('bRR').onclick=()=>rotBy(90);$('bRL').onclick=()=>rotBy(-90);$('bMH').onclick=()=>mirror(false);$('bMV').onclick=()=>mirror(true);
$('bR0').onclick=()=>{OR.rot=0;OR.flip=false;applyOrient()};
$('fRot').addEventListener('change',()=>{OR.rot=+$('fRot').value||0;applyOrient()});
// XY.o origin (image px); XY.u,XY.v: image-frame unit vectors of +X,+Y once "Align X" is done (else they follow the view)
const XY=Object.assign({o:null,u:null,v:null},lsGet('xy')||{});let MODE=null,PTS=[];
const fmt1=v=>{const r=Math.round(v*10)/10;return (r===0?0:r).toFixed(1)};
function axes(){if(XY.u&&XY.v)return[XY.u,XY.v];return[Lt(1,0),Lt(0,-1)]}   // view axes: +X right, +Y up on screen
function toXY(p){const[u,v]=axes(),dx=p.x-XY.o[0],dy=p.y-XY.o[1];return[(dx*u[0]+dy*u[1])*D.pixel_um,(dx*v[0]+dy*v[1])*D.pixel_um]}
function ptTxt(p){if(!XY.o)return `${fmt1(p.x*D.pixel_um)}, ${fmt1(p.y*D.pixel_um)} µm from the image corner`;const q=toXY(p);return `X ${fmt1(q[0])}, Y ${fmt1(q[1])} µm`}
function refreshXY(){D.flakes.forEach(f=>{if(XY.o){const q=toXY(P(f.tx,f.ty));f.cx=q[0];f.cy=q[1]}else f.cx=f.cy=null});
 document.querySelector('#tbl table').classList.toggle('noxy',!XY.o);if(!XY.o&&(sortK==='cx'||sortK==='cy'))sortK='score';
 $('bAx').classList.toggle('on',!!XY.u);draw();if(selId!=null&&byId[selId])select(selId,false);status()}
function setMode(m){MODE=MODE===m?null:m;if(MODE==='axis'&&!XY.o)MODE='origin';
 [['bO','origin'],['bMe','measure']].forEach(([b,k])=>$(b).classList.toggle('on',MODE===k));if(MODE==='axis')$('bAx').classList.add('on');else $('bAx').classList.toggle('on',!!XY.u);
 $('svg').classList.toggle('pick',!!MODE);$('view').classList.toggle('pick',!!MODE);status()}
function axisBtn(){if(XY.u&&MODE!=='axis'){XY.u=XY.v=null;lsSet('xy',XY);refreshXY()}else setMode('axis')}
$('bO').onclick=()=>setMode('origin');$('bMe').onclick=()=>setMode('measure');$('bAx').onclick=axisBtn;
$('bClr').onclick=()=>{XY.o=XY.u=XY.v=null;PTS=[];lsSet('xy',XY);setMode(null);refreshXY();sync()};
V.addHandler('canvas-click',e=>{if(!MODE||!IMG||!e.quick)return;e.preventDefaultAction=true;
 const p=IMG.viewerElementToImageCoordinates(e.position);   // (OSD has already undone the mirroring of click positions)
 if(MODE==='origin'){XY.o=[p.x,p.y];lsSet('xy',XY);setMode(null);refreshXY()}
 else if(MODE==='axis'){const dx=p.x-XY.o[0],dy=p.y-XY.o[1],n=Math.hypot(dx,dy);if(n<1e-6)return;
  // +X towards the clicked point; +Y 90° counter-clockwise of it as displayed now (right-handed, as on screen)
  XY.u=[dx/n,dy/n];const s=Lm(XY.u[0],XY.u[1]);XY.v=Lt(s[1],-s[0]);lsSet('xy',XY);setMode(null);refreshXY()}
 else{if(PTS.length>=2)PTS=[];PTS.push(p);status()}
 sync()});
function copyTxt(t,btn){const ok=()=>{if(btn){const o=btn.textContent;btn.textContent='copied';setTimeout(()=>btn.textContent=o,900)}};
 const fb=()=>{const ta=document.createElement('textarea');ta.value=t;document.body.appendChild(ta);ta.select();try{document.execCommand('copy');ok()}catch(e){}ta.remove()};
 try{navigator.clipboard.writeText(t).then(ok,fb)}catch(e){fb()}}
function wireCopy(el){el.querySelectorAll('button.cp').forEach(b=>b.onclick=()=>copyTxt(b.dataset.c,b))}
function status(){const ax=XY.u?'chip axes (+X along the edge you clicked)':'view axes (+X right, +Y up)';
 let h=XY.o?`Origin set · ${ax}`:'';
 if(MODE==='origin')h=`<b>Click the feature to use as origin</b> (Esc cancels)`;
 else if(MODE==='axis')h=`<b>Click a second point along a chip edge</b>: it sets the +X direction (Esc cancels)`;
 else if(MODE==='measure'&&!PTS.length)h+=`<br><b>Click points on the map</b> to measure (Esc ends)`;
 const cp=p=>{const q=XY.o?toXY(p):[p.x*D.pixel_um,p.y*D.pixel_um];return `<button class="cp" data-c="${fmt1(q[0])}, ${fmt1(q[1])}">copy</button>`};
 PTS.forEach((p,i)=>{h+=`<br>${'AB'[i]}: <b>${ptTxt(p)}</b>${cp(p)}`});
 if(PTS.length===2){const a=PTS[0],b=PTS[1],dd=Math.hypot(b.x-a.x,b.y-a.y)*D.pixel_um;let dl='';
  if(XY.o){const qa=toXY(a),qb=toXY(b);dl=` (ΔX ${fmt1(qb[0]-qa[0])}, ΔY ${fmt1(qb[1]-qa[1])})`}
  h+=`<br>distance A–B: <b>${fmt1(dd)} µm</b>${dl}<button class="cp" data-c="${fmt1(dd)}">copy</button>`}
 h=h.replace(/^<br>/,'');$('cst').innerHTML=h;$('cst').hidden=!h;wireCopy($('cst'))}
// origin, axis arrows and measured points, drawn in screen pixels over the map
function drawMarks(){const M=$('mk');if(!M||!IMG)return;let h='';const sp=(x,y)=>img2scr(x,y);
 if(XY.o){const o=sp(XY.o[0],XY.o[1]),[u,v]=axes(),s=60;
  const tip=(w,col,lab)=>{const q=Lm(w[0],w[1]);const x=o.x+q[0]*s,y=o.y+q[1]*s;
   return `<line x1="${o.x}" y1="${o.y}" x2="${x}" y2="${y}" stroke="${col}"/><circle cx="${x}" cy="${y}" r="2.5" stroke="${col}"/><text x="${x+q[0]*10-4}" y="${y+q[1]*10+4}">${lab}</text>`};
  h+=tip(u,'#ff5050','X')+tip(v,'#50e070','Y')+`<circle cx="${o.x}" cy="${o.y}" r="6" stroke="#ffd400"/><circle cx="${o.x}" cy="${o.y}" r="1.5" stroke="#ffd400"/>`}
 const ps=PTS.map(p=>sp(p.x,p.y));
 if(ps.length===2)h+=`<line x1="${ps[0].x}" y1="${ps[0].y}" x2="${ps[1].x}" y2="${ps[1].y}" stroke="#00e5ff" stroke-dasharray="6 4"/>`;
 ps.forEach((p,i)=>{h+=`<circle cx="${p.x}" cy="${p.y}" r="5" stroke="#00e5ff"/><text x="${p.x+8}" y="${p.y-8}">${'AB'[i]}</text>`});
 M.innerHTML=h}
V.addHandler('rotate',()=>setTimeout(sync,0));V.addHandler('flip',()=>setTimeout(sync,0));
// ---- toggles
let labOn=true;
function layer(k){if(enhItem)enhItem.setOpacity(k==='enh'?1:0);if(linItem)linItem.setOpacity(k==='lin'?1:0);
 [['img','bImg'],['enh','bEnh'],['lin','bLin']].forEach(([kk,b])=>$(b).classList.toggle('on',kk===k))}
$('bImg').onclick=()=>layer('img');
$('bEnh').onclick=()=>{if(enhItem)layer('enh')};
$('bLin').onclick=()=>{if(linItem)layer('lin')};
if(!D.info.enhanced)$('bEnh').disabled=true;
if(!D.info.look)$('bLin').hidden=true;
$('bOut').onclick=e=>{e.target.classList.toggle('on');$('svg').style.display=e.target.classList.contains('on')?'':'none'};
$('bLab').onclick=e=>{e.target.classList.toggle('on');labOn=e.target.classList.contains('on');sync()};
// ---- table
let sortK='score',sortDir=-1,selId=null;
document.querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=(sortK===k)?-sortDir:(k==='id'?1:-1);sortK=k;draw()});
['fT0','fT1','fA','fS','fS1'].forEach(i=>$(i).addEventListener('input',draw));
if(!H100)document.querySelector('#tbl table').classList.add('no100');
if(H100){$('lH').hidden=false;$('lS1').hidden=false;$('fH').addEventListener('change',draw)}
// ---- the toolbar's menus: Image options opens on hover (and on click, for touch screens); Filter and Sort on click;
// a click elsewhere or Esc closes them
let menuT=null;
function openMenu(m,on){document.querySelectorAll('.menu').forEach(x=>x.classList.toggle('open',x===m&&on))}
const MI=$('mImg');
MI.addEventListener('mouseenter',()=>{clearTimeout(menuT);openMenu(MI,true)});
MI.addEventListener('mouseleave',()=>{clearTimeout(menuT);menuT=setTimeout(()=>{if(!MI.contains(document.activeElement)||document.activeElement.tagName!=='INPUT')MI.classList.remove('open')},250)});
document.querySelectorAll('.menu>.tb').forEach(b=>b.onclick=()=>{const m=b.parentNode;openMenu(m,m===MI||!m.classList.contains('open'))});
document.addEventListener('mousedown',e=>{if(!e.target.closest('.menu'))openMenu(null,false)});
$('fRot').addEventListener('blur',()=>{if(!MI.matches(':hover'))MI.classList.remove('open')});
const SORTS=[['score','10x score'],['s100','100x score'],['t','thickness'],['area','flake area'],['bd','best terrace Ø'],['n','number of terraces'],['id','flake number'],['cx','X (from the origin)'],['cy','Y (from the origin)']];
function sortOk(k){return (k!=='s100'||H100)&&((k!=='cx'&&k!=='cy')||XY.o)}
function sortMenu(){$('sortPop').innerHTML='<h4>Sort by</h4>'+SORTS.filter(s=>sortOk(s[0])).map(([k,l])=>`<label><input type="radio" name="srt" value="${k}"${k===sortK?' checked':''}> ${l}</label>`).join('')
  +`<h4>Order</h4><label><input type="radio" name="sdir" value="-1"${sortDir<0?' checked':''}> highest first</label><label><input type="radio" name="sdir" value="1"${sortDir>0?' checked':''}> lowest first</label>`;
 $('sortPop').querySelectorAll('input[name=srt]').forEach(r=>r.onchange=()=>{sortK=r.value;sortDir=r.value==='id'?1:-1;draw()});
 $('sortPop').querySelectorAll('input[name=sdir]').forEach(r=>r.onchange=()=>{sortDir=+r.value;draw()});
 const s=SORTS.find(s=>s[0]===sortK);$('bSort').textContent=`Sort: ${s?s[1]:sortK} ${sortDir<0?'↓':'↑'}`;
 document.querySelectorAll('th[data-k]').forEach(th=>{th.classList.toggle('srt',th.dataset.k===sortK);th.style.setProperty('--arr',sortDir<0?'"▼"':'"▲"')})}
function nFilters(){let n=0;if(+$('fT0').value>D.tmin||+$('fT1').value<D.tmax)n++;if(+$('fA').value>0)n++;if(+$('fS').value>0)n++;
 if(H100&&+$('fS1').value>0)n++;if(H100&&$('fH').checked)n++;return n}
function resetFilters(){$('fT0').value=D.tmin;$('fT1').value=D.tmax;$('fA').value=0;$('fS').value=0;$('fS1').value=0;$('fH').checked=false;
 if(M100)setM100(false);else draw()}
$('bFR').onclick=resetFilters;$('bFR2').onclick=resetFilters;
function visible(f){const t0=+$('fT0').value,t1=+$('fT1').value,a=+$('fA').value,s=+$('fS').value;
 if(f.area<a||(f.score!=null&&f.score<s))return false;
 if(H100&&$('fH').checked&&!H100[f.id])return false;
 if(H100&&+$('fS1').value>0&&!((f.s100||0)>=+$('fS1').value))return false;
 return f.terr.some(tid=>{const T=D.terraces[tid];return T&&T.t>=t0&&T.t<=t1})}
const fmtS=v=>v==null?'–':v.toFixed(2);
let ORDER=[];
function draw(){const rows=D.flakes.filter(visible).sort((a,b)=>{const x=a[sortK]??-1,y=b[sortK]??-1;return (x>y?1:x<y?-1:0)*sortDir});ORDER=rows.map(f=>f.id);
 const vis=new Set(rows.map(f=>f.id));Object.values(byId).forEach(o=>o.g.style.display=vis.has(o.f.id)?'':'none');
 const nf=nFilters();$('bFil').textContent=nf?`Filter (${nf})`:'Filter';$('bFil').classList.toggle('on',nf>0);
 $('bFR2').hidden=!nf;$('nft').textContent=`${rows.length} of ${D.flakes.length} flakes shown`;sortMenu();
 // (the row number: the place in the current order, whatever the sort and the filters)
 $('rows').innerHTML=rows.map((f,i)=>`<tr data-id="${f.id}" class="${f.id===selId?'sel':''}"><td class="rn">${i+1}</td><td>${f.id}${H100&&H100[f.id]&&f.s100==null?'<span class="h1b">100x</span>':''}</td><td>${fmtS(f.score)}</td><td class="s1">${f.s100==null?'':Math.round(f.s100)}</td><td><span class="sw" style="background:${col(f.t)}"></span>${f.t}</td><td class="nt">${f.n}</td><td>${f.area.toFixed(0)}</td><td>${f.bd.toFixed(0)}</td><td class="xy">${f.cx==null?'':fmt1(f.cx)}</td><td class="xy">${f.cy==null?'':fmt1(f.cy)}</td></tr>`).join('');
 $('rows').querySelectorAll('tr').forEach(tr=>tr.onclick=()=>select(+tr.dataset.id,true));sync()}
// 100x (and stacking-domain) images of a flake, written by flakeid.hundred after the 100x imaging
// views of a 100x image: key, label, panel image, full image.  "avg": the frame-averaged main image alone, without the
// two shifted frames (written by flakeid.hundred from 2026-10-04); "ir": the flake's IR (stacking-domain) picture laid
// over the 100x frame (from 2026-10-15)
const VIEWS=[['nat','natural','p','f'],['lin','linear','p','f'],['c8','colour ±8 %','c8','c8'],['sg','green ±4 %','sg','sg'],['deb','debris','deb','deb'],
 ['a','natural · avg frame','a','a'],['a_c8','colour · avg frame','a_c8','a_c8'],['a_sg','green · avg frame','a_sg','a_sg'],
 ['ir','IR · stacking domains','ir','ir'],['mos','mosaic','mp','mos'],['irmos','IR mosaic','irmos','irmos']];
// "mos": the stitched mosaic of the flake's whole terrace (flakeid.hundred, from 2026-10-19: terraces larger than a field);
// "irmos": the same with the IR camera
let VIEW='nat',BIG=false,NOTIR='nat',NOTMOS='nat';
function mosTxt(h,v){return v==='mos'&&h.mos_um?` · mosaic of ${h.mos_n} fields, ${Math.round(h.mos_um[0])} × ${Math.round(h.mos_um[1])} µm`:''}
// h.look (flakeid.look, from 2026-10-27): the copies people look at, shown in place of f / p / a / mos / mp; "linear"
// (only when there is a look) shows the originals, the colours the analysis measures
function viewsOf(h){if(h.domain)return[];const L=h.look||{};
 return VIEWS.map(([k,l,ps,fs])=>k==='lin'?(h.look?[k,l,h[ps]||h[fs],h[fs]]:[k,l,null,null]):[k,l,L[ps]||h[ps]||L[fs]||h[fs],L[fs]||h[fs]]).filter(v=>v[2])}
function pickView(h){const vs=viewsOf(h);return vs.find(v=>v[0]===VIEW)||vs[0]}
// the selected flake's 100x (and stacking-domain) pictures: a card at the bottom right of the map (2026-10-27; before,
// under the table), Hide / Show 100x remembered; the details in the picture's tooltip; click: full resolution
function stTxt(h){const s=h.stats;if(!s)return'';return `${h.s100?`100x score ${Math.round(h.s100[0])} µm² (largest clean area, Ø ${h.s100[1]} µm) · `:''}terrace ${Math.round(s.terrace_um2)} µm²${s.cut?' (cut by the frame)':''} · debris ${s.debris_n} spot${s.debris_n===1?'':'s'} (${s.debris_um2} µm²)${s.steps_um2?` · other thickness ${Math.round(s.steps_um2)} µm²`:''} · clean ${Math.round(s.clean_um2)} µm², largest clean Ø ${s.clean_d_um} µm${h.focus?' · focus '+h.focus:''}`}
function imgs100(id){const hs=(H100&&H100[id])||[];
 return hs.map((h,k)=>{const cur=pickView(h);
  return `<figure class="h100" data-k="${k}"><a href="${cur?cur[3]:h.f}" target="_blank" title="${stTxt(h)}"><img src="${cur?cur[2]:h.p||h.f}" loading="lazy" alt="${h.domain?'Stacking-domain':'100x'} image of flake ${id}"></a><figcaption>${h.domain?'stacking domains':h.lab}${cur?mosTxt(h,cur[0]):''}${h.raw?` · <a href="${h.raw}" download>16-bit</a>`:''}${h.domain&&h.mos?` · <a href="${h.mos}" target="_blank">IR mosaic</a>`:''}</figcaption></figure>`}).join('')}
let HIDE100=!!lsGet('hide100');
function card100(){const hs=(H100&&H100[selId])||[],any=hs.length>0&&!BIG;
 $('p100').hidden=!any||HIDE100;$('p100s').hidden=!any||!HIDE100;if(!any)return;
 const h=first100(selId);$('p100t').innerHTML=`<b>Flake ${selId}</b> · 100x${h&&h.s100?` · score <b>${Math.round(h.s100[0])} µm²</b>`:''}`;
 $('p100b').innerHTML=imgs100(selId)}
$('bHide').onclick=()=>{HIDE100=true;lsSet('hide100',true);card100()};$('p100s').onclick=()=>{HIDE100=false;lsSet('hide100',false);card100()};
// the chosen view applies to every 100x image (and stays when moving to another flake)
function setView(v){VIEW=v;card100();showBig();vbar()}
// the view bar at the bottom centre: the selected flake's 100x views (as the keys 1-9)
function vbar(){const B=$('vbar'),h=first100(selId),vs=h?viewsOf(h):[];if(vs.length<2){B.hidden=true;return}
 const cur=pickView(h);B.innerHTML=vs.map((v,j)=>`<button data-v="${v[0]}" aria-pressed="${v[0]===cur[0]}"><kbd>${j+1}</kbd>${v[1]}</button>`).join('');
 B.querySelectorAll('button').forEach(b=>b.onclick=()=>setView(b.dataset.v));B.hidden=false}
function first100(id){return ((H100&&H100[id])||[]).find(h=>!h.domain)}
// ---- large 100x view: the image fills the map's place, the whole chip small with the flake marked
const MINI={cv:$('mini'),img:null,s:1};
function buildMini(){const I=D.info.image,ts=I.tileSize;let L=I.maxLevel;
 while(L>0&&Math.max(I.width,I.height)/2**(I.maxLevel-L)>480)L--;
 const k=2**(I.maxLevel-L),w=Math.ceil(I.width/k),h=Math.ceil(I.height/k);
 const off=document.createElement('canvas');off.width=w;off.height=h;const c=off.getContext('2d');c.fillStyle='#000';c.fillRect(0,0,w,h);
 MINI.img=off;MINI.s=1/k;
 for(let x=0;x*ts<w;x++)for(let y=0;y*ts<h;y++){const im=new Image();im.onload=()=>{c.drawImage(im,x*ts,y*ts);drawMini()};im.src=`tiles/${BASE.dir}/${L}/${x}_${y}.jpg`}}
function drawMini(){if(!MINI.img||!BIG)return;const c=MINI.cv.getContext('2d'),w=MINI.img.width,h=MINI.img.height;
 // same orientation as the chip map: rotate, then mirror left-right
 const t=OR.rot*Math.PI/180,cs=Math.abs(Math.cos(t)),sn=Math.abs(Math.sin(t)),cw=Math.round(w*cs+h*sn),ch=Math.round(w*sn+h*cs);
 if(MINI.cv.width!==cw||MINI.cv.height!==ch){MINI.cv.width=cw;MINI.cv.height=ch}
 c.setTransform(1,0,0,1,0,0);c.fillStyle='#000';c.fillRect(0,0,cw,ch);c.translate(cw/2,ch/2);if(OR.flip)c.scale(-1,1);c.rotate(t);c.translate(-w/2,-h/2);
 c.drawImage(MINI.img,0,0);const o=byId[selId];if(!o)return;
 const f=o.f,x=f.x*MINI.s,y=f.y*MINI.s,r=Math.max(7,Math.max(f.bw,f.bh)*MINI.s*0.8);
 c.lineWidth=2;c.strokeStyle='#ffd400';c.beginPath();c.arc(x,y,r,0,7);c.stroke();
 c.beginPath();c.moveTo(x-r-8,y);c.lineTo(x-r-2,y);c.moveTo(x+r+2,y);c.lineTo(x+r+8,y);c.moveTo(x,y-r-8);c.lineTo(x,y-r-2);c.moveTo(x,y+r+2);c.lineTo(x,y+r+8);c.stroke()}
function showBig(){if(!BIG)return;const h=first100(selId),cur=h&&pickView(h);
 $('bigImg').src=cur?cur[3]:'';$('bigImg').alt=cur?`100x image of flake ${selId} (${cur[1]})`:'';
 const vs=h?viewsOf(h):[];
 $('bigCap').innerHTML=h?`Flake <b>${selId}</b> · ${h.lab}${h.s100?` · <b>100x score ${Math.round(h.s100[0])} µm²</b>`:''} · <b>${cur[1]}</b>${mosTxt(h,cur[0])} &nbsp; ${vs.map((v,j)=>`<kbd>${j+1}</kbd> ${v[1]}`).join(' &nbsp;')} &nbsp; <kbd>C</kbd> colour <kbd>A</kbd> avg frame${h.ir?' <kbd>I</kbd> IR':''}${h.mos||h.irmos?' <kbd>W</kbd> whole terrace (mosaic)':''} <kbd>↑</kbd><kbd>↓</kbd> flakes <kbd>L</kbd>/<kbd>Esc</kbd> map`
  :`Flake ${selId ?? ''}: no 100x image · <kbd>↑</kbd><kbd>↓</kbd> flakes with 100x images · <kbd>L</kbd>/<kbd>Esc</kbd> back to the map`;drawMini()}
function setBig(on){BIG=on&&!!H100;$('big').hidden=!BIG;$('bBig').classList.toggle('on',BIG);if(BIG){if(!MINI.img)buildMini();showBig()}card100()}
$('bBig').onclick=()=>setBig(!BIG);
// 100x mode: the flakes with 100x pictures only, ranked by the 100x score, the picture large; off: as before
let M100=false,M100prev=null;
function setM100(on){M100=on&&!!H100;$('bM100').classList.toggle('on',M100);
 if(M100){M100prev=[sortK,sortDir,$('fH').checked];sortK='s100';sortDir=-1;$('fH').checked=true}
 else if(M100prev){[sortK,sortDir]=M100prev;$('fH').checked=M100prev[2];M100prev=null}
 draw();setBig(M100);if(M100&&ORDER.length)select(ORDER[0],false)}
if(H100){$('bM100').hidden=false;$('bM100').onclick=()=>setM100(!M100)}$('mini').onclick=()=>setBig(false);
if(H100)$('bBig').hidden=false;
function step(d){let ids=ORDER;if(BIG)ids=ids.filter(i=>first100(i));if(!ids.length)return;
 const i=ids.indexOf(selId);const j=i<0?(d>0?0:ids.length-1):Math.max(0,Math.min(ids.length-1,i+d));select(ids[j],!BIG)}
document.addEventListener('keydown',e=>{const t=e.target,tg=t.tagName;
 if((tg==='INPUT'&&!/^(checkbox|radio|button)$/.test(t.type))||tg==='TEXTAREA'||tg==='SELECT'||e.metaKey||e.ctrlKey||e.altKey)return;
 const k=e.key;let hit=true;
 if(k==='ArrowDown'||k==='ArrowRight')step(1);else if(k==='ArrowUp'||k==='ArrowLeft')step(-1);
 else if(k==='l'||k==='L')setBig(!BIG);else if(k==='Escape'&&document.querySelector('.menu.open'))openMenu(null,false);else if(k==='Escape'&&MODE)setMode(null);else if(k==='Escape'&&BIG)setBig(false);
 else if(k==='r'||k==='R')rotBy(k==='r'?90:-90);else if(k==='h'||k==='H')mirror(false);else if(k==='v'||k==='V')mirror(true);
 else if(k==='o'||k==='O')setMode('origin');else if(k==='x'||k==='X')axisBtn();else if(k==='m'||k==='M')setMode('measure');
 else if(/^[1-9]$/.test(k)){const h=first100(selId);const vs=h?viewsOf(h):[];const v=vs[+k-1];if(v)setView(v[0]);else hit=false}
 else if(k==='i'||k==='I'){const h=first100(selId);if(VIEW==='ir'||VIEW==='irmos')setView(NOTIR);else if(h&&h.ir){NOTIR=VIEW;setView('ir')}else if(h&&h.irmos){NOTIR=VIEW;setView('irmos')}else hit=false}
 else if(k==='w'||k==='W'){const h=first100(selId);   // W: the whole terrace (mosaic) and back; in the IR view its IR mosaic
  if(VIEW==='mos')setView(NOTMOS);else if(VIEW==='irmos')setView(h&&h.ir?'ir':NOTIR);else if(VIEW==='ir'&&h&&h.irmos)setView('irmos');
  else if(h&&h.mos){NOTMOS=VIEW;setView('mos')}else if(h&&h.irmos){NOTIR=VIEW;setView('irmos')}else hit=false}
 else if(k==='c'||k==='C'){const avg=VIEW.startsWith('a');setView(VIEW==='c8'||VIEW==='a_c8'?(avg?'a':'nat'):(avg?'a_c8':'c8'))}
 else if(k==='a'||k==='A'){const m={nat:'a',c8:'a_c8',a:'nat',a_c8:'c8',sg:'a_sg',a_sg:'sg'};const h=first100(selId);const nv=m[VIEW]||'a';if(h&&h[nv==='nat'?'f':nv])setView(nv);else hit=false}
 else hit=false;if(hit)e.preventDefault()});
function zoomTo(x,y,m){if(IMG)V.viewport.fitBounds(IMG.imageToViewportRectangle(x-m,y-m,2*m,2*m))}
function select(id,zoom){selId=id;const o=byId[id];if(!o)return;const f=o.f;
 G.querySelectorAll('polygon.sel,polygon.tsel').forEach(p=>p.classList.remove('sel','tsel'));o.g.querySelectorAll('polygon.fl').forEach(p=>p.classList.add('sel'));
 document.querySelectorAll('#rows tr').forEach(tr=>tr.classList.toggle('sel',+tr.dataset.id===id));
 if(zoom)zoomTo(f.x,f.y,Math.max(f.bw,f.bh)*0.9+60);
 const st=f.sx!=null?`<br>stage <b>${f.sx}, ${f.sy}</b> µm`:'';
 const rel=XY.o?`<br>from the origin: <b>${ptTxt(P(f.tx,f.ty))}</b> <button class="cp" data-c="${fmt1(f.cx)}, ${fmt1(f.cy)}">copy</button>`:'';
 const tl=f.terr.map(tid=>{const T=D.terraces[tid];if(!T)return'';return `<div class="trow" data-t="${tid}"><span class="sw" style="background:${col(T.t)}"></span><b>${T.t} nm</b>${T.alt?' (alt '+T.alt+')':''}${D.material==='graphite'&&T.t<3?' · '+T.layers+' L':''} · Ø ${T.d} µm · ${Math.round(T.area)} µm² · 10x score ${fmtS(T.score)}${T.q!=='good'?' · <i>'+T.q.replace(/;/g,', ')+'</i>':''}</div>`}).join('');
 const more=`focus ${f.focus??'–'} µm · crowding ${f.crowd} · in window ${Math.round(100*(f.fw||0))} % · thick ${Math.round(100*(f.fthick||0))} %`;
 $('detail').innerHTML=`<b title="${more}">Flake ${f.id}</b> · 10x score ${fmtS(f.score)}${f.s100!=null?` · <b>100x score ${Math.round(f.s100)} µm²</b>`:''} · ${f.L}×${f.W} µm · ${Math.round(f.area)} µm²${st}${rel}
 <div class="lbl2">Terraces — click to zoom</div>${tl}`;
 card100();showBig();vbar();wireCopy($('detail'));
 // a terrace: back to the map (from the large 100x view), zoomed onto it and outlined (before 2026-10-27 the zoom was
 // nearly the flake's own, and nothing showed while the large 100x view covered the map)
 $('detail').querySelectorAll('.trow').forEach(el=>el.onclick=()=>{const tid=el.dataset.t,T=D.terraces[+tid];if(!T)return;
  if(BIG)setBig(false);G.querySelectorAll('polygon.tsel').forEach(p=>p.classList.remove('tsel'));
  G.querySelectorAll(`polygon.tr[data-t="${tid}"]`).forEach(p=>p.classList.add('tsel'));
  $('detail').querySelectorAll('.trow').forEach(r=>r.classList.toggle('on',r===el));
  zoomTo(T.tx,T.ty,Math.max(T.L,T.W)/D.pixel_um*0.55+15)});
 const tr=document.querySelector(`#rows tr[data-id="${id}"]`);if(tr&&!zoom)tr.scrollIntoView({block:'nearest'});
 try{history.replaceState(null,'','#f'+id)}catch(e){}}
// deep links: #f12 (flake 12) or #t34 (terrace 34)
function fromHash(){const h=location.hash.slice(1);const m=/^([ft])(\d+)$/.exec(h);if(!m)return;
 if(m[1]==='f')select(+m[2],true);else{const T=D.terraces[+m[2]];if(T){select(T.fl,false);zoomTo(T.tx,T.ty,Math.max(T.L,T.W)/D.pixel_um*0.9+40)}}}
V.addHandler('open',()=>{applyOrient(true);V.viewport.goHome(true);setTimeout(fromHash,50)});window.addEventListener('hashchange',fromHash);
$('keys').innerHTML=`<kbd>↑</kbd><kbd>↓</kbd> next / previous flake · <kbd>R</kbd> rotate · <kbd>H</kbd>/<kbd>V</kbd> mirror · <kbd>O</kbd> origin · <kbd>X</kbd> align X · <kbd>M</kbd> measure${H100?' · <kbd>L</kbd> large 100x · <kbd>1</kbd>–<kbd>9</kbd> 100x views · <kbd>C</kbd> colour · <kbd>A</kbd> avg frame · <kbd>W</kbd> whole terrace':''} · scroll to zoom, drag to pan`;
$('fRot').value=+norm360(+OR.rot||0).toFixed(2);$('bMH').classList.toggle('on',!!OR.flip);orientImgs();refreshXY();
</script></body></html>
"""
