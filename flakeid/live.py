"""Analyse a Micro-Manager tile scan while it is being acquired.

    python -m flakeid.live "Z:/scans/chipA_tiles" hbn 285 --flat flat_6.3x.npz -o chipA_live

The watcher polls the tile folder written by the scan program (python -m flakeid.scope; a shared / synced folder works:
only finished tiles, whose metadata.txt is written after the TIFF, are read).  The scan is divided
into blocks of BLOCK x BLOCK tiles.  As soon as a block and the ring of tiles around it have arrived,
it is stitched on its own (registration within the block), cropped to the block plus MARGIN_UM on
every side and analysed with find_flakes in a worker process.  A flake belongs to the block whose
core tile is nearest to its target point (the same nearest-centre rule the mosaic uses), so flakes
crossing block boundaries are reported once, from a block in which they are complete.

The scan plan (<NAME>_scan.pos next to the <NAME>_tiles folder) gives the tiles to wait for and the
end of the scan; without it the watcher waits until the scan has moved past a block's last row and
stops after --idle seconds without new tiles.

After every block the ranked tables <name>_live_flakes.csv / _terraces.csv are rewritten; when the
scan is complete the 100x target list (<name>_live_targets.pos / .csv, with the 10x focus map in
z10_um) is written, and --then-batch runs the full-mosaic analysis for the zoomable viewer (and
the tile TIFFs are deleted once the viewer is written unless --keep-tiles).  Live flake numbers are the live ranking; the batch
analysis ranks independently (match them by stage position).

Testing: --replay SRC copies the tiles of a finished scan into the watched folder in acquisition
order, one every --interval seconds, to reproduce a scan.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .mm import (DEFAULT_M, StageMap, Tile, _prep, check_linear, compose, fit_layout, full_scale, glitch_frame, prepare_flat,
                 read_frame, read_tile)
from .pipeline import DEFAULT_RANGE_NM, FlakeResult, FlakeSummary, Terrace

_POSITION = re.compile(r"position(\d+)")


class LiveError(RuntimeError):
    """The analysis cannot work (e.g. the flat field cannot be read): stop instead of writing an empty target list."""


def replace_file(tmp, dst, tries=20):
    """os.replace that rides out a moment of "access is denied" on Windows (a virus scanner or the search indexer
    reading the destination; it stopped the analysis on the microscope PC, 2026-10-03): retried for ~10 s."""
    for k in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.5)


def check_flat(path):
    """Fail early, with a useful message, if the flat field cannot be used (every block would fail)."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        here = sorted(str(q) for q in Path.cwd().rglob("flat*.npz"))[:6] if Path.cwd() != Path.home() else []
        raise SystemExit(f"--flat {path}: no such file (looked in {Path.cwd()})."
                         + (f"  Flat fields found under this folder: {', '.join(here)}" if here else ""))
    from .mm import FlatField
    try:
        FlatField.load(p)
    except Exception as exc:
        raise SystemExit(f"--flat {path}: cannot be read as a flat field ({exc})")
    return str(p.resolve())


def _index(tile: Tile) -> int:
    m = _POSITION.search(tile.path.name)
    return int(m.group(1)) if m else -1


def read_plan(folder: Path) -> dict | None:
    """{(col, row): (x, y)} of the planned tiles, from <NAME>_scan.pos next to <NAME>_tiles[_n]."""
    name = re.sub(r"_tiles(_\d+)?$", "", folder.name)
    pos = folder.parent / f"{name}_scan.pos"
    if not pos.is_file():
        return None
    d = json.loads(pos.read_text(encoding="utf-8", errors="replace"))
    plan = {}
    for e in d["map"]["StagePositions"]["array"]:
        xy = next(dp["Position_um"]["array"] for dp in e["DevicePositions"]["array"] if len(dp["Position_um"]["array"]) == 2)
        plan[(int(e["GridCol"]["scalar"]), int(e["GridRow"]["scalar"]))] = (float(xy[0]), float(xy[1]))
    return plan


# ---------------------------------------------------------------------------
# One block (runs in a worker process)
# ---------------------------------------------------------------------------

_FLAT = {}

# each tile's registration input (mm._prep of its flat-corrected frame, ~0.1 s per tile) kept in the worker process: a
# ring tile is a core or ring tile of up to 4 blocks, and blocks analysed before the oxide was fixed are analysed
# again.  Bounded (oldest dropped first); same values as computed afresh.  (A block's frames themselves are read again:
# reading, demosaicking and flat-fielding a tile is ~20 ms; keeping them would take 35 MB per tile)
PREP_CACHE_MB = 256.0
_PREP = {}                      # (path, mtime, size, flat) -> prep (insertion order = age)
_PREP_BYTES = [0]


def _prep_cached(tile, flat_path, corrected):
    try:
        st = tile.path.stat()
        key = (str(tile.path), st.st_mtime_ns, st.st_size, flat_path)
    except OSError:
        key = None
    g = _PREP.pop(key, None) if key is not None else None
    if g is None:
        g = _prep(corrected)
    if key is not None and PREP_CACHE_MB > 0:
        _PREP[key] = g                                    # (re-inserted: now the newest)
        _PREP_BYTES[0] = sum(v.nbytes for v in _PREP.values())
        while _PREP and _PREP_BYTES[0] > PREP_CACHE_MB * 1e6:
            _PREP_BYTES[0] -= _PREP.pop(next(iter(_PREP))).nbytes
    return g


def _flat(path, tile0, fs):
    if path not in _FLAT:
        _FLAT[path] = prepare_flat(path, tile0, fs) if path else (None, 1.0)
    return _FLAT[path]


def analyse_block(tiles: list, core: list, cfg: dict) -> dict:
    """Stitch `tiles` (the block and its ring), analyse the block + margin, keep the flakes owned by `core`."""
    from .io import ImageData
    from .pipeline import find_flakes
    t0 = time.time()
    fs = full_scale(tiles[0])
    flat, fmin = _flat(cfg["flat"], tiles[0], fs)

    shifts = {}

    def corrected(k):
        r = read_frame(tiles[k], shifts.get(k))
        return r, (r if flat is None else flat.apply(r))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:                                     # (4 per worker: workers run in parallel too)
        frames = list(ex.map(corrected, range(len(tiles))))
    bad = [i for i in range(len(tiles)) if glitch_frame(tiles[i])]      # camera garbage frames: left out
    M = np.asarray(cfg["M"], float)
    P = np.array([t.stage_xy for t in tiles])
    h, w = frames[0][0].shape[:2]
    reg = "stage"
    smap = None
    if len(tiles) > 1:
        try:
            smap, stats = fit_layout(tiles, [_prep_cached(t, cfg["flat"], c) for t, (_, c) in zip(tiles, frames)], verbose=False, M_prior=M)
            # a block has few pairs: keep the scan-wide pixel scale / rotation if the fit disagrees
            if abs(smap.pixel_um / math.sqrt(abs(np.linalg.det(M))) - 1) > 0.01:
                smap = None
            else:
                reg = f"{stats['used']}/{stats['pairs']} pairs"
        except RuntimeError:
            smap = None
    if smap is None:
        q = (np.linalg.inv(M) @ (P - P[0]).T).T
        smap = StageMap(M=M, P=P, q=q - q.min(0), frame_wh=(w, h), labels=[t.label for t in tiles],
                        z=[t.z_um for t in tiles], z_units=tiles[0].z_units)

    dark = 0.0 if flat is None else float(np.max(flat.dark))
    k = 0.96 * 65535.0 * fmin / max(fs - dark, 1.0)          # as mm.load_dataset

    def tile16(i):
        r, c = frames[i]
        v = np.clip(c * k, 0, 0.965 * 65535).astype(np.uint16)
        v[r >= fs - 0.5] = 65535
        return v

    mosaic, covered = compose(tile16, smap, dtype=np.uint16, skip=bad)
    del frames
    # crop: the core tiles' footprints + margin
    ci = [smap.labels.index(c) for c in core]
    qc = np.round(smap.q[ci]).astype(int)
    mpx = int(round(cfg["margin_um"] / smap.pixel_um))
    x0, y0 = np.maximum(qc.min(0) - mpx, 0)
    x1 = min(int(qc[:, 0].max()) + w + mpx, mosaic.shape[1])
    y1 = min(int(qc[:, 1].max()) + h + mpx, mosaic.shape[0])
    rgb = np.ascontiguousarray(mosaic[y0:y1, x0:x1])
    cov = covered[y0:y1, x0:x1]
    del mosaic, covered
    cmap = StageMap(M=smap.M, P=smap.P, q=smap.q - np.array([x0, y0]), frame_wh=smap.frame_wh, labels=smap.labels,
                    z=smap.z, z_units=smap.z_units)
    img = ImageData(rgb=rgb, path=Path(cfg["name"]), pixel_um=cmap.pixel_um,
                    pixel_um_source=f"block registration ({reg})",
                    metadata=dict(source="micromanager", data_format=tiles[0].data_format),
                    max_value=65535, clip_value=k * (fs - dark), stage_map=cmap)
    out = dict(core=core, n_tiles=len(tiles), registration=reg, flakes=[], seconds=0.0, oxide=None, gain=0.0)
    if cfg.get("oxide_only"):
        # before the oxide is fixed: only its refinement samples and the substrate level (~1/5 of a full analysis); the
        # block is analysed in full once, with the fixed oxide (before 2026-10-21 every block analysed before the oxide
        # was fixed was analysed again: 5-8 blocks of ~25 on every hBN piece)
        out["oxide_only"] = True
        if cov.mean() >= 0.05:
            try:
                res = find_flakes(img, cfg["material"], cfg["oxide_nm"], refine_oxide_thickness=True,
                                  substrate_hint=cfg.get("substrate"), oxide_only=True, **cfg.get("options", {}))
                out["oxide"], out["gain"], out["refined"] = res.oxide_nm_used, res.params.get("oxide_refine_gain", 0.0), True
                out["substrate"] = np.asarray(res._substrate_rgb, float).tolist()
                if res._oxide_samples is not None and len(res._oxide_samples) > 500:
                    out["samples"] = res._oxide_samples.astype(np.float32)
                    out["clip_max"] = np.asarray(res._clip_max, float).tolist()
                    out["profile"] = res.params.get("profile")
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"
        out["seconds"] = time.time() - t0
        return out
    cc = cfg.get("canvas")

    def to_canvas(res_=None):
        # the block's own tiles for the viewer canvas (display gain from the scan's substrate level once it is fixed):
        # rendered here, pasted by the main process, which alone writes the canvas file (livecanvas.paste).  Never
        # fatal: the canvas only feeds the viewer image
        if cc is None:
            return
        try:
            from . import livecanvas
            sub = cfg.get("substrate") or (res_._substrate_rgb if res_ is not None else None)
            if sub is None:
                v = rgb[::16, ::16].reshape(-1, 3)
                v = v[v.min(1) > 0]
                sub = np.median(v, 0) if len(v) else np.full(3, 20000.0)
            out["canvas"] = livecanvas.render_block(cc, cmap, core, rgb, res_._contrast if res_ is not None else None,
                                                    res_._scale if res_ is not None else 0.5, sub, None)
        except Exception as exc:
            out["canvas_error"] = f"{type(exc).__name__}: {exc}"
    if cov.mean() < 0.05:
        to_canvas()
        return out
    try:
        res = find_flakes(img, cfg["material"], cfg["oxide_nm"], refine_oxide_thickness=cfg["refine"],
                          substrate_hint=cfg.get("substrate"), keep_contrast=cc is not None, **cfg.get("options", {}))
    except Exception as exc:          # e.g. a block that is nearly all off-chip
        out["error"] = f"{type(exc).__name__}: {exc}"
        to_canvas()
        return out
    to_canvas(res)
    out["oxide"], out["gain"] = res.oxide_nm_used, res.params.get("oxide_refine_gain", 0.0)
    out["refined"] = bool(cfg["refine"])
    out["pre_cal"] = not cfg.get("calibrated", True)
    out["substrate"] = np.asarray(res._substrate_rgb, float).tolist()
    if res._oxide_samples is not None and len(res._oxide_samples) > 500:
        out["samples"] = res._oxide_samples.astype(np.float32)
        out["clip_max"] = np.asarray(res._clip_max, float).tolist()
        out["profile"] = res.params.get("profile")
    coreset = set(ci)
    by_id = {}
    for t in res.terraces:
        by_id.setdefault(t.flake_id, []).append(t)
    H, W = rgb.shape[:2]
    polys_t, polys_f = ({}, {})
    if cc is not None:                # outlines of the flakes this block owns, at the real edge (crop pixels)
        from .outline import polygons, refine_flake
        from .zoomview import _outlines
        polys_t, polys_f = _outlines(res)
    for f in res.flakes:
        if cmap.owner(f.target_x_px, f.target_y_px) not in coreset:
            continue
        if cc is not None:
            try:
                rf = refine_flake(rgb, res, f)
            except Exception:
                rf = None
            if rf is not None:
                polys_f[f.flake_id] = polygons(rf[2], rf[0], rf[1], 1.0)
                for tid, m in rf[3].items():
                    polys_t[tid] = polygons(m, rf[0], rf[1], 0.8)
        # cut by the crop (the flake reaches beyond the margin): area and shape are incomplete
        edge = (f.bbox_x_px <= 1 or f.bbox_y_px <= 1 or f.bbox_x_px + f.bbox_w_px >= W - 1
                or f.bbox_y_px + f.bbox_h_px >= H - 1) and not (x0 == 0 and y0 == 0 and cov.all())
        terr = by_id.get(f.flake_id, [])
        if edge:
            f.quality = "block_edge" if f.quality == "good" else f.quality + ";block_edge"
        fd, tds = asdict(f), [asdict(t) for t in terr]
        if cc is not None:
            fd["_poly"] = polys_f.get(f.flake_id, [])
            for td in tds:
                td["_poly"] = polys_t.get(td["terrace_id"], [])
        out["flakes"].append((fd, tds))
    out["clean"] = [c for c in getattr(res, "_clean", []) if cmap.owner(c["target_x_px"], c["target_y_px"]) in coreset]
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

class LiveScan:
    def __init__(self, folder, material, oxide_nm, out_dir, flat=None, block=4, margin_um=400.0, workers=None,
                 poll_s=2.0, idle_s=120.0, refine_oxide=True, options=None, log=print, viewer=True, workers_final=None):
        self.folder, self.out = Path(folder), Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.name = self.folder.name
        # oxide: entered as the wafers' nominal thickness; modelled as measured (profiles/oxide_nominal.json, "90" -> 82)
        from .thickness import model_oxide_nm
        self.oxide_entered = float(oxide_nm)
        self.material, self.oxide_nm, self.flat = material, model_oxide_nm(oxide_nm), (str(flat) if flat else None)
        self.block, self.margin_um, self.poll_s, self.idle_s = int(block), float(margin_um), poll_s, idle_s
        self.workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        # once the microscope has finished scanning (it no longer needs the CPU for the stage, camera and autofocus):
        # one more worker for the rest of the blocks
        self.workers_final = max(self.workers, workers_final or min(self.workers + 1, os.cpu_count() or 1))
        self.refine, self.options, self.log = refine_oxide, options or {}, log
        self.viewer, self.cc = viewer, None      # the viewer canvas (flakeid/livecanvas.py), set up once the plan is known
        self.fpolys, self.tpolys = {}, {}
        self.tiles: dict = {}              # label -> Tile
        self.plan = None
        self.slow = 1                      # grid axis the scan advances along (rows)
        self.done_blocks, self.running = set(), {}
        self.results = {}                  # block -> result
        self.inputs = {}                   # block -> (tiles, core), to re-run a block
        self.oxide_frozen = None
        self.substrate = None              # substrate RGB level (mosaic units), fixed with the oxide
        self.sub_pool = []
        self.sub_all = []                  # substrate level of every block (fallback)
        self.errors = []
        self.n_ok = 0
        self.ox_samples = []               # oxide refined on single blocks (for the log)
        self.ox_pool = []                  # (samples, clip_max, profile) of the blocks before the oxide is fixed
        self.ox_spare = []                 # the same from blocks with few flakes (used only if there is nothing else)
        self.ox_done = set()               # blocks whose oxide pass (cfg oxide_only) has come back
        self.last_new = time.time()
        self.t0 = time.time()
        self.status_path = self.out / f"{self.name}_live_status.json"
        self.logf = open(self.out / f"{self.name}_live_log.txt", "a")
        if self.oxide_nm != self.oxide_entered:
            self.say(f"oxide: {self.oxide_entered:g} nm entered (nominal) -> {self.oxide_nm:g} nm in the model "
                     f"(profiles/oxide_nominal.json; still refined from the image)")

    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:5.0f}s] {msg}"
        self.log(line)
        self.logf.write(line + "\n"); self.logf.flush()

    # -- tiles ------------------------------------------------------------
    def scan_folder(self):
        new = 0
        if self.plan is None:
            self.plan = read_plan(self.folder)
            if self.plan:
                self.say(f"scan plan: {len(self.plan)} tiles")
        for mdf in self.folder.glob("*/metadata.txt"):
            lbl = mdf.parent.name
            if lbl in self.tiles:
                continue
            try:
                if time.time() - mdf.stat().st_mtime < 1.0:      # still being written
                    continue
                t = read_tile(mdf)
            except (OSError, ValueError, KeyError):
                continue
            if t is None or t.grid is None or time.time() - t.path.stat().st_mtime < 1.0:
                continue
            if not self.tiles:
                for msg in check_linear(t):
                    self.say("warning: " + msg)
            self.tiles[lbl] = t
            new += 1
        if new:
            self.last_new = time.time()
            if self.plan is None and len(self.tiles) > 8:
                ts = sorted(self.tiles.values(), key=_index)
                rows = [t.grid[1] for t in ts]
                self.slow = 1 if all(b >= a for a, b in zip(rows, rows[1:])) else 0
        return new

    abort = None                        # callable: True when the scan was given up (multi-piece run: piece failed)
    busy = None                         # callable: True while the microscope is still scanning (default: this scan incomplete)
    on_progress = None                  # callable(n_done_blocks): after blocks were added once the oxide is fixed (provisional targets)
    errors: list                        # error messages of failed blocks
    n_ok = 0                            # blocks analysed without error

    def complete(self) -> bool:
        if self.abort is not None and self.tiles and self.abort():
            return True
        if self.plan is not None:
            return all(self._label(g) in self.tiles for g in self.plan)
        return bool(self.tiles) and time.time() - self.last_new > self.idle_s

    @staticmethod
    def _label(g):
        return f"Pos-{g[0]:03d}_{g[1]:03d}"

    def grids(self):
        return set(self.plan) if self.plan is not None else {t.grid for t in self.tiles.values()}

    def full_blocks(self):
        """Blocks with a complete core and a complete ring of neighbours in the plan (interior blocks): their oxide passes
        run first, before the oxide is fixed, as they give the best oxide fits."""
        if self.plan is None:
            return None
        grids = self.grids()
        out = set()
        for b in {(g[0] // self.block, g[1] // self.block) for g in grids}:
            c0, r0 = b[0] * self.block - 1, b[1] * self.block - 1
            if all((c, r) in grids for c in range(c0, c0 + self.block + 2) for r in range(r0, r0 + self.block + 2)):
                out.add(b)
        return out

    def ready_blocks(self, final=False):
        by_block = {}
        for g in self.grids():
            by_block.setdefault((g[0] // self.block, g[1] // self.block), []).append(g)
        have = {t.grid for t in self.tiles.values()}
        top = max((g[self.slow] for g in have), default=-1)
        out = []
        for b, gs in sorted(by_block.items()):
            if b in self.done_blocks or b in self.running:
                continue
            c0, c1 = b[0] * self.block - 1, (b[0] + 1) * self.block
            r0, r1 = b[1] * self.block - 1, (b[1] + 1) * self.block
            ring = [g for g in self.grids() if c0 <= g[0] <= c1 and r0 <= g[1] <= r1]
            if self.plan is not None:
                ok = all(g in have for g in ring)
            else:
                ok = final or top > (r1 if self.slow == 1 else c1)
            core = [self._label(g) for g in gs if g in have]
            if ok and core:
                self.inputs[b] = ([self.tiles[self._label(g)] for g in ring if g in have], core)
                out.append((b, *self.inputs[b]))
        return out

    # -- results ----------------------------------------------------------
    def merged(self):
        """All owned flakes, de-duplicated and ranked by score -> (FlakeSummary list, Terrace list)."""
        rows = [(f, ts) for r in self.results.values() for f, ts in r["flakes"]]
        rows.sort(key=lambda ft: -(ft[0]["score"] if ft[0]["score"] == ft[0]["score"] else -1))
        kept, pts = [], []
        for f, ts in rows:         # a flake on a block boundary seen by two blocks: keep the better one
            p = (f["stage_x_um"], f["stage_y_um"])
            if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 5.0 for q in pts):
                continue
            kept.append((f, ts)); pts.append(p)
        # global pixel coordinates: from the stage position with the scan-wide matrix (tile 0 at the origin)
        t0 = min(self.tiles.values(), key=_index)
        Minv = np.linalg.inv(self.M)
        wh = np.array(read_frame_shape(t0), float)
        P0, O = (np.array(self.cc["P0"]), np.array(self.cc["O"])) if self.cc else (t0.stage_xy, np.zeros(2))

        def gpx(sx, sy):                       # (with a viewer canvas: its pixel frame)
            return Minv @ (np.array([sx, sy]) - P0) + wh / 2 - O

        def shift(polys, d):
            return [(np.array(p).reshape(-1, 2) + d).round(1).ravel().tolist() for p in polys]
        self.fpolys, self.tpolys = {}, {}

        px_um = math.sqrt(abs(np.linalg.det(self.M)))
        flakes, terraces = [], []
        for rank, (f, ts) in enumerate(kept, 1):
            d = gpx(f["stage_x_um"], f["stage_y_um"]) - np.array([f["target_x_px"], f["target_y_px"]])
            ids = []
            for t in ts:
                t = dict(t)
                tp = t.pop("_poly", None)
                t["flake_id"] = rank
                t["terrace_id"] = len(terraces) + 1
                if tp is not None:
                    self.tpolys[t["terrace_id"]] = shift(tp, d)
                for a, b in (("x_px", 0), ("target_x_px", 0), ("y_px", 1), ("target_y_px", 1)):
                    t[a] = t[a] + d[b]
                t["bbox_x_px"], t["bbox_y_px"] = int(t["bbox_x_px"] + d[0]), int(t["bbox_y_px"] + d[1])
                for a in ("x", "y", "target_x", "target_y"):
                    t[f"{a}_um"] = t[f"{a}_px"] * px_um
                ids.append(t["terrace_id"])
                terraces.append(Terrace(**t))
            f = dict(f)
            fp = f.pop("_poly", None)
            if fp is not None:
                self.fpolys[rank] = shift(fp, d)
            best_old = f["best_terrace_id"]
            f["best_terrace_id"] = ids[[x["terrace_id"] for x in ts].index(best_old)] if ts else 0
            f["flake_id"], f["terrace_ids"] = rank, ";".join(map(str, ids))
            for a, b in (("x_px", 0), ("target_x_px", 0), ("y_px", 1), ("target_y_px", 1)):
                f[a] = f[a] + d[b]
            f["bbox_x_px"], f["bbox_y_px"] = int(f["bbox_x_px"] + d[0]), int(f["bbox_y_px"] + d[1])
            for a in ("x", "y", "target_x", "target_y"):
                f[f"{a}_um"] = f[f"{a}_px"] * px_um
            flakes.append(FlakeSummary(**f))
        self._gpx = gpx
        return flakes, terraces

    def merged_clean(self, flakes):
        """The clean areas of all blocks (avram_new1), de-duplicated (two blocks: the larger), largest first, with the
        flake_id of the merged flake under each (its target within 5 um of the block's flake) and global pixels."""
        rows = sorted((c for r in self.results.values() for c in r.get("clean", []) if c.get("stage_x_um") is not None),
                      key=lambda c: -c["area_um2"])
        kept = []
        for c in rows:
            if not any(math.hypot(c["stage_x_um"] - k["stage_x_um"], c["stage_y_um"] - k["stage_y_um"]) < 5.0 for k in kept):
                kept.append(c)
        fl = [(f.flake_id, f.stage_x_um, f.stage_y_um) for f in flakes if f.stage_x_um is not None]
        out = []
        for i, c in enumerate(kept, 1):
            fid = ""
            fs = c.get("flake_stage")
            if fs and fs[0] is not None and fl:
                j = min(fl, key=lambda q: math.hypot(q[1] - fs[0], q[2] - fs[1]))
                if math.hypot(j[1] - fs[0], j[2] - fs[1]) < 5.0:
                    fid = j[0]
            g = self._gpx(c["stage_x_um"], c["stage_y_um"])
            cr = (list(c.get("contrast") or []) + [float("nan")] * 3)[:3]
            out.append(dict(clean_id=i, flake_id=fid, area_um2=c["area_um2"], thickness_nm=c["thickness_nm"], d_um=c["d_um"],
                            stage_x_um=round(c["stage_x_um"], 2), stage_y_um=round(c["stage_y_um"], 2),
                            target_x_px=round(float(g[0]), 1), target_y_px=round(float(g[1]), 1),
                            contrast_r=cr[0], contrast_g=cr[1], contrast_b=cr[2]))
        return out

    def write_tables(self, final=False):
        flakes, terraces = self.merged()
        tiles = sorted(self.tiles.values(), key=_index)
        res = FlakeResult(image=str(self.folder), material=self.material, oxide_nm=self.oxide_nm,
                          oxide_nm_used=self.oxide_frozen or self.oxide_nm,
                          pixel_um=math.sqrt(abs(np.linalg.det(self.M))), pixel_um_source="live block registration",
                          flakes=flakes, terraces=terraces,
                          params=dict(live=True, block=self.block, margin_um=self.margin_um, blocks=len(self.results), oxide_blocks=self.ox_samples,
                                      final=final, thickness_range_nm=list(DEFAULT_RANGE_NM.get(self.material, (0.0, 100.0))),
                                      oxide_entered_nm=self.oxide_entered,
                                      stage_axis_sign=[1, 1]),
                          metadata=dict(source="micromanager", dataset=str(self.folder),
                                        stage_map=dict(P=[t.stage_xy.tolist() for t in tiles], z=[t.z_um for t in tiles],
                                                       labels=[t.label for t in tiles],
                                                       z_units=tiles[0].z_units if tiles else "drive")))
        stem = self.out / f"{self.name}_live_flakes"
        for suffix, fn in ((".csv", lambda p: res.to_csv(p, p.with_name(f"{self.name}_live_terraces.csv.tmp"))),
                           (".json", res.to_json)):
            tmp = stem.with_suffix(suffix + ".tmp")
            fn(tmp)
            replace_file(tmp, stem.with_suffix(suffix))
        tt = self.out / f"{self.name}_live_terraces.csv.tmp"
        if tt.exists():
            replace_file(tt, self.out / f"{self.name}_live_terraces.csv")
        if (self.options or {}).get("clean_areas"):     # (avram_new1: the clean areas, for its target list)
            write_target_csv(self.out / f"{self.name}_live_clean.csv", self.merged_clean(flakes), list(CLEAN_COLUMNS))
        return res

    def write_status(self, state):
        n_plan = len(self.plan) if self.plan else None
        good = sum(1 for r in self.results.values() for f, _ in r["flakes"] if f["score"] == f["score"] and f["score"] >= 0.75)
        st = dict(state=state, tiles=len(self.tiles), planned=n_plan, blocks_done=len(self.done_blocks),
                  blocks_running=len(self.running), flakes=sum(len(r["flakes"]) for r in self.results.values()),
                  flakes_score_075=good, oxide_nm=self.oxide_frozen, elapsed_s=round(time.time() - self.t0, 1),
                  updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        # (only for people watching: never worth stopping the analysis for)
        try:
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1))
            replace_file(tmp, self.status_path, tries=4)
        except OSError as exc:
            if not getattr(self, "_status_warned", False):
                self.say(f"(status file not updated: {exc}; going on)")
                self._status_warned = True

    # -- main loop --------------------------------------------------------
    def cfg(self):
        return dict(flat=self.flat, M=self.M.tolist(), margin_um=self.margin_um, name=self.name,
                    material=self.material, oxide_nm=self.oxide_frozen or self.oxide_nm,
                    refine=self.refine and self.oxide_frozen is None, options=self.options,
                    substrate=self.substrate, calibrated=self.oxide_frozen is not None, canvas=self.cc)

    def _collect(self, b, r, pending_blocks):
        patches = r.pop("canvas", None)
        if patches and self.cc is not None:
            from . import livecanvas
            try:
                livecanvas.paste(self.cc, patches)
            except Exception as exc:
                r["canvas_error"] = f"{type(exc).__name__}: {exc}"
        if r.get("oxide_only"):
            self.ox_done.add(b)
            if self.oxide_frozen is None:
                msg = f"block {b}: oxide pass, {r['seconds']:.0f}s"
                if r.get("oxide") is not None:
                    msg += f", oxide {r['oxide']:.1f} nm (gain {r['gain']:.2f})"
                if r.get("error"):
                    msg += f"  [{r['error']}]"
                self.say(msg)
                self._pool_oxide(r)
            return                      # not done: analysed in full once the oxide is fixed
        if self.oxide_frozen is not None and r.get("pre_cal") and not self.agrees(r):
            self.say(f"block {b}: started before the oxide / substrate were fixed and its own fit differs; analysing again")
            return                      # not done: ready_blocks queues it again
        self.done_blocks.add(b)
        self.results[b] = r
        msg = (f"block {b}: {len(r['core'])} tiles (+{r['n_tiles'] - len(r['core'])} ring, {r['registration']}), "
               f"{len(r['flakes'])} flakes, {r['seconds']:.0f}s")
        if r.get("oxide") is not None:
            msg += f", oxide {r['oxide']:.1f} nm" + (f" (refined, gain {r['gain']:.2f})" if r.get("refined") else "")
        if r.get("error"):
            msg += f"  [{r['error']}]"
            if r.get("trace") and r["error"] not in self.errors:            # the first time this error occurs: where
                msg += "\n" + r["trace"].rstrip()
            self.errors.append(r["error"])
        else:
            self.n_ok += 1
        if r.get("canvas_error"):
            msg += f"  (viewer image of this block not written: {r['canvas_error']})"
        tiles_left = (len(self.plan) - len(self.tiles)) if self.plan else "?"
        self.say(msg + f"   [{len(self.tiles)} tiles in, {tiles_left} to come, {pending_blocks} blocks queued]")
        # the same error in the first blocks and no block analysed: a setup problem (flat-field path, ...),
        # not something later blocks will get past -> stop instead of reporting an empty piece
        if self.n_ok == 0 and len(self.errors) >= 3 and len(set(self.errors)) == 1:
            raise LiveError(f"the first {len(self.errors)} blocks all failed with: {self.errors[0]}")
        self._pool_oxide(r)

    def _pool_oxide(self, r):
        # oxide thickness: one block's flat material is not representative (a few large flakes dominate
        # and single blocks disagree by several nm), so the refinement samples of the first blocks are
        # pooled, as the whole mosaic is in the batch analysis, and the oxide is fixed from the pool
        # Only blocks whose own refinement clearly worked are pooled: the fit improved a lot (gain) and the value is
        # not at an end of the +-10 % search range.  Edge blocks with residue and debris give poor fits that end up
        # at the range end (bigbntest_2: 256.5 nm, gain 0.07-0.37); a flake-count rule failed on a small piece
        # (bn_test_2: one 20-flake edge block fixed 256.5 nm while two clean blocks gave 303 nm, gain 0.95).
        # The substrate level (for chip detection) comes from the same blocks.
        smp = r.pop("samples", None)
        if self.oxide_frozen is None and r.get("substrate") is not None:
            self.sub_all.append(r["substrate"])
            if self.refine:
                good = (r.get("refined") and r["gain"] >= self.MIN_OXIDE_GAIN
                        and abs(r["oxide"] / self.oxide_nm - 1) < self.OX_WINDOW)
            else:
                good = len(r["flakes"]) >= 5
            if good:
                self.sub_pool.append(r["substrate"])
                self.ox_samples.append(r["oxide"])
            if smp is not None:
                (self.ox_pool if good else self.ox_spare).append((smp, r["clip_max"], r["profile"]))
            if len(self.sub_pool) >= self.MIN_OXIDE_BLOCKS and (
                    not self.refine or sum(len(s) for s, _, _ in self.ox_pool) >= 40000):
                self.freeze_oxide()
            elif len(self.sub_all) >= self.MAX_PREFREEZE:
                # (otherwise every block of a piece whose oxide does not fit is analysed twice, and its provisional
                # targets wait for the end: a graphite piece, microscope PC 2026-10-03, 30 blocks redone)
                self.say(f"oxide: {len(self.sub_all)} blocks without enough good fits; fixing it now")
                self.freeze_oxide()

    MIN_OXIDE_BLOCKS = 3
    MIN_OXIDE_GAIN = 0.5
    OX_WINDOW = 0.19                    # refined oxide within this of the entered value (the fit searches up to +-20 %;
                                        # a value at that end is a fit that ran away, not the oxide)
    MAX_PREFREEZE = 8                   # blocks analysed without enough good oxide fits: fix it anyway (nominal / pooled)

    KEEP_OXIDE_NM = 1.0                 # a block analysed before the oxide was fixed is kept if its own oxide is this close
    KEEP_SUBSTRATE = 0.03               # ... and its substrate level within this fraction (chip detection unchanged)

    def agrees(self, r):
        """A block analysed with its own refined oxide / substrate level close enough to the fixed ones: no second
        analysis (1 nm of oxide moves the thicknesses by less than the fit scatter)."""
        if r.get("error") or r.get("oxide") is None or self.oxide_frozen is None:
            return False
        if self.refine and not (r.get("refined") and r.get("gain", 0) >= self.MIN_OXIDE_GAIN):
            return False
        if abs(r["oxide"] - self.oxide_frozen) > self.KEEP_OXIDE_NM:
            return False
        if self.substrate is not None and r.get("substrate") is not None:
            a, s = np.asarray(r["substrate"], float), np.asarray(self.substrate, float)
            if np.any(np.abs(a / np.maximum(s, 1e-9) - 1) > self.KEEP_SUBSTRATE):
                return False
        return True

    def substrate_check(self, n=24):
        """Substrate colour and brightness of the tiles so far against the blank-substrate reference (flakeid.bgcheck):
        warns early in the scan if a filter, the aperture or the lamp is not as for the reference / flat field."""
        from . import bgcheck
        from .hundred import _mode_rgb
        from .mm import FlatField, full_scale, read_frame
        ts = sorted(self.tiles.values(), key=_index)
        if not ts:
            return
        pick = [ts[int(i)] for i in np.linspace(0, len(ts) - 1, min(n, len(ts)))]
        FF = FlatField.load(self.flat) if self.flat else None
        frames = []
        for t in pick:
            raw = read_frame(t)
            lin = FF.apply(raw) if FF is not None and FF.flat.shape == raw.shape else raw - 128.0
            m, f = _mode_rgb(lin[::2, ::2])
            if f >= 0.3:
                e = t.props.get("Exposure-ms")
                frames.append((m, float(e) if e not in (None, "") else None))
        if not frames:
            self.say("substrate check: no tile with enough bare substrate yet")
            return
        exps = [e for _, e in frames if e]
        bits = int(round(np.log2(full_scale(pick[0]) + 1)))
        res = bgcheck.check(np.median(np.array([m for m, _ in frames]), 0), "10x", self.oxide_frozen or self.oxide_nm,
                            exposure_ms=float(np.median(exps)) if exps else None, bit_depth=bits,
                            flat_info=FF.info if FF is not None else None, frames=frames)
        bgcheck.report(res, log=self.say)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / f"{self.name}_bgcheck.json").write_text(json.dumps(res, indent=1))

    def freeze_oxide(self):
        from .thickness import load_profile, refine_oxide
        def pooled(blocks):
            pool = np.concatenate([s for s, _, _ in blocks])
            pool = pool[np.random.default_rng(0).choice(len(pool), min(len(pool), 20000), replace=False)]
            clip = np.median(np.array([c for _, c, _ in blocks]), 0)
            return refine_oxide(pool.astype(np.float64), self.material, self.oxide_nm, load_profile(blocks[0][2]), clip_max=clip)

        def usable(ox, gain):
            return gain >= self.MIN_OXIDE_GAIN and abs(ox / self.oxide_nm - 1) < self.OX_WINDOW
        if not self.refine:
            self.oxide_frozen = self.oxide_nm
        elif self.ox_pool:
            ox, gain = pooled(self.ox_pool)
            if usable(ox, gain):
                self.oxide_frozen = ox
                how = f"pooled samples of {len(self.ox_pool)} block(s), gain {gain:.2f}"
            else:                                   # the pool disagrees with its own blocks: trust the blocks
                self.oxide_frozen = float(np.median(self.ox_samples))
                how = f"median of {len(self.ox_samples)} block(s); the pooled fit was poor ({ox:.1f} nm, gain {gain:.2f})"
            self.say(f"oxide fixed at {self.oxide_frozen:.1f} nm ({how}; single blocks gave {', '.join(f'{v:.1f}' for v in self.ox_samples)})")
        elif self.ox_spare:
            ox, gain = pooled(self.ox_spare)
            if usable(ox, gain):
                self.oxide_frozen = ox
                self.say(f"oxide fixed at {ox:.1f} nm (no block fitted well on its own; pooled {len(self.ox_spare)} blocks, gain {gain:.2f})")
            else:
                self.oxide_frozen = self.oxide_nm
                self.say(f"WARNING: the oxide thickness could not be determined (best {ox:.1f} nm, gain {gain:.2f}); using the nominal "
                         f"{self.oxide_nm:.1f} nm.  Thicknesses and scores may be off: give the right value with --no-refine-oxide.")
        else:
            self.oxide_frozen = self.oxide_nm
            self.say(f"oxide: no flat material to refine it; nominal {self.oxide_nm:.1f} nm used")
        if not self.sub_pool and self.sub_all:
            self.sub_pool = list(self.sub_all)
        self.ox_pool, self.ox_spare = [], []
        if self.sub_pool:
            self.substrate = np.median(np.array(self.sub_pool), 0).tolist()
            self.say(f"substrate level for chip detection: {np.round(self.substrate).tolist()}")
        try:
            self.substrate_check()
        except Exception as exc:                      # a check must never stop the analysis
            self.say(f"substrate check failed: {exc}")
        # everything analysed so far used its own oxide / chip detection: analysed again unless its own values are
        # close to the fixed ones (3 nm of oxide triples the fit residuals; a block with a strip of surround can lose its
        # chip mask)
        redo = [b for b, r in self.results.items() if not self.agrees(r)]
        for b in redo:
            self.done_blocks.discard(b)
            del self.results[b]
        kept = len(self.results)
        if redo or kept:
            self.say(f"earlier blocks: {kept} kept (own oxide within {self.KEEP_OXIDE_NM:g} nm of the fixed value), "
                     f"{len(redo)} to analyse again with the fixed oxide and substrate level")

    def run(self, build_viewer=True):
        """Analyse the scan as it comes in; build_viewer=False: the caller writes the target list first and then calls
        finish_viewer (the microscope gets the targets a few minutes earlier)."""
        self.say(f"watching {self.folder} ({self.workers} workers while scanning, {self.workers_final} after; blocks of "
                 f"{self.block}x{self.block} tiles, margin {self.margin_um:.0f} um)")
        self.M = None
        ctx = get_context("spawn")
        full = None
        n_reported, said_final = None, False        # (None: the first provisional list as soon as the oxide is fixed)
        with ProcessPoolExecutor(max_workers=self.workers_final, mp_context=ctx) as pool:
            finished = False
            while True:
                self.scan_folder()
                if self.M is None and self.tiles:
                    self.M = scan_matrix(self.folder, DEFAULT_M)
                    if self.viewer and self.plan:
                        from . import livecanvas
                        t0 = min(self.tiles.values(), key=_index)
                        # the canvas is written block by block from several processes: always on a local disk (on a
                        # network share, e.g. -o in the run folder, every block took 15-60x longer); only the finished
                        # viewer goes to the output folder
                        self.cc = livecanvas.setup(livecanvas.local_dir(self.name), self.name, self.plan, self.M, read_frame_shape(t0))
                        self.say(f"viewer canvas {self.cc['shape'][1]} x {self.cc['shape'][0]} px (on disk, filled block by block)")
                complete = self.complete()
                ready = self.ready_blocks(final=complete)
                if self.oxide_frozen is None and self.plan is not None:
                    # before the oxide is fixed: interior blocks first (the best oxide fits).  Edge blocks only fill
                    # workers that would otherwise wait: the first interior block needs ~9 rows scanned, and holding
                    # everything back for it left the workers idle for half the piece.  (With --no-refine-oxide these are
                    # full analyses, kept if their substrate level agrees with the fixed one, else analysed again.)
                    if full is None:
                        full = self.full_blocks()
                    ready = [x for x in ready if x[0] in full] + [x for x in ready if x[0] not in full]
                if self.oxide_frozen is None and self.refine:
                    ready = [x for x in ready if x[0] not in self.ox_done]      # (each block's oxide pass once)
                if complete and not ready and not self.running and self.oxide_frozen is None:
                    self.freeze_oxide()                 # small scan: fewer well-determined blocks than needed
                    ready = self.ready_blocks(final=True)
                scanning = self.busy() if self.busy is not None else not complete
                cap = self.workers if (scanning or self.oxide_frozen is None) else self.workers_final
                if not scanning and not said_final and self.workers_final > self.workers:
                    self.say(f"scanning finished: {self.workers_final} workers from now on")
                    said_final = True
                for b, tiles, core in ready:
                    if len(self.running) >= cap:
                        break
                    cfg = self.cfg()
                    # before the oxide is fixed: the oxide pass only (each block is then analysed in full once)
                    cfg["oxide_only"] = self.oxide_frozen is None and self.refine
                    self.running[b] = pool.submit(analyse_block, tiles, core, cfg)
                if self.running:
                    done, _ = wait(list(self.running.values()), timeout=self.poll_s, return_when=FIRST_COMPLETED)
                    for b in [b for b, f in self.running.items() if f in done]:
                        fut = self.running.pop(b)
                        try:
                            r = fut.result()
                        except Exception as exc:
                            import traceback
                            # (the worker's own traceback comes along as the cause: where in the block it failed)
                            r = dict(core=[], n_tiles=0, registration="-", flakes=[], seconds=0, oxide=None, gain=0,
                                     error=f"{type(exc).__name__}: {exc}",
                                     trace="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                        self._collect(b, r, len(ready))
                    if done:
                        self.write_tables()
                        if (self.on_progress is not None and self.oxide_frozen is not None
                                and (n_reported is None or len(self.done_blocks) - n_reported >= self.PROGRESS_BLOCKS)
                                and not complete_all(self)):
                            n_reported = len(self.done_blocks)
                            try:
                                self.on_progress(n_reported)
                            except Exception as exc:          # provisional targets must never stop the analysis
                                self.say(f"provisional target list not written: {type(exc).__name__}: {exc}")
                else:
                    if complete and not self.ready_blocks(final=True) and self.oxide_frozen is not None:
                        finished = True
                    else:
                        time.sleep(self.poll_s)
                self.write_status("running")
                if finished:
                    break
        res = self.write_tables(final=True)
        self.res = res
        if build_viewer:
            self.finish_viewer()
        self.write_status("done")
        self.say(f"scan complete: {len(self.tiles)} tiles, {len(res.flakes)} flakes "
                 f"({sum(1 for f in res.flakes if f.score >= 0.75)} scored >= 0.75)")
        if self.errors:
            failed = sum(1 for r in self.results.values() if r.get("error"))
            if self.n_ok == 0:
                raise LiveError(f"no block could be analysed ({len(self.errors)} failures, e.g. {self.errors[0]})")
            if failed:
                self.say(f"WARNING: {failed} of {len(self.results)} blocks failed ({self.errors[-1]}); flakes in those areas are missing")
        return res

    PROGRESS_BLOCKS = 2                 # on_progress after every this many new blocks

    def finish_viewer(self):
        if self.cc is None:
            return
        from . import livecanvas
        t1 = time.time()
        vdir = self.out / f"{self.name}_viewer"
        livecanvas.build_viewer(self.cc, self.res, self.fpolys, self.tpolys, vdir)
        livecanvas.cleanup(self.cc)
        self.cc = None
        self.say(f"viewer written in {time.time() - t1:.0f} s: {vdir / 'index.html'}")
        self.viewer_path = vdir / "index.html"


def complete_all(live):
    """All blocks of the plan analysed or running (the final list follows within a minute: no provisional one)."""
    if live.plan is None:
        return False
    return not live.ready_blocks(final=True) and not live.running and live.complete()


def read_frame_shape(tile: Tile) -> tuple:
    import cv2
    a = cv2.imread(str(tile.path), cv2.IMREAD_UNCHANGED)
    return a.shape[1], a.shape[0]


def scan_matrix(folder: Path, default):
    """Pixel -> stage matrix: from <NAME>_chip.json (the scan script's calibration) or the default."""
    name = re.sub(r"_tiles(_\d+)?$", "", folder.name)
    cj = folder.parent / f"{name}_chip.json"
    try:
        M = np.array(json.loads(cj.read_text())["M"], float)
        if M.shape == (2, 2) and abs(np.linalg.det(M)) > 0.1:
            return M
    except (OSError, ValueError, KeyError):
        pass
    return np.asarray(default, float)


TARGET_COLUMNS = ["order", "label", "flake_id", "score", "thickness_nm", "terrace_d_um", "area_um2", "x_px", "y_px",
                  "stage_x_um", "stage_y_um", "z_um", "z10_um", "quality", "terrace_poly_um"]


def terrace_polys_um(rows, ter_csv, tpolys=None, M=None, offset=(0.0, 0.0), min_um=60.0):
    """{flake_id: "x1 y1;x2 y2;..."}: each row's chosen terrace (graphite: the row's terrace_id; else the flake's best
    terrace) as a polygon in the target list's stage um (10x stage + offset, as stage_x_um), at most
    fieldmosaic.POLY_MAX_VERTICES vertices, for the microscope's 100x / IR mosaics of large terraces.  The outline from
    the viewer canvas (tpolys, the live scan's pixel frame) where there is one, else the terrace's bounding box.  Pixel
    -> stage: the scan matrix M about the terrace's own target point (whose stage position the table holds); without M,
    fitted to the table's terraces.  Terraces smaller than min_um both ways are left out (they fit one 100x field)."""
    from .fieldmosaic import format_poly, poly_area, simplify_poly
    from . import targets as T
    try:
        ter = {str(r.get("terrace_id")): r for r in T.read_flakes(ter_csv)}
    except (OSError, ValueError):
        return {}
    if M is None:
        X = np.array([[T._f(r.get("target_x_px")), T._f(r.get("target_y_px")), 1.0, T._f(r.get("stage_x_um")), T._f(r.get("stage_y_um"))]
                      for r in ter.values()], float)
        X = X[np.isfinite(X).all(1)] if len(X) else X
        if len(X) < 3:
            return {}
        M = np.linalg.lstsq(X[:, :3], X[:, 3:], rcond=None)[0][:2].T
    M = np.asarray(M, float)
    px_um = math.sqrt(abs(np.linalg.det(M)))
    out = {}
    for r in rows:
        tid = r.get("terrace_id") or r.get("best_terrace_id")
        t = ter.get(str(tid).split(".")[0]) if tid not in (None, "") else None
        if t is None:
            continue
        try:
            bw, bh = float(t["bbox_w_px"]), float(t["bbox_h_px"])
            if max(bw, bh) * px_um < min_um:
                continue
            tp = np.array([float(t["target_x_px"]), float(t["target_y_px"])])
            ts = np.array([float(t["stage_x_um"]), float(t["stage_y_um"])])
        except (KeyError, TypeError, ValueError):
            continue
        polys = [np.array(p, float).reshape(-1, 2) for p in (tpolys or {}).get(int(float(tid)), []) if len(p) >= 6]
        if polys:
            P = max(polys, key=poly_area)
        else:
            x0, y0 = float(t["bbox_x_px"]), float(t["bbox_y_px"])
            P = np.array([[x0, y0], [x0 + bw, y0], [x0 + bw, y0 + bh], [x0, y0 + bh]])
        S = (P - tp) @ M.T + ts + np.asarray(offset, float)
        out[int(r["flake_id"])] = format_poly(simplify_poly(S))
    return out


def focus_map_tiles(tiles: list, z_scale=12.0):
    """10x focus Z at (x, y) interpolated between the tile centres (the scan's own focus surface, plane or
    quadratic, sampled at every tile); nearest tile outside them.  None without tile Z."""
    P = np.array([t.stage_xy for t in tiles])
    z = np.array([np.nan if t.z_um is None else t.z_um for t in tiles], float)
    ok = np.isfinite(z)
    if ok.sum() < 3:
        return None
    z = z if tiles[0].z_units == "um" else z / z_scale
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    try:
        lin = LinearNDInterpolator(P[ok], z[ok])
    except Exception:              # tiles on a line
        lin = None
    near = NearestNDInterpolator(P[ok], z[ok])

    def f(x, y):
        v = float(lin(x, y)) if lin is not None else float("nan")
        return v if v == v else float(near(x, y))
    return f


def thickness_for(spec, material):
    """--thickness entries ("MAX" or "MIN-MAX", for every material or as "MATERIAL:...") -> (min, max) nm for material,
    or None.  A single number is a maximum only (thin flakes are kept)."""
    rng = None
    for x in spec or []:
        m, _, r = x.rpartition(":")
        if m and m.lower() not in (material or "").lower():
            continue
        lo, hi = (float(v) for v in r.split("-", 1)) if "-" in r else (0.0, float(r))
        if m or rng is None:
            rng = (lo, hi)
    return rng


# flags that keep a flake off the 100x list (it stays in the flake table and the viewer):
#   chip_edge       touches the chip edge: mostly edge debris and shadows at 10x
#   thick_branch    hBN whose colour fits both ~20 and ~150 nm, decided thick (branch.resolve: labelled direction,
#                   boundary from the chip's own two colour groups)
#   possibly_thick  the same without a branch calibration for this profile / oxide (thin-film-model margin only)
#   bulk/no_signal  thick graphite / black slivers (hard flags since 2026-10-02; listed for older tables)
#   thick_context   hBN of ambiguous colour with clear thick-only colours (orange/red/purple) at its edge or within 8 um
# Inclusive about thickness (2026-10-25): a flake that may be inside the thickness window is a target.  possibly_thick,
# thick_branch, thick_context (its colour may be a thick film's, but also a thin one's) and saturated (a clipped channel:
# its thickness a lower-bound fit) are no longer excluded; the flags stay in the tables.  Before:
# ("saturated", "chip_edge", "thick_branch", "possibly_thick", "bulk", "no_signal", "thick_context").
TARGET_EXCLUDE = ("chip_edge", "bulk", "no_signal")
# provisional target lists (written while the rest of the piece is analysed, imaged at 100x meanwhile): also without the
# flakes whose thickness depends on chip-wide decisions (hBN thin / thick colour branch from all of the chip's flakes)
PARTIAL_EXCLUDE = TARGET_EXCLUDE + ("two_branch", "ambiguous_thickness")
PARTIAL_MAX = 30
MIN_TARGET_D_UM = 30.0
MIN_TARGETS = 10            # a piece whose flakes all score low still gets this many 100x targets (best scores first)
FILL_MIN_D_UM = 10.0        # ... each at least this wide (best terrace), so the fill is not specks
# hBN 100x order and ranking (Jordan, 2026-10-22):
#   --target-order size (default): the same targets, imaged largest best terrace first (groups of --size-chunk targets,
#     a short stage path within each), so a run cut short (stopped, piece skipped) has the biggest flakes; 'path' = the
#     shortest stage path over the whole list, as before.  The 100x time per target (~7.6 s) matters more than the travel.
#   --target-rank score (default) | size: which flakes become targets.  'size' ranks the candidate pool by
#     logit(score) + 2 ln(best terrace d um) (flakeid.targets.rank_value) instead of the score.
#   Every hBN final list also gets <piece>_targets_alt.csv beside <piece>_targets.csv: the flakes the OTHER ranking
#     would have chosen instead (alt = add: not imaged in this run) or not at all (alt = drop: imaged only because of
#     this ranking), with their scores, best terrace sizes and places under both rankings (rank_score, rank_size).
#   The trial: a run of one hBN chip with the size ranking (the setting applies to every hBN piece of the run):
#     C:\FlakeID\run_flakeid.bat scope --set target_rank=size  (flakeid.scope settings target_rank; or by hand:
#     python -m flakeid.live RUN_DIR -o OUT --target-rank size), then the next runs as usual (default 'score').
#     In its designer review, the keeps among the alt = drop flakes (in the list only because of the size ranking) are
#     the gain; the alt = add flakes (the score ranking's choices, not imaged) are the possible loss: image them later
#     from <piece>_targets_alt.csv if the loss has to be measured rather than estimated (10x scores, sizes and pictures
#     are in the viewer).  Simulated on 21 hBN pieces (2026-10-22): 15 of 80 flakes differ per piece.
TARGET_ORDERS = ("size", "path")
# --hbn-max-targets (Jordan, 2026-10-23): with --target-order size, an hBN piece's targets (chosen as before: the best
#   scores, up to --max-targets / the piece's area x targets per cm2) are cut to the HBN_MAX_TARGETS with the largest
#   best terrace.  Replayed on 21 hBN pieces (80 targets each): 60 largest = 75 % of the 100x shots, 114 of the 123
#   keeps; 40 largest = 50 %, 97 keeps, and every hBN flake used in a design so far was size rank 3-28 (40 chosen,
#   2026-10-23).  Provisional lists: their largest PARTIAL_MAX of the same pool.  0 = no cut.
#   Graphite: not cut (size does not predict its keeps).  48 since 2026-10-25 (inclusive thickness: ~7 of a piece's 40
#   were likely-thick flakes on multitest chip_1; 8 more slots keep the thin ones, the thick ones are told apart at 100x).
#   The microscope now gives each hBN piece its own number in the run file (hbn_max_targets: its area x the hBN pictures
#   per cm2, 30 by default; Avram, 2026-10-27), which takes the place of this one; pieces without one: this.
HBN_MAX_TARGETS = 48
# graphite ranking by the kept-at-100x model (flakeid.target_model, profiles/target_graphite.json; Jordan, 2026-10-23):
#   --graphite-rank model (default): within the ribbon / 1-4 layer / thicker groups (graphite.reserve), the flakes most
#     likely to be kept at 100x first, and the targets imaged in that order (groups of --size-chunk, a short stage path
#     within each); 'tiers' = graphite.select's own order (layer tier, oversize last, score) and the shortest stage path,
#     as before.  Without the model file: as 'tiers'.
#   The model: a logistic on 37 flake-level 10x numbers fitted on whether past targets were kept at 100x (706 graphite
#     targets, 5 runs; tools/fit_target_model.py).  Replayed with the 60 % thick reserve, keeps when only the top K per
#     piece are imaged (of 156): K 30: 96 vs 79 by the tier order, K 40: 111 vs 98, K 50: 128 vs 117, K 60: 139 vs 129.
#   --graphite-max-targets N: at most N imaged per graphite piece (graphite.reserve of the ranked list: ribbon places
#     and the thick fraction kept); 0 = no cut beyond --max-targets.
GRAPHITE_RANKS = ("model", "tiers")
GRAPHITE_MAX_TARGETS = 0
# --target-mode (which flakes are 100x targets):
#   standard (default): everything above (the learned score, the quality flags, the hBN size and graphite model
#     rankings).
#   avram_new1 (Avram, 2026-10-25): every flake whose clean area is at least --min-clean-um2 (default 400 um2), any
#     material; the score and the quality flags are not used.  The clean area of a flake: its largest region of one
#     colour (a terrace: pixels within the segmentation's colour tolerance of each other) times the fraction of that
#     region's full-resolution pixels that match its colour (fill).  The 100x picture is taken in the middle of that
#     region; targets are imaged largest clean area first (groups of --size-chunk, a short stage path within each).
#     No limit on their number (--max-targets, the piece's area x targets per cm2, does not apply).
#   avram_new1 analyses the piece with every region kept (find_flakes include_rejected: the quality rules only flag),
#   so a clean flake the rules would reject (colour fit, thickness, flatness, ...) is still found; regions flagged as not
#   being material (NOT_MATERIAL: residue haze, too faint) are not targets.
TARGET_MODES = ("standard", "avram_new1")
MIN_CLEAN_UM2 = 400.0
NOT_MATERIAL = ("residue_like", "in_haze", "low_contrast", "no_signal")


def live_options(a, material=None):
    """find_flakes options of the live analysis: --target-mode avram_new1 keeps every region and looks for the clean
    areas of --clean-thickness (default the material's thickness window; see TARGET_MODES)."""
    if getattr(a, "target_mode", None) != "avram_new1":
        return None
    rng = getattr(a, "clean_thickness", None) or DEFAULT_RANGE_NM.get(str(material).lower(), (0.0, 100.0))
    return {"include_rejected": True, "clean_areas": (float(rng[0]), float(rng[1]))}


CLEAN_COLUMNS = ("clean_id", "flake_id", "area_um2", "thickness_nm", "d_um", "stage_x_um", "stage_y_um", "target_x_px", "target_y_px",
                 "contrast_r", "contrast_g", "contrast_b")


def clean_rows(clean_csv, min_clean_um2, log=print):
    """avram_new1: target rows from the piece's clean areas (<name>_live_clean.csv) of at least min_clean_um2, largest
    first: flake_id the identified flake there (else 10000 + clean_id, for the 100x labels), the 100x point the area's
    middle."""
    from . import targets as T
    out = []
    for c in T.read_flakes(clean_csv):
        a = T._f(c.get("area_um2"), 0.0)
        if not a >= min_clean_um2:
            continue
        fid = str(c.get("flake_id") or "").split(".")[0]
        out.append(dict(flake_id=int(fid) if fid.isdigit() else 10000 + int(c["clean_id"]), clean_id=c["clean_id"],
                        clean_area_um2=a, area_um2=a, best_thickness_nm=c["thickness_nm"], best_terrace_d_um=c["d_um"],
                        score="", quality="", terrace_id="", target_x_px=c["target_x_px"], target_y_px=c["target_y_px"],
                        stage_x_um=c["stage_x_um"], stage_y_um=c["stage_y_um"]))
    out.sort(key=lambda q: -q["clean_area_um2"])
    log(f"targets (avram_new1): {len(out)} clean areas (one colour of the right thickness, whatever the flake's outline) "
        f">= {min_clean_um2:g} um2" + (f" (largest {out[0]['clean_area_um2']:.0f} um2)" if out else ""))
    return out


def clean_area_rows(rows, ter_csv, min_clean_um2, thickness=None, log=print):
    """avram_new1: the flakes (rows of the flake table) whose clean area (see TARGET_MODES) is at least min_clean_um2,
    largest first, each pointed at its clean region (target point, stage position, thickness, outline) and with
    clean_area_um2."""
    from . import targets as T
    best = {}
    for t in T.read_flakes(ter_csv):
        fl = T._f(t.get("fill"))
        a = T._f(t.get("area_um2"), 0.0) * (1.0 if fl != fl else min(max(fl, 0.0), 1.0))
        if thickness is not None and not thickness[0] <= T._f(t.get("thickness_nm")) <= thickness[1]:
            continue
        if set(str(t.get("quality") or "").split(";")) & set(NOT_MATERIAL):
            continue
        k = str(t.get("flake_id"))
        if a == a and (k not in best or a > best[k][0]):
            best[k] = (a, t)
    out = []
    for r in rows:
        b = best.get(str(r.get("flake_id")))
        if b is None or b[0] < min_clean_um2:
            continue
        a, t = b
        q = dict(r, clean_area_um2=round(a, 1), best_terrace_id=t.get("terrace_id"), terrace_id=t.get("terrace_id"),
                 best_thickness_nm=t.get("thickness_nm"), best_terrace_d_um=t.get("inscribed_diameter_um"),
                 target_x_px=t.get("target_x_px"), target_y_px=t.get("target_y_px"), quality=t.get("quality", ""))
        if t.get("stage_x_um") not in (None, "", "None"):
            q.update(stage_x_um=t.get("stage_x_um"), stage_y_um=t.get("stage_y_um"))
        q.pop("layers", None)
        out.append(q)
    out.sort(key=lambda q: -q["clean_area_um2"])
    log(f"targets (avram_new1): {len(out)} of {len(rows)} flakes have a clean area (largest one-colour region x its fill) "
        f">= {min_clean_um2:g} um2" + (f" (largest {out[0]['clean_area_um2']:.0f} um2)" if out else ""))
    return out


def target_opts(a):
    """write_targets' order / ranking options from the command line (getattr: callers with older argument sets)."""
    hm = getattr(a, "hbn_max_targets", None)
    gm = getattr(a, "graphite_max_targets", None)
    return dict(target_mode=getattr(a, "target_mode", None) or "standard",
                min_clean_um2=MIN_CLEAN_UM2 if getattr(a, "min_clean_um2", None) is None else float(a.min_clean_um2),
                target_order=getattr(a, "target_order", None) or "size", target_rank=getattr(a, "target_rank", None) or "score",
                size_chunk=getattr(a, "size_chunk", None), size_max=HBN_MAX_TARGETS if hm is None else int(hm),
                graphite_rank=getattr(a, "graphite_rank", None) or "model",
                graphite_max=GRAPHITE_MAX_TARGETS if gm is None else int(gm))


def piece_target_opts(a, p):
    """target_opts for one piece of the run file: its own hBN cut (hbn_max_targets, from its area x the hBN pictures per
    cm2 on the microscope; 2026-10-27) when it has one."""
    to = target_opts(a)
    if p.get("hbn_max_targets") is not None:
        to["size_max"] = int(p["hbn_max_targets"])
    return to


def write_targets(res_csv: Path, out_stem: Path, tiles: list, min_score: float, offset, z_scale=12.0, log=print, copy_to=None,
                  max_targets=None, thickness=None, min_d=None, exclude=TARGET_EXCLUDE, material=None, oxide_nm=None,
                  oxide_entered=None, graphite_thick_frac=None, graphite_ribbons=None, canvas=None, min_targets=MIN_TARGETS, tpolys=None, M=None,
                  target_order="size", target_rank="score", size_chunk=None, alt=True, size_max=None,
                  graphite_rank="model", graphite_max=None, target_mode="standard", min_clean_um2=MIN_CLEAN_UM2):
    """100x target list (.pos + .csv) from the live flake table, with the 10x focus Z of each flake from the tile Z.

    The best-scoring flakes above min_score, at most max_targets of them (the 100x time is bounded whatever the
    density of the piece), within thickness (min, max nm) if given, whose best terrace is at least min_d um across, and
    without any of the `exclude` flags.
    min_targets: if fewer qualify, the best-scoring other flakes (any score, best terrace >= FILL_MIN_D_UM, still
    within thickness and without the exclude flags) fill the list up to this many (e.g. graphite on 285 nm oxide,
    where every flake scores low).
    The .csv is always written (only a header if no flake qualifies), so a waiting microscope script goes on;
    copy_to: also write the .csv there (multi-piece runs: <run folder>/<piece>_targets.csv).
    tpolys, M: the live scan's terrace outlines (pixels) and pixel -> stage matrix, for terrace_poly_um (the chosen
    terrace's outline in stage um: the microscope's mosaics of terraces larger than a 100x field; terrace_polys_um).
    target_order ('size' | 'path'), target_rank ('score' | 'size'), size_chunk, size_max: hBN only (see TARGET_ORDERS
    and HBN_MAX_TARGETS above; graphite and other materials: score ranking (graphite: its layer tiers), shortest-path
    order, no size cut).
    alt: hBN, also write <stem>_alt.csv (and beside copy_to) with the flakes the other ranking would add / drop.
    graphite_rank ('model' | 'tiers'), graphite_max: graphite only (see GRAPHITE_RANKS above).  Every target gets
    p_keep100 (flakeid.target_model; nan without the material's model file): for hBN only written, not used.
    target_mode 'avram_new1', min_clean_um2: the flakes by their clean area instead (see TARGET_MODES above)."""
    from . import targets as T
    from . import target_model as TM
    rows = T.read_flakes(res_csv)
    ter_csv = Path(str(res_csv).replace("_flakes.csv", "_terraces.csv"))
    rank = "score"
    by_model = graphite_sel = False
    clean = target_mode == "avram_new1"
    clean_csv = Path(str(res_csv).replace("_flakes.csv", "_clean.csv"))
    if clean and clean_csv.is_file():
        above = clean_rows(clean_csv, float(min_clean_um2), log=log)
    elif clean:                                 # (a table without clean areas: its terraces by their clean area)
        above = clean_area_rows(rows, ter_csv, float(min_clean_um2), thickness=thickness, log=log)
    elif material == "graphite" and (oxide_nm or oxide_entered):
        # graphene / graphite: layer number from the measured contrast peaks, thinnest terrace first, not too big
        from . import graphite as G
        graphite_sel = True
        nrib = G.RIBBON_PLACES if graphite_ribbons is None else int(graphite_ribbons)
        gfrac = G.THICK_FRACTION if graphite_thick_frac is None else graphite_thick_frac
        above = G.select(ter_csv, oxide_nm, min_score=min_score, exclude=exclude, log=log, oxide_nominal=oxide_entered,
                         ribbons=nrib > 0)
        if thickness is not None:
            above = [r for r in above if thickness[0] <= T._f(r["best_thickness_nm"]) <= thickness[1]]
        if graphite_rank == "model" and TM.annotate(above, rows, ter_csv, "graphite", log):
            # (stable: equal or missing probabilities keep select()'s order; reserve takes each group in this order)
            p = [T._f(r.get("p_keep100")) for r in above]
            above = [above[i] for i in sorted(range(len(above)), key=lambda i: -p[i] if p[i] == p[i] else 1.0)]
            by_model = True
            log("graphite targets ranked by the kept-at-100x model (profiles/target_graphite.json) within the ribbon / "
                "layer groups (--graphite-rank)")
        above = G.reserve(above, max_targets, gfrac, log=log, ribbon_places=nrib)
    else:
        if material == "hbn" and oxide_nm:
            # colour fits both ~20 and ~150 nm: thin or thick from this chip's two colour groups (labelled direction)
            from . import branch as B
            ter_path = Path(str(res_csv).replace("_flakes.csv", "_terraces.csv"))
            prof = ""
            try:
                prof = Path(json.loads(Path(str(res_csv).replace(".csv", ".json")).read_text())["params"].get("profile") or "").stem
            except (OSError, ValueError, KeyError):
                pass
            bm = B.load(prof or "bx51_xc10_mm_raw16", "hbn", oxide_nm)
            if bm is not None and ter_path.is_file():
                dec = B.resolve(T.read_flakes(ter_path), bm.cal, log=log)
                for r in rows:
                    if dec.get(r.get("best_terrace_id")) == "thick":
                        r["quality"] = (r.get("quality") or "") + ";thick_branch"
        rank = target_rank if material == "hbn" and target_rank in T.RANKS else "score"
        above = T.select(rows, min_score=min_score, thickness=thickness, min_d=min_d or None, exclude_flags=exclude, rank=rank)
        if rank != "score":
            log(f"targets ranked by '{rank}': logit(score) + {T.SIZE_RANK_W:g} ln(best terrace d) (--target-rank)")
    if rows and material != "graphite" and not clean:
        log(f"targets: {len(above)} of {len(rows)} flakes have score >= {min_score}, terrace >= {min_d or 0:g} um and none of "
            f"{', '.join(exclude)}")
    sel = above[:max_targets] if max_targets and not clean else above      # (avram_new1: no limit, every clean area)
    if len(sel) < len(above):
        log(f"{len(above)} flakes qualify{f' at {thickness[0]:g}-{thickness[1]:g} nm' if thickness else ''}: "
            f"the first {len(sel)} in priority order are targets (--max-targets)")
    sel_ranked = sel                            # (before the --min-targets fill: the comparison list is about the ranking)
    want = 0 if clean else min(min_targets or 0, max_targets or 10 ** 9)     # (avram_new1: only flakes above the clean area)
    if len(sel) < want:
        have = {str(r["flake_id"]) for r in sel}
        more = [r for r in T.select(rows, thickness=thickness, min_d=FILL_MIN_D_UM, exclude_flags=exclude)
                if str(r["flake_id"]) not in have][:want - len(sel)]
        if more:
            log(f"only {len(sel)} flakes qualify: {len(more)} more of the best-scoring flakes added to reach "
                f"--min-targets {want} (scores {T._f(more[0].get('score')):.2f} to {T._f(more[-1].get('score')):.2f})")
            if material == "graphite":
                more = [dict(r, layers="") for r in more]   # same columns as the graphite rows
            sel = list(sel) + more
    hbn = material == "hbn"
    if material in ("hbn", "graphite") and not clean:
        TM.annotate(sel, rows, ter_csv, material, log)
    if graphite_sel and graphite_max and len(sel) > int(graphite_max):
        n0 = len(sel)
        sel = G.reserve(sel, int(graphite_max), gfrac, log=log, ribbon_places=nrib)
        log(f"{n0} targets: the first {len(sel)} in " + ("kept-at-100x order" if by_model else "priority order")
            + " are imaged, ribbon places and the thick fraction kept (--graphite-max-targets)")
    order = "clean" if clean else target_order if hbn and target_order in TARGET_ORDERS else ("model" if by_model else "path")
    if hbn and order == "size" and size_max and len(sel) > size_max:
        n0 = len(sel)
        big = sorted(range(n0), key=lambda i: (-T._f(sel[i].get("best_terrace_d_um"), 0), i))[:int(size_max)]
        sel = [sel[i] for i in sorted(big)]
        log(f"{n0} targets: the {len(sel)} with the largest best terrace are imaged (--hbn-max-targets; smallest kept "
            f"{min(T._f(r.get('best_terrace_d_um'), 0) for r in sel):.0f} um)")
    chunk = int(size_chunk or T.SIZE_CHUNK)
    tg = []
    if sel:
        tg = T.make_targets(sel, offset=offset, focus10=focus_map_tiles(tiles, z_scale), log=log, order=order, chunk=chunk)
        if order == "size" and len(tg) > 1:
            xy = np.array([[t["stage_x_um"], t["stage_y_um"]] for t in tg])
            log(f"(the shortest path over the same targets: {T.path_length(xy, T.order_path(xy)) / 1000:.1f} mm; "
                "--target-order path)")
        try:
            tpu = terrace_polys_um(sel, Path(str(res_csv).replace("_flakes.csv", "_terraces.csv")), tpolys, M, offset)
        except Exception as e:                  # (never costs the target list: the microscope takes single pictures)
            log(f"(terrace outlines not written: {type(e).__name__}: {e})")
            tpu = {}
        for t in tg:
            t["terrace_poly_um"] = tpu.get(int(t["flake_id"]), "")
        T.write_pos(out_stem.with_suffix(".pos"),
                    [T.position_entry(t["label"], (t["stage_x_um"], t["stage_y_um"]), t["z_um"],
                                      props={"flake_id": t["flake_id"], "thickness_nm": f"{t['thickness_nm']:.1f}",
                                             "score": f"{t['score']:.3f}", "source": res_csv.name})
                     for t in tg])
    else:
        log("no flakes with a clean area >= " + f"{min_clean_um2:g} um2: empty target list" if clean
            else f"no flakes with score >= {min_score}: empty target list")
    if tg and canvas is not None and copy_to:
        try:
            write_target_crops(canvas, tg, offset, Path(copy_to).parent / (Path(copy_to).name.split("_targets")[0] + "_targets_crops"))
        except Exception as e:
            log(f"(target pictures not written: {e})")
    for dst in [out_stem.with_suffix(".csv")] + ([Path(copy_to)] if copy_to else []):
        write_target_csv(dst, tg, list(tg[0]) if tg else TARGET_COLUMNS)
        log(f"wrote {dst} ({len(tg)} targets)")
    if hbn and alt and not clean:
        try:                                    # (never costs the target list)
            alt_dst = [out_stem.with_name(out_stem.name + "_alt.csv")]
            if copy_to:
                alt_dst.append(Path(copy_to).with_name(Path(copy_to).stem + "_alt.csv"))
            write_alt_targets(rows, sel_ranked, rank, max_targets, dict(min_score=min_score, thickness=thickness, min_d=min_d or None,
                                                                 exclude_flags=exclude),
                              alt_dst, offset, focus_map_tiles(tiles, z_scale), log)
        except Exception as e:
            log(f"(alternative target list not written: {type(e).__name__}: {e})")
    return out_stem.with_suffix(".csv")


def write_target_csv(dst, recs, cols):
    """A target list (.csv, via a .tmp and a rename)."""
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for t in recs:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
    replace_file(tmp, dst)


def write_alt_targets(rows, sel, rank, max_targets, filt, dsts, offset, focus10, log=print):
    """<piece>_targets_alt.csv: what the other ranking (score <-> size) would have changed in this target list.
    alt = add: the other ranking's target, not in this list (not imaged); alt = drop: in this list only because of
    this ranking.  Same columns as the target list (labels alt_NN_...) + alt, rank_score, rank_size (1-based places
    in the qualifying pool under each ranking); the adds in the other ranking's order, then the drops."""
    from . import targets as T
    other = "size" if rank == "score" else "score"
    places, alt_sel = {}, []
    for rk in T.RANKS:
        pool = T.select(rows, rank=rk, **filt)
        places[rk] = {str(r["flake_id"]): k + 1 for k, r in enumerate(pool)}
        if rk == other:
            alt_sel = pool[:max_targets] if max_targets else pool
    this_ids = {str(r["flake_id"]) for r in sel}
    alt_ids = {str(r["flake_id"]) for r in alt_sel}
    add = [r for r in alt_sel if str(r["flake_id"]) not in this_ids]
    drop = [r for r in sel if str(r["flake_id"]) not in alt_ids]
    recs = []
    if add or drop:
        recs = T.make_targets(add + drop, offset=offset, focus10=focus10, log=lambda m: None, order="score")
        for t in recs:
            fid = str(t["flake_id"])
            t["label"] = "alt_" + t["label"]
            t["alt"] = "add" if fid in alt_ids else "drop"
            t["rank_score"] = places["score"].get(fid, "")
            t["rank_size"] = places["size"].get(fid, "")
    cols = list(recs[0]) if recs else TARGET_COLUMNS + ["alt", "rank_score", "rank_size"]
    for dst in dsts:
        write_target_csv(dst, recs, cols)
    log(f"ranking '{other}' instead of '{rank}' (comparison only): {len(add)} other flakes in, {len(drop)} of these "
        f"{len(sel)} out, {len(this_ids & alt_ids)} in both -> {dsts[-1].name}")


def write_target_crops(cc, tg, offset, out_dir, half=180):
    """A 10x picture of each target flake from the viewer canvas, for the microscope's focus-by-hand window and the
    re-targeting: <out_dir>/<label>.jpg, ~390 um across, the 100x field marked where the target is."""
    import cv2
    from . import livecanvas
    can, _ = livecanvas._maps(cc)
    H, W = cc["shape"]
    Minv, P0, O, wh = np.array(cc["Minv"]), np.array(cc["P0"]), np.array(cc["O"]), np.array(cc["wh"], float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fw, fh = wh / 9.54                                     # the 100x field in 10x pixels (same camera)
    for t in tg:
        q = np.array([t["stage_x_um"] - offset[0], t["stage_y_um"] - offset[1]])     # 10x stage position
        cx, cy = (Minv @ (q - P0) + wh / 2 - O).round().astype(int)
        img = np.zeros((2 * half, 2 * half, 3), np.uint8)
        x0, y0 = cx - half, cy - half
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * half, W), min(y0 + 2 * half, H)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        img[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = can[sy0:sy1, sx0:sx1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (int(half - fw / 2), int(half - fh / 2)), (int(half + fw / 2), int(half + fh / 2)), (0, 190, 255), 2)
        cv2.drawMarker(img, (half, half), (0, 190, 255), cv2.MARKER_CROSS, 14, 1)
        cv2.imwrite(str(out_dir / f"{t['label']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def find_manifest(run_dir: Path, name=None, since=None, exclude=()):
    """The run file (<name>_run_<n>.json) written by the scan program in multi mode: the given one, else the newest
    (modified after `since`, if given) that is not in `exclude`."""
    if name:
        name = name[:-4] if name.endswith(".tmp") else name
        p = run_dir / name
        return p if (p.is_file() or p.with_name(p.name + ".tmp").is_file()) and p.name not in exclude else None
    # also a run file left as <name>.json.tmp: on a network share the script's rename can fail
    cands = {}
    for m in list(run_dir.glob("*_run_*.json")) + list(run_dir.glob("*_run_*.json.tmp")):
        key = m.name[:-4] if m.name.endswith(".tmp") else m.name
        if key not in exclude and (since is None or m.stat().st_mtime >= since):
            cands[key] = max(cands.get(key, 0), m.stat().st_mtime)
    ms = [run_dir / k for k in cands]
    return max(ms, key=lambda m: cands[m.name]) if ms else None


def watch_run(run_dir: Path, a, log=print):
    """Multi-piece runs: follow a run file and analyse its pieces (process_run).  With --continuous, keep
    going: after each run, wait for the next new run file in the folder (stop with Ctrl-C)."""
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    seen, rc, first = set(), 0, True
    try:
        while True:
            log(f"waiting for a{' new' if seen else ''} run file (*_run_*.json) in {run_dir} ...")
            man_path = None
            while man_path is None:
                # the first run: the given one, or any (--existing), or one written from 10 min before the start;
                # later runs: any run file not followed yet that was written after this command started
                since = None if (first and (a.run or a.existing)) else t_start - (600 if first else 0)
                man_path = find_manifest(run_dir, a.run if first else None, since=since, exclude=seen)
                if man_path is None:
                    time.sleep(2)
            seen.add(man_path.name)
            first = False
            rc |= process_run(run_dir, man_path, a, out, log)
            if not getattr(a, "continuous", False):
                return rc
            log(f"run {man_path.name} finished; --continuous: watching for the next run (Ctrl-C to stop)")
    except KeyboardInterrupt:
        log("stopped")
        return rc


def piece_material(p, tiles, a, log=print):
    """(material, oxide_nm) of a run-file piece: as entered at the microscope (run file, then scan_params.json),
    else the command line; (None, None) if neither."""
    stored = (p.get("material"), p.get("oxide_nm"))
    if not all(stored):
        from .mm import scan_params
        sp = scan_params(tiles)
        stored = (sp["material"], sp["oxide_nm"]) if sp else (None, None)
    if all(stored):
        mat = "hbn" if str(stored[0]).lower() in ("bn", "hbn") else str(stored[0]).lower()
        ox = float(stored[1])
        given = (a.material, a.oxide_nm)
        if (given[0] != "auto" and given[0] != mat) or (given[1] != "auto" and float(given[1]) != ox):
            log(f"{p['name']}: the microscope recorded {mat} on {ox:g} nm; using that, not {given[0]} {given[1]} from the command line")
        return mat, ox
    if a.material != "auto" and a.oxide_nm != "auto":
        return a.material, float(a.oxide_nm)
    return None, None


def pick_flat100(spec, tiles_dir):
    """The 100x flat for a piece: --flat100 FILE, or the OXIDE_NM=FILE entry nearest the piece's oxide."""
    if not spec:
        return None
    spec = [spec] if isinstance(spec, str) else list(spec)
    plain = [x for x in spec if "=" not in x]
    per = {}
    for x in spec:
        if "=" in x:
            k, v = x.split("=", 1)
            per[float(k)] = v
    if not per:
        return plain[0] if plain else None
    from .mm import scan_params
    sp = scan_params(tiles_dir)
    if sp is None:
        return plain[0] if plain else None
    return per[min(per, key=lambda o: abs(o - float(sp["oxide_nm"])))]


EXPORT_DONE = "export_done.json"


def mark_exported(piece_out: Path, what):
    """<results>/<piece>/export_done.json: {"100x": time, "ir": time} once that part of the piece's export has ended
    (the scan program copies the piece to the share then, while the next pieces are imaged)."""
    try:
        d = Path(piece_out) / EXPORT_DONE
        old = json.loads(d.read_text()) if d.is_file() else {}
        old[what] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = d.with_name(d.name + ".tmp")
        tmp.write_text(json.dumps(old))
        replace_file(tmp, d)
    except (OSError, ValueError):
        pass


def export_100x(run_dir: Path, man_path: Path, manifest, viewers: dict, a, log=print, analysing=None):
    """Every piece's 100x images into its viewer (flakeid.hundred), the 100x TIFFs deleted unless --keep-tiles: each
    piece as soon as the microscope has finished its 100x imaging (run file: piece state 'imaged') and its viewer is
    built, so the export runs while the next piece is imaged (it took ~12 min after the end of a 2-piece run when done
    all at the end, microscope PC 2026-10-03); stacking-domain images (taken after all 100x) in a second pass at the end
    of the run.  analysing: a threading.Event set while pieces are still being analysed (viewers still to come)."""
    from .hundred import export
    t0 = time.time()
    done_pieces, domains_done = set(), set()

    def one(name, vdir, second=False):
        f100 = run_dir / f"{name}_100x"
        flat100 = pick_flat100(getattr(a, "flat100", None), run_dir / f"{name}_tiles")
        if not f100.is_dir():
            log(f"{name}: no 100x images")
            return
        try:
            if flat100 and not second:
                log(f"{name}: 100x flat field {flat100}")
            busy = analysing is not None and analysing.is_set()
            export(f100, vdir, flat=flat100, flat_domain=getattr(a, "flat_domain", None), keep_tiles=not a.delete_tiles, log=log, skip_done=second,
                   workers=2 if busy else None)
        except Exception as exc:
            log(f"{name}: 100x export failed ({type(exc).__name__}: {exc}); the 100x TIFFs are kept")
        mark_exported(Path(vdir).parent, "ir" if second else "100x")
    log("100x images go into the viewers as each piece's 100x imaging ends")
    while True:
        man = manifest()
        st = man.get("state")
        pst = {q["name"]: q.get("state") for q in man.get("pieces", [])}
        ended = st in ("done", "stopped")
        for name, vdir in list(viewers.items()):
            if name not in done_pieces and (pst.get(name) == "imaged" or ended):
                one(name, vdir)
                done_pieces.add(name)
        if ended:
            for name in sorted(done_pieces - domains_done):
                if (run_dir / f"{name}_domains").is_dir():
                    one(name, viewers[name], second=True)
                domains_done.add(name)
            if analysing is None or not analysing.is_set():
                missing = set(viewers) - done_pieces
                if not missing:
                    return
        if time.time() - t0 > a.wait_100x_h * 3600:
            log("gave up waiting for the end of the run; export the 100x images later with python -m flakeid.hundred")
            return
        time.sleep(10)


def process_run(run_dir: Path, man_path: Path, a, out: Path, log=print):
    """Analyse the pieces of one run file one after the other as they are scanned, and write each piece's
    target list into the run folder for the microscope script."""
    log(f"run file {man_path.name}")

    def manifest():
        # the newer of <name>.json and a leftover <name>.json.tmp (a rename on a network share can fail)
        for _ in range(10):
            try:
                tmp = man_path.with_name(man_path.name + ".tmp")
                ps = [q for q in (man_path, tmp) if q.is_file()]
                q = max(ps, key=lambda q: q.stat().st_mtime)
                return json.loads(q.read_text())
            except (OSError, ValueError):
                time.sleep(0.5)
        return {"pieces": [], "state": "?"}

    done, mats, viewers, finishing = [], {}, {}, []
    analysing = threading.Event()
    analysing.set()
    exporter = None
    if not getattr(a, "no_100x", False):
        exporter = threading.Thread(target=export_100x, args=(run_dir, man_path, manifest, viewers, a, log, analysing),
                                    daemon=True, name="100x export")
        exporter.start()
    while True:
        man = manifest()
        todo = [p for p in man["pieces"] if p["name"] not in done]
        if not todo:
            break
        p = todo[0]
        name, tiles = p["name"], run_dir / p["tiles"]
        if (run_dir / f"{name}_targets.csv").exists() and not getattr(a, "redo", False):
            log(f"{name}: already has a target list ({name}_targets.csv), skipped (--redo to analyse it again)")
            done.append(name)
            continue
        if p["state"] == "failed" and not tiles.exists():
            log(f"{name}: scan failed, skipped")
            done.append(name)
            continue
        if not tiles.exists() or not (run_dir / f"{name}_scan.pos").exists():
            if man.get("state") in ("stopped", "done"):
                done.append(name)
                continue
            # the microscope script was stopped: the run file no longer changes and this piece never starts
            ps = [q for q in (man_path, man_path.with_name(man_path.name + ".tmp")) if q.is_file()]
            age = time.time() - max(q.stat().st_mtime for q in ps) if ps else 0
            if age > a.stale_min * 60:
                log(f"the run file has not changed for {age / 60:.0f} min and {name} has no tiles: the scan seems to have "
                    f"stopped; {len(todo)} piece(s) left unanalysed")
                break
            time.sleep(2)
            continue
        mat, ox = piece_material(p, tiles, a, log)
        if mat is None:
            log(f"{name}: no material / oxide stored in the run file or {tiles.name}/scan_params.json and none given on the "
                f"command line: skipped (run again with MATERIAL OXIDE_NM)")
            done.append(name)
            continue
        log(f"==== {name}: {mat} on {ox:g} nm oxide")
        mats[name] = (mat, ox)
        # 100x targets: the piece's own number when the microscope gave one (its area x targets per cm2), else --max-targets
        mt = int(p.get("max_targets") or 0) or a.max_targets
        if p.get("max_targets"):
            log(f"{name}: up to {mt} 100x targets (piece area {float(p.get('area_mm2') or 0) / 100:.2f} cm2)")
        if mat == "hbn" and p.get("hbn_max_targets") is not None:
            log(f"{name}: hBN, " + (f"the {int(p['hbn_max_targets'])} with the largest best terrace imaged"
                                    if p["hbn_max_targets"] else "every target imaged") + " (its area)")
        live = LiveScan(tiles, mat, ox, out / name, flat=a.flat, block=a.block, margin_um=a.margin,
                        workers=a.workers, idle_s=a.idle, refine_oxide=not a.no_refine_oxide, log=log, viewer=not a.no_viewer,
                        workers_final=a.workers_final, options=live_options(a, mat))

        def failed(nm=name):
            st = next((q["state"] for q in manifest()["pieces"] if q["name"] == nm), "")
            return st == "failed"
        live.abort = failed
        live.busy = lambda: manifest().get("state") == "scanning"

        def provisional(n, nm=name, lv=live, mat_=mat, mt_=mt, p_=p):
            # the flakes of the blocks analysed so far, for the microscope to image at 100x while the rest is analysed
            # (hBN cut by size: the largest PARTIAL_MAX of the final list's pool, so they are mostly in the final list)
            to = piece_target_opts(a, p_)
            if lv.material == "hbn" and to["size_max"] and to["target_order"] == "size":
                mt_p, to["size_max"] = mt_, min(PARTIAL_MAX, to["size_max"])
            else:
                mt_p = min(PARTIAL_MAX, mt_ or PARTIAL_MAX)
            write_targets(out / nm / f"{lv.name}_live_flakes.csv", out / nm / f"{lv.name}_live_targets_partial",
                          sorted(lv.tiles.values(), key=_index), a.min_score, a.offset, log=lambda m: None,
                          copy_to=run_dir / f"{nm}_targets_partial.csv", max_targets=mt_p,
                          thickness=thickness_for(a.thickness, mat_), min_d=a.min_d, exclude=PARTIAL_EXCLUDE, material=lv.material,
                          oxide_nm=lv.oxide_frozen, oxide_entered=lv.oxide_entered, graphite_thick_frac=getattr(a, "graphite_thick_frac", None), graphite_ribbons=getattr(a, "ribbons", None),
                          canvas=lv.cc, tpolys=lv.tpolys, M=lv.M, alt=False, **to)
            with open(run_dir / f"{nm}_targets_partial.csv") as fh:
                n_t = max(0, sum(1 for _ in fh) - 1)
            lv.say(f"provisional target list after {n} blocks: {n_t} flakes ({nm}_targets_partial.csv)")
        if not getattr(a, "no_partial", False):
            live.on_progress = provisional
        try:
            live.run(build_viewer=False)
        except LiveError as exc:
            log(f"ERROR on {name}: {exc}.  No target list written for it (the microscope script keeps waiting for "
                f"{name}_targets.csv); fix the problem and run this command again: pieces with a target list are skipped.")
            return 2
        # the piece's entry again (its target numbers may have been written after the analysis started on it)
        p = next((q for q in (manifest().get("pieces") or []) if q.get("name") == name), p)
        mt2 = int(p.get("max_targets") or 0) or a.max_targets
        if mt2 != mt:
            log(f"{name}: up to {mt2} 100x targets (the run file's number for the piece, read again: {mt} before)")
            mt = mt2
        write_targets(out / name / f"{live.name}_live_flakes.csv", out / name / f"{live.name}_live_targets",
                      sorted(live.tiles.values(), key=_index), a.min_score, a.offset, log=live.say,
                      copy_to=run_dir / f"{name}_targets.csv", max_targets=mt, min_targets=a.min_targets,
                      thickness=thickness_for(a.thickness, mat), min_d=a.min_d, material=live.material, oxide_nm=live.oxide_frozen or live.oxide_nm,
                      oxide_entered=live.oxide_entered, graphite_thick_frac=getattr(a, "graphite_thick_frac", None), graphite_ribbons=getattr(a, "ribbons", None), canvas=live.cc,
                      tpolys=live.tpolys, M=live.M, **piece_target_opts(a, p))

        def finish(live=live, name=name, tiles=tiles):
            # the viewer (and then the tile deletion) after the target list, in the background: the next piece's
            # analysis starts at once (a viewer took ~90 s on the microscope PC, all of it time the next piece's
            # provisional targets came later and the 100x imaging waited)
            try:
                live.finish_viewer()
            except Exception as exc:
                import traceback
                log(f"ERROR building the viewer of {name}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            if getattr(live, "viewer_path", None):
                viewers[name] = live.viewer_path.parent
            if a.delete_tiles and not a.then_batch and getattr(live, "viewer_path", None):
                from .cli import delete_tiles
                delete_tiles(tiles, [out / name / f"{live.name}_live_flakes.csv"], live.viewer_path.parent)
        th = threading.Thread(target=finish, daemon=True, name=f"viewer {name}")
        th.start()
        finishing.append(th)
        done.append(name)
    for th in finishing:
        th.join()
    analysing.clear()
    log(f"all {len(done)} pieces analysed")
    if exporter is not None:
        if viewers:
            log("waiting for the end of the 100x imaging (exports each piece's 100x images as it ends) ...")
            exporter.join()
        if viewers and a.no_designer:
            log("designer run: not made here (--no-designer); the vdW Stack Designer builds it from the results on the share")
        elif viewers:
            # the run's flakes for the vdW Stack Designer: <results>/designer/ (copied to the share with the results;
            # added from the designer page, "Scan results")
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
                from make_designer_bundle import main as bundle
                log("designer bundle ...")
                bundle([str(out), "--run", out.name, "-o", str(out / "designer")])
            except Exception as exc:
                log(f"designer bundle not made ({type(exc).__name__}: {exc}); make it later with tools/make_designer_bundle.py")
    rc = 0
    if a.then_batch:
        from .cli import main as batch
        for name in done:
            tiles = run_dir / f"{name}_tiles"
            if not tiles.exists():
                continue
            if name not in mats:
                continue
            log(f"full-mosaic analysis of {name} for the viewer ...")
            args = [str(tiles), mats[name][0], str(mats[name][1]), "--viewer", "--json", "-o", str(out / name)]
            if a.flat:
                args += ["--flat", a.flat]
            if not a.delete_tiles:
                args.append("--keep-tiles")
            rc |= batch(args)
    return rc


def replay(src: Path, dst: Path, interval: float, log=print):
    """Copy a finished scan into `dst` tile by tile in acquisition order (TIFF first, then metadata.txt)."""
    tiles = sorted((t for t in (read_tile(m) for m in src.glob("*/metadata.txt")) if t), key=_index)
    dst.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"_tiles(_\d+)?$", "", dst.name)
    sname = re.sub(r"_tiles(_\d+)?$", "", src.name)
    for suf in ("_scan.pos", "_chip.json"):
        if (src.parent / f"{sname}{suf}").is_file():
            shutil.copy(src.parent / f"{sname}{suf}", dst.parent / f"{name}{suf}")
    if (src / "scan_params.json").is_file():
        shutil.copy(src / "scan_params.json", dst / "scan_params.json")
    log(f"replay: {len(tiles)} tiles from {src}, one every {interval:g} s")
    for t in tiles:
        d = dst / t.label
        d.mkdir(exist_ok=True)
        shutil.copy(t.path, d / t.path.name)
        shutil.copy(t.path.parent / "metadata.txt", d / "metadata.txt")
        time.sleep(interval)
    log("replay: all tiles copied")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m flakeid.live", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("folder", help="the scan's tile folder (<NAME>_tiles), or the run folder of a multi-piece scan "
                                  "(the scan program's multi mode; it holds <name>_run_<n>.json)")
    p.add_argument("material", nargs="?", default="auto",
                   help="graphite | hbn; default auto: as entered at the microscope (multi-piece run file, or the tile "
                        "folder's scan_params.json).  In a run folder, a piece's stored material and oxide take "
                        "precedence; values given here are used for pieces without them")
    p.add_argument("oxide_nm", nargs="?", default="auto", help="SiO2 thickness in nm; default auto (as entered at the microscope)")
    p.add_argument("--flat", help="flat field .npz")
    p.add_argument("-o", "--out-dir", required=True)
    p.add_argument("--block", type=int, default=4, help="block size in tiles (default 4 = 4x4)")
    p.add_argument("--margin", type=float, default=400.0, help="context around each block (um, default 400)")
    p.add_argument("--workers", type=int, help="worker processes while the microscope is scanning (default: min(4, cores - 1))")
    p.add_argument("--workers-final", type=int, help="worker processes once the scanning is over (default: one more, at most "
                                                     "the number of cores)")
    p.add_argument("--no-partial", action="store_true", help="multi-piece run folder: no provisional target lists "
                                                             "(<piece>_targets_partial.csv) while a piece is still being analysed")
    p.add_argument("--idle", type=float, default=120.0, help="without a scan plan: stop after this many s without new tiles")
    p.add_argument("--no-refine-oxide", action="store_true")
    p.add_argument("--min-score", type=float, default=0.75, help="target list: min flake score (default 0.75)")
    p.add_argument("--max-targets", type=int, default=80,
                   help="target list: at most this many flakes per piece, the best-scoring first (default 80, about 35 min "
                        "of 100x imaging with 3 positions per flake; 0 = no limit)")
    p.add_argument("--min-targets", type=int, default=MIN_TARGETS, metavar="N",
                   help=f"target list: at least this many flakes per piece (default {MIN_TARGETS}); if fewer reach "
                        f"--min-score, the best-scoring other flakes at least {FILL_MIN_D_UM:g} um across are added (0 = off)")
    p.add_argument("--graphite-thick-frac", type=float, default=None, metavar="F",
                   help="graphite target lists: fraction of --max-targets kept for flakes thicker than 4 layers (default "
                        "0.6; the rest go to 1-4 layer graphene, either group's unused places to the other)")
    p.add_argument("--ribbons", type=int, default=None, metavar="N",
                   help="graphite target lists: places kept for ribbons (long, narrow, isolated terraces of a few nm, "
                        "targeted whatever their width or score; flakeid.graphite.RIBBON; default 10, 0 = no ribbon rule)")
    p.add_argument("--target-mode", choices=TARGET_MODES, default="standard",
                   help="which flakes are 100x targets: standard (default): the learned score and the quality flags; "
                        "avram_new1: every flake whose clean area (its largest one-colour region x its fill) is at least "
                        "--min-clean-um2, largest first")
    p.add_argument("--min-clean-um2", type=float, default=MIN_CLEAN_UM2, metavar="UM2",
                   help=f"--target-mode avram_new1: the smallest clean area (um2, default {MIN_CLEAN_UM2:g})")
    p.add_argument("--clean-thickness", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="--target-mode avram_new1: the thickness window of the clean areas (nm; default the material's, "
                        "hBN 0-100, graphite 0.3-25)")
    p.add_argument("--target-order", choices=TARGET_ORDERS, default="size",
                   help="hBN target lists: imaging order.  size (default): largest best terrace first, a short stage path "
                        "within groups of --size-chunk (a run cut short has the biggest flakes); path: the shortest stage "
                        "path over the whole list (before 2026-10-22).  Graphite and other materials: path")
    p.add_argument("--hbn-max-targets", type=int, default=HBN_MAX_TARGETS, metavar="N",
                   help=f"hBN target lists with --target-order size: of the targets chosen (best scores, --max-targets), "
                        f"only the N with the largest best terrace (default {HBN_MAX_TARGETS}; 0 = all)")
    p.add_argument("--graphite-rank", choices=GRAPHITE_RANKS, default="model",
                   help="graphite target lists: model (default): the flakes most likely kept at 100x first within the "
                        "ribbon / layer groups, imaged in that order (profiles/target_graphite.json); tiers: layer tier, "
                        "then the score, shortest stage path (before 2026-10-23)")
    p.add_argument("--graphite-max-targets", type=int, default=GRAPHITE_MAX_TARGETS, metavar="N",
                   help="graphite target lists: at most N imaged per piece, the first in --graphite-rank order with the "
                        f"ribbon places and thick fraction kept (default {GRAPHITE_MAX_TARGETS}: no cut beyond --max-targets)")
    p.add_argument("--size-chunk", type=int, default=None, metavar="N",
                   help="--target-order size: targets per group (default 10; 1 = strictly by size)")
    p.add_argument("--target-rank", choices=("score", "size"), default="score",
                   help="hBN target lists: which flakes are targets.  score (default): the best scores; size (trial): "
                        "logit(score) + 2 ln(best terrace d).  Either way <piece>_targets_alt.csv lists the flakes the "
                        "other ranking would add / drop")
    p.add_argument("--min-d", type=float, default=MIN_TARGET_D_UM, metavar="UM",
                   help=f"target list: only flakes whose best terrace is at least this wide (inscribed diameter, um; "
                        f"default {MIN_TARGET_D_UM:g}; 0 = any)")
    p.add_argument("--thickness", action="append", default=[], metavar="[MATERIAL:]MAX or [MATERIAL:]MIN-MAX",
                   help="target list: only flakes whose best terrace is at most MAX nm thick (or within MIN-MAX), for every "
                        "material or per material, e.g. --thickness hbn:80 (all hBN up to 80 nm; graphite unlimited)")
    p.add_argument("--offset", nargs=2, type=float, default=(24.0, -18.3), metavar=("DX", "DY"),
                   help="10x -> 100x objective offset added to the targets (stage um; default 24.0 -18.3)")
    p.add_argument("--no-designer", action="store_true",
                   help="multi-piece run folder: do not build the designer run at the end (the vdW Stack Designer builds it "
                        "from the results on the share, on a faster computer)")
    p.add_argument("--no-viewer", action="store_true",
                   help="do not build the zoomable viewer during the live analysis (it is built from the analysed blocks, "
                        "without reading the tiles again; needs the scan plan <NAME>_scan.pos)")
    p.add_argument("--flat-domain", default=None, metavar="NPZ",
                   help="IR flat field (scan: Flat field, IR (domain camera)): the stacking-domain images are divided by it")
    p.add_argument("--flat100", action="append", default=[],
                   help="100x flat field .npz for the 100x images in the viewer (else a fixed black level).  One file for all "
                        "pieces, or one per oxide as OXIDE_NM=FILE (repeat; each piece uses the nearest oxide), e.g. "
                        "--flat100 285=flats/flat_100x_285nm.npz --flat100 90=flats/flat_100x_90nm.npz")
    p.add_argument("--no-100x", action="store_true", help="multi-piece run folder: do not wait for the 100x images and put "
                                                          "them into the viewers (python -m flakeid.hundred does it later)")
    p.add_argument("--wait-100x-h", type=float, default=6.0, help="hours to wait for the end of the run's 100x imaging (default 6)")
    p.add_argument("--then-batch", action="store_true",
                   help="after the scan: full-mosaic analysis with the zoomable viewer (find_flakes --viewer --json)")
    p.add_argument("--keep-tiles", action="store_true",
                   help="keep the tile TIFFs.  By default they are deleted once the viewer and the tables are written "
                        "(metadata.txt kept; with no viewer, e.g. --no-viewer, nothing is deleted).  Keep them while the "
                        "models are being developed (a new material, calibration work): deleted tiles cannot be analysed again")
    p.add_argument("--delete-tiles", action="store_true", help=argparse.SUPPRESS)      # (the default now; still accepted)
    p.add_argument("--run", help="multi-piece run folder: the run file to follow (default: the newest one written from "
                                 "10 min before this command on)")
    p.add_argument("--existing", action="store_true", help="multi-piece run folder: also accept an older run file")
    p.add_argument("--redo", action="store_true",
                   help="multi-piece run folder: analyse pieces again even if their <piece>_targets.csv exists")
    p.add_argument("--stale-min", type=float, default=15.0,
                   help="multi-piece run folder: give up waiting for the next piece after the run file has not changed "
                        "for this many minutes (the scan was stopped; default 15)")
    p.add_argument("--continuous", action="store_true",
                   help="multi-piece run folder: after a run, keep watching and follow every new run file "
                        "(stop with Ctrl-C)")
    p.add_argument("--replay", help="test: copy this finished scan into FOLDER tile by tile")
    p.add_argument("--interval", type=float, default=0.9, help="replay: seconds per tile (default 0.9, the scan rate)")
    a = p.parse_args(argv)
    a.delete_tiles = not a.keep_tiles
    if a.delete_tiles and a.no_viewer and not a.then_batch:
        print("--no-viewer: the tile TIFFs are kept (the viewer is what is kept of the images)")
    a.flat = check_flat(a.flat)
    folder = Path(a.folder)
    is_tiles = folder.name.endswith("_tiles") or re.search(r"_tiles_\d+$", folder.name) or any(folder.glob("*/metadata.txt"))
    if not a.replay and folder.is_dir() and not is_tiles:
        return watch_run(folder, a)
    if a.replay:
        if folder.exists() and any(folder.iterdir()):
            p.error(f"replay target {folder} is not empty")
        threading.Thread(target=replay, args=(Path(a.replay), folder, a.interval), daemon=True).start()
        while not folder.exists():
            time.sleep(0.2)
    if "auto" in (a.material, a.oxide_nm):
        from .mm import resolve_material
        t0 = time.time()
        while not (folder / "scan_params.json").exists() and time.time() - t0 < 60:   # written just before the first tile
            time.sleep(1)
        try:
            a.material, a.oxide_nm = resolve_material(folder, a.material, a.oxide_nm)
        except ValueError as exc:
            p.error(str(exc))
        print(f"{folder.name}: {a.material} on {a.oxide_nm:g} nm oxide (from the scan parameters)")
    a.oxide_nm = float(a.oxide_nm)
    live = LiveScan(folder, a.material, a.oxide_nm, a.out_dir, flat=a.flat, block=a.block, margin_um=a.margin,
                    workers=a.workers, idle_s=a.idle, refine_oxide=not a.no_refine_oxide, viewer=not a.no_viewer,
                    workers_final=a.workers_final, options=live_options(a, a.material))
    try:
        live.run(build_viewer=False)
    except LiveError as exc:
        live.say(f"ERROR: {exc}.  No target list written.")
        return 2
    csv_path = Path(a.out_dir) / f"{live.name}_live_flakes.csv"
    write_targets(csv_path, Path(a.out_dir) / f"{live.name}_live_targets",
                  sorted(live.tiles.values(), key=_index), a.min_score, a.offset, log=live.say,
                  max_targets=a.max_targets, min_targets=a.min_targets, thickness=thickness_for(a.thickness, a.material), min_d=a.min_d, material=live.material, oxide_nm=live.oxide_frozen or live.oxide_nm,
                  oxide_entered=live.oxide_entered, graphite_thick_frac=getattr(a, "graphite_thick_frac", None), graphite_ribbons=getattr(a, "ribbons", None),
                  tpolys=live.tpolys, M=live.M, **target_opts(a))
    live.finish_viewer()
    if a.delete_tiles and not a.then_batch and getattr(live, "viewer_path", None):
        from .cli import delete_tiles
        delete_tiles(folder, [csv_path], live.viewer_path.parent)
    if a.then_batch:
        from .cli import main as batch
        live.say("full-mosaic analysis for the viewer ...")
        args = [str(folder), a.material, str(a.oxide_nm), "--viewer", "--json", "-o", a.out_dir]
        if a.flat:
            args += ["--flat", a.flat]
        if not a.delete_tiles:
            args.append("--keep-tiles")
        return batch(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
