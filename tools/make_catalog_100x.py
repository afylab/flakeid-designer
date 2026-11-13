#!/usr/bin/env python3
"""Designer catalog of the flakes imaged at 100x (only those), from live-analysed Micro-Manager runs.

    python tools/make_catalog_100x.py ~/FlakeResults/multitest/chip_1 ~/FlakeResults/multitest/chip_2 -o designer_100x
    cd designer_100x && python3 -m http.server 8000         # then open http://localhost:8000

Per flake with a 100x image (the chip viewer's 100x/ folder, from `python -m flakeid.hundred --viewer`):
  * cut-out: the connected material around the imaged terrace in the natural 100x JPEG (colour more than
    --material-tol from the bare substrate), at 1/--downscale of the 100x resolution;
  * terrace: the 100x terrace outline of the export (flakeid.hundred), with its debris / steps outlines and clean
    area kept for the designer;
  * thickness of the terrace from the 10x measured calibrations: graphene 1-4 layers from the green-contrast peaks,
    hBN thin / thick decided per chip where the colour fits both ~20 and ~150 nm.  Flakes whose terrace is now
    judged unusable (thick hBN, bulk graphite, black slivers) are left out (counted in the log);
  * terrace mosaic (the scan imaged the fields around a terrace larger than one picture; flakeid.fieldmosaic): the
    terrace followed into the mosaic (same colour rule as the export), and the flake's outline and cut-out taken from
    the mosaic, so they are not cut off at the picture's edges (kept in the picture's pixels: frame x0 / y0 < 0 and
    outlines beyond w / h; --no-mosaic-outline: the picture only).  Debris, steps and the clean area stay the picture's;
    the picture's terrace is kept when the mosaic's does not agree with it inside the picture;
  * outline: the cut-out's colour mask refined on the picture at the cut-out's scale (flakeid.outline100: the edge
    at half the flake's contrast, not in the halo outside it, and flake parts the 10x outline missed; outline_refined,
    else outline_why: why the colour mask's outline is kept); the cut-out, its transparency and the flake's area follow
    it (--no-refine-outline: the colour mask as before 2026-10-22);
  * position: placed on the chip overview (the viewer's 1/4-scale level) from the 100x stage position;
  * hBN: defects of the terrace in the natural 100x picture (flakeid.defects100: largest clean disc, step-edge
    density; terraces[0].defects) and from them and the label model's p_quality the score p_keep and tier keep /
    review / reject of the designer-decision model (profiles/screen100_hbn.json "pick"); "reject" stays in the catalog.
Flakes imaged more than once: the image with the least noise.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_afm_set import load_viewer_data, results_file, viewer_dir  # noqa: E402
from make_catalog import pack  # noqa: E402
from make_catalog_live import OV_LEVELS_DOWN, Pyramid  # noqa: E402
from flakeid import branch as B  # noqa: E402
from flakeid import defects100 as D100  # noqa: E402
from flakeid import graphite as GR  # noqa: E402
from flakeid.look import CANVAS_SUBSTRATE, look  # noqa: E402
from flakeid import netscreen as NS  # noqa: E402
from flakeid import outline100 as O100  # noqa: E402
from flakeid import screen100 as S100  # noqa: E402
from flakeid.hundred import (_M100, _centre_is_substrate, _mode_rgb, _terrace_at, dt_view, screen_record,  # noqa: E402
                             to_stage, SEED_AREA_HI, SEED_AREA_LO)
from flakeid.outline import polygons  # noqa: E402
from flakeid.mm import StageMap  # noqa: E402

ATLAS = 4096
THUMB_W = 320                         # full-frame thumbnails for the flake picker (packed into thumbs_NN.jpg)
# views made here (re-analysed pictures): JPEG quality 95 at full colour resolution (4:4:4; half colour resolution
# showed blotchy texture near flakes, 2026-10-15)
JPEG_VIEW = [cv2.IMWRITE_JPEG_QUALITY, 95] + ([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
                                              if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") else [])



def t100x(R, material=None):
    """The target terrace's thickness from its 100x colour as the scan program measured it (flakeid.thick100, in the
    picture's _regions.json since py-2026-10-30a): dict(t, alts, agree ("yes" / "alt" / "no" against the 10x), poor),
    or None (older runs; a picture where it could not be measured).  (2026-11-05u) Graphite: refitted from the colour kept
    there (thick100.fit: since 2026-11-12 graphite's own fitted 100x model, all three channels), so a Rebuild
    corrects runs measured before."""
    x = R.get("t100") if isinstance(R, dict) else None
    if not isinstance(x, dict) or x.get("t") is None:
        return None
    if str(material).lower() == "graphite" and x.get("contrast") is not None:
        ox = x.get("oxide_nm", R.get("oxide_nm"))
        if ox is not None:
            from flakeid import thick100 as T
            x = T.fit(x["contrast"], "graphite", float(ox), x.get("t10")) or x
    return dict(t=x["t"], alts=list(x.get("alts") or []), agree=x.get("agree"), poor=bool(x.get("poor")))

def h100(viewer: Path) -> dict:
    h = (viewer / "index.html").read_text()
    i = h.index("const H100=") + 11
    return json.loads(h[i:h.index(";\n", i)])


def read_bytes(path, tries=4, wait_s=2.0):
    """A picture's bytes from the share, read completely: a JPEG must end with its end-of-image marker (a share that
    hiccups mid-read gave short pictures, 20261014_run_1); retried with a pause, else IOError (the build stops with
    the reason instead of using a broken picture)."""
    import time
    path, err = Path(path), None
    for k in range(tries):
        try:
            b = path.read_bytes()
            if path.suffix.lower() not in (".jpg", ".jpeg") or (b[:2] == b"\xff\xd8" and b.rstrip(b"\x00")[-2:] == b"\xff\xd9"):
                return b
            err = f"incomplete JPEG ({len(b)} bytes)"
        except OSError as e:
            err = f"{type(e).__name__}: {e}"
        if k < tries - 1:
            time.sleep(wait_s * (k + 1))
    raise IOError(f"could not read {path} completely after {tries} tries ({err}): is the share reachable?")


def read_img(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread of a picture on the share, read completely (read_bytes) and decoded, or IOError."""
    a = cv2.imdecode(np.frombuffer(read_bytes(path), np.uint8), flags)
    if a is None:
        raise IOError(f"{path} could not be decoded")
    return a


def copy_img(src, dst):
    """shutil.copy of a picture from the share, read completely first (read_bytes)."""
    Path(dst).write_bytes(read_bytes(src))


def noise(path, flags=cv2.IMREAD_GRAYSCALE):
    a = read_img(path, flags).astype(np.float32)
    return float((a - cv2.GaussianBlur(a, (0, 0), 2)).std())


class Stage:
    """The run's natural 100x pictures read from the share once each (in parallel: IO_THREADS) into a folder on this
    computer; the substrate colour, the choice among repeated shots, the cut-out and the copy then read the local copy
    (before: each picture read and decoded 3-4 times over the network, 2026-10-16).  Other files (high contrast,
    thickness change, IR) are copied in the same threads (copy()); wait() for them before using the output."""
    IO_THREADS = 6

    def __init__(self, folder: Path):
        from concurrent.futures import ThreadPoolExecutor
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(self.IO_THREADS)
        self.local, self.mode, self.jobs = {}, {}, []

    def _one(self, src):
        import hashlib
        b = read_bytes(src)
        dst = self.dir / (hashlib.sha1(str(src).encode()).hexdigest()[:16] + Path(src).suffix)
        dst.write_bytes(b)
        # the substrate estimate at 1/4 resolution (decoded scaled: ~10x less work than the full picture, the mode of
        # 4x4 averages instead of every 4th pixel)
        a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        m = _mode_rgb(cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32), step=4) if a is not None else None
        return src, dst, m

    def fetch(self, paths, progress=None):
        todo = [Path(p) for p in dict.fromkeys(paths)]
        for k, (src, dst, m) in enumerate(self.pool.map(self._one, todo), 1):
            self.local[str(src)], self.mode[str(src)] = dst, m
            if progress is not None:
                progress(k, len(todo))

    def path(self, src):
        return self.local.get(str(src), Path(src))

    def substrate(self, paths):
        """chip_substrate from the staged estimates."""
        ms = [self.mode[str(p)] for p in paths if self.mode.get(str(p)) is not None]
        if not ms:
            return chip_substrate(paths)
        good = [m for m, f in ms if f >= 0.2] or [m for m, _ in ms]
        return np.median(np.array(good), axis=0)

    def copy(self, src, dst):
        """copy_img in a worker thread (a staged picture: from this computer)."""
        self.jobs.append(self.pool.submit(lambda: Path(dst).write_bytes(self.path(src).read_bytes()) if str(src) in self.local
                                          else copy_img(src, dst)))

    def wait(self):
        jobs, self.jobs = self.jobs, []
        for j in jobs:
            j.result()                                   # (an error of a copy stops the build with its reason)

    def close(self):
        self.wait()
        self.pool.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)


def poly_of(mask, simplify=1.5):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), simplify, True).reshape(-1, 2)
    return [round(float(v) + 0.5, 1) for v in c.ravel()]


def scale_polys(polys, k, ox=0.0, oy=0.0):
    return [[round((p[i] - (ox if i % 2 == 0 else oy)) * k, 1) for i in range(len(p))] for p in polys]


def chip_substrate(paths):
    """Substrate colour of a run's natural 100x JPEGs (one white balance per run): median of the per-image most
    common colour over the images where it covers a good part of the frame."""
    ms = []
    for pth in paths:
        a = cv2.cvtColor(read_img(pth), cv2.COLOR_BGR2RGB)[::4, ::4].astype(np.float32)
        m, frac = _mode_rgb(a, step=4)
        ms.append((m, frac))
    good = [m for m, f in ms if f >= 0.2] or [m for m, _ in ms]
    return np.median(np.array(good), axis=0)


def region_10x(f_poly, smap, R, shape, margin_um=5.0, org=(0, 0)):
    """The flake's 10x outline (mosaic px) drawn into the 100x frame, grown by margin_um: limits the cut-out to the
    flake that was found at 10x (residue haze around it would otherwise join the mask).  org: where the 100x frame's
    top-left lies in the image of that shape (a terrace mosaic around it)."""
    Minv = np.linalg.inv(_M100())
    c = np.array(R["frame_wh"], float) / 2.0 + np.asarray(org, float)
    base = np.asarray(R["stage_um"], float) - np.asarray(R["offset_100x_um"], float)
    m = np.zeros(shape, np.uint8)
    for p in f_poly:
        pts = np.array(p, float).reshape(-1, 2)
        st = np.array([smap.to_stage(x, y) for x, y in pts])
        q = (st - base) @ Minv.T + c
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    r = max(1, int(round(margin_um / float(R["px_um"]))))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def misfit(R, rgb, sub, expect):
    """Why an older 100x analysis (before 2026-10-14) does not show the target terrace, or None: the frame centre is
    bare substrate (an off-centre shot), the terrace is partly substrate (the old hole filling), or its size or colour
    does not fit the 10x terrace."""
    a = rgb.astype(np.float32)
    if _centre_is_substrate(a, np.maximum(np.asarray(sub, np.float32), 1.0))[0]:
        return "centre is substrate"
    tm = np.zeros(a.shape[:2], np.uint8)
    for p in R.get("terrace") or []:
        cv2.fillPoly(tm, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], 1)
    if tm.sum() < 50:
        return "no terrace"
    sm = cv2.GaussianBlur(a, (0, 0), 2.0)
    if float((np.abs(sm / np.maximum(sub, 1.0) - 1.0).max(axis=2) < 0.025)[tm > 0].mean()) > 0.1:
        return "substrate in the terrace"
    ea = float(expect.get("area_um2") or 0)
    if ea and not SEED_AREA_LO <= float(R.get("terrace_um2") or 0) / ea <= SEED_AREA_HI:
        return "terrace size"
    c = expect.get("contrast")
    if c is not None:
        g = float(np.median(a[tm > 0][:, 1]) / max(float(sub[1]), 1.0) - 1.0)
        if abs(g - 0.75 * float(c[1])) / (0.1 + abs(0.75 * float(c[1]))) >= 1.0:
            return "terrace colour"
    return None


MOSAIC_MIN_IOU = 0.6                  # the mosaic's terrace must agree this well with the 100x picture's inside it


def load_mosaic(viewer, e, R, wh):
    """The terrace mosaic around a flake's 100x picture (flakeid.fieldmosaic, written by the export with the main
    picture's look), at the picture's scale: dict(rgb, cov (covered pixels), ox, oy (the picture's top-left in it,
    whole px), mos (the designer's placement, main picture px), file) or None."""
    if not e.get("mos"):
        return None
    mj = viewer / re.sub(r"\.jpg$", ".json", e["mos"])
    if not (mj.is_file() and (viewer / e["mos"]).is_file()):
        return None
    try:
        J = json.loads(mj.read_text())
        Ww, Hh = wh
        r = float(np.sqrt(abs(np.linalg.det(np.asarray(J["um_per_px"], float))))) / float(R["px_um"])
        (mw, mh), (mcx, mcy) = J["size_px"], J["main_centre_px"]
        mos = dict(w=round(mw * r, 1), h=round(mh * r, 1), x=round(mcx * r - Ww / 2, 1), y=round(mcy * r - Hh / 2, 1),
                   n=len(J.get("fields") or []), um=J.get("size_um"))
        a = cv2.imdecode(np.frombuffer(read_bytes(viewer / e["mos"]), np.uint8), cv2.IMREAD_COLOR)
        if a is None:
            return None
        if abs(r - 1.0) > 2e-3:
            a = cv2.resize(a, (int(round(a.shape[1] * r)), int(round(a.shape[0] * r))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        # uncovered parts are black; JPEG blurs their edge, so a few pixels are taken off the covered area
        cov = cv2.erode((rgb.max(axis=2) > 8).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        return dict(rgb=rgb, cov=cov, ox=int(round(mos["x"])), oy=int(round(mos["y"])), mos=mos, file=e["mos"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def paste(mask, shape, ox, oy):
    """mask (100x picture px) placed at (ox, oy) in an empty image of shape (clipped)."""
    out = np.zeros(shape, mask.dtype)
    h, w = mask.shape
    x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + w, shape[1]), min(oy + h, shape[0])
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return out


def mosaic_terrace(M, tm):
    """The picture's terrace tm (100x picture px) followed into the mosaic: the connected area of its colour
    (flakeid.hundred._terrace_at, the export's own terrace rule) around it, in mosaic px; None when the mosaic's terrace
    does not agree with the picture's inside the picture (a mis-stitched or differently lit mosaic: the picture's
    terrace is kept)."""
    rgb, cov, ox, oy = M["rgb"], M["cov"], M["ox"], M["oy"]
    shape = rgb.shape[:2]
    T0 = paste(tm.astype(np.uint8), shape, ox, oy)
    if T0.sum() < 50:
        return None
    lin = rgb.astype(np.float32)
    tc = np.median(lin[T0 > 0], axis=0)
    dist = cv2.distanceTransform(np.pad(T0, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    win = max(5, min(90, int(dist.max() * 0.7)))
    _, _, T = _terrace_at(lin, tc, int(cy), int(cx), win)
    if T is None:
        return None
    T = (T.astype(bool) & cov) | (T0 > 0)
    lab, _ = ndimage.label(T)
    keep = np.unique(lab[T0 > 0])
    T = np.isin(lab, keep[keep > 0])
    inside = paste(np.ones(tm.shape, np.uint8), shape, ox, oy) > 0
    a, b = T & inside, T0 > 0
    if (a & b).sum() / max((a | b).sum(), 1) < MOSAIC_MIN_IOU:
        return None
    return T


def cut_by_cover(mask, cov):
    """The mask reaches the edge of what the mosaic covers (or of the mosaic): the terrace goes on beyond it."""
    out = np.pad(~cov, 1, constant_values=True).astype(np.uint8)
    near = cv2.dilate(out, np.ones((3, 3), np.uint8))[1:-1, 1:-1] > 0
    return bool((mask & near).any())


def chip_to_stage(smap, px10, w_um, h_um):
    """The chip's micrometres (the designer's chip map: 10x mosaic px x px10) -> 10x stage um, as an affine
    [[a, b, tx], [c, d, ty]] fitted to the scan's stage map on a 5 x 5 grid (the chips of a block share the stage's
    frame: their AFM list is made in it, with the block's reference corners)."""
    src, dst = [], []
    for u in np.linspace(0, w_um, 5):
        for v in np.linspace(0, h_um, 5):
            st = smap.to_stage(u / px10, v / px10)
            src.append([u, v, 1.0])
            dst.append([float(st[0]), float(st[1])])
    sol, *_ = np.linalg.lstsq(np.array(src), np.array(dst), rcond=None)
    return [[round(float(sol[0, 0]), 7), round(float(sol[1, 0]), 7), round(float(sol[2, 0]), 3)],
            [round(float(sol[0, 1]), 7), round(float(sol[1, 1]), 7), round(float(sol[2, 1]), 3)]]


def flake_mask(rgb, terrace_mask, tol, sub=None, limit=None):
    """Connected material touching the terrace: pixels whose colour differs from the substrate by more than tol
    (thin residue strands removed by an opening)."""
    if sub is None:
        sub, _ = _mode_rgb(rgb.astype(np.float32), step=4)
    sm = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 2.0)
    fg = (np.abs(sm / np.maximum(sub, 1.0) - 1.0).max(axis=2) > tol)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    if limit is not None:
        fg &= limit
    fg |= terrace_mask
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    lab, _ = ndimage.label(fg)
    keep = np.unique(lab[terrace_mask & (lab > 0)])
    m = np.isin(lab, keep[keep > 0])
    return ndimage.binary_fill_holes(m)


# Materials whose outline is refined on the 100x picture (refine_fm), and why not for the others.  Graphite since
# 2026-10-22: on graphite_20261019_run_5 45 of 72 refined (median area 0.96 of the colour mask's; the pictures: the
# neighbours and thick debris the mask had swallowed taken off (5-f0051 0.51, -f0067 0.61, -f0061 0.67, -f0025 0.76,
# -f0046 0.91), edges on the flake's; none worse); thin 1-2 layer flakes (contrast 0.08-0.20) mostly keep the colour
# mask ("low contrast": outline100's noise floor is at least 0.04, terrace a must reach 3x that).
OUTLINE_MATERIALS = ("hbn", "graphite")
OFF_WHY = {}


def refine_fm(a, src, fm, terrace, cov, px_cut, mat="hbn"):
    """The flake mask fm (made by flake_mask on src) refined on the picture at the cut-out's scale
    (flakeid.outline100.refine_mask): (mask, result) with result None when not asked for (--no-refine-outline); the
    mask unchanged when the colour mask's outline is kept (result's why).  Materials in OUTLINE_MATERIALS only."""
    if a.no_refine_outline:
        return fm, None
    if mat not in OUTLINE_MATERIALS:
        return fm, dict(used=False, why="not refined (%s)" % OFF_WHY.get(mat, "not checked for this material"), ms=0.0)
    terrace = terrace.astype(bool)
    try:
        fm2, r = O100.refine_mask(src, fm, terrace, 1.0 / a.downscale, px_cut, cover=cov)
    except Exception as ex:                             # (one picture must not stop the build)
        return fm, dict(used=False, why=f"not refined ({type(ex).__name__}: {ex})", ms=0.0)
    if r["used"] and cov is not None:
        fm2 &= cov | terrace
    return fm2, r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--downscale", type=float, default=2.0, help="cut-outs at 1/N of the 100x resolution (default 2)")
    ap.add_argument("--material-tol", type=float, help="colour difference from the substrate that counts as flake "
                    "(default 0.10 hBN, 0.06 graphite)")
    ap.add_argument("--hbn-max-nm", type=float, default=100.0)
    ap.add_argument("--no-screen", action="store_true", help="keep flakes the 100x screening rejects (flakeid.screen100)")
    ap.add_argument("--no-mosaic-outline", action="store_true", help="outlines from the 100x picture only, also where "
                    "the scan took a terrace mosaic around it (before 2026-10-21)")
    ap.add_argument("--only", help="a JSON file {piece name: [flake numbers]}: only these flakes (an update of a built "
                    "run, tools/make_designer_bundle.py --update): other pieces, the chip overviews and the other 100x "
                    "pictures are not read")
    ap.add_argument("--overview", action="append", default=[], help="with --only: this piece's chip overview is made too "
                    "(a piece new to the bundle; repeat)")
    ap.add_argument("--no-reanalyse", action="store_true", help="do not redo older 100x analyses whose terrace does not fit "
                    "the 10x one (off-centre shots, substrate in the terrace) from the natural JPEG")
    ap.add_argument("--no-refine-outline", action="store_true", help="the flake outline of the colour mask, not refined on "
                    "the 100x picture (flakeid.outline100; before 2026-10-22)")
    a = ap.parse_args(argv)
    a.only_map = ({k: {int(x) for x in v} for k, v in json.loads(Path(a.only).read_text()).items()} if a.only else None)
    a.left_out = {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flakes, cutouts, scans, thumbs = [], [], {}, []
    screens = {m: S100.Screen(m) for m in ("hbn", "graphite")}
    (out / "100x").mkdir(exist_ok=True)
    stage = Stage(out / "_stage")
    try:
        _pieces(a, out, flakes, cutouts, scans, thumbs, screens, stage)
    finally:
        stage.close()
    D100.score_flakes(flakes, screens["hbn"], log=print)   # (hBN: the decision model's p_keep / tier, all flakes together)
    # graphite: the image network's p_keep / tier when its model file is there (flakeid.netscreen), else the pick model's
    NS.score_flakes(flakes, NS.bundle_picture(out), log=print)
    n_over = want_max(flakes)
    if n_over:
        print(f"  {n_over} flakes thicker than the wanted range at 100x: 100x score 0")
    _write(out, flakes, cutouts, scans, thumbs, a.left_out)


def want_max(flakes):
    """Terrace a of a flake thicker at 100x than the wanted range (t_over, set above): 100x score 0, tier reject (the
    designer: automatically discarded); the screening's own score kept in p_keep_screen.  Returns how many."""
    n = 0
    for f in flakes:
        for t in f.get("terraces") or []:
            if t.get("t_over") is not None:
                t["p_keep_screen"], t["p_keep"], t["tier"] = t.get("p_keep"), 0.0, "reject"
                n += 1
    return n


def graphite_thickness(rows, ox, stem="", log=print):
    """Thickness of every graphite terrace (rows: terrace_id -> *_live_terraces.csv row) from its stored 10x colour:
    1-4 layers from the measured contrast peaks, thicker from the graphite colour model.  Returns (thick_info, skip):
    terrace_id -> dict(t, layers, alt, note), and terrace_id -> why it is left out (bulk graphite, black sliver).  Also
    used by tools/patch_thickness.py to update a built run's numbers without rebuilding its pictures."""
    thick_info, skip = {}, {}
    from flakeid.thickness import ThicknessEstimator, load_profile, refine_oxide
    # thicker than 4 layers: the graphite colour model (profile fitted on graphite_20261019_run_1, 2026-10-19)
    # on the stored terrace colours, whatever model the 10x analysis used (the hBN-calibrated one before put
    # 2-12 nm at 8-11 nm); oxide refined for this model on the chip's own terraces
    gp = load_profile("bx51_xc10_mm_raw16_graphite")
    cols = np.array([[float(r["contrast_r"]), float(r["contrast_g"]), float(r["contrast_b"])] for r in rows.values()])
    ox_g = ox
    if len(cols) >= 30:
        o, gain = refine_oxide(cols, "graphite", ox, gp)
        ox_g = o if gain > 0.05 else ox
    est_g = ThicknessEstimator("graphite", ox_g, gp)
    cmin = est_g.curve.min(axis=0)
    log(f"{stem}: graphite thickness from the graphite colour model, oxide {ox_g:.1f} nm (10x analysis {ox:.1f})")
    cal = GR.load_calibration()
    cal = cal if cal and abs(ox - cal["oxide_nm"]) <= cal.get("oxide_tol_nm", 6.0) else None
    for tid, r in rows.items():
        c = [float(r["contrast_r"]), float(r["contrast_g"]), float(r["contrast_b"])]
        if max(c) < -0.97:
            skip[tid] = "black sliver"
        elif c[1] < cmin[1] - 0.03 and c[2] / c[1] > 0.8:
            skip[tid] = "bulk graphite"
        elif cal and (n := GR.layers_from_contrast(c, cal)):
            thick_info[tid] = dict(t=round(n * GR.LAYER_NM, 2), layers=n, alt="", note="measured contrast peaks")
        else:
            f = est_g.fit(np.array(c))
            thick_info[tid] = dict(t=round(f.thickness_nm, 1), layers=round(f.thickness_nm / GR.LAYER_NM, 1),
                                   alt=";".join(f"{x:g}" for x in f.alternatives_nm),
                                   note="10x colour fit, graphite model" + (" (off the model: wide range)" if f.off_model else ""))
    return thick_info, skip


PROGRESS = None          # (a callable(fraction 0..1, text): set by tools/make_designer_bundle.py for its progress bar)


def _progress(frac, text):
    if PROGRESS is not None:
        try:
            PROGRESS(max(0.0, min(1.0, frac)), text)
        except Exception:
            pass


def _pieces(a, out, flakes, cutouts, scans, thumbs, screens, stage):
    n_res = max(1, len(a.results))
    for i_res, res in enumerate(Path(r).expanduser() for r in a.results):
        _progress(i_res / n_res, f"{Path(res).name}: reading the 10x results")
        D = load_viewer_data(res)
        viewer = viewer_dir(res)
        stem = D["name"].replace("_tiles", "")
        only = a.only_map.get(stem) if a.only_map is not None else None
        if a.only_map is not None and not only:
            continue
        mat, ox, px10 = D["material"], float(D["oxide"]), float(D["pixel_um"])
        H = h100(viewer)
        H_all = H
        if only is not None:
            H = {k: v for k, v in H.items() if int(k) in only}
        rows = {int(r["terrace_id"]): r for r in csv.DictReader(open(results_file(res, "*_live_terraces.csv")))}
        F = {f["id"]: f for f in D["flakes"]}
        T = {int(k): v for k, v in D["terraces"].items()}
        sm = D["stage_map"]
        smap = StageMap(M=np.array(sm["M"]), P=np.array(sm["P"]), q=np.array(sm["q"]), frame_wh=tuple(sm["wh"]),
                        labels=[], z=[])

        # thickness of every terrace from the measured calibrations
        thick_info, skip = {}, {}
        if mat == "graphite":
            thick_info, skip = graphite_thickness(rows, ox, stem)
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
                        tb.append(dict(r, quality=r["quality"] + ";two_branch", _tn=dec[1], _tk=dec[3]))
                dec = B.resolve(tb, bm.cal, log=lambda s: print(f"{stem}: {s}"))
                for r in tb:
                    tid = int(r["terrace_id"])
                    if dec.get(r["terrace_id"]) == "thick":
                        skip[tid] = "thick hBN (~150 nm)"
                    else:
                        thick_info[tid] = dict(t=round(r["_tn"], 1), layers=None, alt=f"{r['_tk']:.0f}",
                                               note="thin of two colour readings (labelled calibration)")

        ov = None
        ov_px_um = px10 * 2 ** OV_LEVELS_DOWN
        if only is None or stem in a.overview:          # (an update keeps the bundle's chip overview)
            pyr = Pyramid(viewer)
            ov = pyr.level_image(pyr.top - OV_LEVELS_DOWN)
            # (shown with the look of the pictures, flakeid.look: the viewer's tiles/image have a grey substrate)
            cv2.imwrite(str(out / f"chip_{stem}.jpg"), cv2.cvtColor(look(ov, (CANVAS_SUBSTRATE,) * 3), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])

        n_ok, skipped, px_cut, n_redo = 0, {}, None, {}
        n_mos_used, n_mos_kept = 0, {}                   # terraces / outlines from the terrace mosaics
        ms_def = []                                      # (time of the hBN defects per flake)
        n_ref, ref_why, ms_ref = 0, {}, []               # outlines refined (flakeid.outline100), why not, time
        nat = [viewer / e["f"] for imgs in H.values() for e in imgs if not e.get("domain")]
        stage.fetch(nat, lambda k, n: _progress((i_res + 0.5 * k / max(n, 1)) / n_res,
                                                f"{stem}: 100x pictures {k} of {n}"))
        if only is not None:                            # (the substrate from more than the few pictures updated)
            more = [viewer / e["f"] for imgs in H_all.values() for e in imgs if not e.get("domain")]
            more = [q for q in more if q not in set(nat)][::max(1, len(more) // 12)][:12]
            stage.fetch(more)
            nat = nat + more
        sub100 = stage.substrate(nat)
        added = set()
        tol = a.material_tol or (0.10 if mat == "hbn" else 0.06)
        for j_fl, (fid_s, imgs) in enumerate(H.items()):
            _progress((i_res + 0.5 + 0.5 * j_fl / max(len(H), 1)) / n_res, f"{stem}: flake {j_fl + 1} of {len(H)}")
            fid = int(fid_s)
            f = F.get(fid)
            if f is None:
                continue
            imgs = [e for e in imgs if not e.get("domain")]  # (stacking-domain pictures are not the flake's 100x view)
            if not imgs:
                continue
            # a focused image first (a blurred one has the least fine detail, so "least noise" alone picks it), then the
            # least noisy
            # (noise at half resolution, from the local copies; a single picture needs no choice)
            e = imgs[0] if len(imgs) == 1 else min(imgs, key=lambda e: (e.get("focus") not in ("auto", "manual", None),
                                                                        noise(stage.path(viewer / e["f"]), cv2.IMREAD_REDUCED_GRAYSCALE_2)))
            rj = viewer / re.sub(r"\.jpg$", "_regions.json", e["f"])
            if not rj.is_file():
                continue
            R = json.loads(rj.read_text())
            tid = int(f["terr"][0])                         # the terrace the 100x target was centred on
            expect = dict(area_um2=T[tid].get("area"), contrast=T[tid].get("c"))
            rgb = cv2.cvtColor(read_img(stage.path(viewer / e["f"])), cv2.COLOR_BGR2RGB)
            new_c8 = new_dt = None
            why = None if (R.get("seed") is not None or a.no_reanalyse) else misfit(R, rgb, sub100, expect)
            if why:
                # analysed before the off-centre / hole-fill fix (2026-10-14) and the terrace does not fit the 10x one:
                # again from the natural JPEG (linear, white-balanced: the analysis only uses colour ratios)
                feat, reg, extra = S100.features(rgb.astype(np.float32), mat, ox, T[tid]["t"], substrate=sub100, expect=expect)
                R.update({kk: reg[kk] for kk in ("terrace", "debris", "steps", "terrace_um2", "debris_n", "debris_um2",
                                                 "steps_um2", "clean_um2", "clean_d_um", "cut_by_frame", "seed")},
                         fine=extra["fine"], wrinkles=extra["wrinkles"], screen=screen_record(screens[mat], feat, mat),
                         reanalysed=why)
                new_c8 = extra["c8"]
                new_dt = dt_view(extra["dt"], reg["terrace"], rgb) if "dt" in extra else None
                n_redo[why] = n_redo.get(why, 0) + 1
            scr = dict(R.get("screen") or {})
            if scr.get("features") and screens[mat].ok:  # scored with the current profile (models are refitted)
                ft = dict(scr["features"], t_terrace_nm=(thick_info.get(tid) or T[tid])["t"])   # (thickness rules)
                scr.update(screen_record(screens[mat], ft, mat), features=scr["features"])
            seed = R.get("seed") or {}
            if seed.get("found") is False:
                skipped["100x shot missed the flake"] = skipped.get("100x shot missed the flake", 0) + 1
                continue
            if seed.get("fits") is False:
                # a flake is there but its terrace does not fit the 10x one: the screen may have judged the wrong thing
                scr.update(tier="review", keep=True)
            if not a.no_screen and (scr.get("keep") is False or (scr.get("keep") is None and scr.get("tier") == "reject")):
                # the screening model (fitted on the user's 100x labels) judges this flake not worth keeping (tier
                # "reject" with keep true: the graphite pick model's screen-out, kept in the designer as screened out)
                skipped["screened out (debris, terraces, wrinkles, size)"] = skipped.get("screened out (debris, terraces, wrinkles, size)", 0) + 1
                continue
            b100 = R.get("branch100") or {}
            if mat == "hbn" and b100.get("decided"):
                # (the 100x colour chose one of the 10x readings, flakeid.hundred.branch100: over this chip's own guess)
                if str(skip.get(tid, "")).startswith("thick hBN"):
                    skip.pop(tid)
                thick_info[tid] = dict(t=round(float(b100["chosen"]), 1), layers=None,
                                       alt=";".join(f"{v:g}" for v in b100["readings"] if v != b100["chosen"]),
                                       note="of the 10x colour readings, the one the 100x colour fits")
            t100 = t100x(R, mat)
            if t100 and str(skip.get(tid, "")).startswith("thick hBN"):
                skip.pop(tid)                       # (2026-11-12: the 100x thickness is trusted; the window below)
            if tid in skip:
                skipped[skip[tid]] = skipped.get(skip[tid], 0) + 1
                continue
            t10 = T[tid]
            ti = thick_info.get(tid, dict(t=t10["t"], layers=t10["layers"] if mat == "graphite" else None,
                                          alt=t10["alt"], note="10x colour fit"))
            # (2026-11-12, Avram: "trust the 100x thickness ... a flake that's over 100nm should just get like 0") the
            # window on the 100x thickness when there is one: thicker -> kept, its 100x score 0 (want_max below);
            # without a 100x reading, the 10x one decides as before (left out)
            if mat == "hbn" and not t100 and ti["t"] > a.hbn_max_nm:
                skipped["thicker than window"] = skipped.get("thicker than window", 0) + 1
                continue
            Hh, Ww = rgb.shape[:2]
            tm = np.zeros((Hh, Ww), np.uint8)
            for p in R["terrace"]:
                cv2.fillPoly(tm, [np.round(np.array(p).reshape(-1, 2)).astype(np.int32)], 1)
            if tm.sum() < 50:
                skipped["no 100x terrace"] = skipped.get("no 100x terrace", 0) + 1
                continue
            if float(rgb.mean()) < 20:
                skipped["black 100x image"] = skipped.get("black 100x image", 0) + 1
                continue
            tdev = float(np.abs(np.median(rgb[tm.astype(bool)].astype(np.float32), 0) / np.maximum(sub100, 1) - 1).max())
            if tdev < tol:
                skipped["no flake visible at 100x"] = skipped.get("no flake visible at 100x", 0) + 1
                continue
            fpoly = [q for q in f["poly"] if len(q) >= 6] or [q for i in f["terr"] for q in T[int(i)]["poly"] if len(q) >= 6]
            # a terrace mosaic around the picture (a terrace larger than one field): the terrace, the flake's outline
            # and the cut-out from the mosaic, so they are not cut off at the picture's edges.  Everything is worked
            # out in the image used (src: the picture, or the mosaic with the picture's top-left at mx, my) and stored
            # in the picture's pixels as before (the outlines then reach beyond its edges: the designer shows them on
            # the mosaic, "Whole terrace")
            Mz = None if a.no_mosaic_outline else load_mosaic(viewer, e, R, (Ww, Hh))
            Tz = mosaic_terrace(Mz, tm) if Mz is not None else None
            if Mz is not None and Tz is None:
                n_mos_kept["did not agree"] = n_mos_kept.get("did not agree", 0) + 1
            if Tz is not None:
                src, mx, my, cov, tsrc = Mz["rgb"], Mz["ox"], Mz["oy"], Mz["cov"], Tz.astype(np.uint8)
            else:
                src, mx, my, cov, tsrc = rgb, 0, 0, None, tm
            Hs, Ws = src.shape[:2]
            # (the 10x outline is placed by the stage position: of no use when the flake is off-centre)
            lim = (region_10x(fpoly, smap, R, (Hs, Ws), org=(mx, my)) if fpoly and not (R.get("seed") or {}).get("moved")
                   else None)
            if cov is not None:                         # (uncovered mosaic is black: far from the substrate colour)
                lim = cov if lim is None else (lim & cov)
            fm = flake_mask(src, tsrc.astype(bool), tol, sub100, lim)
            if cov is not None:
                fm &= cov | tsrc.astype(bool)
            # the outline refined on the picture (flakeid.outline100): the cut-out's box, its transparency and the
            # flake's area follow it
            px_c = float(R["px_um"]) * a.downscale
            fm, oref = refine_fm(a, src, fm, tsrc, cov, px_c, mat)
            ys, xs = np.nonzero(fm)
            pad = 8
            x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
            x1, y1 = min(int(xs.max()) + pad + 1, Ws), min(int(ys.max()) + pad + 1, Hs)
            k = 1.0 / a.downscale
            if Tz is not None and max(x1 - x0, y1 - y0) * k > ATLAS - 8:
                # (a cut-out too big for an atlas sheet: the picture's terrace only)
                n_mos_kept["too big"] = n_mos_kept.get("too big", 0) + 1
                src, mx, my, cov, tsrc, Tz = rgb, 0, 0, None, tm, None
                Hs, Ws = Hh, Ww
                lim = region_10x(fpoly, smap, R, tm.shape) if fpoly and not (R.get("seed") or {}).get("moved") else None
                fm = flake_mask(rgb, tm.astype(bool), tol, sub100, lim)
                fm, oref = refine_fm(a, rgb, fm, tm, None, px_c, mat)
                ys, xs = np.nonzero(fm)
                x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
                x1, y1 = min(int(xs.max()) + pad + 1, Ww), min(int(ys.max()) + pad + 1, Hh)
            if Tz is not None:
                n_mos_used += 1
            if oref is not None:
                ms_ref.append(oref["ms"])
                if oref["used"]:
                    n_ref += 1
                else:
                    w_ = oref["why"].split(" (")[0]
                    ref_why[w_] = ref_why.get(w_, 0) + 1
            crop = look(src[y0:y1, x0:x1], sub100)            # (the cut-out as people see it: flakeid.look)
            alpha = (cv2.dilate(fm.astype(np.uint8), np.ones((5, 5), np.uint8))[y0:y1, x0:x1] * 255)
            cw, ch = int(round((x1 - x0) * k)), int(round((y1 - y0) * k))
            crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_AREA)
            cutouts.append(np.dstack([crop, alpha]))
            if oref is not None and oref["used"]:           # (src px, canvas convention -> cut-out px)
                kx, ky = cw / float(x1 - x0), ch / float(y1 - y0)
                outline = [round((v - (x0 if i % 2 == 0 else y0)) * (kx if i % 2 == 0 else ky), 1)
                           for i, v in enumerate(oref["outline_src"])]
            else:
                outline = poly_of(cv2.resize(fm[y0:y1, x0:x1].astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST))
            ref_info = {} if oref is None else dict(outline_refined=bool(oref["used"]),
                                                    **({} if oref["used"] else dict(outline_why=oref["why"])))
            px = float(R["px_um"]) * a.downscale                # um per cut-out pixel
            X0, Y0 = x0 - mx, y0 - my                         # the cut-out's top-left in the picture's px (< 0: mosaic)
            # place on the chip overview: cut-out pixel -> stage (10x frame) -> 10x mosaic px -> overview px
            st = to_stage([[X0, Y0, x1 - mx, y1 - my]], R["stage_um"], R["frame_wh"], offset=R["offset_100x_um"])[0]
            p0 = smap.to_pixel(st[0], st[1])
            ovs = px / ov_px_um                                 # cut-out px -> overview px
            tpolys = scale_polys(polygons(tsrc, 0, 0, 1.0) if Tz is not None else R["terrace"], k, x0, y0)
            dist = cv2.distanceTransform(np.pad(tsrc.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            ty, tx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            stats = {kk: R.get(kk) for kk in ("clean_um2", "clean_d_um", "debris_n", "debris_um2", "steps_um2", "cut_by_frame")}
            t_area = R["terrace_um2"]
            if Tz is not None:                              # (debris, steps, clean area: the picture's own analysis)
                t_area = round(float(Tz.sum()) * float(R["px_um"]) ** 2, 1)
                stats.update(cut_by_frame=cut_by_cover(Tz, cov), terrace_from="mosaic",
                             terrace_um2_picture=R["terrace_um2"])
            elif Mz is not None:                            # (a mosaic, but the picture's terrace kept: an update
                stats.update(terrace_from="picture")        # of the bundle does not try it again)
            fidc = f"{stem}-f{fid:04d}"
            # the whole 100x frame: thumbnail for the picker, full natural and colour +-8 % images for the viewer
            th = look(cv2.resize(rgb, (THUMB_W, int(round(THUMB_W * Hh / Ww))), interpolation=cv2.INTER_AREA), sub100)
            thumbs.append(th)
            name = Path(e["f"]).name
            c8 = e.get("c8") and viewer / e["c8"]
            stage.copy(viewer / e["f"], out / "100x" / f"{stem}_{name}")
            img_look = None                                 # (the copy people look at, flakeid.look; img stays linear
            lkf = (e.get("look") or {}).get("f")            # for the designer's measurements)
            if lkf and (viewer / lkf).is_file():
                img_look = f"100x/{stem}_{Path(lkf).name}"
                stage.copy(viewer / lkf, out / img_look)
            img_c8 = None
            if new_c8 is not None:                          # (re-analysed: its own views)
                img_c8 = f"100x/{stem}_{Path(name).stem}_c8.jpg"
                cv2.imwrite(str(out / img_c8), cv2.cvtColor(new_c8, cv2.COLOR_RGB2BGR), JPEG_VIEW)
            elif c8 and c8.is_file():
                img_c8 = f"100x/{stem}_{c8.name}"
                stage.copy(c8, out / img_c8)
            dtp = viewer / re.sub(r"\.jpg$", "_dt.jpg", e["f"])
            img_dt = None
            if new_dt is not None:
                img_dt = f"100x/{stem}_{Path(name).stem}_dt.jpg"
                cv2.imwrite(str(out / img_dt), new_dt, JPEG_VIEW)
            elif new_c8 is None and dtp.is_file():
                img_dt = f"100x/{stem}_{dtp.name}"
                stage.copy(dtp, out / img_dt)
            img_ir = None                                   # the IR (stacking-domain) picture laid over this frame
            if e.get("ir") and (viewer / e["ir"]).is_file():
                img_ir = f"100x/{stem}_{Path(e['ir']).name}"
                stage.copy(viewer / e["ir"], out / img_ir)
            img_mos, mos, img_mos_look = None, None, None   # the whole terrace (the scan's mosaic, flakeid.fieldmosaic)
            mj = viewer / re.sub(r"\.jpg$", ".json", e["mos"]) if e.get("mos") else None
            if mj is not None and mj.is_file() and (viewer / e["mos"]).is_file():
                try:                                        # placed in the main picture's pixels: same scale, main at 0,0
                    J = json.loads(mj.read_text())
                    r = float(np.sqrt(abs(np.linalg.det(np.asarray(J["um_per_px"], float))))) / float(R["px_um"])
                    (mw, mh), (mcx, mcy) = J["size_px"], J["main_centre_px"]
                    mos = dict(w=round(mw * r, 1), h=round(mh * r, 1), x=round(mcx * r - Ww / 2, 1), y=round(mcy * r - Hh / 2, 1),
                               n=len(J.get("fields") or []), um=J.get("size_um"))
                    img_mos = f"100x/{stem}_{Path(e['mos']).name}"
                    stage.copy(viewer / e["mos"], out / img_mos)
                    lkm = (e.get("look") or {}).get("mos")
                    if lkm and (viewer / lkm).is_file():
                        img_mos_look = f"100x/{stem}_{Path(lkm).name}"
                        stage.copy(viewer / lkm, out / img_mos_look)
                except (OSError, ValueError, KeyError, TypeError):
                    img_mos, mos, img_mos_look = None, None, None
            terr = dict(id=f"{fidc}.a", t=ti["t"], tlo=ti["t"], thi=ti["t"], alt=ti["alt"], layers=ti["layers"],
                        area=t_area, d=round(2 * float(dist.max()) * float(R["px_um"]), 1),
                        L=t10["L"], W=t10["W"], c=[int(round((tx - x0) * k)), int(round((ty - y0) * k))],
                        poly=max(tpolys, key=len), quality=t10["q"], score=t10["score"], usable=True,
                        terrace_id=tid, thickness_source=ti["note"], t100x=t100, glue=R.get("glue"),
                        **(dict(t_over=a.hbn_max_nm) if mat == "hbn" and t100 and t100["t"] > a.hbn_max_nm else {}),
                        debris=scale_polys(R["debris"], k, X0, Y0), steps=scale_polys(R.get("steps", []), k, X0, Y0),
                        fine=[dict(dt_nm=q["dt_nm"], area_um2=q["area_um2"], poly=scale_polys(q["poly"], k, X0, Y0))
                              for q in R.get("fine", [])],
                        p_keep=scr.get("p_keep"), p_quality=scr.get("p_quality"), tier=scr.get("tier") or "keep",
                        seed={kk: seed.get(kk) for kk in ("moved", "fits", "offset_um")} if seed else None,
                        reanalysed=R.get("reanalysed"),
                        wrinkles=scale_polys(R.get("wrinkles", []), k, X0, Y0) if mat == "graphite" else [],
                        wrinkle={kk: (scr.get("features") or {}).get(kk) for kk in
                                 ("wrinkle_n", "wrinkle_free_ratio", "wrinkle_coherence")}, **stats)
            flakes.append(dict(id=fidc, scan=stem, material=mat, oxide_nm=round(ox, 1), rank=fid, score=f["score"],
                               px_um=round(px, 5), w=cw, h=ch, origin_px=[round(p0[0] / (2 ** OV_LEVELS_DOWN) / ovs, 1),
                                                                         round(p0[1] / (2 ** OV_LEVELS_DOWN) / ovs, 1)],
                               origin_um=[round(p0[0] * px10, 1), round(p0[1] * px10, 1)], stage_origin_um=None,
                               stage_um=R["stage_um"], area=round(float(fm.sum()) * float(R["px_um"]) ** 2, 1),
                               d=terr["d"], L=t10["L"], W=t10["W"], cropped=bool(stats.get("cut_by_frame")),
                               source=e["f"], image100=e["f"], img=f"100x/{stem}_{name}", img_c8=img_c8, img_dt=img_dt, img_ir=img_ir,
                               **(dict(img_look=img_look) if img_look else {}),
                               **(dict(img_mos=img_mos, mos=mos) if img_mos else {}),
                               **(dict(img_mos_look=img_mos_look) if img_mos and img_mos_look else {}),
                               # (IR on another of its pictures only: a bundle update does not look for it again)
                               **(dict(ir_none=True) if img_ir is None and any(q.get("ir") for q in imgs) else {}),
                               frame=dict(w=Ww, h=Hh, x0=X0, y0=Y0, k=k, thumb=len(thumbs) - 1),
                               outline=outline, **ref_info, terraces=[terr]))
            if mat == "hbn":                                # terrace a's defects (the decision screening, score_flakes)
                try:
                    dfx = D100.flake_defects(rgb, flakes[-1])
                except Exception as ex:                     # (one picture must not stop the build)
                    dfx = None
                    print(f"  {fidc}: defects not measured ({type(ex).__name__}: {ex})")
                if dfx is not None:
                    ms_def.append(dfx.pop("ms"))
                    terr["defects"] = dfx
            px_cut = px
            n_ok += 1
            added.add(fid)
        stage.wait()
        a.left_out[stem] = sorted(int(k) for k in H if int(k) not in added)
        if px_cut is not None and ov is not None:
            scans[stem] = dict(file=f"chip_{stem}.jpg", w=ov.shape[1], h=ov.shape[0], scale=round(px_cut / ov_px_um, 6),
                               px_um=px_cut, material=mat, oxide_nm=round(ox, 1), stage_origin_um=None, stage_sign=[1, 1],
                               chip_to_stage=chip_to_stage(smap, px10, ov.shape[1] * ov_px_um, ov.shape[0] * ov_px_um))
        print(f"{stem} ({mat}): {n_ok} flakes with 100x images in the catalog"
              + (f"; left out: {', '.join(f'{v} {k}' for k, v in skipped.items())}" if skipped else "")
              + (f"; re-analysed from the JPEG: {', '.join(f'{v} {k}' for k, v in n_redo.items())}" if n_redo else "")
              + (f"; outlines from the terrace mosaic: {n_mos_used}" if n_mos_used else "")
              + (f"; outlines refined on the 100x picture: {n_ref} of {len(ms_ref)}, {np.median(ms_ref):.0f} ms each"
                 + (f" (colour mask kept: {', '.join(f'{v} {k}' for k, v in sorted(ref_why.items()))})" if ref_why else "")
                 if ms_ref else "")
              + (f"; terrace defects: {len(ms_def)} flakes, {np.median(ms_def):.0f} ms each" if ms_def else "")
              + (f" (picture only: {', '.join(f'{v} {k}' for k, v in n_mos_kept.items())})" if n_mos_kept else ""))



def _write(out, flakes, cutouts, scans, thumbs, left_out=None):
    pos = pack_big([(c.shape[1], c.shape[0]) for c in cutouts])
    n_sheets = max(p[0] for p in pos) + 1 if pos else 0
    sheets = [np.zeros((ATLAS, ATLAS, 4), np.uint8) for _ in range(n_sheets)]
    used_h = [0] * n_sheets
    for f, c, (s, x, y) in zip(flakes, cutouts, pos):
        sheets[s][y:y + c.shape[0], x:x + c.shape[1]] = c
        used_h[s] = max(used_h[s], y + c.shape[0])
        f["atlas"], f["rect"] = s, [x, y, c.shape[1], c.shape[0]]
    for s, sh in enumerate(sheets):
        cv2.imwrite(str(out / f"flakes_{s:02d}.webp"), cv2.cvtColor(sh[:used_h[s]], cv2.COLOR_RGBA2BGRA),
                    [cv2.IMWRITE_WEBP_QUALITY, 88])
    # thumbnail sheets: fixed grid, THUMB_W x th cells
    th_h = max([t.shape[0] for t in thumbs] or [1])
    cols, rows = ATLAS // THUMB_W, ATLAS // th_h
    per = cols * rows
    thumb_files = []
    for k0 in range(0, len(thumbs), per):
        chunk = thumbs[k0:k0 + per]
        nr = (len(chunk) + cols - 1) // cols
        sheet = np.zeros((nr * th_h, cols * THUMB_W, 3), np.uint8)
        for j, t in enumerate(chunk):
            r, c = divmod(j, cols)
            sheet[r * th_h:r * th_h + t.shape[0], c * THUMB_W:(c + 1) * THUMB_W] = t
        fn = f"thumbs_{k0 // per:02d}.jpg"
        cv2.imwrite(str(out / fn), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 84])
        thumb_files.append(dict(file=fn, w=sheet.shape[1], h=sheet.shape[0]))
    for f in flakes:
        j = f["frame"].pop("thumb")
        sh, jj = divmod(j, per)
        r, c = divmod(jj, cols)
        f["frame"]["thumb"] = [sh, c * THUMB_W, r * th_h, THUMB_W, int(round(THUMB_W * f["frame"]["h"] / f["frame"]["w"]))]
    ng = sum(1 for f in flakes if f["material"] == "graphite")
    subtitle = (f"{len(flakes)} flakes imaged at 100x · {len(flakes) - ng} hBN, {ng} graphite · "
                "thicknesses from measured 10x calibrations · positions and sizes in µm")
    ctx = ("You help design van der Waals heterostructures from a catalogue of exfoliated flakes that were found by an "
           "automatic 10x scan (BX51, XC10 camera) and then imaged at 100x; every flake in the catalogue has a 100x image. "
           "Materials: graphite/graphene on ~82-90 nm SiO2, hBN on ~305 nm SiO2. Thicknesses come from measured calibrations "
           "of the 10x colour: graphene 1-4 layers from measured contrast peaks (reliable to the layer; 0.34 nm per layer), "
           "thicker graphite from the thin-film model (roughly +-20-30 %); hBN from the thin-film model (+-10-20 %), except "
           "where the colour fits both ~20 and ~150 nm, decided from the user's labels (thin kept; 'alt' gives the other "
           "reading). Thick hBN and bulk graphite are already excluded. Large graphite gates are hard to use (they short to "
           "other layers), so prefer graphite that is not much larger than the device; for hBN bigger is better.")
    cat = dict(version=2, demo=True, subtitle=subtitle, assistant_context=ctx,
               note="flakes imaged at 100x (multitest run); terrace outlines, debris and steps from the 100x images; "
                    "thicknesses from the measured 10x calibrations (graphene layer peaks, labelled hBN thin/thick).",
               atlas_size=ATLAS, atlases=[f"flakes_{s:02d}.webp" for s in range(n_sheets)], thumbs=thumb_files,
               scans=scans, flakes=flakes, left_out=left_out or {})
    (out / "catalog.json").write_text(json.dumps(cat, separators=(",", ":")))
    if (ROOT / "tools" / "designer_page.html").is_file():   # (the old standalone page; not in the packaged designer app)
        shutil.copy(ROOT / "tools" / "designer_page.html", out / "index.html")
    print(f"{len(flakes)} flakes, {n_sheets} atlases -> {out}")


def pack_big(sizes):
    """Shelf packing on ATLAS-wide sheets (make_catalog.pack is fixed to 2048)."""
    pos, sheet, x, y, shelf = [], 0, 0, 0, 0
    for w, h in sizes:
        if x + w > ATLAS:
            x, y, shelf = 0, y + shelf + 2, 0
        if y + h > ATLAS:
            sheet, x, y, shelf = sheet + 1, 0, 0, 0
        pos.append((sheet, x, y))
        x += w + 2
        shelf = max(shelf, h)
    return pos


if __name__ == "__main__":
    main()
