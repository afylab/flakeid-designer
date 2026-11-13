#!/usr/bin/env python3
"""Prototype: hBN terraces in 100x images from the log brightness.

    python tools/terraces100.py <100x folder or images...> [--bundle <run>/designer] [-o out] [--sheet sheet.jpg]

A thickness step changes the colour by a few %; in the log of the brightness (each channel) that is the same step in a
dark region as in a bright one, so a fixed edge threshold finds steps everywhere on the flake.  The flake (non-substrate
region at the image centre) is split along those edges, neighbouring pieces of the same colour (within MERGE) are
merged, the rest of the flake is given to the nearest piece (watershed), and pieces of at least MIN_UM2 are terraces.
Writes one picture per image (terraces outlined and numbered, their colour against the main terrace) and a CSV.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_stack_bundle import flake_region  # noqa: E402

UM_PX = 0.1143 * 2          # working at half resolution
MERGE = 0.015               # neighbours closer than this in every log channel (~1.5 %) are one terrace
MIN_UM2 = 5.0
MIN_D_UM = 1.5              # narrower: the blurred rim along a flake edge, not a terrace
K, LO, HI = 400.0, 18, 40   # log scaling and Canny thresholds (as tools/fit_stack_design.log_edges)


def segment(img, outline=None):
    """(labels, regions) for one 100x picture (BGR, natural colour).  labels: 0 off the flake, 1.. terraces (largest 1).
    outline: the flake's outline (half-resolution px of this picture), else the non-substrate region at the centre."""
    h = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if outline is not None:
        F = np.zeros(h.shape[:2], np.uint8)
        cv2.fillPoly(F, [np.round(outline).astype(np.int32)], 1)
    else:
        F = flake_region(h, overlay_box=(0, 0, 0, 0))
    if F is None:
        return h, None, []
    lg = np.log(cv2.GaussianBlur(h.astype(np.float32) + 1.0, (0, 0), 1.2))
    e = np.zeros(F.shape, np.uint8)
    for ch in cv2.split(lg):
        u8 = np.clip((ch - np.median(ch[F > 0])) * K + 128, 0, 255).astype(np.uint8)
        e |= cv2.Canny(u8, LO, HI)
    inner = cv2.erode(F, np.ones((5, 5), np.uint8))
    e = cv2.dilate(e, np.ones((2, 2), np.uint8)) & inner
    free = (inner > 0) & (e == 0)
    n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    # mean log colour per piece; merge neighbours of the same colour (union-find over touching pieces)
    area = np.bincount(lab.ravel(), minlength=n).astype(float)
    mean = np.stack([np.bincount(lab.ravel(), weights=lg[..., c].ravel(), minlength=n) for c in range(3)], 1) / np.maximum(area, 1)[:, None]
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    big = [i for i in range(1, n) if area[i] * UM_PX ** 2 >= 1.0]
    dl = cv2.dilate(lab.astype(np.float32), np.ones((5, 5), np.uint8)).astype(np.int32)       # a neighbour across the edge line
    pairs = set()
    for a_, b_ in zip(lab[free].ravel(), dl[free].ravel()):
        if a_ != b_ and a_ > 0 and b_ > 0:
            pairs.add((min(a_, b_), max(a_, b_)))
    # also across the 2-3 px edge lines: compare pieces whose dilations overlap
    dl2 = cv2.dilate(lab.astype(np.float32), np.ones((7, 7), np.uint8)).astype(np.int32)
    el = (e > 0) & (inner > 0)
    er = cv2.erode(np.where(lab > 0, lab, 10 ** 6).astype(np.float32), np.ones((7, 7), np.uint8)).astype(np.int32)
    for a_, b_ in zip(dl2[el].ravel(), er[el].ravel()):
        if 0 < a_ < 10 ** 6 and 0 < b_ < 10 ** 6 and a_ != b_:
            pairs.add((min(a_, b_), max(a_, b_)))
    bigs = set(big)
    for a_, b_ in sorted(pairs, key=lambda p: np.abs(mean[p[0]] - mean[p[1]]).max()):
        if a_ in bigs and b_ in bigs and np.abs(mean[a_] - mean[b_]).max() < MERGE:
            ra, rb = find(a_), find(b_)
            if ra != rb:
                par[rb] = ra
    root = np.array([find(i) for i in range(n)])
    keep = [r for r in sorted(set(root[big])) if area[root == r].sum() * UM_PX ** 2 >= MIN_UM2]
    markers = np.zeros(lab.shape, np.int32)
    for k, r in enumerate(sorted(keep, key=lambda r: -area[root == r].sum()), 1):
        markers[np.isin(lab, np.flatnonzero(root == r)) & free] = k
    if not keep:
        return h, None, []
    ws = cv2.watershed(cv2.GaussianBlur(h, (0, 0), 1.0), np.where(F > 0, markers, 0).astype(np.int32) + 0)
    out = np.where((F > 0) & (ws > 0), ws, 0)
    regs = []
    m0 = lg[out == 1].mean(0)
    for k in range(1, int(out.max()) + 1):
        r = out == k
        if not r.any():
            continue
        d = lg[r].mean(0) - m0
        dist = cv2.distanceTransform(np.pad(r.astype(np.uint8), 1), cv2.DIST_L2, 5)
        if k > 1 and 2 * float(dist.max()) * UM_PX < MIN_D_UM:
            continue
        regs.append(dict(k=k, area_um2=round(float(r.sum()) * UM_PX ** 2, 1), d_um=round(2 * float(dist.max()) * UM_PX, 1),
                         dB=round(float(np.expm1(d[0])) * 100, 1), dG=round(float(np.expm1(d[1])) * 100, 1), dR=round(float(np.expm1(d[2])) * 100, 1)))
    return h, out, regs


_MODELS = {}

# Graphite: its own optics for the 100x maps.  (2026-11-12) The one 100x model of flakeid.thick100.model100 (the graphite
# camera and optical constants with the 100x stray light and NA fitted to GR11' - GR15' colours), as every 100x thickness;
# until then GRAPHITE_PROFILE's 10x model x GRAPHITE_C100 (0.85), which fit those colours 4 x worse (median 0.034 against
# 0.008) and put 4 - 6 nm graphite at 3 - 10 nm.  hBN keeps bx51_xc10_mm_raw16 (and screen100.sensitivity) as before.
LAYER_NM = {"graphite": 0.335, "hbn": 0.333}


def _root_on_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def contrast_model(material, oxide_nm):
    """The thin-film model at 100x for a material: flakeid.thick100.model100 (graphite: its own 100x optics; hBN and
    others: bx51_xc10_mm_raw16 at NA 0.9)."""
    _root_on_path()
    from flakeid.thick100 import model100
    return model100(material, oxide_nm)


def contrast100(material, oxide_nm, ts):
    """The model's contrast (n, 3) R G B at 100x for thicknesses ts."""
    return np.asarray(contrast_model(material, oxide_nm).contrast([float(t) for t in ts]), float)


def model_ratio(material, oxide_nm, t_ref, grid):
    """Thin-film model (100x NA, the scan camera): (1 + contrast(t)) / (1 + contrast(t_ref)) for t in grid, (n, 3) R G B."""
    return (1.0 + contrast100(material, oxide_nm, grid)) / (1.0 + contrast100(material, oxide_nm, [t_ref])[0])


_G100 = {}


def sensitivity100(material, oxide_nm, t_nm):
    """d ln I / d t per nm (R, G, B) at thickness t: hBN screen100.sensitivity (unchanged); graphite from its own model
    (contrast_model) over +-0.1 nm (a layer is 0.335 nm; screen100's +-0.5 nm spans three)."""
    if material != "graphite":
        _root_on_path()
        from flakeid.screen100 import sensitivity
        return sensitivity(material, oxide_nm, t_nm)
    key = (material, round(float(oxide_nm)), round(float(t_nm), 2))
    if key not in _G100:
        t = max(float(t_nm), 0.15)
        c = contrast100(material, oxide_nm, [t - 0.1, t, t + 0.1])
        _G100[key] = (c[2] - c[0]) / 0.2 / (1.0 + c[1])
    return _G100[key]


_LOG_RATIO = {}


def thickness_from_ratio(ratio_rgb, material, oxide_nm, t_ref, t_max=400.0):
    """Thickness whose model colour ratio to t_ref matches the measured one: (t, rms log error, runner-up t or None).
    Graphite: on a finer grid (0.05 nm from 0.1 nm: a layer is 0.335 nm), up to 100 nm, runner-up 3 nm away."""
    if material == "graphite":
        grid, far_nm = np.arange(0.1, min(t_max, 100.0), 0.05), 3.0
    else:
        grid, far_nm = np.arange(1.0, t_max, 0.5), 15.0
    # (the model over the whole grid took 0.1-0.2 s, once for every terrace far from terrace a: remembered per flake)
    key = (material, round(float(oxide_nm), 1), round(float(t_ref), 3), float(t_max))
    if key not in _LOG_RATIO:
        if len(_LOG_RATIO) > 64:
            _LOG_RATIO.clear()
        _LOG_RATIO[key] = np.log(model_ratio(material, oxide_nm, t_ref, grid))
    err = np.sqrt(((_LOG_RATIO[key] - np.log(np.maximum(ratio_rgb, 1e-3))) ** 2).mean(1))
    i = int(np.argmin(err))
    far = np.abs(grid - grid[i]) > far_nm
    j = int(np.argmin(np.where(far, err, 9.0)))
    alt = float(grid[j]) if err[j] < err[i] + 0.01 else None
    return float(grid[i]), float(err[i]), alt


def extra_terraces(img_path, flake, main, min_um2=20.0, min_d_um=4.0, max_err=0.08, sub_tol=0.10):
    """The flake's 100x terraces besides `main` (its terrace "a"): regions of the log-brightness segmentation that lie
    mostly outside it, each with a thickness from its colour against `main` (thin-film model).  Coordinates as the
    designer's (cut-out px, flake["px_um"]).  Returns a list of terrace dicts (no ids)."""
    img = cv2.imread(str(img_path))
    fr = flake.get("frame")
    if img is None or not fr:
        return []
    k = float(fr.get("k", 0.5))
    off = np.array([fr["x0"] * k, fr["y0"] * k])
    outline = np.array(flake["outline"], float).reshape(-1, 2) + off
    h, lab, regs = segment(img, outline)
    if lab is None:
        return []
    A = np.zeros(lab.shape, np.uint8)
    cv2.fillPoly(A, [np.round(np.array(main["poly"], float).reshape(-1, 2) + off).astype(np.int32)], 1)
    if A.sum() < 50:
        return []
    lin = h.astype(np.float64)
    ref = lin[A > 0].mean(0)                                         # B G R
    Fm = np.zeros(lab.shape, np.uint8)
    cv2.fillPoly(Fm, [np.round(outline).astype(np.int32)], 1)
    ring = (cv2.dilate(Fm, np.ones((41, 41), np.uint8)) > 0) & (cv2.dilate(Fm, np.ones((11, 11), np.uint8)) == 0)
    sub = np.median(lin[ring], axis=0) if ring.sum() > 200 else None    # bare substrate around the flake
    out = []
    for r in regs:
        m = lab == r["k"]
        if r["area_um2"] < min_um2 or r["d_um"] < min_d_um or (A[m] > 0).mean() > 0.5:
            continue
        col = lin[m].mean(0)
        if sub is not None and np.abs(col / np.maximum(sub, 1e-6) - 1).max() < sub_tol:
            continue                                                 # bare substrate inside the scan's outline
        ratio = (col / np.maximum(ref, 1e-6))[::-1]                  # R G B
        t, err, alt = thickness_from_ratio(ratio, flake["material"], flake.get("oxide_nm") or 300.0, main["t"])
        if err > max_err:
            continue                                                 # colour no hBN thickness explains: debris, residue, a gap
        mm = m.astype(np.uint8)
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        poly = (cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(float) + 0.5 - off)
        dist = cv2.distanceTransform(np.pad(mm, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c.reshape(-1, 2).astype(np.float32))
        px = float(flake["px_um"])
        out.append(dict(t=round(t, 1), tlo=round(t, 1), thi=round(t, 1), alt="" if alt is None else f"{alt:g}", layers=None,
                        area=r["area_um2"], d=r["d_um"], L=round(max(rw, rh) * px, 1), W=round(min(rw, rh) * px, 1),
                        c=[int(round(cx - off[0])), int(round(cy - off[1]))], poly=[round(v, 1) for v in poly.ravel().tolist()],
                        quality="terrace_100x", score=None, usable=True,
                        thickness_source=f"100x colour against terrace a ({main['t']:g} nm), thin-film model; fit error {100 * err:.1f} %",
                        colour_vs_a=[round(100 * (v - 1), 1) for v in ratio], fine=[], debris=[], steps=[], wrinkles=[]))
    return out


def draw(h, out, regs, title):
    v = h.copy()
    cols = [(0, 230, 255), (255, 120, 40), (80, 255, 80), (255, 80, 255), (60, 140, 255), (255, 255, 90), (200, 160, 255), (120, 255, 200)]
    for r in regs:
        m = (out == r["k"]).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(v, cs, -1, cols[(r["k"] - 1) % len(cols)], 1, cv2.LINE_AA)
        ys, xs = np.nonzero(cv2.erode(m, np.ones((5, 5), np.uint8)) if cv2.erode(m, np.ones((5, 5), np.uint8)).any() else m)
        cv2.putText(v, str(r["k"]), (int(np.median(xs)) - 4, int(np.median(ys)) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    b = np.zeros((22, v.shape[1], 3), np.uint8)
    cv2.putText(b, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([b, v])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--update", nargs=2, metavar=("DESIGNER_DIR", "RUN_DIR"), help="recompute the hBN step terraces of an "
                    "existing designer bundle from the run's 100x pictures")
    ap.add_argument("-o", "--out", default="terraces100")
    ap.add_argument("--sheet", help="also a contact sheet of every picture")
    ap.add_argument("--bundle", action="append", default=[], help="designer run folder(s): each picture's flake outline from it "
                    "(picture <nnn>_<n>_F<id>_... in <scan>/100x)")
    a = ap.parse_args(argv)
    if a.update:
        update_bundle(*a.update)
        return 0
    files = []
    for p in a.inputs:
        P = Path(p)
        files += sorted(q for q in P.glob("*.jpg") if not any(s in q.stem for s in ("_c8", "_sg", "_dt", "_avg", "_debris"))) if P.is_dir() else [P]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    flakes = {}
    for b in a.bundle:
        for fl in json.loads((Path(b) / "bundle.json").read_text())["flakes"]:
            flakes[fl["id"].split(":")[-1].lower()] = fl                     # <scan>-f0010
    rows, tiles = [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        m = re.search(r"_F(\d+)", f.stem)
        fl = flakes.get(f"{f.parent.parent.name}-f{int(m.group(1)):04d}".lower()) if m else None
        if flakes and (fl is None or fl.get("material") != "hbn"):
            continue                                                    # (hBN only: what this prototype is for)
        ol = None
        if fl is not None and fl.get("frame"):
            ol = np.array(fl["outline"], float).reshape(-1, 2) + [fl["frame"]["x0"] / 2.0, fl["frame"]["y0"] / 2.0]
        h, lab, regs = segment(img, ol)
        if lab is None:
            continue
        t = f"{f.stem}: {len(regs)} terrace{'s' if len(regs) != 1 else ''}" + "".join(
            f"  {r['k']}: {r['area_um2']:.0f}um2 {r['dR']:+.0f}/{r['dG']:+.0f}/{r['dB']:+.0f}%" for r in regs[1:4])
        v = draw(h, lab, regs, t)
        cv2.imwrite(str(out / f"{f.stem}_terraces.jpg"), v, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append(v)
        for r in regs:
            rows.append(dict(image=f.name, **r))
        print(t)
    with open(out / "terraces100.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "k", "area_um2", "d_um", "dR", "dG", "dB"])
        w.writeheader()
        w.writerows(rows)
    if a.sheet and tiles:
        cols = 3
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(a.sheet, cv2.resize(sheet, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return 0



# ---------------------------------------------------------------- terraces bounded by straight step lines
# hBN terraces and flake edges are bounded by straight lines along the crystal directions (30 deg apart).  Faint steps
# (a few 0.1 nm, visible only in the high-contrast view) are found as straight lines in the thickness-change map: a
# matched filter (step across the line, averaged over STEP_WIN_UM along it) at every direction, each candidate checked
# to be a sharp step between flat levels that holds along its whole length.  The lines are extended to the next line
# or the flake edge, the flake is split along them, and neighbouring pieces at the same level not separated by a
# detected step are joined.

STEP_WIN_UM = 8.0           # length over which a step is averaged (shortest line)
STEP_MIN_NM = 0.12          # smallest step (equivalent thickness, nm)
STEP_EXTEND_UM = 20.0       # a step line is extended this far to meet another line or the edge
LEVEL_JOIN_NM = 0.15        # neighbouring pieces closer than this (and no detected step between) are one terrace


def thickness_change_map(lin_rgb, F, A, material, oxide_nm, t_a):
    """Equivalent thickness change (nm) from terrace A's colour: the colour change along the thin-film model's
    direction for a thickness change at t_a (screen100's dt map), over the whole flake."""
    _root_on_path()
    from flakeid.hundred import _flatten
    rf = _flatten(cv2.GaussianBlur(lin_rgb / np.median(lin_rgb[A > 0], axis=0), (0, 0), 1.0))
    g = np.asarray(sensitivity100(material, oxide_nm, t_a), float)
    return (((rf - 1.0) @ g) / float(g @ g)).astype(np.float32)


def crystal_directions(img, F, px):
    """Straight edges near the flake outline (line segment detector on the picture): the crystal direction theta0
    (edges at theta0 + n x 30 deg, image frame, deg), the fraction of edge length within 1.5 deg of those, the edges."""
    import math
    h, w = F.shape
    r = max(3, int(round(1.4 / px)))
    rim = (cv2.dilate(F, np.ones((2 * r + 1,) * 2, np.uint8)) > 0) & (cv2.erode(F, np.ones((2 * r + 1,) * 2, np.uint8)) == 0)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    ang, wts = [], []
    for ch in range(3):
        sg = lsd.detect(cv2.GaussianBlur(img[..., ch], (0, 0), 1.0))[0]
        for x1, y1, x2, y2 in (sg.reshape(-1, 4) if sg is not None else []):
            L = math.hypot(x2 - x1, y2 - y1) * px
            t = np.linspace(0, 1, 9)
            if L >= 5 and rim[np.clip((y1 + t * (y2 - y1)).astype(int), 0, h - 1), np.clip((x1 + t * (x2 - x1)).astype(int), 0, w - 1)].mean() > 0.8:
                ang.append(math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180)
                wts.append(L)
    if not ang:
        return None, 0.0, []
    ang, wts = np.array(ang), np.array(wts)
    hm = np.histogram(ang % 30, bins=120, range=(0, 30), weights=wts)[0].astype(float)
    hm = np.convolve(np.r_[hm[-4:], hm, hm[:4]], np.ones(5) / 5, mode="same")[4:-4]
    th = (int(np.argmax(hm)) + 0.5) * 0.25
    dev = np.abs(((ang - th + 15) % 30) - 15)
    return float(th), float(wts[dev <= 1.5].sum() / wts.sum()), list(zip(ang.tolist(), wts.tolist()))


RESIDUE_ROUGH_NM = 0.5      # a terrace this rough inside (robust spread of the thickness map) and smaller than
RESIDUE_MAX_UM2 = 300.0     # this is residue (tape residue, rim debris), not a terrace (user marks, 2026-10-06)
BIG_STEP_NM = 2.0           # areas this far from terrace a's level are outlined from the map itself (large steps)
RIM_UM = 4.0                # a line lying all within this of the flake edge is the edge's blurred rim, not a step


def step_lines(d, inner, px, F=None, step_min=None, win_um=None, rim_um=None):
    """Straight steps in the thickness-change map d (nm) inside `inner`: [(step nm (signed), angle deg, p0, p1, length um)]."""
    c = step_candidates(d, inner, px, F, win_um=win_um, floor=step_min)
    return select_steps(c, d.shape, px, step_min=step_min, rim_um=rim_um)


def step_candidates(d, inner, px, F=None, win_um=None, floor=None):
    """Every verified straight step (all directions, step >= floor nm): [(step, angle, p0, p1, length um, rim um)], rim =
    80th percentile of the line's distance from the flake edge.  The expensive part; select_steps picks from these."""
    import math
    h, w = d.shape
    win_um = STEP_WIN_UM if win_um is None else win_um
    floor = STEP_MIN_NM if floor is None else floor
    win = max(8, int(round(win_um / px)))
    o1, o2 = max(1, int(round(0.11 / px))), max(3, int(round(0.57 / px)))       # rows averaged on each side of the line
    cands = []
    # the search area with the filters' reach around it (the flake is often a small part of the picture): each angle's
    # rotated picture made only over it, in the same pixel grid as the whole picture's (moved by whole pixels)
    ys_, xs_ = np.nonzero(inner)
    if not ys_.size:
        return []
    mg = win + 2 * o2 + 4
    cy0, cy1, cx0, cx1 = max(0, ys_.min() - mg), min(h, ys_.max() + mg + 1), max(0, xs_.min() - mg), min(w, xs_.max() + mg + 1)
    dc, ic = np.ascontiguousarray(d[cy0:cy1, cx0:cx1]), np.ascontiguousarray(inner[cy0:cy1, cx0:cx1])
    box = np.array([[cx0, cy0], [cx1, cy0], [cx0, cy1], [cx1, cy1]], float)
    for a in np.arange(0.0, 180.0, 1.0):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        cs_, sn_ = abs(M[0, 0]), abs(M[0, 1])
        W2, H2 = int(h * sn_ + w * cs_), int(h * cs_ + w * sn_)
        M[0, 2] += W2 / 2 - w / 2
        M[1, 2] += H2 / 2 - h / 2
        q_ = box @ M[:, :2].T + M[:, 2]
        ox, oy = np.floor(q_.min(0)).astype(int)
        Wc, Hc = (np.ceil(q_.max(0)).astype(int) - (ox, oy) + 1).tolist()
        Mc = M.copy()
        Mc[:, 2] += M[:, :2] @ [cx0, cy0] - np.array([ox, oy], float)
        H2 = Hc
        dr = cv2.warpAffine(dc, Mc, (Wc, Hc), flags=cv2.INTER_LINEAR)
        mr = cv2.warpAffine(ic, Mc, (Wc, Hc), flags=cv2.INTER_NEAREST)
        n_ = o2 - o1 + 1
        up = cv2.blur(dr, (1, n_))
        st = np.zeros_like(dr)
        c_ = (o1 + o2) // 2
        st[o2 + 1:-o2 - 1] = up[o2 + 1 + c_:H2 - o2 - 1 + c_] - up[o2 + 1 - c_:H2 - o2 - 1 - c_]
        ok = cv2.erode(mr, np.ones((2 * o2 + 3, 1), np.uint8)) > 0
        st = np.where(ok, st, 0)
        sa = cv2.blur(st, (win, 1))
        va = cv2.blur(ok.astype(np.float32), (win, 1))
        vals = sa[va > 0.95]
        if vals.size < 100:
            continue
        thr = max(floor, 5 * 1.4826 * float(np.median(np.abs(vals - np.median(vals)))))
        hit = (np.abs(sa) > thr) & (va > 0.95)
        hit &= np.abs(sa) >= cv2.dilate(np.abs(sa), np.ones((2 * o2 + 1, 1), np.uint8)) - 1e-6
        Mi = cv2.invertAffineTransform(M)
        for y in np.unique(np.nonzero(hit)[0]):
            xs = np.flatnonzero(hit[y])
            for r in np.split(xs, np.flatnonzero(np.diff(xs) > 3) + 1):
                if len(r) < 2:
                    continue
                x0, x1 = r[0] - win // 2, r[-1] + win // 2
                if (x1 - x0) * px < win_um:
                    continue
                cands.append((float(a), Mi @ [x0 + ox, y + oy, 1], Mi @ [x1 + ox, y + oy, 1], (x1 - x0) * px))
    verify = _step_verifier(d, inner, px, floor)
    dedge = cv2.distanceTransform(np.pad(F, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * px if F is not None else None
    good = []
    for a, p0, p1, L in cands:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        s_ = verify(p0, p1)
        if s_ is None:
            continue
        rim = 1e9                                        # (only for verified lines: most candidates fail)
        if dedge is not None:
            t = np.linspace(0, 1, 25)
            xs = np.clip((p0[0] + t * (p1[0] - p0[0])).astype(int), 0, w - 1)
            ys = np.clip((p0[1] + t * (p1[1] - p0[1])).astype(int), 0, h - 1)
            rim = float(np.percentile(dedge[ys, xs], 80))
        good.append((s_, a, p0, p1, L, rim))
    return good


def _step_verifier(d, inner, px, floor):
    """verify(p0, p1): the step (nm, signed) across the straight line p0-p1 of the thickness-change map d if it is a
    sharp step between flat levels that holds along the line's whole length (1 um pieces: >= 85 % of the same sign, the
    mean at least 4 standard errors), at least `floor` nm, all inside `inner`; else None."""
    import math
    innerf = inner.astype(np.float32)                    # (once: converted again for every candidate it was ~1/3 of the time)
    # the profile across a line is sampled at the same offsets for every candidate: the bands' means as weights, once
    # (verify is called for thousands of candidates; its many small means were most of its time)
    ss = np.arange(-1.6, 1.61, 0.1) / px
    sv = ss * px
    npp, nm_ = (sv > 0.35) & (sv < 0.9), (sv < -0.35) & (sv > -0.9)
    fp, fm = (sv >= 0.9) & (sv <= 1.6), (sv <= -0.9) & (sv >= -1.6)
    w_np, w_nm, w_fp, w_fm = (b_ / b_.sum() for b_ in (npp, nm_, fp, fm))
    w_step = w_np - w_nm
    k = max(1, int(round(1.0 / px)))

    def verify(p0, p1):
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-u[1], u[0]])
        L = np.linalg.norm(p1 - p0)
        ts = np.arange(0, L, 1.0)
        X = (p0[0] + ts[None, :] * u[0] + ss[:, None] * n[0]).astype(np.float32)
        Y = (p0[1] + ts[None, :] * u[1] + ss[:, None] * n[1]).astype(np.float32)
        if cv2.remap(innerf, X, Y, cv2.INTER_NEAREST).mean() < 0.9:
            return None
        P = cv2.remap(d, X, Y, cv2.INTER_LINEAR)
        pr = P.mean(1)
        a_np, a_nm = float(pr @ w_np), float(pr @ w_nm)
        step = a_np - a_nm
        if abs(float(pr @ w_fp) - a_np) > 0.35 * abs(step) or abs(float(pr @ w_fm) - a_nm) > 0.35 * abs(step):
            return None
        nb = P.shape[1] // k                             # (1 um pieces along the line: block means, as one array operation)
        pcs = (w_step @ P[:, :nb * k]).reshape(nb, k).mean(1)
        if len(pcs) < 4 or np.mean(np.sign(pcs) == np.sign(step)) < 0.85 or abs(step) < floor:
            return None
        if abs(pcs.mean()) < 4 * pcs.std(ddof=1) / math.sqrt(len(pcs)):
            return None
        return float(step)
    return verify


def select_steps(good, shape, px, step_min=None, rim_um=None):
    """From step_candidates: steps >= step_min, not within rim_um of the edge, one detection per step (strongest first)."""
    step_min = STEP_MIN_NM if step_min is None else step_min
    rim_um = RIM_UM if rim_um is None else rim_um
    h, w = shape
    good = sorted([g_ for g_ in good if abs(g_[0]) >= step_min and g_[5] >= rim_um], key=lambda g_: -abs(g_[0]) * g_[4])
    expl = np.zeros((h, w), np.uint8)
    out = []
    wexp = max(5, int(round(1.5 / px)) | 1)
    for g_ in good:
        p0, p1 = g_[2], g_[3]
        t = np.linspace(0, 1, 40)
        xs = np.clip((p0[0] + t * (p1[0] - p0[0])).astype(int), 0, w - 1)
        ys = np.clip((p0[1] + t * (p1[1] - p0[1])).astype(int), 0, h - 1)
        if expl[ys, xs].mean() > 0.4:
            continue
        out.append(g_[:5])
        cv2.line(expl, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1, wexp)
    return out


def split_by_steps(d, F, lines, px, join_nm=None, rim_um=None, walls=None):
    """The flake split along the step lines (extended to the next line or the edge); pieces at the same level not
    separated by a detected step joined.  Returns [(mask, level nm)], largest first."""
    h, w = F.shape
    join_nm = LEVEL_JOIN_NM if join_nm is None else join_nm
    rim_um = RIM_UM if rim_um is None else rim_um
    # split and measure off the flake's rim (its blurred edge shifts the colour); the rim goes to the nearest piece after
    F2 = cv2.erode(F, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(2 * rim_um * 0.75 / px)) | 1),) * 2))
    if F2.sum() < 100:
        F2 = cv2.erode(F, np.ones((max(3, int(round(1.0 / px)) | 1),) * 2, np.uint8))
    base = np.zeros((h, w), np.uint8)
    for _, _, p0, p1, _ in lines:
        cv2.line(base, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1, 3 if px < 0.15 else 2)
    if walls is not None:
        base |= walls                                                # (outlines of large steps: like detected steps)
    wall = base.copy()
    for _, _, p0, p1, _ in lines:
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        own = np.zeros((h, w), np.uint8)
        cv2.line(own, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1, 3 if px < 0.15 else 2)
        other = (base > 0) & (own == 0)
        for q, dn in ((p1, u), (p0, -u)):
            for k in range(3, int(STEP_EXTEND_UM / px)):
                x, y = np.round(q + dn * k).astype(int)
                if not (0 <= x < w and 0 <= y < h) or F2[y, x] == 0 or other[y, x]:
                    cv2.line(wall, tuple(np.round(q).astype(int)), (int(x), int(y)), 1, 3 if px < 0.15 else 2)
                    break
    n, lab = cv2.connectedComponents(((F2 > 0) & (wall == 0)).astype(np.uint8), connectivity=4)
    if n <= 1:
        return []
    par = list(range(n))

    def fnd(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    kk = np.ones((11, 11), np.uint8) if px < 0.15 else np.ones((7, 7), np.uint8)
    dl = cv2.dilate(lab.astype(np.float32), kk).astype(np.int32)
    er = -cv2.dilate(-np.where(lab > 0, lab, 10 ** 6).astype(np.float32), kk).astype(np.int32)
    sel = F2 > 0
    pairs = set()
    for A_, B_ in ((lab, dl), (dl, er)):
        for a_, b_ in zip(A_[sel].ravel(), B_[sel].ravel()):
            if 0 < a_ < 10 ** 6 and 0 < b_ < 10 ** 6 and a_ != b_:
                pairs.add((min(a_, b_), max(a_, b_)))
    lvl0 = {i: float(np.median(d[lab == i])) for i in range(1, n)}
    for a_, b_ in sorted(pairs, key=lambda p_: abs(lvl0[p_[0]] - lvl0[p_[1]])):
        ra, rb = fnd(a_), fnd(b_)
        if ra == rb:
            continue
        ma = np.isin(lab, [i for i in range(1, n) if fnd(i) == ra])
        mb = np.isin(lab, [i for i in range(1, n) if fnd(i) == rb])
        if abs(float(np.median(d[ma])) - float(np.median(d[mb]))) >= join_nm:
            continue
        B = (cv2.dilate(ma.astype(np.uint8), kk) > 0) & (cv2.dilate(mb.astype(np.uint8), kk) > 0) & (wall > 0)
        if B.any() and base[B].mean() > 0.5:
            continue                                                 # a detected step between them
        par[rb] = ra
    groups = {}
    for i in range(1, n):
        groups.setdefault(fnd(i), []).append(i)
    from scipy import ndimage
    G = np.zeros((h, w), np.int32)
    for j, ids in enumerate(groups.values(), 1):
        G[np.isin(lab, ids)] = j
    # slivers (under 10 um2 off the rim): into the neighbour they share the most boundary with
    ng = len(groups)
    for _ in range(3):
        sizes = np.bincount(G.ravel(), minlength=ng + 1)
        small = [j for j in range(1, ng + 1) if 0 < sizes[j] * px * px < 10.0]
        if not small:
            break
        for j in small:
            m = G == j
            if not m.any():
                continue
            ring = (cv2.dilate(m.astype(np.uint8), kk) > 0) & ~m & (G > 0)
            nb = np.bincount(G[ring].ravel(), minlength=ng + 1)
            nb[j] = 0
            if nb.max() > 0:
                G[m] = int(np.argmax(nb))
    lvls = {j: float(np.median(d[G == j])) for j in range(1, len(groups) + 1) if (G == j).any()}
    _, (iy, ix) = ndimage.distance_transform_edt(G == 0, return_indices=True)
    Gf = np.where(F > 0, G[iy, ix], 0)                               # walls and rim: the nearest piece
    out = []
    for j in lvls:
        m = cv2.morphologyEx((Gf == j).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        if m.any():
            out.append((m, lvls[j]))
    out.sort(key=lambda r: -int(r[0].sum()))
    return out


def prepare(img_path, flake, main):
    """One flake's 100x picture (path or BGR picture) made ready for the step search: dict(h (picture, half resolution BGR), F, A (masks),
    lin, d (thickness-change map), inner (search area), crystal_deg / crystal_edge_frac, off, px) or None."""
    img = img_path if isinstance(img_path, np.ndarray) else cv2.imread(str(img_path))
    fr = flake.get("frame")
    if img is None or not fr:
        return None
    k = float(fr.get("k", 0.5))
    h = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    px = float(flake["px_um"])
    off = np.array([fr["x0"] * k, fr["y0"] * k])
    F = np.zeros(h.shape[:2], np.uint8)
    cv2.fillPoly(F, [np.round(np.array(flake["outline"], float).reshape(-1, 2) + off).astype(np.int32)], 1)
    A = np.zeros_like(F)
    cv2.fillPoly(A, [np.round(np.array(main["poly"], float).reshape(-1, 2) + off).astype(np.int32)], 1)
    if A.sum() < 50:
        return None
    lin = cv2.cvtColor(h, cv2.COLOR_BGR2RGB).astype(np.float32) + 0.5
    d = cv2.GaussianBlur(thickness_change_map(lin, F, A, flake["material"], flake.get("oxide_nm") or 300.0, main["t"]), (0, 0), 0.8)
    th, frac, _ = crystal_directions(h, F, px)
    inner = cv2.erode(F, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(round(3.0 / px)) | 1),) * 2))
    far = cv2.dilate((np.abs(d - np.median(d[A > 0])) > 2.5).astype(np.uint8), np.ones((11, 11), np.uint8))
    inner = (inner & (1 - far)).astype(np.uint8)
    return dict(h=h, F=F, A=A, lin=lin, d=d, inner=inner, crystal_deg=th, crystal_edge_frac=frac, off=off, px=px,
                edge_um=edge_width(h, F, px),
                flake=flake, main=main)


def terrace_features(P, m, lines):
    """Residue cues of one terrace mask m: straight = fraction of its boundary inside the flake (not on the flake edge)
    that lies on a detected step line; rough = robust spread (nm) of the thickness-change map inside it (off its rim);
    solidity = area / convex hull; edge = fraction of its boundary on the flake edge."""
    px, F, d = P["px"], P["F"], P["d"]
    mm = m.astype(np.uint8)
    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    h, w = F.shape
    fe = cv2.erode(F, np.ones((max(3, int(round(1.2 / px)) | 1),) * 2, np.uint8))
    on_edge = fe[np.clip(c[:, 1], 0, h - 1), np.clip(c[:, 0], 0, w - 1)] == 0
    L = np.zeros((h, w), np.uint8)
    for _, _, p0, p1, _ in lines:
        cv2.line(L, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1, max(3, int(round(1.4 / px)) | 1))
    on_line = L[np.clip(c[:, 1], 0, h - 1), np.clip(c[:, 0], 0, w - 1)] > 0
    inner_b = ~on_edge
    straight = float(on_line[inner_b].mean()) if inner_b.any() else 1.0
    er = cv2.erode(mm, np.ones((max(3, int(round(1.0 / px)) | 1),) * 2, np.uint8)) > 0
    v = d[er] if er.sum() > 30 else d[m]
    rough = float(1.4826 * np.median(np.abs(v - np.median(v))))
    hull = cv2.convexHull(c.astype(np.int32))
    sol = float(mm.sum() / max(cv2.contourArea(hull), 1.0))
    return dict(straight=round(straight, 3), rough=round(rough, 3), solidity=round(sol, 3), edge=round(float(on_edge.mean()), 3))


def terraces_from(P, lines, join_nm=None, rim_um=None, min_um2=20.0, min_d_um=2.0, rough_nm=None):
    """Terraces (designer dicts, cut-out px) from chosen step lines: the flake split along them (split_by_steps); the
    piece terrace a stands on is left out; thicknesses terrace a's plus the change (far from it: colour-ratio fit)."""
    flake, main, d, F, lin, A, off, px = P["flake"], P["main"], P["d"], P["F"], P["lin"], P["A"], P["off"], P["px"]
    # large steps (beyond the straight-step search): the outline of the areas far from terrace a's level
    big = (np.abs(d - float(np.median(d[A > 0]))) > BIG_STEP_NM).astype(np.uint8) & F
    big = cv2.morphologyEx(cv2.morphologyEx(big, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    walls = np.zeros_like(F)
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(big)
    for i in range(1, nb):
        if sb[i, cv2.CC_STAT_AREA] * px * px >= min_um2:
            cs_, _ = cv2.findContours((lb == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(walls, cs_, -1, 1, 2)
    walls &= cv2.erode(F, np.ones((5, 5), np.uint8))                 # (not along the flake's own edge)
    parts = split_by_steps(d, F, lines, px, join_nm=join_nm, rim_um=rim_um, walls=walls)
    ca = np.round(np.asarray(main["c"], float) + off).astype(int)
    out = []
    ref = lin[A > 0].mean(0)
    for m, lv in parts:
        if m[min(max(ca[1], 0), m.shape[0] - 1), min(max(ca[0], 0), m.shape[1] - 1)]:
            continue                                                 # the piece terrace a stands on
        area = float(m.sum()) * px * px
        mm = m.astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(mm, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        dd = 2 * float(dist.max()) * px
        if area < min_um2 or dd < min_d_um:
            continue
        if abs(lv) <= 2.0:
            t, how = main["t"] + lv, f"terrace a ({main['t']:g} nm) {lv:+.2f} nm from the 100x thickness-change map"
        else:
            ratio = lin[m].mean(0) / np.maximum(ref, 1e-6)
            t, err, _ = thickness_from_ratio(ratio, flake["material"], flake.get("oxide_nm") or 300.0, main["t"])
            if err > 0.08:
                continue
            how = f"100x colour against terrace a ({main['t']:g} nm), thin-film model; fit error {100 * err:.1f} %"
        feats = terrace_features(P, m, lines)
        if feats["rough"] > (RESIDUE_ROUGH_NM if rough_nm is None else rough_nm) and area < RESIDUE_MAX_UM2:
            continue                                                 # residue, not a terrace
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 1.2, True).reshape(-1, 2).astype(float) + 0.5 - off
        cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c.reshape(-1, 2).astype(np.float32))
        out.append(dict(t=round(float(t), 2), tlo=round(float(t), 2), thi=round(float(t), 2), alt="", layers=None, area=round(area, 1),
                        d=round(dd, 1), L=round(max(rw, rh) * px, 1), W=round(min(rw, rh) * px, 1),
                        c=[int(round(cx - off[0])), int(round(cy - off[1]))], poly=[round(v, 1) for v in poly.ravel().tolist()],
                        quality="terrace_100x", score=None, usable=True, thickness_source=how, step_nm=round(lv, 2),
                        fine=[], debris=[], steps=[], wrinkles=[], features=feats))
    return out



# ---------------------------------------------------------------- terraces as regions of the thickness-change map
# (2026-10-21, the default; Jordan: "If finding regions is better than lines, you should absolutely do this!")  The
# step-line terraces above cut the flake along the extensions of every straight step found: ragged pieces, many at
# terrace a's own level, ~10 s a flake.  Here the thickness-change map is split directly into patches of one level:
#   1. edge-preserving smoothing (bilateral, SMOOTH_NM colour scale), the gradient's magnitude (nm/um);
#   2. long straight steps (long_steps: the matched filter above, averaged over LONG_WIN_UM): the gradient misses
#      steps of a few 0.01 nm that hold over 20-50 um (2-f0041's central triangle); they are walls in the gradient,
#      cut the flat cores, and two regions on opposite sides of one are never joined (not even through a third);
#   3. markers = flat cores (gradient under FLAT_NM_UM), the gradient's watershed from them over the flake off its
#      blurred rim;
#   4. neighbours joined, the most similar first, while the LOCAL step across their shared boundary (median of d within
#      BAND_UM on each side) is under JOIN_NM (or NOISE_Z x the spread there, or JOIN_REL x the level: the map is less
#      exact far from terrace a) -- local, so a slow drift of the level across a terrace does not matter;
#   5. slivers into the neighbour with the longest shared boundary; pieces within A_LEVEL_NM of terrace a's level into
#      terrace a's region unless a long step separates them (and they lie A_FRAC of that step from a's level); areas
#      far from a's level (BIG_STEP_NM) one region unless their colours differ by THICK_COLOUR;
#   6. each region's level is the median of d (far from terrace a: thickness from the colour ratio, as above); rough
#      small regions are residue, small blotches with ragged outlines and regions whose boundary step is faint against
#      the map's spread there (TEXTURE_NM, TEXTURE_Z) texture: neither is a terrace;  out of focus (edge_width over
#      FOCUS_UM) only the areas far from a's level, far out of focus nothing;
#   7. polygons: the region boundaries split into arcs between junctions, each arc simplified once and shared by the two
#      regions on either side (no gaps, no overlaps), long segments turned onto the crystal directions.
# On Jordan's labels (2026-10-22, the 20 label-set flakes; boundaries within 1.5 um): of the region boundaries 0.84 of
# those Jordan drew found (the lines 0.75; precision 0.60 vs 0.58), 0.81 of the step lines Jordan accepted (the lines
# 0.85: partly their own earlier output; the regions without the long steps 0.74); of the terraces shown 0.80 / 0.71
# (the lines 0.73 / 0.56), accepted lines 0.74 / 0.74 (1-f0033 0.84 / 0.97, 2-f0041 0.77 / 0.59, 4-f0002 0.60 / 0.66;
# the lines 0.90 / 0.95, 0.92 / 0.85, 0.65 / 0.85); terrace overlap 0.67 vs 0.53 (22 vs 16 of 30 at least 0.5); 178
# terraces vs 331 (28 vs 65 within 0.15 nm of a's level, none vs 3 on the two out-of-focus flakes); ~1.4 s a flake vs
# ~9 s.  Still weak: heavily textured flakes (2-f0003: 34 terraces, precision 0.38 against Jordan's few steps; 4-f0013,
# 2-f0062), and the same picture JPEG re-encoded gives boundaries that agree 0.74 (the lines 0.86): the region
# terraces follow small changes of the picture more (the long steps found, the flat cores).  The thresholds of step 6
# and of the focus were chosen on these same flakes (and the 231 hBN flakes of bn_20261020_1-4 for the focus).

METHOD = "regions"          # "regions" (the default since 2026-10-21) or "lines" (the step-line terraces above)
JOIN_NM = 0.12              # neighbours whose local step is under this are one terrace ...
NOISE_Z = 0.75              # ... or under this many times the spread of the map on either side (texture)
JOIN_REL = 0.25             # ... or under this fraction of the larger |level| (the map is less exact far from terrace a)
SMOOTH_NM = 0.08            # bilateral filter: colour scale (nm) ...
SMOOTH_UM = 1.2             # ... and spatial scale (um)
FLAT_NM_UM = 0.12           # gradient under this (nm/um): a flat core (a marker)
BAND_UM = 1.6               # width on each side of a boundary for the local step
RIM_IN_UM = 1.0             # the flake's blurred rim: left out of the split, given to the nearest region
SLIVER_UM2 = 20.0           # regions smaller than this ...
SLIVER_D_UM = 1.6           # ... or narrower (inscribed circle) are slivers
SIMPLIFY_UM = 0.5           # Douglas-Peucker tolerance of the outlines along the flake edge ...
SIMPLIFY_IN_UM = 1.0        # ... and between two regions (steps)
SNAP_DEG = 6.0              # segments within this of a crystal direction ...
SNAP_MIN_UM = 3.0           # ... and at least this long are turned onto it
LONG_WIN_UM = 15.0          # long steps: averaged over this length ...
LONG_MIN_UM = 15.0          # ... at least this long ...
LONG_STEP_NM = 0.08         # ... and this high (the verified step across the line)
LONG_DEG = 2.0              # ... searched every this many degrees
LONG_EXT_UM = 20.0          # a long step inside a region is taken on up to this far beyond its ends
LONG_RIM_UM = 6.0           # a long step this close to the flake edge (median) is its blurred rim: it cuts nothing
LONG_STEPS = True           # long steps searched at all (graphite: no, see GRAPHITE)
A_LEVEL_NM = 0.15           # a region this close to terrace a's level whose local step to it is under A_COST x the
A_COST = 2.0                # joining threshold is terrace a (unless a long step separates them, and it lies at least
A_FRAC = 0.33               # this fraction of that step away from a's level)
THICK_COLOUR = 0.05         # areas far from a's level: neighbours closer than this in every log colour are one
BLOB_MAX_UM2 = 80.0         # texture blotches: regions smaller than this with less of their outline on a long step
BLOB_STRAIGHT = 0.4         # than this (on Jordan's labels their outline's shape, solidity and crystal directions told
                            # them from terraces no better than chance; a straight step along the outline did)
TEXTURE_NM = 0.3            # texture (any size): a region whose step along its boundary is under this (about one
TEXTURE_Z = 2.0             # layer, 0.33 nm) and under this many times the map's spread there (0: off)
FOCUS_UM = 1.6              # the flake's edge wider than this (edge_width; twice the in-focus median): out of focus,
FOCUS_RIM = 1.5             # terraces only BIG_STEP_NM or more from a's level, off a rim this many edge widths wide;
                            # more than twice FOCUS_UM: none (the colours there are the substrate's blurred in)
A_MIN_UM2 = 100.0           # terrace a's own region smaller than this or narrower (inscribed circle) than A_MIN_D_UM:
A_MIN_D_UM = 5.0            # its anchor fell in a sliver (1-f0086: 21 um2 of a 3109 um2 terrace), the catalog's shape
                            # kept (on bn_20261020_1-2 the next smallest is 272 um2, 10 um across: 2-f0003)
A_PICK_NM = 0.35            # terrace a at 100x: the biggest region (cut to the catalog's polygon) within this of a's
                            # level (about one layer, 0.33 nm), when bigger than a's own: the anchor can fall in a small
                            # piece beside the flat area (6-f0098: 936 of 4242 um2; Jordan, 2026-10-22: 'the biggest
                            # terrace first'); 0: off
A_WIGGLE_MAX = 1.25         # terrace a's region, or the one picked for it, whose outline inside the flake wanders more
                            # than this (ring_wiggle): residue or texture, not a terrace (6-f0098: residue picked as
                            # terrace a, 1.34; clean terraces 1.02-1.18; Jordan, 2026-10-22: 'evident from the boundary
                            # geometry'); neither: the catalog's shape
CLIP_A = True               # terrace a's region cut to the catalog's polygon (the piece holding the anchor): grown past
                            # it, the region ran over folds and cracks into residue and thicker pieces (6-f0098,
                            # 4-f0069, 6-f0003; Jordan, 2026-10-22)

# Materials (2026-10-22).  The region terraces and terrace a's own region / the biggest at its level
# (add_step_terraces) run for the materials in TERRACE_MATERIALS (tools/make_designer_bundle.py, update_bundle); the
# settings above are hBN's, and a material's entry in MATERIAL_SETTINGS overrides some of them while one of its flakes
# is worked on (material_settings).  The map is in nm of the material's own thickness and a graphite layer (0.335 nm)
# is an hBN layer (0.333 nm): the layer-sized tolerances (JOIN_NM, A_LEVEL_NM, A_PICK_NM, TEXTURE_NM) stay as they are.
# Graphite (graphite_20261019_run_1 / _5, judged on the pictures; TODO.md):
#   * its own optics (contrast_model: thick100.model100): a 1 -> 2 layer step reads 0.33 nm in the map;
#   * LONG_STEPS off: on thick graphite (~5 nm, where the map's sensitivity goes through zero) the long-step search
#     found straight "steps" in the colour noise and cut terrace a down (5-f0032: 2595 -> 271 um2, 5-f0021: 7683 ->
#     1886 um2); graphite's steps are a layer or more (8 % in colour): the regions find them without;
#   * BIG_STEP_NM 0.5 (1.5 layers): the map is linear within about a layer (1 -> 3 layers reads 0.60 nm for 0.67,
#     1 -> 4 layers 0.85 for 1.0); levels farther off take their thickness from the colour fit (thickness_from_ratio);
#   * A_WIGGLE_MAX off: on graphite it refused only terrace a regions within 1 % of the catalog's shape whose torn
#     edges and fold strips wander (5-f0059, -f0094, -f0229, -f0027, -f0069), never residue;
#   * terraces under GRAPHITE_MIN_LAYERS layers are the substrate (gaps, slits), and thicknesses within LAYER_SNAP of
#     a whole number of layers up to 4 are that number (layers; t = n x 0.335 nm), as the catalog's 1-4 layers.
TERRACE_MATERIALS = ("hbn", "graphite")
MATERIAL_SETTINGS = {"graphite": dict(LONG_STEPS=False, BIG_STEP_NM=0.5, A_WIGGLE_MAX=0.0)}
GRAPHITE_MIN_LAYERS = 0.5
LAYER_SNAP = 0.25


class material_settings(object):
    """with material_settings(material): the module's settings overridden by MATERIAL_SETTINGS[material], put back
    after (one flake at a time in a process, as the builds work)."""

    def __init__(self, material):
        self.over = MATERIAL_SETTINGS.get(material) or {}
        self.old = {}

    def __enter__(self):
        g = globals()
        self.old = {k: g[k] for k in self.over}
        g.update(self.over)
        return self

    def __exit__(self, *exc):
        globals().update(self.old)
        return False


def graphite_layers(t):
    """(t, layers) of a graphite terrace: a whole number of layers up to 4 (t = n x 0.335 nm) when within LAYER_SNAP of
    one, else layers to 0.1 (as the catalog's)."""
    L = LAYER_NM["graphite"]
    n = t / L
    if 1 <= round(n) <= 4 and abs(n - round(n)) <= LAYER_SNAP:
        return round(round(n) * L, 2), int(round(n))
    return t, round(n, 1)


def _disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def edge_width(h, F, px):
    """How sharp the picture is: the median width (um, 10-90 %) of the flake's edge, across its outline every ~1 um in
    the log brightness of the channel where flake and substrate differ most (None: too little clear edge).  In focus
    0.7-0.9 um (the 231 hBN flakes of bn_20261020_1-4: median 0.8, 95 % under 1.2); the two Jordan called 'out of
    focus, useless' 1.8 and 4.0 um."""
    Fi = cv2.erode(F, _disk(2.0 / px)) > 0
    ring = (cv2.dilate(F, _disk(4.0 / px)) > 0) & (cv2.dilate(F, _disk(2.0 / px)) == 0)
    if Fi.sum() < 50 or ring.sum() < 50:
        return None
    lg = np.log(h.astype(np.float32) + 0.5)
    lg = lg[..., None] if lg.ndim == 2 else lg                       # (a grey picture: its one channel)
    c = int(np.argmax([abs(np.median(lg[..., i][Fi]) - np.median(lg[..., i][ring])) for i in range(lg.shape[2])]))
    g = cv2.GaussianBlur(lg[..., c], (0, 0), 0.7)
    cs, _ = cv2.findContours(F.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    cc = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    n = len(cc)
    i = np.arange(0, n, max(1, int(round(1.0 / px))))
    t = cc[(i + 3) % n] - cc[(i - 3) % n]                               # the outline's direction, the normal across it
    ln = np.hypot(t[:, 0], t[:, 1])
    i, t = i[ln > 0], t[ln > 0] / ln[ln > 0, None]
    nv = np.stack([t[:, 1], -t[:, 0]], 1)
    ss = np.arange(-3.0, 3.01, 0.1) / px                              # (px along the normal)
    H, W = F.shape
    pin = np.clip(np.round(cc[i] - 2.0 * nv).astype(int), 0, [W - 1, H - 1])
    nv[F[pin[:, 1], pin[:, 0]] == 0] *= -1                            # (pointing out of the flake)
    X = cc[i, 0, None] + ss[None, :] * nv[:, 0, None]
    Y = cc[i, 1, None] + ss[None, :] * nv[:, 1, None]
    ok = (X.min(1) >= 1) & (Y.min(1) >= 1) & (X.max(1) <= W - 2) & (Y.max(1) <= H - 2)
    if ok.sum() < 10:
        return None
    pr = cv2.remap(g, X[ok].astype(np.float32), Y[ok].astype(np.float32), cv2.INTER_LINEAR)
    a = np.median(pr[:, ss * px < -2.0], axis=1)
    b = np.median(pr[:, ss * px > 2.0], axis=1)
    keep = np.abs(b - a) >= 0.03                                      # (a clear step: not along a touching flake or debris)
    if keep.sum() < 10:
        return None
    u = (pr[keep] - a[keep, None]) / (b - a)[keep, None]
    k0 = np.argmax(np.abs(np.diff(pr[keep], axis=1)), axis=1)         # the steepest point; 10 % and 90 % on either side
    ws = []
    for r, k in zip(u, k0):
        lo = k
        while lo > 0 and r[lo] > 0.1:
            lo -= 1
        hi = k + 1
        while hi < len(r) - 1 and r[hi] < 0.9:
            hi += 1
        ws.append((ss[hi] - ss[lo]) * px)
    return round(float(np.median(ws)), 2)


def long_steps(d, inner, px, F=None):
    """Long straight steps (>= LONG_MIN_UM, >= LONG_STEP_NM) in the thickness-change map d inside `inner`, also faint
    ones: [(step nm (signed), angle deg, p0, p1, length um)] as step_lines.  step_candidates' matched filter averaged over
    LONG_WIN_UM, every LONG_DEG, on the map at half resolution (4x fewer pixels; the lines are long); each angle's hits
    taken as connected runs (one candidate per run, its line fitted through the run), the candidates verified at full
    resolution strongest first and only where no line found already explains them (step_candidates spends most of its
    time checking thousands of candidates: here a few dozen; ~1 s a flake)."""
    H0, W0 = d.shape
    d2 = cv2.resize(d, (W0 // 2, H0 // 2), interpolation=cv2.INTER_AREA)
    i2 = cv2.resize(inner.astype(np.uint8), (W0 // 2, H0 // 2), interpolation=cv2.INTER_NEAREST)
    q2 = 2.0 * px
    h, w = d2.shape
    win = max(8, int(round(LONG_WIN_UM / q2)))
    o1, o2 = max(1, int(round(0.11 / q2))), max(3, int(round(0.57 / q2)))
    ys_, xs_ = np.nonzero(i2)
    if not ys_.size:
        return []
    mg = win + 2 * o2 + 4
    cy0, cy1, cx0, cx1 = max(0, ys_.min() - mg), min(h, ys_.max() + mg + 1), max(0, xs_.min() - mg), min(w, xs_.max() + mg + 1)
    dc, ic = np.ascontiguousarray(d2[cy0:cy1, cx0:cx1]), np.ascontiguousarray(i2[cy0:cy1, cx0:cx1])
    box = np.array([[cx0, cy0], [cx1, cy0], [cx0, cy1], [cx1, cy1]], float)
    n_ = o2 - o1 + 1
    c_ = (o1 + o2) // 2
    cands = []
    for a in np.arange(0.0, 180.0, LONG_DEG):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)       # (as step_candidates: the crop in the whole grid)
        q_ = box @ M[:, :2].T + M[:, 2]
        ox, oy = np.floor(q_.min(0)).astype(int)
        Wc, Hc = (np.ceil(q_.max(0)).astype(int) - (ox, oy) + 1).tolist()
        Mc = M.copy()
        Mc[:, 2] += M[:, :2] @ [cx0, cy0] - np.array([ox, oy], float)
        dr = cv2.warpAffine(dc, Mc, (Wc, Hc), flags=cv2.INTER_LINEAR)
        mr = cv2.warpAffine(ic, Mc, (Wc, Hc), flags=cv2.INTER_NEAREST)
        up = cv2.blur(dr, (1, n_))
        st = np.zeros_like(dr)
        st[o2 + 1:-o2 - 1] = up[o2 + 1 + c_:Hc - o2 - 1 + c_] - up[o2 + 1 - c_:Hc - o2 - 1 - c_]
        ok = cv2.erode(mr, np.ones((2 * o2 + 3, 1), np.uint8)) > 0
        st[~ok] = 0
        sa = cv2.blur(st, (win, 1))
        va = cv2.blur(ok.astype(np.float32), (win, 1)) > 0.95
        vals = sa[va][::3]                                           # (a third of them: enough for the spread)
        if vals.size < 40:
            continue
        thr = max(LONG_STEP_NM, 5 * 1.4826 * float(np.median(np.abs(vals - np.median(vals)))))
        asa = np.abs(sa)
        hit = (asa > thr) & va & (asa >= cv2.dilate(asa, np.ones((2 * o2 + 1, 1), np.uint8)) - 1e-6)
        if not hit.any():
            continue
        # runs along the rows (gaps of a few px bridged; a line a little off this angle steps over a row or two)
        nr, lab, stt, _ = cv2.connectedComponentsWithStats(cv2.dilate(hit.astype(np.uint8), np.ones((1, 7), np.uint8)), connectivity=8)
        Mi = cv2.invertAffineTransform(M)
        for i in range(1, nr):
            x, y, ww, hh = stt[i, :4]
            if (ww - 6 + win) * q2 < LONG_MIN_UM:
                continue
            sub = (lab[y:y + hh, x:x + ww] == i) & hit[y:y + hh, x:x + ww]
            yy, xx = np.nonzero(sub)
            if xx.size < 3:
                continue
            wt = asa[y:y + hh, x:x + ww][sub]
            sgn = np.sign(float((sa[y:y + hh, x:x + ww][sub] * wt).sum()))
            # (one sign per run: a run of both signs is two steps side by side, rare: the stronger sign's hits)
            same = np.sign(sa[y:y + hh, x:x + ww][sub]) == sgn
            yy, xx, wt = yy[same], xx[same], wt[same]
            if xx.size < 3 or xx.max() - xx.min() < 2:
                continue
            k_, b_ = np.polyfit(xx.astype(float), yy.astype(float), 1, w=wt) if np.ptp(yy) else (0.0, float(yy[0]))
            xa, xb = float(xx.min() - win // 2), float(xx.max() + win // 2)
            p0 = (Mi @ [xa + x + ox, k_ * (xa + win // 2) + b_ + y + oy, 1] + 0.5) * 2.0 - 0.5   # (full resolution)
            p1 = (Mi @ [xb + x + ox, k_ * (xb - win // 2) + b_ + y + oy, 1] + 0.5) * 2.0 - 0.5
            L = float(np.hypot(*(p1 - p0))) * px
            cands.append((float(wt.mean()) * L, float(a), p0, p1, L))
    verify = _step_verifier(d, inner, px, LONG_STEP_NM)
    dedge = cv2.distanceTransform(np.pad(F, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * px if F is not None else None
    h, w = H0, W0
    expl = np.zeros((h, w), np.uint8)
    wexp = max(5, int(round(5.0 / px)) | 1)                          # (parallel ones within ~2.5 um: the same step)
    out = []
    for _, a, p0, p1, L in sorted(cands, key=lambda c: -c[0]):
        t = np.linspace(0, 1, 40)
        xs = np.clip((p0[0] + t * (p1[0] - p0[0])).astype(int), 0, w - 1)
        ys = np.clip((p0[1] + t * (p1[1] - p0[1])).astype(int), 0, h - 1)
        if expl[ys, xs].mean() > 0.4:
            continue                                                 # (the same step at the next angle or row)
        if dedge is not None and float(np.percentile(dedge[ys, xs], 80)) < RIM_UM:
            continue                                                 # the flake edge's blurred rim
        s_ = verify(p0, p1)
        if s_ is None:
            continue
        out.append((s_, float(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180), p0, p1, L))
        cv2.line(expl, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1, wexp)
    return out


def _line_sides(lab, lines, px):
    """{region: {line index: +1 / -1}}: the side of each long step a watershed region lies on (it has pixels within
    ~1 um of the line, beyond its own 2 px, on that side only)."""
    H, W = lab.shape
    out = {}
    hw = max(4.0, 1.0 / px)
    for li, (_, _, p0, p1, _) in enumerate(lines):
        L = float(np.hypot(*(p1 - p0)))
        if L < 2:
            continue
        u = (p1 - p0) / L
        n = np.array([-u[1], u[0]])
        x0, x1 = int(max(0, min(p0[0], p1[0]) - hw - 1)), int(min(W, max(p0[0], p1[0]) + hw + 2))
        y0, y1 = int(max(0, min(p0[1], p1[1]) - hw - 1)), int(min(H, max(p0[1], p1[1]) + hw + 2))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        t = (xx - p0[0]) * u[0] + (yy - p0[1]) * u[1]
        s = (xx - p0[0]) * n[0] + (yy - p0[1]) * n[1]
        sel = (t >= 0) & (t <= L) & (np.abs(s) <= hw) & (np.abs(s) > 2.0)
        lb = lab[y0:y1, x0:x1][sel]
        sd = s[sel] > 0
        m = lb > 0
        if not m.any():
            continue
        nmax = int(lb.max()) + 1
        cp = np.bincount(lb[m & sd], minlength=nmax)
        cm = np.bincount(lb[m & ~sd], minlength=nmax)
        for i in np.flatnonzero((cp > 2) ^ (cm > 2)):
            out.setdefault(int(i), {})[li] = 1 if cp[i] > 2 else -1
    return out


def oversegment(d, F2, px):
    """Watershed regions of the smoothed map's gradient inside F2 from its flat cores: (labels int32, gradient nm/um)."""
    from skimage.measure import label as cc_label
    from skimage.segmentation import watershed
    sp = max(1.0, SMOOTH_UM / px)
    ds = d.astype(np.float32)
    for _ in range(2):
        ds = cv2.bilateralFilter(ds, int(2 * round(2 * sp) + 1), float(SMOOTH_NM), float(sp))
    gx = cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3) / (8 * px)
    gy = cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3) / (8 * px)
    g = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.0)
    low = ((g < FLAT_NM_UM) & (F2 > 0)).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, _disk(0.3 / px)) > 0
    mk = cc_label(low.astype(np.int32), background=0, connectivity=1).astype(np.int32)
    small = np.bincount(mk.ravel()) * px * px < 0.5
    small[0] = True
    mk[small[mk]] = 0
    return watershed(g, mk, mask=F2 > 0).astype(np.int32), g


def split_along_steps(G, lines, px, F=None):
    """Regions of G (0 outside) that long steps run through (the same region on both sides, ~1 um away, along most of a
    step's length: its flat core spanned the faint step) cut along them: each such step taken on along its line to the
    region's edge or the next step (at most LONG_EXT_UM; a step does not end inside a flake), and each piece of the
    region (all its steps' cuts together: three sides of a triangle enclose it, none alone does) a region of its own.
    Steps along the flake's edge (median distance under LONG_RIM_UM: its blurred rim) do not cut.  Returns (G with new
    labels, indices of the steps that cut)."""
    H, W = G.shape
    G = G.copy()
    dE = cv2.distanceTransform(np.pad(F, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * px if F is not None else None
    inside = []
    for li, (_, _, p0, p1, L) in enumerate(lines):
        n_s = max(4, int(L / 0.5))
        u = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-6)
        nv = np.array([-u[1], u[0]]) * (1.0 / px)
        q = p0 + np.linspace(0.05, 0.95, n_s)[:, None] * (p1 - p0)
        qi = np.clip(np.round(q).astype(int), 0, [W - 1, H - 1])
        if dE is not None and float(np.median(dE[qi[:, 1], qi[:, 0]])) < LONG_RIM_UM:
            continue
        a = np.clip(np.round(q + nv).astype(int), 0, [W - 1, H - 1])
        b = np.clip(np.round(q - nv).astype(int), 0, [W - 1, H - 1])
        la, lb = G[a[:, 1], a[:, 0]], G[b[:, 1], b[:, 0]]
        ok = (la > 0) & (lb > 0)
        if ok.mean() < 0.5 or (la[ok] == lb[ok]).mean() < 0.5:
            continue                                                 # (the regions have a boundary there already)
        inside.append((li, int(np.bincount(la[ok & (la == lb)]).argmax())))
    if not inside:
        return G, []
    walls = np.zeros((H, W), np.uint8)
    for li, _ in inside:
        _, _, p0, p1, _ = lines[li]
        cv2.line(walls, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1, 1)
    cut = walls.copy()
    E = LONG_EXT_UM / px
    seg = []                                                         # (each step's line, as far as it may be taken on)
    for li, _ in inside:
        _, _, p0, p1, _ = lines[li]
        L = max(float(np.hypot(*(p1 - p0))), 1e-6)
        seg.append((li, p0, (p1 - p0) / L, L))
    for li, r in inside:
        _, _, p0, p1, _ = lines[li]
        u = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-6)
        for q0, dn in ((p1, u), (p0, -u)):
            # as far as the region's edge, or where it meets another step taken on as far (two sides of a triangle
            # meet at its corner: neither goes past it), at most LONG_EXT_UM
            lim = E
            for lj, pj, uj, Lj in seg:
                if lj == li:
                    continue
                M = np.array([[dn[0], -uj[0]], [dn[1], -uj[1]]])
                if abs(np.linalg.det(M)) < 1e-3:
                    continue
                t, s_ = np.linalg.solve(M, pj - q0)
                if 0 < t < lim and -E <= s_ <= Lj + E:
                    lim = float(t)
            x, y = np.round(q0).astype(int)
            for k in range(1, int(lim) + 1):
                x2, y2 = np.round(q0 + dn * k).astype(int)
                if not (0 <= x2 < W and 0 <= y2 < H) or G[y2, x2] != r:
                    break
                x, y = x2, y2
            cv2.line(cut, tuple(np.round(q0).astype(int)), (int(x), int(y)), 1, 1)
    used = []
    nxt = int(G.max()) + 1
    for r in sorted(set(r for _, r in inside)):
        reg = G == r
        m = reg & (cut == 0)
        n, cc = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
        if n <= 2:
            continue                                                 # (the steps end inside the region: no cut)
        sizes = np.bincount(cc[m], minlength=n)
        if np.sort(sizes[1:])[-2] * px * px < SLIVER_UM2:
            continue                                                 # (only corners cut off)
        # (the cut's own pixels to a piece beside them)
        cc = np.where(m, cc, cv2.dilate(cc.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32))
        big = int(np.argmax(sizes))
        for c in range(1, n):
            if c != big:
                G[reg & (cc == c)] = nxt
                nxt += 1
        used += [li for li, r_ in inside if r_ == r]
    return G, sorted(used)


class _Regions:
    """Region adjacency with boundary-band samples (for the local steps) and the long steps' sides, for the joining."""

    def __init__(self, lab, d, px, band_um, sides=None):
        from scipy import ndimage
        self.d, self.px = d.ravel(), px
        H, W = lab.shape
        r = max(1, int(round(band_um / px)))
        k = _disk(r)
        self.near = {}        # i -> {j: flat pixel indices of region j within the band around region i}
        self.blen = {}        # i -> {j: shared boundary length (px)}
        self.pix, self.size = {}, {}
        self.side = {i: dict(v) for i, v in (sides or {}).items()}
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None:
                continue
            y0, y1 = max(0, sl[0].start - r - 1), min(H, sl[0].stop + r + 1)
            x0, x1 = max(0, sl[1].start - r - 1), min(W, sl[1].stop + r + 1)
            sub = lab[y0:y1, x0:x1]
            m = (sub == i).astype(np.uint8)
            ys, xs = np.nonzero(m)
            self.pix[i] = (ys + y0) * W + xs + x0
            self.size[i] = int(ys.size)
            ring = (cv2.dilate(m, k) > 0) & (m == 0) & (sub > 0)
            yy, xx = np.nonzero(ring)
            jj = sub[yy, xx]
            idx = (yy + y0) * W + xx + x0
            o = np.argsort(jj, kind="stable")
            jj, idx = jj[o], idx[o]
            cut = np.flatnonzero(np.diff(jj)) + 1
            self.near[i] = {int(js[0]): ids for js, ids in zip(np.split(jj, cut), np.split(idx, cut)) if js.size}
            ring1 = (cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0) & (m == 0) & (sub > 0)
            self.blen[i] = {int(j): int(c) for j, c in zip(*np.unique(sub[ring1], return_counts=True))}
            self.side.setdefault(i, {})
        self.alive = set(self.pix)
        self.stat = {}        # (i, j) -> (median, robust spread) of near[i][j]

    def _stat(self, i, j):
        s = self.stat.get((i, j))
        if s is None:
            v = self.d[self.near[i][j]]
            m = float(np.median(v))
            s = self.stat[(i, j)] = (m, 1.4826 * float(np.median(np.abs(v - m))))
        return s

    def apart(self, i, j):
        """On opposite sides of a long step?"""
        a, b = self.side[i], self.side[j]
        return any(b.get(li, s) != s for li, s in a.items())

    def local_step(self, i, j):
        """(j's level - i's level near their boundary, robust spread there, larger |level|) or None."""
        a, b = self.near[j].get(i), self.near[i].get(j)
        if a is None or b is None or a.size < 3 or b.size < 3:
            return None
        (ma, sa), (mb, sb) = self._stat(j, i), self._stat(i, j)
        return mb - ma, math.sqrt((sa * sa + sb * sb) / 2), max(abs(ma), abs(mb))

    def level(self, i):
        return float(np.median(self.d[self.pix[i]]))

    def merge(self, i, j):
        """j into i."""
        self.alive.discard(j)
        self.pix[i] = np.concatenate([self.pix[i], self.pix.pop(j)])
        self.size[i] += self.size.pop(j)
        self.side[i].update(self.side.pop(j))
        for o, v in self.near.pop(j).items():
            for key in ((i, o), (o, i), (j, o), (o, j)):
                self.stat.pop(key, None)
            if o == i:
                continue
            self.near[i][o] = np.union1d(self.near[i][o], v) if o in self.near[i] else v
            w = self.near[o].pop(j, None)
            if w is not None:
                self.near[o][i] = np.union1d(self.near[o][i], w) if i in self.near[o] else w
        self.near[i].pop(j, None)
        for o, c in self.blen.pop(j).items():
            if o == i:
                continue
            self.blen[i][o] = self.blen[i].get(o, 0) + c
            self.blen[o][i] = self.blen[o].get(i, 0) + self.blen[o].pop(j, 0)
        self.blen[i].pop(j, None)

    def labels(self, shape):
        """Label image 1..n of the regions (largest first) and their ids in that order."""
        ids = sorted(self.alive, key=lambda i: -self.size[i])
        G = np.zeros(shape[0] * shape[1], np.int32)
        for k, i in enumerate(ids, 1):
            G[self.pix[i]] = k
        return G.reshape(shape), ids


def _slivers(R, px, shape):
    """Small or thin regions into the neighbour with the longest shared boundary (not across a long step)."""
    H, W = shape
    for _ in range(4):
        changed = False
        for i in sorted(R.alive, key=lambda i: R.size[i]):
            if i not in R.alive:
                continue
            area = R.size[i] * px * px
            if area >= SLIVER_UM2:
                if area >= 200.0:
                    continue
                ys, xs = np.unravel_index(R.pix[i], (H, W))
                y0, x0 = ys.min(), xs.min()
                m = np.zeros((ys.max() - y0 + 3, xs.max() - x0 + 3), np.uint8)
                m[ys - y0 + 1, xs - x0 + 1] = 1
                if 2 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max()) * px >= SLIVER_D_UM:
                    continue
            nb = [(c, o) for o, c in R.blen[i].items() if c > 0 and o in R.alive and not R.apart(i, o)]
            if not nb:
                continue
            R.merge(max(nb)[1], i)
            changed = True
        if not changed:
            break


def regions_by_level(d, F, px, lines=(), lin=None, anchor=None, A=None, join_nm=None, rim_um=None, info=None):
    """The flake split into regions of one level of d (steps 3-5 above): (label image 1..n largest first (0 off the
    flake), levels nm, k_a (the region of terrace a: at `anchor` (x, y) or holding most of A), apart_a {k: True if a
    long step (with a level change) or a shared boundary separates region k from terrace a's}, indices of the long
    steps that cut a region)."""
    from scipy import ndimage
    join_nm = JOIN_NM if join_nm is None else join_nm
    rim_um = RIM_IN_UM if rim_um is None else rim_um
    H, W = F.shape
    F2 = cv2.erode(F, _disk(rim_um / px))
    if F2.sum() < 100:
        F2 = cv2.erode(F, _disk(0.5 / px))
    lab, _ = oversegment(d, F2, px)
    if info is not None:
        info["n_initial"] = int(lab.max())
    R = _Regions(lab, d, px, BAND_UM)

    def cost(i, j):                          # local step over the threshold for it (joined under 1)
        if R.apart(i, j):
            return None
        s = R.local_step(i, j)
        if s is None:
            return None
        return abs(s[0]) / max(join_nm, NOISE_Z * s[1], JOIN_REL * s[2])
    heap = []
    for i in R.alive:
        for j in R.near[i]:
            if i < j:
                c = cost(i, j)
                if c is not None and c < 1:
                    heap.append((c, i, j))
    heapq.heapify(heap)
    while heap:
        c, a, b = heapq.heappop(heap)
        if a not in R.alive or b not in R.alive or b not in R.near[a]:
            continue
        c2 = cost(a, b)
        if c2 is None or c2 >= 1:
            continue
        if c2 > c + 1e-9:
            heapq.heappush(heap, (c2, a, b))
            continue
        i, j = (a, b) if R.size[a] >= R.size[b] else (b, a)
        R.merge(i, j)
        for o in list(R.near[i]):
            c3 = cost(i, o)
            if c3 is not None and c3 < 1:
                heapq.heappush(heap, (c3, min(i, o), max(i, o)))
    _slivers(R, px, (H, W))
    # long steps inside a region (its flat core spanned them): the region cut along them; then the regions once more,
    # now knowing each long step's sides
    used = []
    if len(lines):
        G1, _ = R.labels((H, W))
        G1, used = split_along_steps(G1, lines, px, F)
        R = _Regions(G1, d, px, BAND_UM, _line_sides(G1, lines, px))
        _slivers(R, px, (H, W))
    # terrace a's region: pieces at its level (a ridge or a noisy boundary cut them off: '+0.0x nm' terraces) join it
    Af = None if A is None else (A.ravel() > 0)
    ka = None
    if anchor is not None:
        x, y = int(round(anchor[0])), int(round(anchor[1]))
        if 0 <= x < W and 0 <= y < H:
            q = y * W + x
            ka = next((i for i in R.alive if np.any(R.pix[i] == q)), None)
    if ka is None and Af is not None:
        cnt = [(int(np.count_nonzero(Af[R.pix[i]])), i) for i in R.alive]
        ka = max(cnt)[1] if cnt and max(cnt)[0] > 0 else None
    if ka is not None:
        la = R.level(ka)
        changed = True
        while changed:
            changed = False
            for o in list(R.blen[ka]):
                if o in R.alive and o != ka and not R.apart(ka, o) and abs(R.level(o) - la) < A_LEVEL_NM \
                        and (cost(ka, o) or 0.0) < A_COST:
                    R.merge(ka, o)
                    changed = True
        # areas far from terrace a's level: one region unless clearly another colour (the map is not linear there)
        if lin is not None:
            L3 = np.log(np.maximum(lin.reshape(-1, 3), 1e-3))
            far = lambda i: abs(R.level(i) - la) > BIG_STEP_NM
            changed = True
            while changed:
                changed = False
                for i in sorted(R.alive, key=lambda i: -R.size[i]):
                    if i not in R.alive or i == ka or not far(i):
                        continue
                    ci = L3[R.pix[i]].mean(0)
                    for o in list(R.blen[i]):
                        if o in R.alive and o != ka and o != i and far(o) and not R.apart(i, o) \
                                and np.abs(L3[R.pix[o]].mean(0) - ci).max() < THICK_COLOUR:
                            R.merge(i, o)
                            changed = True
    if info is not None:
        info["n_merged"] = len(R.alive)
    # relabelled 1..n (largest first); the rim and the rest of the flake to the nearest region
    G2, ids = R.labels((H, W))
    lv = [R.level(i) for i in ids]
    # (a region at a's level is a terrace of its own only where something separates it from a's: a long step it lies at
    # least A_FRAC of that step away from, beside the step or as a whole (both sides of a step at one level: the step
    # was a blur gradient or texture, 1-f0049's out-of-focus triangle at +0.00 nm beyond a -0.17 nm step; half the
    # step lost 2-f0041's 2490 um2 terrace beyond a -0.55 nm step Jordan accepted, 0.25 nm beside it), or a boundary
    # with a's region whose local step is too big for the join above; not one cut off by a third region)
    def apart_a(i):
        if ka is None or i == ka:
            return False
        if R.apart(ka, i):
            cross = [abs(lines[li][0]) for li, s in R.side[ka].items() if R.side[i].get(li, s) != s]
            s = R.local_step(ka, i)                                  # (beside it: the levels there, or the regions')
            return max(abs(s[0]) if s is not None else 0.0, abs(R.level(i) - R.level(ka))) >= A_FRAC * min(cross)
        return R.blen[ka].get(i, 0) > 0
    apart = {k: apart_a(i) for k, i in enumerate(ids, 1)}
    _, (iy, ix) = ndimage.distance_transform_edt(G2 == 0, return_indices=True)
    Gf = np.where(F > 0, G2[iy, ix], 0).astype(np.int32)
    return Gf, lv, (ids.index(ka) + 1 if ka is not None else 0), apart, used


# ---------------------------------------------------------------- shared-boundary polygons

def _arcs(L):
    """Region boundaries of label image L (0 = outside) on the pixel-corner lattice, as arcs between junctions:
    [(corner points (x, y) float array, left label, right label, closed)]."""
    Lp = np.pad(L, 1)
    hz = Lp[:-1, :] != Lp[1:, :]            # between pixel (r, c) and (r+1, c): corner row r+1, cols c..c+1
    vt = Lp[:, :-1] != Lp[:, 1:]            # between pixel (r, c) and (r, c+1): corner col c+1, rows r..r+1
    adj = {}

    def add(p, q):
        adj.setdefault(p, []).append(q)
        adj.setdefault(q, []).append(p)
    for r, c in zip(*np.nonzero(hz)):
        add((int(c), int(r) + 1), (int(c) + 1, int(r) + 1))
    for r, c in zip(*np.nonzero(vt)):
        add((int(c) + 1, int(r)), (int(c) + 1, int(r) + 1))

    def side(p, q):
        mx, my = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
        dx, dy = q[0] - p[0], q[1] - p[1]
        lx, ly = mx + 0.5 * dy, my - 0.5 * dx
        rx, ry = mx - 0.5 * dy, my + 0.5 * dx
        return int(Lp[int(math.floor(ly)), int(math.floor(lx))]), int(Lp[int(math.floor(ry)), int(math.floor(rx))])
    used = set()
    out = []
    junc = set(p for p, v in adj.items() if len(v) != 2)
    for start in sorted(junc) + list(adj):
        for q0 in adj[start]:
            e = (start, q0) if start < q0 else (q0, start)
            if e in used:
                continue
            pts = [start]
            p, q = start, q0
            while True:
                used.add((p, q) if p < q else (q, p))
                pts.append(q)
                if q == start or len(adj[q]) != 2:
                    break
                nq = adj[q][0] if adj[q][0] != p else adj[q][1]
                if ((q, nq) if q < nq else (nq, q)) in used:
                    break
                p, q = q, nq
            lft, rgt = side(pts[0], pts[1])
            out.append((np.array(pts, float) - 1.0, lft, rgt, pts[0] == pts[-1]))   # (padded -> picture corners)
            junc.add(start)                                          # (a loop: its start is now a fixed point)
    return out


def _simplify(a, eps, closed):
    if len(a) <= 2:
        return a
    if closed:
        s = cv2.approxPolyDP(a[:-1].astype(np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(float)
        return np.vstack([s, s[:1]])
    return cv2.approxPolyDP(a.astype(np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2).astype(float)


def _snap(a, theta0, px):
    """Long segments within SNAP_DEG of a crystal direction (theta0 + n x 30 deg) turned onto it (about their
    midpoint); a vertex between two turned segments moves to the lines' intersection, the arc's ends stay."""
    if theta0 is None or len(a) < 2:
        return a
    n = len(a) - 1
    lines = []
    for s in range(n):
        p, q = a[s], a[s + 1]
        L = float(np.hypot(*(q - p))) * px
        ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        dv = ((ang - theta0 + 15.0) % 30.0) - 15.0
        if L >= SNAP_MIN_UM and abs(dv) <= SNAP_DEG and abs(dv) > 1e-6:
            t = math.radians(ang - dv)
            lines.append(((p + q) / 2.0, np.array([math.cos(t), math.sin(t)])))
        else:
            lines.append(None)
    b = a.copy()
    for v in range(1, n):
        l1, l2 = lines[v - 1], lines[v]
        if l1 is None and l2 is None:
            continue
        if l1 is not None and l2 is not None:
            M = np.array([l1[1], -l2[1]]).T
            if abs(np.linalg.det(M)) > 1e-3:
                t = np.linalg.solve(M, l2[0] - l1[0])
                x = l1[0] + t[0] * l1[1]
                if np.hypot(*(x - a[v])) * px < 1.0:
                    b[v] = x
            continue
        ln = l1 if l1 is not None else l2       # one turned neighbour: the vertex onto its line (foot of the normal)
        x = ln[0] + float((a[v] - ln[0]) @ ln[1]) * ln[1]
        if np.hypot(*(x - a[v])) * px < 1.0:
            b[v] = x
    return b


def region_polygons(G, px, theta0=None, snap=True):
    """The outer ring of each region of G (1..n): {k: (m, 2) pixel-corner coordinates}, from the shared arcs."""
    eps, eps_in = SIMPLIFY_UM / px, SIMPLIFY_IN_UM / px
    arcs = []
    for a, l, r, closed in _arcs(G):
        s = _simplify(a, eps if 0 in (l, r) else eps_in, closed)
        if snap and not closed:
            s = _snap(s, theta0, px)
        arcs.append((s, l, r))
    rings = {}
    for k in range(1, int(G.max()) + 1):
        own = [s if l == k else s[::-1] for s, l, r in arcs if l == k or r == k]
        if not own:
            continue
        starts = {}
        for i, s in enumerate(own):
            starts.setdefault(tuple(np.round(s[0], 6)), []).append(i)
        used, best = set(), None
        for i0 in range(len(own)):
            if i0 in used:
                continue
            ring, i = [], i0
            while i is not None and i not in used:
                used.add(i)
                ring.append(own[i][:-1])
                nxt = [j for j in starts.get(tuple(np.round(own[i][-1], 6)), []) if j not in used]
                i = nxt[0] if nxt else None
            R = np.vstack(ring)
            ar = abs(cv2.contourArea(R.astype(np.float32)))
            if best is None or ar > best[0]:
                best = (ar, R)
        rings[k] = best[1]
    return rings


def crystal_fraction(ring, F, px, theta0):
    """Fraction of a polygon's length inside the flake (off its edge) in segments >= 2 um within SNAP_DEG of a crystal
    direction (None without crystal directions or an inside boundary): steps run straight along them, blotches not."""
    if theta0 is None:
        return None
    H, W = F.shape
    dE = cv2.distanceTransform(np.pad(F, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    tot = good = 0.0
    q = np.vstack([ring, ring[:1]])
    for p0, p1 in zip(q[:-1], q[1:]):
        L = float(np.hypot(*(p1 - p0))) * px
        mx, my = np.clip(np.round((p0 + p1) / 2.0 - 0.5).astype(int), 0, [W - 1, H - 1])
        if dE[my, mx] * px < 1.2 or L <= 0:
            continue                                                 # (along the flake edge)
        tot += L
        ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        if L >= 2.0 and abs(((ang - theta0 + 15.0) % 30.0) - 15.0) <= SNAP_DEG:
            good += L
    return round(good / tot, 3) if tot > 0 else None


def terraces_from_regions(P, lines=(), join_nm=None, rim_um=None, min_um2=20.0, min_d_um=2.0, rough_nm=None, info=None):
    """Terraces (designer dicts as terraces_from, cut-out px) from the regions of the thickness-change map (above),
    `lines` the long steps (long_steps).  Terrace a's region is left out, and so are regions at its level that no long
    step separates from it, residue and texture; out of focus (P's edge_um, else edge_width) see FOCUS_UM.  info gets
    G, levels, rings, k_a, cut_by, flags (why each other region is not a terrace) and edge_um."""
    flake, main, d, F, lin, A, off, px = P["flake"], P["main"], P["d"], P["F"], P["lin"], P["A"], P["off"], P["px"]
    info = {} if info is None else info
    ca = np.asarray(main["c"], float) + off
    # out of focus (the flake's edge wider than FOCUS_UM): faint steps and levels are the blur's, not the flake's; only
    # areas far from terrace a's level (clear colours), measured off the rim the blur widened; far out of focus nothing
    # (2-f0004, 4.0 um: its blurred rim, 6 um wide, a '+6.4 nm' band; Jordan: 'out of focus, useless')
    blur = P["edge_um"] if "edge_um" in P else edge_width(P["h"], F, px)
    soft = blur is not None and blur > FOCUS_UM
    info["edge_um"] = blur
    if soft and blur > 2 * FOCUS_UM:
        info.update(G=(F > 0).astype(np.int32), levels=[0.0], rings={}, k_a=1, cut_by=[], flags={}, n_initial=1)
        return []                                                    # (the whole flake terrace a's)
    if soft:
        lines = ()
        rim_um = max(RIM_IN_UM if rim_um is None else rim_um, FOCUS_RIM * blur)
    G, lv, ka, apart, used = regions_by_level(d, F, px, lines=lines, lin=lin, anchor=ca, A=A, join_nm=join_nm,
                                              rim_um=rim_um, info=info)
    th = P.get("crystal_deg")
    rings = region_polygons(G, px, theta0=th)
    info.update(G=G, levels=lv, rings=rings, k_a=ka, cut_by=used)
    la = lv[ka - 1] if ka else 0.0
    ref = lin[A > 0].mean(0)
    out, flags = [], {}
    RB = None                                                        # (the final regions' adjacency, when needed)
    for k in range(1, int(G.max()) + 1):
        if k == ka or k not in rings:
            continue
        lvk = lv[k - 1]
        if abs(lvk - la) < A_LEVEL_NM and not apart.get(k):
            flags[k] = "a level"
            continue                                                 # (cut off from a's region by another one)
        if soft and abs(lvk - la) <= BIG_STEP_NM:
            flags[k] = "out of focus"
            continue
        m = G == k
        area = float(m.sum()) * px * px
        mm = m.astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(mm, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        dd = 2 * float(dist.max()) * px
        if area < min_um2 or dd < min_d_um:
            flags[k] = "small"
            continue
        if abs(lvk) <= BIG_STEP_NM:
            t, how = main["t"] + lvk, f"terrace a ({main['t']:g} nm) {lvk:+.2f} nm from the 100x thickness-change map"
        else:
            ratio = lin[m].mean(0) / np.maximum(ref, 1e-6)
            t, err, _ = thickness_from_ratio(ratio, flake["material"], flake.get("oxide_nm") or 300.0, main["t"])
            if err > 0.08:
                flags[k] = "colour"
                continue
            how = f"100x colour against terrace a ({main['t']:g} nm), thin-film model; fit error {100 * err:.1f} %"
        layers = None
        if flake["material"] == "graphite":
            if t < GRAPHITE_MIN_LAYERS * LAYER_NM["graphite"]:
                flags[k] = "substrate"
                continue                                             # (a gap or slit in the flake)
            t, layers = graphite_layers(float(t))
        feats = terrace_features(P, m, lines)
        feats["crystal"] = crystal_fraction(rings[k], F, px, th)
        if feats["rough"] > (RESIDUE_ROUGH_NM if rough_nm is None else rough_nm) and area < RESIDUE_MAX_UM2:
            flags[k] = "residue"
            continue
        if area < BLOB_MAX_UM2 and feats["straight"] < BLOB_STRAIGHT:
            flags[k] = "blotch"
            continue                                                 # texture, not a terrace
        if TEXTURE_Z > 0:
            if RB is None:
                RB = _Regions(G, d, px, BAND_UM)
            st, sp = boundary_step(RB, k)
            if st is not None and st < TEXTURE_NM and st < TEXTURE_Z * sp:
                flags[k] = "texture"
                continue                                             # (any size: 4-f0002's 100-300 um2 ones)
        poly = rings[k] - off                                       # (pixel corners: the designer's coordinates already,
        cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)  # pixel i spanning [i, i+1])
        (_, _), (rw, rh), _ = cv2.minAreaRect(poly.astype(np.float32))
        out.append(dict(t=round(float(t), 2), tlo=round(float(t), 2), thi=round(float(t), 2), alt="", layers=layers, area=round(area, 1),
                        d=round(dd, 1), L=round(max(rw, rh) * px, 1), W=round(min(rw, rh) * px, 1),
                        c=[int(round(cx - off[0])), int(round(cy - off[1]))], poly=[round(v, 1) for v in poly.ravel().tolist()],
                        quality="terrace_100x", score=None, usable=True, thickness_source=how, step_nm=round(lvk, 2),
                        fine=[], debris=[], steps=[], wrinkles=[], features=feats))
    info["flags"] = flags
    return out


def boundary_step(R, k):
    """(|local step| (nm), the map's spread there (nm)) along region k's boundary with its neighbours (_Regions R),
    averaged over the boundary's length; (None, None) without neighbours."""
    st = sp = wt = 0.0
    for o, c in R.blen.get(k, {}).items():
        s = R.local_step(k, o) if c > 0 else None
        if s is not None:
            st, sp, wt = st + c * abs(s[0]), sp + c * s[1], wt + c
    return (st / wt, sp / wt) if wt else (None, None)


def lines_info(P, lines):
    off = P["off"]
    return [dict(step_nm=round(s_, 2), angle_deg=round(a_ % 180, 1), p=[round(v, 1) for v in (p0 - off).tolist() + (p1 - off).tolist()],
                 length_um=round(L, 1)) for s_, a_, p0, p1, L in lines]


def a_region(P, ri, min_um2=None, min_d_um=None):
    """Terrace a's own 100x region (ri: terraces_from_regions' info; the region holding a's anchor, the pieces at a's
    level joined to it) as the fields a terrace's shape has: dict(poly, area, d, L, W) (cut-out px), or None when there
    is none to trust: no regions (no picture, far out of focus: the whole flake a's), out of focus (blur over FOCUS_UM:
    the levels near a's are the blur's), the anchor not inside it, or smaller than min_um2 / narrower than min_d_um
    (default A_MIN_UM2 / A_MIN_D_UM).  Like the other terraces: the region's outer ring (a terrace wholly inside it
    stays inside the polygon; a region in pieces: its largest, which must hold the anchor), its area and inscribed
    circle the region's own pixels inside that ring."""
    G, rings, ka, blur = ri.get("G"), ri.get("rings") or {}, ri.get("k_a"), ri.get("edge_um")
    if G is None or not ka or ka not in rings or (blur is not None and blur > FOCUS_UM):
        return None
    off, px = P["off"], P["px"]
    x, y = (int(round(v)) for v in np.asarray(P["main"]["c"], float) + off)
    if not (0 <= y < G.shape[0] and 0 <= x < G.shape[1]) or G[y, x] != ka:
        return None                                              # (a found by its polygon only: its anchor elsewhere)
    R = np.zeros_like(G, np.uint8)                               # (a region in pieces: the ring is its largest's)
    cv2.fillPoly(R, [np.round(rings[ka]).astype(np.int32)], 1)
    if cv2.pointPolygonTest(rings[ka].astype(np.float32), (x + 0.5, y + 0.5), True) * px < -1.0:
        return None                                              # (in another piece; on a's edge (2-f0024): 0.3 um out)
    ring = rings[ka]
    cat = P["main"].get("poly")
    if CLIP_A and cat:                                           # (cut to the catalog's polygon: the anchor's piece)
        C = np.zeros_like(R)
        cv2.fillPoly(C, [np.round(np.asarray(cat, float).reshape(-1, 2) + off).astype(np.int32)], 1)
        n, lab = cv2.connectedComponents(R & C)
        yy, xx = min(max(y, 0), G.shape[0] - 1), min(max(x, 0), G.shape[1] - 1)
        k = lab[yy, xx]
        if not k:                                                # (anchor on the cut: the largest piece)
            if n < 2:
                return None
            k = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
        R = (lab == k).astype(np.uint8)
        cs, _ = cv2.findContours(R, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        ring = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.7, True).reshape(-1, 2).astype(float)
        if len(ring) < 3:
            return None
    m = ((G == ka) & (R > 0)).astype(np.uint8)
    area = float(m.sum()) * px * px
    dd = 2 * float(cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5).max()) * px
    if area < (A_MIN_UM2 if min_um2 is None else min_um2) or dd < (A_MIN_D_UM if min_d_um is None else min_d_um):
        return None
    poly = ring - off
    (_, _), (rw, rh), _ = cv2.minAreaRect(poly.astype(np.float32))
    return dict(poly=[round(v, 1) for v in poly.ravel().tolist()], area=round(area, 1), d=round(dd, 1),
                L=round(max(rw, rh) * px, 1), W=round(min(rw, rh) * px, 1))


def step_terraces(img_path, flake, main, min_um2=20.0, min_d_um=2.0, step_min=None, join_nm=None, rim_um=None, win_um=None,
                  method=None):
    """The flake's terraces besides the one holding terrace a's anchor, with thicknesses (METHOD: regions of the
    thickness-change map, or the step-line terraces); also the crystal directions.  Coordinates as the designer's
    (cut-out px).  img_path: the picture's path, or the picture itself (BGR).  Returns (terraces, info); info's
    step_lines: the long steps among the regions' boundaries (regions) or every step line (lines); a_region: terrace
    a's own region (a_region; regions only, else None).  Under the flake's material_settings."""
    with material_settings(flake.get("material")):
        return _step_terraces(img_path, flake, main, min_um2, min_d_um, step_min, join_nm, rim_um, win_um, method)


def _step_terraces(img_path, flake, main, min_um2, min_d_um, step_min, join_nm, rim_um, win_um, method):
    P = prepare(img_path, flake, main)
    if P is None:
        return [], {}
    a_reg = None
    if (method or METHOD) == "lines":
        lines = step_lines(P["d"], P["inner"], P["px"], P["F"], step_min=step_min, win_um=win_um, rim_um=rim_um)
        out = terraces_from(P, lines, join_nm=join_nm, rim_um=rim_um, min_um2=min_um2, min_d_um=min_d_um)
    else:
        blur = P.get("edge_um")
        lines = long_steps(P["d"], P["inner"], P["px"], P["F"]) if LONG_STEPS and (blur is None or blur <= FOCUS_UM) else []
        ri = {}
        out = terraces_from_regions(P, lines, join_nm=join_nm, min_um2=min_um2, min_d_um=min_d_um, info=ri)
        a_reg = a_region(P, ri)
    th, frac = P["crystal_deg"], P["crystal_edge_frac"]
    info = dict(crystal_deg=None if th is None else round(th, 2), crystal_edge_frac=round(frac, 2), step_lines=lines_info(P, lines),
                a_region=a_reg)
    return out, info


def ring_wiggle(ring, F, px, smooth_um=4.0, rim_um=1.5):
    """How much a region's outline wanders where it is inside the flake (off the edge, rim_um): its length over the
    length of the same line smoothed over smooth_um; None with too little of it inside.  Steps between terraces run
    straight (bn_20261020 runs: median 1.15 for terrace a); residue and texture borders wander (6-f0098's: 1.34)."""
    c0 = np.asarray(ring, float).reshape(-1, 2)
    q = np.vstack([c0, c0[:1]]); pts = []
    for p0, p1 in zip(q[:-1], q[1:]):
        n = max(1, int(np.hypot(*(p1 - p0))))
        pts += [p0 + (p1 - p0) * i / n for i in range(n)]
    c = np.array(pts); n = len(c)
    H, W = F.shape
    dE = cv2.distanceTransform(np.pad(F.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    inner = dE[np.clip(c[:, 1].astype(int), 0, H - 1), np.clip(c[:, 0].astype(int), 0, W - 1)] * px > rim_um
    if inner.sum() * px < 20.0:
        return None
    k = max(3, int(round(smooth_um / px)))
    cs = np.cumsum(np.vstack([c[-k:], c, c[:k]]), axis=0)
    sm = (cs[2 * k:] - np.vstack([np.zeros((1, 2)), cs[:-2 * k - 1]])) / (2 * k + 1)
    seg = np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T)
    segs = np.hypot(*np.diff(np.vstack([sm, sm[:1]]), axis=0).T)
    return float(seg[inner].sum() / max(segs[inner].sum(), 1e-6))


def _clipped(poly, cat, px):
    """A terrace polygon (cut-out px) cut to the catalog's polygon: (ring, kept fraction of the ring's pixels, inscribed
    diameter um) of the largest piece, or None."""
    P = np.asarray(poly, float).reshape(-1, 2); C = np.asarray(cat, float).reshape(-1, 2)
    lo = np.floor(np.minimum(P.min(0), C.min(0))) - 2
    W, H = (np.ceil(np.maximum(P.max(0), C.max(0)) - lo) + 4).astype(int)
    mp, mc = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    cv2.fillPoly(mp, [np.round(P - lo).astype(np.int32)], 1)
    cv2.fillPoly(mc, [np.round(C - lo).astype(np.int32)], 1)
    n, lab = cv2.connectedComponents(mp & mc)
    if n < 2 or not mp.sum():
        return None
    k = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    m = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ring = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.7, True).reshape(-1, 2).astype(float) + lo
    if len(ring) < 3:
        return None
    dd = 2 * float(cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5).max()) * px
    return ring, float(m.sum()) / float(mp.sum()), dd


def _flake_mask(f, pts):
    """The flake's outline as a mask in a frame holding pts (cut-out px): (mask, frame origin)."""
    O = np.asarray(f["outline"], float).reshape(-1, 2)
    P = np.vstack([O] + [np.asarray(p, float).reshape(-1, 2) for p in pts])
    lo = np.floor(P.min(0)) - 4
    W, H = (np.ceil(P.max(0) - lo) + 8).astype(int)
    F = np.zeros((H, W), np.uint8)
    cv2.fillPoly(F, [np.round(O - lo).astype(np.int32)], 1)
    return F, lo


def wiggle_of(f, poly):
    """ring_wiggle of a terrace polygon (cut-out px) inside the flake's outline; None when it cannot be told."""
    if len(f.get("outline") or []) < 6:
        return None
    F, lo = _flake_mask(f, [poly])
    return ring_wiggle(np.asarray(poly, float).reshape(-1, 2) - lo, F, f["px_um"])


def pick_a(f, a, ex, cat):
    """Terrace a at 100x the biggest region at its level (A_PICK_NM; inside the catalog's polygon cat): when one of the
    other terraces ex is bigger than a's own region, a takes its shape, anchor and thickness (id, screening kept) and
    a's region goes into ex in its place.  Returns True when a changed."""
    if not A_PICK_NM or not cat:
        return False
    px, best = f["px_um"], None
    for j, t in enumerate(ex):
        if t.get("step_nm") is None or abs(t["step_nm"]) > A_PICK_NM:
            continue
        cl = _clipped(t["poly"], cat, px)
        if cl is None:
            continue
        area = t["area"] * cl[1]
        if area > a["area"] and (best is None or area > best[1]):
            w = wiggle_of(f, cl[0])
            if A_WIGGLE_MAX and w is not None and w > A_WIGGLE_MAX:
                continue                                         # (residue's wandering outline)
            best = (j, area, cl)
    if best is None:
        return False
    j, area, (ring, frac, dd) = best
    t = ex.pop(j)
    old = dict(t=a["t"], tlo=a.get("tlo", a["t"]), thi=a.get("thi", a["t"]), alt="",
               layers=a.get("layers") if f.get("material") == "graphite" else None, area=a["area"], d=a["d"],
               L=a["L"], W=a["W"], c=a["c"], poly=a["poly"], quality="terrace_100x", score=None, usable=True,
               thickness_source=f"terrace a's own 100x region before the biggest region at its level became terrace a",
               step_nm=round(-t["step_nm"], 2), fine=[], debris=[], steps=[], wrinkles=[], features={})
    (_, _), (rw, rh), _ = cv2.minAreaRect(ring.astype(np.float32))
    c = t["c"]
    if frac < 0.999:                                             # (cut: the anchor in the kept piece)
        R = np.asarray(ring, np.float32)
        if cv2.pointPolygonTest(R, (float(c[0]) + 0.5, float(c[1]) + 0.5), False) < 0:
            M = cv2.moments(R)
            c = [int(round(M["m10"] / M["m00"])), int(round(M["m01"] / M["m00"]))] if M["m00"] else c
    a.update(poly=[round(v, 1) for v in ring.ravel().tolist()], area=round(area, 1),
             d=round(min(dd, t["d"]) if frac < 0.999 else t["d"], 1), L=round(max(rw, rh) * px, 1),
             W=round(min(rw, rh) * px, 1), c=c, t=t["t"], tlo=t["tlo"], thi=t["thi"],
             thickness_source=t.get("thickness_source"), picked_100x=True)
    if f.get("material") == "graphite":
        a["layers"] = t.get("layers")
    ex.append(old)
    return True


def add_step_terraces(f, img_path, log=print):
    """Replace a designer flake's 100x terraces (quality terrace_100x) with the terraces of its 100x image
    (step_terraces), ids <flake>.b, .c, ... (terrace a and its anchor unchanged); store the crystal directions and the
    step lines on the flake (crystal_deg: edges at crystal_deg + n x 30 deg in the cut-out's frame).  Terrace a's
    shape (poly, area, d, L, W) becomes its own 100x region (a_region: the other terraces cut out of it; Jordan,
    2026-10-22), the catalog's kept as poly_10x / size_10x (and put back first when the terraces are found again: the
    search and the defects measure inside the catalog's polygon, the hBN pick model was fitted on that); its id,
    thickness, anchor and screening stay.  No region to trust (out of focus, no picture, too small): the catalog's
    shape.  Under the flake's material_settings (graphite: its layers kept with the catalog's shape too).  Returns the
    number added."""
    with material_settings(f.get("material")):
        return _add_step_terraces(f, img_path, log)


def _add_step_terraces(f, img_path, log):
    f["terraces"] = [t for t in f["terraces"] if not str(t.get("quality", "")).startswith("terrace_100x")]
    if not f["terraces"]:
        return 0
    a = f["terraces"][0]
    if a.get("poly_10x"):                                        # (found before: the catalog's shape back)
        a.update(a.pop("size_10x", None) or {}, poly=a.pop("poly_10x"))
        a.pop("picked_100x", None)
    a.pop("region_100x", None)
    try:
        ex, info = step_terraces(img_path, f, a)
    except Exception as e:                                       # (one bad picture must not stop the bundle)
        log(f"  {f['id']}: 100x terraces not found ({e})")
        return 0
    reg = info.pop("a_region", None)
    if reg:
        cat = a["poly"]
        keys = ("area", "d", "L", "W", "c", "t", "tlo", "thi", "thickness_source") + (("layers",) if f.get("material") == "graphite" else ())
        a["poly_10x"], a["size_10x"] = cat, {k: a.get(k) for k in keys if k in a}
        w_reg = wiggle_of(f, reg["poly"]) if A_WIGGLE_MAX else None
        a.update(reg)
        if pick_a(f, a, ex, cat):
            log(f"  {f['id']}: terrace a = the biggest region at its level ({a['area']:.0f} um2, {a['t']:g} nm)")
        elif w_reg is not None and w_reg > A_WIGGLE_MAX:          # (a's own region residue-like, nothing better)
            a.update(a.pop("size_10x"), poly=a.pop("poly_10x"))
            a["region_100x"] = f"not used: its outline wanders ({w_reg:.2f}), residue or texture"
            log(f"  {f['id']}: terrace a's 100x region looks like residue (outline {w_reg:.2f}): the catalog's shape")
    base = a["id"].rsplit(".", 1)[0]
    for j, t in enumerate(sorted(ex, key=lambda t: -t["area"])):
        t["id"] = f"{base}.{chr(ord('b') + j)}" if j < 25 else f"{base}.z{j}"
        t["p_keep"], t["tier"] = None, None
        f["terraces"].append(t)
    f.update(info)
    return len(ex)


def _bundle_view(f, designer_dir):
    """The bundle's natural-colour view of a flake (a file of its own, img: name, or a piece of a sheet, img: {s, x, y,
    w, h}): a path, a picture (BGR) or None."""
    v = f.get("img")
    for base in (Path(designer_dir), Path(designer_dir) / "100x"):
        if isinstance(v, str) and (base / v).is_file():
            return base / v
        if isinstance(v, dict) and v.get("s") and (base / v["s"]).is_file():
            s = cv2.imread(str(base / v["s"]))
            if s is not None:
                return s[v["y"]:v["y"] + v["h"], v["x"]:v["x"] + v["w"]].copy()
    return None


def _read_picture(path):
    """A picture (BGR) read whole first (np.fromfile: also a path cv2.imread cannot open, a share), or None."""
    try:
        return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def _scan_pictures(run_dir, scan, cache):
    """(folder, entry) of a scan's 100x pictures that have a <stem>_regions.json, listed once per cache (a dict kept
    over one update): entry = {"names": [picture], "at": {picture: (stage_um, frame_wh)} (read when first needed)}."""
    folder = Path(run_dir) / scan / "100x"
    key = str(folder)
    if key not in cache:
        try:
            names = set(os.listdir(key))
        except OSError:
            names = set()
        pics = sorted(n[:-len("_regions.json")] + ".jpg" for n in names if n.endswith("_regions.json"))
        cache[key] = dict(names=[q for q in pics if q in names], at={})
    return folder, cache[key]


def flake_picture(f, designer_dir, run_dir=None, log=print, cache=None):
    """A bundle flake's own 100x picture: the run's picture <run>/<scan>/100x/<stem>.jpg whose <stem>_regions.json has
    the flake's stage position (and frame size), what a full build reads (the catalog's copy of it, byte for byte), else
    the bundle's natural-colour view of it (_bundle_view: the same pixels re-encoded, JPEG quality 95: the region
    terraces follow such small changes more than the step lines did: on 7-f0009 5 terraces from the view, 2 from the
    run's picture).  The pictures with the flake's number (_F<nnn>_) are looked at first, then the scan's others: a
    picture's number is not always the flake's (in bn_20261020_6 and _7 about half differ: 7-f0009's picture is
    F004_65nm_100x_022), numbers repeat within a scan (one per 10x pass) and a resumed scan is numbered anew
    (20261007_3).  Several at the position (repeated shots): the one with the view's pixels.  cache: a dict kept over
    one update (each scan's folder listed, each _regions.json read once).  A path, a picture (BGR) or None."""
    cache = {} if cache is None else cache
    fr = f.get("frame") or {}
    st = f.get("stage_um")
    view = None
    if run_dir is not None and st:
        scan, num = f["id"].split(":")[-1].rsplit("-f", 1)
        folder, c = _scan_pictures(run_dir, scan, cache)
        tag = "_F%03d_" % int(num)
        hits = []
        for group in ([q for q in c["names"] if tag in q], [q for q in c["names"] if tag not in q]):
            for q in group:
                if q not in c["at"]:
                    try:
                        R = json.loads((folder / (q[:-4] + "_regions.json")).read_text(encoding="utf-8"))
                        c["at"][q] = (R.get("stage_um"), R.get("frame_wh"))
                    except (OSError, ValueError):
                        c["at"][q] = (None, None)
                s_, wh = c["at"][q]
                if s_ and float(np.hypot(*(np.asarray(s_, float) - np.asarray(st, float)))) < 1.0                         and (not fr.get("w") or list(wh or [fr["w"], fr["h"]]) == [fr["w"], fr["h"]]):
                    hits.append(q)
            if hits:
                break
        if len(hits) > 1:
            view = _bundle_view(f, designer_dir)
            v = _read_picture(view) if isinstance(view, Path) else view
            if v is not None:
                def diff(q):
                    a = _read_picture(folder / q)
                    return float(np.abs(a.astype(np.float32) - v).mean()) if a is not None and a.shape == v.shape else 1e9
                hits.sort(key=diff)
            else:
                log(f"  {f['id']}: {len(hits)} 100x pictures at its stage position, the first taken ({hits[0]})")
        if hits:
            return folder / hits[0]
    view = _bundle_view(f, designer_dir) if view is None else view
    if view is None:
        log(f"  {f['id']}: no 100x picture of its own (none in the run at its stage position, no view in the bundle)")
    return view


def update_bundle(designer_dir, run_dir, only=None):
    """Recompute the 100x terraces (TERRACE_MATERIALS) of an existing designer bundle from the flakes' 100x pictures (flake_picture: the
    run's pictures, else the bundle's own views), without rebuilding the bundle's images."""
    p = Path(designer_dir) / "bundle.json"
    b = json.loads(p.read_text(encoding="utf-8"))
    n_new = n_fl = n_view = 0
    cache = {}                                                   # (each scan's folder listed once)
    for f in b["flakes"]:
        if f["material"] not in TERRACE_MATERIALS or (only and only not in f["id"]):
            continue
        img = flake_picture(f, designer_dir, run_dir, cache=cache)
        from_run = isinstance(img, Path) and run_dir is not None and Path(run_dir) in img.parents
        if isinstance(img, Path):
            a = _read_picture(img)
            if a is None and from_run:                           # (unreadable: the view, not no terraces at all)
                print(f"  {f['id']}: {img} could not be read: the bundle's view")
                img, from_run = _bundle_view(f, designer_dir), False
                a = _read_picture(img) if isinstance(img, Path) else img
            img = a
        if img is None:
            continue
        n_view += not from_run
        k = add_step_terraces(f, img)
        if "n_step_lines" in f:                                   # (slim's count of the detail file's lines)
            f["n_step_lines"] = len(f.get("step_lines") or [])
        n_new += k
        n_fl += bool(k)
    # (in one step: a designer reading the run meanwhile gets the old or the new file, never half of one; the step lines
    # go into the bundle itself, which the page prefers to a detail file's, 2026-10-16)
    tmp = p.with_name(".bundle.json.part")
    tmp.write_text(json.dumps(b, separators=(",", ":")), encoding="utf-8")
    import time
    for k in range(10):                                          # (Windows: a virus scanner or indexer may hold it)
        try:
            os.replace(tmp, p)
            break
        except OSError:
            if k == 9:
                raise
            time.sleep(0.5)
    print(f"{p}: {n_new} step terraces on {n_fl} flakes" + (f" ({n_view} from the bundle's views: no run picture)" if n_view else ""))

if __name__ == "__main__":
    raise SystemExit(main())
