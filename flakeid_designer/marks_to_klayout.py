#!/usr/bin/env python3
"""Place microscope pictures of a chip onto its layout by the alignment marks, for designing in KLayout.

    python3 flakeid_designer/marks_to_klayout.py <folder of pictures, or pictures> \
        --gds <your layout>.gds [--add-to existing.lys] [--lyp layers.lyp] [-o out/prefix]

Writes
  <out>.lys          KLayout session: the layout with every picture placed on it (File > Restore Session).  Each
                     picture's landmarks are the marks it was fitted to, so KLayout's own landmark alignment still works.
  <out>.gds          a copy of the layout with extra layers (nothing of the original changed; 253 / 254 with --design):
                       250/k  the marks found in picture k, where the fit puts them (small dots; compare with the marks)
                       251/k  the outline of picture k
                       252/k  flake outlines traced from picture k (only with --outline-max-um-px; off by default)
                       253/0  where gate / sample layers of the design touch with no dielectric between (shorts);
                       253/1  sample - sample contacts (the designer's contact check, whole stack)
                       254/i  outline of design layer i (bottom = 0)
  <out>_check.jpg    each picture with the fitted layout marks drawn on it (green) and the marks found (red)
  <out>_fit.json     per picture: um per pixel, rotation, the marks used and the residuals
  <out>_images/      copies of the pictures the session points to (--no-copy: point to the originals)
With --replace-pictures SESSION.lys only that session changes (its pictures; copies in <SESSION>_images/), plus the
check picture and _fit.json.

How it works.  The fiducials are taken from the layout itself: every small polygon (or group of touching polygons) up
to --max-fiducial-um across, e.g. the two-L marks at (+-45, +-45) um and the squares along the diagonals of the pads
template.  In each picture the gold is found by colour (brighter than the substrate in red and green, much less in
blue), and its compact pieces become candidate marks.  Pairs of candidates are matched with pairs of fiducials whose
distance agrees with the pieces' sizes; the transform (shift, rotation, scale) that explains the most fiducials wins
and is refined by least squares (and, for marks 20 px or larger, by correlating the rendered mark with the picture).
No scale needs to be given.  A picture without enough marks (e.g. 2x or 5x) is placed by matching it to a picture
already placed (SIFT features), and then checked against the fiducials it does show.

Orientation.  Pictures are assumed to be taken upright: only rotations within --max-rot (45 deg) of 0 are
considered, which also settles the pads template's 180 deg symmetry.  A picture taken another way round:
name.jpg@180 (or --rot180 for all).  Mirrored pictures: --mirror.

Layer properties.  The session uses your KLayout .lyp (--lyp, remembered in ~/.flakeid_designer.json, shared with the
designer's "Export to KLayout"), plus entries for this tool's layers and any layer the .lyp does not list.
--add-to SESSION.lys adds the result as a new tab to an existing session (backup copy saved next to it first); close
or reload that session in KLayout afterwards, or KLayout's next save of it overwrites the new tab.
--replace-pictures SESSION.lys swaps the pictures of an existing session for these (e.g. the latest fabrication step's):
only the session's pictures change (backup copy first); its layout, layers and view stay, no GDS or new session is
written.  Close the session in KLayout without saving it first, as for --add-to.

Needs numpy, opencv-python and gdstk (pip3 install gdstk).
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
import xml.sax.saxutils as xu

import numpy as np
import cv2

try:
    import gdstk
except ImportError:          # (the designer imports this module for the live camera's calibration and checks gdstk first)
    gdstk = None

LAYER_FOUND, LAYER_FRAME, LAYER_OUTLINE = 250, 251, 252
LAYER_CONTACT, LAYER_DESIGN = 253, 254       # from a designer stack design (--design): contacts, layer outlines


# ----------------------------------------------------------------------------------------------- layout fiducials

def load_layout(path, cell_name=None):
    if gdstk is None:
        raise RuntimeError("reading the layout needs gdstk (python -m pip install gdstk)")
    lib = gdstk.read_gds(path)
    if cell_name:
        cells = [c for c in lib.cells if c.name == cell_name]
        if not cells:
            sys.exit("no cell %r in %s" % (cell_name, path))
        return lib, cells[0]
    tops = [c for c in lib.top_level() if not c.name.startswith("$$$")]

    def area(c):
        bb = c.bounding_box()
        return 0 if bb is None else (bb[1][0] - bb[0][0]) * (bb[1][1] - bb[0][1])
    return lib, max(tops, key=area)


def layout_fiducials(top, layers=None, max_um=60.0, min_um=1.0):
    """Small polygons (touching ones grouped) -> list of dict(c, size, fill, kind, polys)."""
    polys = []
    for p in top.get_polygons():
        if layers and p.layer not in layers:
            continue
        (x0, y0), (x1, y1) = p.bounding_box()
        if max(x1 - x0, y1 - y0) <= max_um:
            polys.append((np.array([x0, y0, x1, y1]), p))
    # group polygons whose boxes touch (a mark drawn as two Ls touching at a corner is one fiducial)
    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    boxes = np.array([b for b, _ in polys]) if polys else np.zeros((0, 4))
    for i in range(len(polys)):
        b = boxes[i]
        near = np.nonzero((boxes[:, 0] <= b[2] + 1e-3) & (boxes[:, 2] >= b[0] - 1e-3) &
                          (boxes[:, 1] <= b[3] + 1e-3) & (boxes[:, 3] >= b[1] - 1e-3))[0]
        for j in near:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(len(polys)):
        groups.setdefault(find(i), []).append(i)
    feats, seen = [], set()
    for idx in groups.values():
        b = boxes[idx]
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        w, h = x1 - x0, y1 - y0
        if min(w, h) < min_um or max(w, h) > max_um or not 0.6 < w / h < 1.67:
            continue
        c = ((x0 + x1) / 2, (y0 + y1) / 2)
        key = (round(c[0], 2), round(c[1], 2))
        if key in seen:                       # the same shape drawn more than once
            continue
        seen.add(key)
        area = sum(abs(polys[i][1].area()) for i in idx)
        fill = min(1.0, area / (w * h))
        feats.append(dict(c=np.array(c), size=float(max(w, h)), fill=fill,
                          kind="square" if fill > 0.85 else "mark", polys=[polys[i][1].points for i in idx]))
    return feats


# --------------------------------------------------------------------------------------------- picture handling

def template_fiducials(gds, cell=None, layers=None, max_um=60.0):
    """The layout's alignment marks: (fiducials, (cell name, mark layers)).  Without `layers`: the chip's metal layer
    (the one with the most metal among layers holding at least 2 small shapes); a device design on other layers
    (contacts, gates) is not on the chip yet when the pictures are taken."""
    lib, top = load_layout(gds, cell)
    if not layers:
        area, has = {}, {}
        for p in top.get_polygons():
            area[p.layer] = area.get(p.layer, 0.0) + abs(p.area())
        for l in area:
            has[l] = len(layout_fiducials(top, {l}, max_um))
        cand = [l for l in area if has[l] >= 2]
        layers = {max(cand, key=lambda l: area[l])} if cand else None
    return layout_fiducials(top, layers, max_um), (top.name, layers)


def fit_good(fit):
    """A fit good enough to place a picture (or calibrate a camera) by its marks."""
    return fit is not None and fit["rms"] < 3 * fit["sim"].scale + 0.5 and (
        fit["n_resolved"] >= 3 or (fit["n_resolved"] == 2 and (fit["shape"] or 0) > 0.5) or fit["n"] >= 6)


def calibrate_frame(im, tpl, mirror=False, um_guess=None, rot_guess=None, narrow=False):
    """A camera's scale from one picture of a chip's alignment marks (the live view's "Calibrate from marks"): the
    marks found and matched to the layout, any rotation.  Returns a dict: ok, um_per_px, rotation_deg (the picture
    against the layout, counterclockwise), n_marks, n_resolved, rms_um, max_um, why (when not ok), check (the
    picture with the layout drawn in green and the marks used circled in red, BGR)."""
    fit = None
    if narrow and um_guess and rot_guess is not None:
        # another objective of this camera is calibrated: the same rotation, its scale x the magnification ratio; a
        # narrow search first (a fraction of the time: at 5x the whole template is in view), the full one if it fails
        fit = fit_picture(im, tpl, mirror, float(rot_guess), smin=um_guess / 1.4, smax=um_guess * 1.4, max_rot=8.0,
                          log=lambda *a: None, max_cand=24)
    if fit is None:
        smin, smax = (um_guess / 4.0, um_guess * 4.0) if um_guess else (0.01, 30.0)
        # (24 candidates: the largest pieces; at 5x the whole template is in view and 60 took ~25 s for the same result)
        fit = fit_picture(im, tpl, mirror, 0.0, smin=smin, smax=smax, max_rot=180.0, log=lambda *a: None, max_cand=24)
    if fit is None:
        return dict(ok=False, why="no two marks of the layout found in the picture (are they in focus and in view?)")
    sim = fit["sim"]
    n_l = sum(1 for i, d in enumerate(fit["assign"]) if d >= 0 and tpl[i]["kind"] == "mark")
    # a camera's scale needs more than placing a picture does: 3 evenly spaced squares also fit at twice the scale (a
    # test at 50x gave 2x the true scale that way), so at least 2 of the two-L marks, or 6 marks in all
    out = dict(ok=fit_good(fit) and (n_l >= 2 or fit["n"] >= 6), um_per_px=sim.scale, rotation_deg=sim.angle, n_marks=fit["n"],
               n_resolved=fit["n_resolved"], n_l_marks=n_l, rms_um=fit["rms"], max_um=fit["max"])
    if not out["ok"]:
        out["why"] = ("only %d marks matched (%d of them two-L marks), rms %.2f um: bring the four two-L marks at the middle of the "
                      "chip into view (at 5x-10x: more of the chip), in focus" % (fit["n"], n_l, fit["rms"]))
    fit["name"] = "live"
    out["check"] = check_picture(im, fit, tpl)
    return out


def read_picture(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit("cannot read %s" % path)
    if im.ndim == 2 and im.dtype == np.uint16:      # raw scan-camera frame (Bayer, see flakeid/mm.py)
        im = cv2.cvtColor(im, cv2.COLOR_BayerBG2BGR)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    if im.dtype != np.uint8:
        f = im.astype(np.float32)
        lo, hi = np.percentile(f, 0.5), np.percentile(f, 99.9)
        im = np.clip((f - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im


def gold_map(im, bright=1.45, diff=0.4):
    """Gold mask and a continuous gold score (0..1), from colour relative to the substrate (median colour)."""
    f = im.astype(np.float32) + 1.0
    sub = np.median(f.reshape(-1, 3), axis=0)
    r = f / sub
    b, g, rr = r[..., 0], r[..., 1], r[..., 2]
    gray = float(np.ptp(sub)) < 0.05 * float(sub.mean()) and float(np.abs(np.diff(im[::7, ::7].astype(int), axis=2)).max()) < 3
    if gray:                                          # monochrome picture: gold is just the bright part
        lum = r.mean(axis=2)
        score = np.clip((lum - 1.0) / (bright - 1.0), 0, 2) / 2
    else:
        lo = np.minimum(rr, g)
        score = np.clip(np.minimum((lo - 1.0) / (bright - 1.0), (lo - b) / diff), 0, 2) / 2
        score[b < 0.6] = 0            # pure yellow (Stream's text overlay) has almost no blue; gold keeps the substrate's
    return score >= 0.5, score, sub


def candidates(mask, max_frac=0.3):
    """Compact gold pieces -> list of (x, y, size, fill, kind) in file pixels, largest first.  Pieces whose boxes
    touch (a two-L mark that came out as two Ls) are grouped first."""
    h, w = mask.shape
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    lim = max_frac * min(h, w)
    idx = [i for i in range(1, n) if max(st[i, 2], st[i, 3]) <= lim]
    if not idx:
        return []
    b = np.array([[st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]] for i in idx], float)
    parent = list(range(len(idx)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(idx)):
        g = max(1.5, 0.04 * max(b[i, 2] - b[i, 0], b[i, 3] - b[i, 1]))
        near = np.nonzero((b[:, 0] <= b[i, 2] + g) & (b[:, 2] >= b[i, 0] - g) &
                          (b[:, 1] <= b[i, 3] + g) & (b[:, 3] >= b[i, 1] - g))[0]
        for j in near:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(len(idx)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for gi in groups.values():
        x0, y0 = b[gi, 0].min(), b[gi, 1].min()
        x1, y1 = b[gi, 2].max(), b[gi, 3].max()
        bw, bh = x1 - x0, y1 - y0
        size = max(bw, bh)
        if size < 2 or size > lim or not 0.6 <= bw / bh <= 1.67:
            continue
        if x0 <= 0 or y0 <= 0 or x1 >= w or y1 >= h:          # cut by the picture edge
            continue
        area = float(sum(st[idx[i], 4] for i in gi))
        cx = sum(cen[idx[i], 0] * st[idx[i], 4] for i in gi) / area
        cy = sum(cen[idx[i], 1] * st[idx[i], 4] for i in gi) / area
        fill = area / float(bw * bh)
        if size >= 10:
            if fill < 0.12:
                continue
            kind = "square" if fill > 0.7 else ("mark" if fill < 0.55 else "any")
        else:
            kind = "any"
        out.append((cx, cy, size, fill, kind))
    out.sort(key=lambda t: -t[2])
    return out


def to_uv(xy, w, h):
    """File pixel (x right, y down, pixel centres at integers) -> KLayout image pixel (origin at the centre, y up)."""
    xy = np.asarray(xy, float)
    return np.stack([xy[..., 0] + 0.5 - w / 2.0, h / 2.0 - 0.5 - xy[..., 1]], axis=-1)


def from_uv(uv, w, h):
    uv = np.asarray(uv, float)
    return np.stack([uv[..., 0] - 0.5 + w / 2.0, h / 2.0 - 0.5 - uv[..., 1]], axis=-1)


# ------------------------------------------------------------------------------------------------ the transform
# um = A @ (m * u, v) + t with A = scale * rotation (a complex number a), m = -1 for a mirrored picture.

class Sim(object):
    def __init__(self, a, t, mirror=False):
        self.a, self.t, self.mirror = complex(a), np.asarray(t, float), mirror

    def _m(self, uv):
        uv = np.array(uv, float)
        if self.mirror:
            uv[..., 0] *= -1
        return uv

    def fwd(self, uv):
        uv = self._m(uv)
        z = (uv[..., 0] + 1j * uv[..., 1]) * self.a
        return np.stack([z.real, z.imag], axis=-1) + self.t

    def inv(self, xy):
        z = ((np.asarray(xy, float) - self.t) @ np.array([1, 1j])) / self.a
        return self._m(np.stack([z.real, z.imag], axis=-1))

    @property
    def scale(self):
        return abs(self.a)

    @property
    def angle(self):
        return math.degrees(math.atan2(self.a.imag, self.a.real))

    def matrix(self):
        a, m = self.a, (-1 if self.mirror else 1)
        return [[a.real * m, -a.imag, self.t[0]], [a.imag * m, a.real, self.t[1]], [0, 0, 1]]


def fit_sim(uv, xy, mirror=False):
    uv = np.array(uv, float)
    if mirror:
        uv[:, 0] *= -1
    pc = uv @ np.array([1, 1j])
    qc = np.asarray(xy, float) @ np.array([1, 1j])
    mp, mq = pc.mean(), qc.mean()
    a = (np.conj(pc - mp) * (qc - mq)).sum() / (np.abs(pc - mp) ** 2).sum()
    t = mq - a * mp
    return Sim(a, [t.real, t.imag], mirror)


# --------------------------------------------------------------------------------------------- matching

def kinds_ok(dk, tk):
    return dk == "any" or dk == tk


def kind_table(tpl, det_kind):
    return np.array([[kinds_ok(dk, f["kind"]) for dk in det_kind] for f in tpl], bool).reshape(len(tpl), len(det_kind))


def match(sim, det_uv, det_size, det_kind, tpl, w, h, kk=None):
    """Each fiducial -> the candidate it explains (or -1); score = matched - 0.25 * clearly visible but missing."""
    if kk is None:
        kk = kind_table(tpl, det_kind)
    pred = sim.inv(np.array([f["c"] for f in tpl]))
    psize = np.array([f["size"] for f in tpl]) / sim.scale
    inside = (np.abs(pred[:, 0]) < w / 2 - psize / 2) & (np.abs(pred[:, 1]) < h / 2 - psize / 2)
    d = np.hypot(pred[:, None, 0] - det_uv[None, :, 0], pred[:, None, 1] - det_uv[None, :, 1])
    tol = np.maximum(2.0, 0.25 * psize)[:, None]
    ok = (d < tol) & (det_size[None, :] > 0.5 * psize[:, None] - 2) & (det_size[None, :] < 1.6 * psize[:, None] + 2)
    ok &= kk
    d = np.where(ok, d, np.inf)
    assign = np.full(len(tpl), -1)
    used = set()
    for ti in np.argsort(d.min(axis=1)):
        if not np.isfinite(d[ti].min()):
            break
        for di in np.argsort(d[ti]):
            if not np.isfinite(d[ti, di]):
                break
            if di not in used:
                assign[ti] = di
                used.add(di)
                break
    matched = assign >= 0
    missing = inside & ~matched & (psize >= 6)
    # a resolved mark (>= 10 px) is real evidence; a speck of a few pixels fits by chance far more easily
    wt = np.where(det_size[assign[matched]] >= 10, 1.0, 0.35) if matched.any() else np.zeros(0)
    return assign, float(wt.sum() - 0.25 * missing.sum()), pred, psize


def shape_score(sim, tpl, assign, mask, w, h):
    """Mean IoU of the rendered fiducials (marks >= 12 px) with the gold mask: tells the mark's orientation."""
    ious = []
    for ti, f in enumerate(tpl):
        if assign[ti] < 0 or f["kind"] != "mark" or f["size"] / sim.scale < 12:
            continue
        r = np.zeros(mask.shape, np.uint8)
        for p in f["polys"]:
            px = from_uv(sim.inv(p), w, h)
            cv2.fillPoly(r, [np.round(px * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
        c = from_uv(sim.inv(f["c"][None]), w, h)[0]
        s = int(f["size"] / sim.scale * 0.8) + 3
        x0, x1 = max(int(c[0]) - s, 0), min(int(c[0]) + s + 1, w)
        y0, y1 = max(int(c[1]) - s, 0), min(int(c[1]) + s + 1, h)
        a, b = r[y0:y1, x0:x1] > 0, mask[y0:y1, x0:x1]
        u = (a | b).sum()
        if u:
            ious.append((a & b).sum() / float(u))
    return float(np.mean(ious)) if ious else None


def refine_marks(sim, tpl, assign, score, w, h, det_uv):
    """Shift each matched mark (>= 20 px) to where the rendered mark correlates best with the gold score."""
    uv = det_uv.copy()
    for ti, f in enumerate(tpl):
        di = assign[ti]
        if di < 0 or f["size"] / sim.scale < 20:
            continue
        ps = f["size"] / sim.scale
        c = from_uv(sim.inv(f["c"][None]), w, h)[0]
        half = int(ps * 0.75) + 2
        srch = int(max(3, 0.15 * ps))
        ss = 4
        cx, cy = int(round(c[0])), int(round(c[1]))
        tplim = np.zeros((2 * half * ss + ss, 2 * half * ss + ss), np.float32)
        off = np.array([cx - half, cy - half], float)        # template pixel k <-> picture pixel cx - half + k
        for p in f["polys"]:
            px = (from_uv(sim.inv(p), w, h) - off + 0.5) * ss
            cv2.fillPoly(tplim, [np.round(px * 4).astype(np.int32)], 1.0, lineType=cv2.LINE_8, shift=2)
        tplim = cv2.resize(tplim, (2 * half + 1, 2 * half + 1), interpolation=cv2.INTER_AREA)
        x0, y0 = cx - half - srch, cy - half - srch
        x1, y1 = cx + half + srch + 1, cy + half + srch + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        win = score[y0:y1, x0:x1].astype(np.float32)
        res = cv2.matchTemplate(win, tplim, cv2.TM_CCOEFF_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(res)
        if best < 0.5:
            continue

        def sub(v0, vm, vp):
            den = vm - 2 * v0 + vp
            return 0.0 if den >= 0 else 0.5 * (vm - vp) / den
        dx = sub(res[by, bx], res[by, bx - 1], res[by, bx + 1]) if 0 < bx < res.shape[1] - 1 else 0.0
        dy = sub(res[by, bx], res[by - 1, bx], res[by + 1, bx]) if 0 < by < res.shape[0] - 1 else 0.0
        shift = np.array([bx + dx - srch, by + dy - srch])
        uv[di] = to_uv((c + shift)[None], w, h)[0]
    return uv


def angdiff(a, b):
    return abs(((a - b + 180) % 360) - 180)


def fit_picture(im, tpl, mirror=False, expect_deg=0.0, smin=0.01, smax=30.0, max_cand=60, log=print, max_rot=45.0):
    h, w = im.shape[:2]
    mask, score, _ = gold_map(im)
    cand = candidates(mask)[:max_cand]
    if len(cand) < 2:
        return None
    det_xy = np.array([c[:2] for c in cand])
    det_uv = to_uv(det_xy, w, h)
    det_size = np.array([c[2] for c in cand], float)
    det_kind = [c[4] for c in cand]
    T = np.array([f["c"] for f in tpl])
    tsize = np.array([f["size"] for f in tpl])
    tkind = [f["kind"] for f in tpl]
    kk = kind_table(tpl, det_kind)
    nt = len(tpl)
    ia, ib = np.nonzero(~np.eye(nt, dtype=bool))
    tvec = (T[ib] - T[ia]) @ np.array([1, 1j])
    tdist = np.abs(tvec)
    m = -1.0 if mirror else 1.0

    def srange(size_px, size_um):
        tol = 1.5 + 0.15 * size_px
        return size_um / (size_px + tol), size_um / max(size_px - tol, 0.7)

    hyps = []
    nd = len(cand)
    for i in range(nd):
        for j in range(i + 1, nd):
            pv = (det_uv[j, 0] * m - det_uv[i, 0] * m) + 1j * (det_uv[j, 1] - det_uv[i, 1])
            dp = abs(pv)
            if dp < 5:
                continue
            s = tdist / dp
            lo_a, hi_a = srange(det_size[i], tsize[ia])
            lo_b, hi_b = srange(det_size[j], tsize[ib])
            ok = (s >= np.maximum(lo_a, lo_b)) & (s <= np.minimum(hi_a, hi_b)) & (s >= smin) & (s <= smax)
            for k in np.nonzero(ok)[0]:
                if not (kinds_ok(det_kind[i], tkind[ia[k]]) and kinds_ok(det_kind[j], tkind[ib[k]])):
                    continue
                a = tvec[k] / pv
                if angdiff(math.degrees(np.angle(a)), expect_deg) > max_rot + 2:     # pictures are taken upright
                    continue
                p0 = det_uv[i, 0] * m + 1j * det_uv[i, 1]
                t = T[ia[k]] @ np.array([1, 1j]) - a * p0
                hyps.append((a, t))
    if not hyps:
        return None
    # score every hypothesis (dedupe near-identical ones first)
    seen, scored = set(), []
    for a, t in hyps:
        key = (round(math.degrees(np.angle(a)) / 2), round(math.log(abs(a)) * 50), round(t.real / (abs(a) * 4)), round(t.imag / (abs(a) * 4)))
        if key in seen:
            continue
        seen.add(key)
        sim = Sim(a, [t.real, t.imag], mirror)
        assign, sc, _, _ = match(sim, det_uv, det_size, det_kind, tpl, w, h, kk)
        if (assign >= 0).sum() >= 2:
            scored.append((sc, sim, assign))
    if not scored:
        return None
    best = max(s[0] for s in scored)
    # all solutions as good as the best (symmetric layouts give several); refine each by least squares
    sols = []
    for sc, sim, assign in scored:
        if sc < best - 0.01:
            continue
        sel = assign >= 0
        sim = fit_sim(det_uv[assign[sel]], T[sel], mirror)
        if angdiff(sim.angle, expect_deg) > max_rot:
            continue
        assign, sc2, _, _ = match(sim, det_uv, det_size, det_kind, tpl, w, h, kk)
        if any(abs(((sim.angle - o.angle + 180) % 360) - 180) < 3 and abs(sim.scale / o.scale - 1) < 0.02 and
               np.hypot(*(sim.t - o.t)) < 10 * sim.scale for _, o, _ in sols):
            continue
        sols.append((sc2, sim, assign))
    if not sols:
        return None
    # orientation: mark shape first, then the expected rotation
    ranked = []
    for sc, sim, assign in sols:
        shp = shape_score(sim, tpl, assign, mask, w, h)
        dang = abs(((sim.angle - expect_deg + 180) % 360) - 180)
        ranked.append((sc, shp, dang, sim, assign))
    top_sc = max(r[0] for r in ranked)
    ranked = [r for r in ranked if r[0] >= top_sc - 0.01]
    shapes = [r[1] for r in ranked if r[1] is not None]
    if shapes:
        smax_ = max(shapes)
        ranked = [r for r in ranked if r[1] is not None and r[1] >= smax_ - 0.08]
    ranked.sort(key=lambda r: r[2])
    sc, shp, dang, sim, assign = ranked[0]
    ambiguous = sorted(set(round(r[3].angle) for r in ranked))
    # refine: mark correlation, then least squares, twice
    for _ in range(2):
        uv = refine_marks(sim, tpl, assign, score, w, h, det_uv)
        sel = assign >= 0
        if sel.sum() < 2:
            break
        sim = fit_sim(uv[assign[sel]], T[sel], mirror)
        assign2, sc, _, _ = match(sim, uv, det_size, det_kind, tpl, w, h, kk)
        if (assign2 >= 0).sum() >= sel.sum():
            assign, det_uv = assign2, uv
    sel = assign >= 0
    res = np.hypot(*(sim.fwd(det_uv[assign[sel]]) - T[sel]).T)
    return dict(sim=sim, assign=assign, det_uv=det_uv, det_size=det_size, det_kind=det_kind, n=int(sel.sum()),
                score=sc, shape=shp, rms=float(np.sqrt(np.mean(res ** 2))), max=float(res.max()), mask=mask,
                n_cand=len(cand), n_resolved=int((det_size[assign[sel]] >= 10).sum()))


# ----------------------------------------------------------------------------------- chaining to a placed picture

def overlay_free(im):
    """Mask (255 = use) without the camera software's overlay: yellow text and the white scale-bar box."""
    f = im.astype(np.float32) + 1.0
    sub = np.median(f.reshape(-1, 3), axis=0)
    r = f / sub
    yellow = (np.minimum(r[..., 1], r[..., 2]) > 1.3) & (r[..., 0] < 0.6)
    white = (f.min(axis=2) > 225)
    bad = cv2.dilate((yellow | white).astype(np.uint8), np.ones((25, 25), np.uint8))
    return np.where(bad > 0, 0, 255).astype(np.uint8)


def sift_place(im, ref_im, ref_sim, mirror=False):
    """Place `im` by SIFT matching to an already placed picture; returns a Sim or None."""
    sift = cv2.SIFT_create(nfeatures=4000)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
    ma, mb = overlay_free(im), overlay_free(ref_im)
    h, w = im.shape[:2]
    best = None
    for f_new in (1.0, 0.5, 0.25, 0.125):
        for f_ref in (1.0, 0.5, 0.25, 0.125):
            if f_new < 1 and f_ref < 1:
                continue
            a, am, b, bm = g, ma, gr, mb
            if f_new < 1:
                a = cv2.resize(g, None, fx=f_new, fy=f_new, interpolation=cv2.INTER_AREA)
                am = cv2.resize(ma, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
            if f_ref < 1:
                b = cv2.resize(gr, None, fx=f_ref, fy=f_ref, interpolation=cv2.INTER_AREA)
                bm = cv2.resize(mb, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)
            ka, da = sift.detectAndCompute(a, am)
            kb, db = sift.detectAndCompute(b, bm)
            if da is None or db is None or len(ka) < 8 or len(kb) < 8:
                continue
            mt = cv2.BFMatcher().knnMatch(da, db, k=2)
            good = [p[0] for p in mt if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) < 8:
                continue
            pa = np.float32([ka[x.queryIdx].pt for x in good]) / f_new
            pb = np.float32([kb[x.trainIdx].pt for x in good]) / f_ref
            M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0 / f_ref)
            if M is None:
                continue
            n = int(inl.sum())
            pin = pa[inl.ravel() > 0]
            if n < 12 or np.ptp(pin[:, 0]) < 0.2 * w or np.ptp(pin[:, 1]) < 0.2 * h:     # matches all in one corner
                continue
            if best is None or n > best[0]:
                best = (n, M, pa[inl.ravel() > 0], pb[inl.ravel() > 0])
    if best is None or best[0] < 12:
        return None, 0
    n, M, pa, pb = best
    hr, wr = ref_im.shape[:2]
    xy = ref_sim.fwd(to_uv(pb, wr, hr))
    sim = fit_sim(to_uv(pa, w, h), xy, mirror)
    if not 1e-4 < sim.scale / ref_sim.scale < 1e4:
        return None, 0
    return sim, n


def _norm(g):
    g = g.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), 25)
    return g / (g.std() + 1e-6)


def ncc_place(im, ref_im, ref_sim, rot_deg=0.0, mirror=False, log=None):
    """Place a picture inside a coarser placed one: search the scale ratio by normalized correlation (same rotation as
    the reference, plus rot_deg), then refine shift / rotation / scale by ECC.  Returns (Sim, correlation) or (None, 0)."""
    g = _norm(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    gr = _norm(cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY))
    h, w = g.shape
    hr, wr = gr.shape
    if rot_deg % 360:
        M0 = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_deg, 1.0)
        g = cv2.warpAffine(g, M0, (w, h))
    best = None
    for r in np.geomspace(0.03, 0.9, 90):          # new pixel size / ref pixel size
        tw, th = int(round(w * r)), int(round(h * r))
        if tw < 24 or th < 18 or tw >= wr or th >= hr:
            continue
        t = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gr, t, cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(res)
        if best is None or v > best[0]:
            best = (v, r, loc)
    if best is None or best[0] < 0.35:
        return None, 0.0
    v, r, (x0, y0) = best
    # ECC refinement at the reference's resolution: warp maps template (new, scaled by r) into the reference
    t = cv2.resize(g, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_AREA)
    warp = np.array([[1, 0, x0], [0, 1, y0]], np.float32)
    try:
        _, warp = cv2.findTransformECC(t, gr, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        pass
    # points of the new picture (file px) -> reference file px
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [w / 2.0, h / 2.0]], np.float64)
    pts = src
    if rot_deg % 360:
        pts = np.hstack([pts, np.ones((len(pts), 1))]) @ M0.T
    pts = pts * r
    ref_px = np.hstack([pts, np.ones((len(pts), 1))]) @ warp.astype(np.float64).T
    sim = fit_sim(to_uv(src, w, h), ref_sim.fwd(to_uv(ref_px, wr, hr)), mirror)
    if log:
        log("    correlation %.2f at %.3f x the reference's pixel size" % (v, r))
    return sim, float(v)


# ---------------------------------------------------------------------------------------------- flake outlines

def flake_outlines(im, gold, sub, um_px, min_area_um2=4.0, tol=0.12):
    f = im.astype(np.float32) + 1.0
    r = f / sub
    dev = np.abs(r - 1.0).max(axis=2)
    white = (r.min(axis=2) > 1.5) & (np.ptp(f, axis=2) < 40)            # scale-bar box and the like
    m = (dev > tol) & ~cv2.dilate(gold.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~white
    m = m.astype(np.uint8)
    k = int(min(15, max(1, round(0.3 / um_px))))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if cv2.contourArea(c) * um_px ** 2 < min_area_um2:
            continue
        c = cv2.approxPolyDP(c, max(0.7, 0.15 / um_px), True)
        if len(c) >= 3:
            out.append(c.reshape(-1, 2).astype(float))
    return out


# --------------------------------------------------------------------------------------------------- outputs

def write_gds_quietly(lib, path):
    """gdstk warns (on stderr, from C) about KLayout's long context properties; they are KLayout's own and fine."""
    import warnings
    sys.stderr.flush()
    fd = os.dup(2)
    try:
        with open(os.devnull, "w") as dn:
            os.dup2(dn.fileno(), 2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lib.write_gds(path, max_points=8000)   # default 199 cuts flake outlines into strips
    finally:
        os.dup2(fd, 2)
        os.close(fd)


PALETTE = ["#ffcc00", "#ff80a8", "#80ff8d", "#c080ff", "#ff9d9d", "#80a8ff", "#ffc280", "#9dff80", "#ff8000", "#00c0ff"]
CONFIG = os.path.join(os.path.expanduser("~"), ".flakeid_designer.json")      # shared with the designer ("klayout" key)


def config_get(key, default=None):
    try:
        with open(CONFIG, encoding="utf-8") as fh:
            return (json.load(fh).get("klayout") or {}).get(key, default)
    except (OSError, ValueError, AttributeError):
        return default


def config_set(**kw):
    """Read-modify-write: the designer keeps its libraries in the same file."""
    try:
        try:
            with open(CONFIG, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except OSError:
            cfg = {}
        cfg.setdefault("klayout", {}).update(kw)
        tmp = CONFIG + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=1)
        os.replace(tmp, CONFIG)
    except (OSError, ValueError):
        pass


def design_to_chip(design, fit, im_shape):
    """Designer world (um, the design's frame) -> chip um, through the design's stack picture: world -> stored picture
    pixel (inverse of the designer's placement) -> this picture's pixel -> KLayout pixel -> chip (the marks fit)."""
    g = design["simg"]
    h, w = im_shape[:2]
    k = w / float(g["w"])                      # the designer may keep a reduced copy of the picture
    th = math.radians(g.get("rot", 0.0))
    c, sn = math.cos(-th), math.sin(-th)
    mir = -1.0 if g.get("mirror") else 1.0

    def f(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        d = pts - np.array([g["tx"], g["ty"]], float)
        x = (c * d[:, 0] - sn * d[:, 1]) * mir / g["s"] + g["w"] / 2.0
        y = (sn * d[:, 0] + c * d[:, 1]) / g["s"] + g["h"] / 2.0
        px = np.stack([x * k - 0.5, y * k - 0.5], axis=1)     # canvas pixel edges -> pixel centres at integers
        return fit["sim"].fwd(to_uv(px, w, h))
    return f


def design_shapes(design, fn):
    """-> list of (layer, datatype, polygon in chip um, name, colour) for the contacts and the design's layers."""
    import base64
    out = []
    ct = design.get("contacts")
    if ct and ct.get("hit_b64"):
        W, H, res = int(ct["W"]), int(ct["H"]), float(ct["res"])
        hit = np.frombuffer(base64.b64decode(ct["hit_b64"]), np.uint8)[:W * H].reshape(H, W)
        for val, dt, nm, col in ((2, 0, "contact: short (gate involved)", "#ff2d6f"),
                                 (1, 1, "contact: sample - sample", "#8f6bff")):
            m = (hit == val).astype(np.uint8)
            if not m.any():
                continue
            m = np.pad(m, 1)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cs:
                cnt = cnt.reshape(-1, 2).astype(float) - 1
                if len(cnt) < 3:
                    continue
                world = np.stack([ct["bb"][0] + (cnt[:, 0] + 0.5) * res, ct["bb"][1] + (cnt[:, 1] + 0.5) * res], axis=1)
                out.append((LAYER_CONTACT, dt, fn(world), nm, col))
    for i, L in enumerate(design.get("layers") or []):
        poly = L.get("poly") or []
        if len(poly) >= 6:
            out.append((LAYER_DESIGN, i, fn(np.array(poly, float).reshape(-1, 2)),
                        "design %d: %s (%s)" % (i + 1, L.get("name", ""), L.get("role", "")), PALETTE[i % len(PALETTE)]))
    return out


def own_props(layers):
    """Outline-only layer entries (so the pictures show through); layers = [(layer, datatype, name, colour)].
    The export's own helper layers (marks found, picture frames, flake outlines) start invalid (greyed, not drawn)."""
    e = xu.escape
    out = []
    for l, d, name, col in layers:
        valid = "false" if l in (LAYER_FOUND, LAYER_FRAME, LAYER_OUTLINE) else "true"
        out.append("""<properties>
     <frame-color>%s</frame-color>
     <fill-color>%s</fill-color>
     <frame-brightness>0</frame-brightness>
     <fill-brightness>0</fill-brightness>
     <dither-pattern>I1</dither-pattern>
     <line-style/>
     <valid>%s</valid>
     <visible>true</visible>
     <transparent>false</transparent>
     <width>%d</width>
     <marked>false</marked>
     <xfill>false</xfill>
     <animation>0</animation>
     <name>%s</name>
     <source>%d/%d@1</source>
    </properties>""" % (col, col, valid, 2 if l in (LAYER_OUTLINE, LAYER_CONTACT) else 1, e(name), l, d))
    return out


def read_lyp(path):
    """A KLayout .lyp -> list of tabs (name, [<properties> XML]) and the (layer, datatype) pairs it covers."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    tabs_el = root.findall("layer-properties") if root.tag == "layer-properties-tabs" else [root]
    tabs, covered = [], set()
    for t in tabs_el:
        props = [ET.tostring(c, encoding="unicode").strip() for c in t if c.tag == "properties"]
        tabs.append((t.findtext("name") or "", props))
        for src in t.iter("source"):
            m = re.match(r"\s*(\d+)/(\d+)", src.text or "")
            if m:
                covered.add((int(m.group(1)), int(m.group(2))))
    return tabs, covered


def tabs_xml(tabs):
    e = xu.escape
    return "\n".join("   <layer-properties>\n    %s\n    %s\n   </layer-properties>" % (
        "\n    ".join(props), "<name>%s</name>" % e(name) if name else "<name/>") for name, props in tabs)


def layout_xml(name, gds_path):
    e = xu.escape
    return """ <layout>
  <name>%s</name>
  <file-path>%s</file-path>
  <save-options-valid>false</save-options-valid>
 </layout>
""" % (e(name), e(os.path.abspath(gds_path)))


def image_annotations(images):
    """The session's <annotation> entries placing each picture (img::Object)."""
    e = xu.escape
    anns = []
    for im in images:
        mtx = im["sim"].matrix()
        # (plain floats: numpy 2 prints a numpy number as "np.float64(...)", which KLayout cannot read, 2026-10-14)
        mstr = "(%r,%r,%r) (%r,%r,%r) (0,0,1)" % tuple(round(float(v), 12) for row in mtx[:2] for v in row)
        lm = ",".join("%r" % round(float(v), 6) for p in im["landmarks"] for v in p)
        val = ("color:matrix=%s;min_value=0;max_value=255;is_visible=true;z_position=%d;brightness=0;contrast=0;"
               "gamma=1;red_gain=1;green_gain=1;blue_gain=1;landmarks=[%s];color_mapping=[0,'#000000';1,'#ffffff';];"
               "file='%s'" % (mstr, im["z"], lm, im["session_path"].replace("\\", "/").replace("'", "\\'")))
        anns.append("   <annotation>\n    <class>img::Object</class>\n    <value>%s</value>\n   </annotation>" % e(val))
    return anns


def view_xml(name, title, cell, images, view_box, layer_tabs):
    e = xu.escape
    anns = image_annotations(images)
    x0, y0, x1, y1 = (float(v) for v in view_box)
    return """ <view>
  <title>%s</title>
  <active-cellview-index>0</active-cellview-index>
  <display>
   <x-left>%r</x-left>
   <x-right>%r</x-right>
   <y-bottom>%r</y-bottom>
   <y-top>%r</y-top>
   <min-hier>0</min-hier>
   <max-hier>10</max-hier>
   <cellpaths>
    <cellpath>
     <cellname>%s</cellname>
    </cellpath>
   </cellpaths>
  </display>
  <cellviews>
   <cellview>
    <layout-ref>%s</layout-ref>
    <tech-name/>
    <hidden-cells>
    </hidden-cells>
   </cellview>
  </cellviews>
  <bookmarks>
  </bookmarks>
  <current-layer-property-tab>0</current-layer-property-tab>
  <layer-properties-tabs>
%s
  </layer-properties-tabs>
  <annotations>
%s
  </annotations>
 </view>
""" % (e(title), x0, x1, y0, y1, e(cell), e(name), layer_tabs, "\n".join(anns))


def session_layout_names(text):
    return set(xu.unescape(n) for n in re.findall(r"<layout>\s*<name>(.*?)</name>", text, re.S))


def keep_existing(path):
    """An export about to replace a file (e.g. DQD11.gds saved from KLayout after drawing on it): a dated copy of it
    first, <file>.bak-<date>-<time> (2026-10-21: the files of an export were overwritten without one)."""
    if os.path.isfile(path):
        backup = "%s.bak-%s" % (path, time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(path, backup)
        print("%s existed: kept as %s" % (os.path.basename(path), os.path.basename(backup)))


def add_to_session(lys, layout_block, view_block):
    """Append a layout and a view (a new tab) to an existing session, after a backup copy of it."""
    with open(lys, encoding="utf-8") as fh:
        text = fh.read()
    if "</session>" not in text:
        raise ValueError("%s does not look like a KLayout session" % lys)
    backup = "%s.bak-%s" % (lys, time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(lys, backup)
    i = text.rfind("</layout>")
    if i >= 0:
        i = text.index("\n", i) + 1 if "\n" in text[i:] else i + len("</layout>")
    else:
        i = text.find("<view>")
        i = text.rfind("\n", 0, i) + 1 if i >= 0 else text.rfind("</session>")
    text = text[:i] + layout_block + text[i:]
    j = text.rfind("</session>")
    text = text[:j] + view_block + text[j:]
    tmp = lys + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, lys)
    return backup


IMG_ANN = re.compile(r"[ \t]*<annotation>\s*<class>img::Object</class>.*?</annotation>[ \t]*\n?", re.S)


def replace_session_pictures(lys, images):
    """Swap the pictures (img::Object annotations) of an existing session for `images`, after a backup copy; rulers,
    layers, view and layout stay.  The views changed: those showing pictures and the session's own layout
    (<session>.gds, as the export writes it), else every view showing pictures, else the first view.
    -> (backup path, number of views changed)."""
    with open(lys, encoding="utf-8") as fh:
        text = fh.read()
    if "</session>" not in text:
        raise ValueError("%s does not look like a KLayout session" % lys)
    views = [m for m in re.finditer(r"<view>.*?</view>", text, re.S)]
    if not views:
        raise ValueError("%s has no view to put the pictures in" % lys)
    files = {xu.unescape(n): xu.unescape(f) for n, f in re.findall(
        r"<layout>\s*<name>(.*?)</name>\s*<file-path>(.*?)</file-path>", text, re.S)}
    own = os.path.splitext(os.path.basename(lys))[0].lower()

    def shows_own(v):
        refs = [xu.unescape(r) for r in re.findall(r"<layout-ref>(.*?)</layout-ref>", v.group(0), re.S)]
        return any(os.path.splitext(os.path.basename(files.get(r, r).replace("\\", "/")))[0].lower() == own for r in refs)
    with_pics = [v for v in views if IMG_ANN.search(v.group(0))]
    pick = [v for v in with_pics if shows_own(v)] or with_pics or views[:1]
    new = "\n".join(image_annotations(images)) + "\n"
    out, last = [], 0
    for v in pick:
        body = v.group(0)
        m = IMG_ANN.search(body)
        if m:
            body = body[:m.start()] + new + IMG_ANN.sub("", body[m.start():])
        elif "</annotations>" in body:
            body = body.replace("</annotations>", new + "  </annotations>", 1)
        elif "<annotations/>" in body:
            body = body.replace("<annotations/>", "<annotations>\n" + new + "  </annotations>", 1)
        else:
            body = body.replace("</view>", " <annotations>\n" + new + "  </annotations>\n </view>", 1)
        out += [text[last:v.start()], body]
        last = v.end()
    out.append(text[last:])
    backup = "%s.bak-%s" % (lys, time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(lys, backup)
    tmp = lys + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    os.replace(tmp, lys)
    return backup, len(pick)


def session_pictures(a, placed, images, img_dir):
    """The placed pictures as the session shows them (copies in img_dir unless --no-copy), finest on top."""
    order = sorted(placed, key=lambda f: -f["sim"].scale)
    sess = []
    for z, f in enumerate(order):
        src = f["path"]
        if a.no_copy:
            sp = src
        else:
            os.makedirs(img_dir, exist_ok=True)
            ext = os.path.splitext(src)[1].lower()
            if ext in (".tif", ".tiff"):          # KLayout reads 8-bit pictures; write the converted one
                sp = os.path.join(img_dir, os.path.splitext(os.path.basename(src))[0] + ".png")
                cv2.imwrite(sp, images[src])
            else:
                sp = os.path.join(img_dir, os.path.basename(src))
                shutil.copy2(src, sp)
        lm = [f["det_uv"][di] for di in f["assign"] if di >= 0]
        sess.append(dict(sim=f["sim"], z=z, landmarks=lm, session_path=os.path.abspath(sp)))
    return sess, order


def check_picture(im, fit, tpl):
    vis = im.copy()
    h, w = im.shape[:2]
    sim = fit["sim"]
    lw = max(1, int(round(min(h, w) / 700)))
    for f in tpl:
        for p in f["polys"]:
            px = from_uv(sim.inv(p), w, h)
            if np.all((px[:, 0] < -50) | (px[:, 0] > w + 50) | (px[:, 1] < -50) | (px[:, 1] > h + 50)):
                continue
            cv2.polylines(vis, [np.round(px * 4).astype(np.int32)], True, (60, 255, 60), lw, cv2.LINE_AA, shift=2)
    for ti, di in enumerate(fit["assign"]):
        if di >= 0:
            c = from_uv(fit["det_uv"][di][None], w, h)[0]
            r = max(3, int(fit["det_size"][di] * 0.6))
            cv2.circle(vis, (int(round(c[0])), int(round(c[1]))), r, (40, 40, 255), lw, cv2.LINE_AA)
    txt = "%s  %.4f um/px  rot %.2f deg  %d marks  rms %.2f um" % (
        fit["name"], sim.scale, sim.angle, fit["n"], fit["rms"])
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


EXPECT_UM = {}       # picture file name -> its um per pixel when known (--expect-um: the designer's step photos)


def scale_ok(name, sim, tol=0.25):
    """A placement's scale near the picture's known one (else a wrong match: e.g. 0.002 um/px for a 10x picture of
    DQD11, 2026-10-21); True when the scale is not known."""
    ex = EXPECT_UM.get(name)
    ok = ex is None or abs(sim.scale / ex - 1) <= tol
    if not ok:
        print("%-28s a placement at %.4f um/px is not this picture's scale (%.4f): not used" % (name, sim.scale, ex))
    return ok


def place_by_match(fit, best, im, tpl, T, mirror):
    sim, n, refname, how0 = best
    h, w = im.shape[:2]
    mask, score, _ = gold_map(im)
    cand = candidates(mask)
    det_uv = to_uv(np.array([c[:2] for c in cand]).reshape(-1, 2), w, h)
    det_size = np.array([c[2] for c in cand], float)
    det_kind = [c[4] for c in cand]
    if len(cand):
        assign = match(sim, det_uv, det_size, det_kind, tpl, w, h)[0]
    else:
        assign = np.full(len(tpl), -1)
    sel = assign >= 0
    how = "matched to %s (%s)" % (refname, how0)
    nres = int((det_size[assign[sel]] >= 10).sum()) if sel.any() else 0
    if sel.sum() >= 3 or nres >= 2:
        sim2 = fit_sim(det_uv[assign[sel]], T[sel], mirror)
        if abs(sim2.scale / sim.scale - 1) < 0.05 and abs(((sim2.angle - sim.angle + 180) % 360) - 180) < 2:
            sim = sim2
            how += ", refined on %d marks" % sel.sum()
    res = np.hypot(*(sim.fwd(det_uv[assign[sel]]) - T[sel]).T) if sel.any() else np.array([np.nan])
    fit.update(sim=sim, assign=assign, det_uv=det_uv, det_size=det_size, det_kind=det_kind, n=int(sel.sum()),
               rms=float(np.sqrt(np.nanmean(res ** 2))) if sel.any() else float("nan"),
               max=float(np.nanmax(res)) if sel.any() else float("nan"), shape=None, mask=mask, how=how)
    print("%-28s %.4f um/px  rotation %7.2f deg  %s; %d marks agree%s" % (
        fit["name"], sim.scale, sim.angle, how, fit["n"], ", rms %.2f um" % fit["rms"] if fit["n"] else ""))


def main():
    if gdstk is None:
        sys.exit("This tool needs gdstk:  pip3 install gdstk")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("pictures", nargs="+", help="pictures of the chip, or folders of them (any objective; "
                    "name@180 = a picture taken rotated by 180 deg)")
    ap.add_argument("--gds", required=True, help="layout with the alignment marks (e.g. pads_template.gds or your device's GDS)")
    ap.add_argument("-o", "--out", help="output prefix (default: <picture folder>/<folder name>_klayout/<folder name>)")
    ap.add_argument("--lyp", help="KLayout layer properties for the session (remembered as the default; default: %s)" % (
        config_get("lyp") or "none saved yet"))
    ap.add_argument("--design", metavar="DESIGN.json", help="a stack design from the designer (with its stack picture): "
                    "its direct contacts (layer 253) and layer outlines (254) are put on the chip through that picture")
    ap.add_argument("--add-to", metavar="SESSION.lys", help="also add the result as a new tab to this KLayout session "
                    "(a backup copy of it is saved next to it first)")
    ap.add_argument("--replace-pictures", metavar="SESSION.lys", help="swap the pictures of this existing session for "
                    "these (e.g. the latest fabrication step's); only the session's pictures change (backup copy first): "
                    "no GDS or new session is written")
    ap.add_argument("--max-rot", type=float, default=45.0,
                    help="pictures are taken upright: rotations further than this from 0 (or name@angle) are not "
                    "considered (default 45 deg)")
    ap.add_argument("--cell", help="top cell (default: the largest top cell)")
    ap.add_argument("--fiducial-layers", help="GDS layers holding the marks, e.g. 50 or 50,51 (default: the layer with the "
                    "most metal among those with small shapes, i.e. the chip's gold, not your device design)")
    ap.add_argument("--max-fiducial-um", type=float, default=60.0, help="largest mark size in the layout (default 60 um)")
    ap.add_argument("--rot180", action="store_true", help="all pictures were taken rotated by 180 deg")
    ap.add_argument("--mirror", action="store_true", help="pictures are mirrored")
    ap.add_argument("--no-copy", action="store_true", help="session points to the original pictures instead of copies")
    ap.add_argument("--fits-json", metavar="OUT.json", help="only place the pictures on the layout and write where each one "
                    "lies (its pixels -> layout um) to this file; no session or GDS (the designer's aligned photos)")
    ap.add_argument("--expect-um", metavar="SCALES.json", help="{picture file name: um per pixel} of pictures whose scale "
                    "is known: their mark search stays near it and placements far from it (> 25 %%) are not used")
    ap.add_argument("--outline-max-um-px", type=float, default=0,
                    help="trace flake outlines from pictures at most this many um per pixel (default 0 = none; "
                         "the 50x / 100x outlines are noisy, the pictures themselves are clearer)")
    a = ap.parse_args()
    res = dict(ok=False, message="", lys=None, gds=None, check_jpg=None, fit_json=None, added_to=None, replaced_in=None,
               backup=None,
               pictures=[])
    try:
        run(a, res)
        res["ok"] = True
    except SystemExit as ex:
        res["message"] = str(ex.code) if ex.code not in (None, 0) else res["message"]
        if ex.code in (None, 0):
            res["ok"] = True
    except Exception as ex:                      # reported to the designer, which shows the message
        import traceback
        traceback.print_exc(file=sys.stdout)
        res["message"] = "%s: %s" % (type(ex).__name__, ex)
    if not res["ok"]:
        print("error: %s" % res["message"])
    sys.stdout.flush()
    print("RESULT " + json.dumps(res))
    sys.exit(0 if res["ok"] else 1)


PICTURE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def expand_pictures(args, rot0):
    pics = []
    for p in args:
        rot = rot0
        if "@" in os.path.basename(p) and not os.path.exists(p):
            p, r = p.rsplit("@", 1)
            rot = float(r)
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            for n in sorted(os.listdir(p)):
                if n.lower().endswith(PICTURE_EXT) and not n.startswith(".") and not n.endswith("_check.jpg"):
                    pics.append((os.path.join(p, n), rot))
        elif os.path.exists(p):
            pics.append((p, rot))
        else:
            sys.exit("no such picture or folder: %s" % p)
    if not pics:
        sys.exit("no pictures found in %s" % ", ".join(args))
    return pics


def place_pictures(a, pics, gds_in):
    """The pictures [(path, rotation)] placed on the layout's alignment marks (fit_picture), the weak fits checked and
    the pictures without enough marks matched to placed ones (sift_place / ncc_place).  Returns (lib, top, tpl, T,
    fits, images, placed)."""
    lib, top = load_layout(gds_in, a.cell)
    layers = set(int(x) for x in a.fiducial_layers.split(",")) if a.fiducial_layers else None
    tpl, (_, layers) = template_fiducials(gds_in, a.cell, layers, a.max_fiducial_um)
    nm = sum(f["kind"] == "mark" for f in tpl)
    print("layout %s, cell %s, mark layer %s: %d fiducials (%d marks, %d squares)" % (
        gds_in, top.name, ",".join(map(str, sorted(layers))) if layers else "all", len(tpl), nm, len(tpl) - nm))
    if len(tpl) < 2:
        sys.exit("fewer than 2 alignment marks found in the layout; try --max-fiducial-um or --fiducial-layers")
    T = np.array([f["c"] for f in tpl])

    fits, images = [], {}
    for path, rot in pics:
        im = read_picture(path)
        images[path] = im
        name = os.path.basename(path)
        ex = EXPECT_UM.get(name)                         # (its scale known: the mark search only near it)
        fit = fit_picture(im, tpl, a.mirror, rot, max_rot=a.max_rot, **(dict(smin=ex / 1.3, smax=ex * 1.3) if ex else {}))
        if fit is not None and fit_good(fit) and not scale_ok(name, fit["sim"]):
            fit = None
        good = fit_good(fit)
        if fit is not None:
            fit["name"], fit["path"], fit["rot"] = name, path, rot
        if good:
            fit["how"] = "marks"
            print("%-28s %.4f um/px  rotation %7.2f deg  %2d marks  rms %.2f um  max %.2f um" % (
                name, fit["sim"].scale, fit["sim"].angle, fit["n"], fit["rms"], fit["max"]))
        else:
            print("%-28s not enough marks (%s); will try matching to another picture" % (
                name, "none found" if fit is None else "%d" % fit["n"]))
            fit = dict(name=name, path=path, rot=rot, how=None, prev=fit)
        fits.append(fit)

    # weak mark fits (few, small marks) are checked against a strong picture by feature matching
    strong = [f for f in fits if f["how"] == "marks" and (f["n_resolved"] >= 4 or f["n"] >= 8)]
    for k, f in enumerate(fits):
        if f["how"] != "marks" or f in strong or not strong:
            continue
        im = images[f["path"]]
        best = None
        for ref in strong:
            sim, n = sift_place(im, images[ref["path"]], ref["sim"], a.mirror)
            if sim is not None and not scale_ok(f["name"], sim):
                sim = None
            if sim is None or angdiff(sim.angle, f["rot"]) > a.max_rot:
                sim, c = ncc_place(im, images[ref["path"]], ref["sim"], f["rot"] - ref["rot"], a.mirror)
                n = int(c * 100) if sim is not None else 0
            if sim is not None and angdiff(sim.angle, f["rot"]) > a.max_rot:
                sim = None
            if sim is not None and (best is None or n > best[1]):
                best = (sim, n)
        if best is None:
            continue
        h, w = im.shape[:2]
        d = np.hypot(*(best[0].fwd(np.array([[0.0, 0.0], [w / 2.0, h / 2.0]])) - f["sim"].fwd(np.array([[0.0, 0.0], [w / 2.0, h / 2.0]]))).T).max()
        f["checked"] = True
        if d > max(5.0, 10 * f["sim"].scale):     # e.g. 2 marks taken for the wrong pair of a symmetric set
            dang = ((best[0].angle - f["sim"].angle + 180) % 360) - 180
            why = " (turned %.0f deg: its few marks cannot tell 0 from 180)" % dang if abs(abs(dang) - 180) < 5 else ""
            print("%-28s its marks fit disagrees with the other pictures by %.0f um%s: placed by matching instead" % (
                f["name"], d, why))
            fits[k] = dict(name=f["name"], path=f["path"], rot=f["rot"], how=None, prev=f)

    # pictures without enough marks: matched to a placed picture (SIFT, in passes so that a picture placed this way
    # can serve the next, e.g. 10x -> 50x -> 100x; correlation only when SIFT finds nothing), then checked / refined
    # with the marks they show
    for mode in ("sift", "sift", "sift", "ncc"):
        for fit in fits:
            if fit["how"]:
                continue
            im = images[fit["path"]]
            h, w = im.shape[:2]
            best = None
            for ref in fits:
                if not ref["how"]:
                    continue
                if mode == "sift":
                    sim, n = sift_place(im, images[ref["path"]], ref["sim"], a.mirror)
                    how = "%d features" % n
                else:
                    sim, c = ncc_place(im, images[ref["path"]], ref["sim"], fit["rot"] - ref["rot"], a.mirror)
                    n, how = int(c * 100), "correlation %.2f" % c
                    if c < 0.5:
                        sim = None
                if sim is not None and (angdiff(sim.angle, fit["rot"]) > a.max_rot or not scale_ok(fit["name"], sim)):
                    sim = None
                if sim is not None and (best is None or n > best[1]):
                    best = (sim, n, ref["name"], how)
            if best is None:
                continue
            place_by_match(fit, best, im, tpl, T, a.mirror)
    for fit in fits:
        if not fit["how"]:
            print("%-28s could not be placed (no marks, no match to a placed picture): left out" % fit["name"])

    for f in fits:
        if f.get("how") == "marks" and f["n"] <= 2 and not f.get("checked"):
            print("warning: %s rests on 2 marks only and no other picture could check it: it may be off by one mark "
                  "spacing; add a lower-magnification picture of the same place" % f["name"])
            f["warning"] = "2 marks only, unchecked"
    placed = [f for f in fits if f.get("how")]
    if not placed:
        sys.exit("no picture could be placed (no alignment marks found, and none to match to)")
    return lib, top, tpl, T, fits, images, placed


def write_fits(path, fits, images):
    """--fits-json: each picture's placement: its pixels (x right, y down) -> layout um, as a 3 x 3 matrix, with how it
    was found (marks / matched to another picture) and the marks' residuals."""
    out = []
    for f in fits:
        im = images.get(f["path"])
        h, w = im.shape[:2] if im is not None else (0, 0)
        e = dict(name=f["name"], path=f["path"], how=f.get("how"), w=w, h=h, warning=f.get("warning"))
        if f.get("how"):
            # pixel (x, y) -> uv (x - w/2 + 0.5, h/2 - 0.5 - y) -> um by the similarity
            S = np.array(f["sim"].matrix(), float)
            P = np.array([[1.0, 0.0, -w / 2.0 + 0.5], [0.0, -1.0, h / 2.0 - 0.5], [0.0, 0.0, 1.0]])
            rms = f.get("rms")
            e.update(px_to_um=np.round(S @ P, 9).tolist(), um_per_px=round(float(f["sim"].scale), 6),
                     angle_deg=round(float(f["sim"].angle), 3), marks=int(f.get("n") or 0),
                     rms_um=round(float(rms), 3) if rms is not None and math.isfinite(rms) else None)   # (no NaN: not JSON)
        out.append(e)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)


def run(a, result):
    pics = expand_pictures(a.pictures, 180.0 if a.rot180 else 0.0)
    if getattr(a, "expect_um", None):
        with open(a.expect_um, encoding="utf-8") as fh:
            EXPECT_UM.update({os.path.basename(k): float(v) for k, v in json.load(fh).items() if v})
    design = None
    if a.design:
        with open(os.path.expanduser(a.design), encoding="utf-8") as fh:
            design = json.load(fh)
        dp = os.path.abspath(design.get("picture") or "")
        if not design.get("simg") or not os.path.isfile(dp):
            sys.exit("the design has no stack picture to place it by (as-built: Stack picture..., Align)")
        if dp not in [p for p, _ in pics]:
            pics.append((dp, 180.0 if a.rot180 else 0.0))
        design["picture"] = dp
    gds_in = os.path.expanduser(a.gds)
    if not os.path.exists(gds_in):
        sys.exit("no such layout: %s" % gds_in)
    first = pics[0][0]
    if getattr(a, "fits_json", None):
        lib, top, tpl, T, fits, images, placed = place_pictures(a, pics, gds_in)
        write_fits(a.fits_json, fits, images)
        result.update(fits_json=a.fits_json, pictures=[dict(name=f["name"], how=f.get("how")) for f in fits])
        return
    replace = os.path.abspath(os.path.expanduser(a.replace_pictures)) if a.replace_pictures else None
    if replace and (not os.path.isfile(replace) or not replace.lower().endswith(".lys")):
        sys.exit("no such session: %s" % replace)
    if a.out:
        out = os.path.abspath(os.path.expanduser(a.out))
    elif replace:
        out = os.path.splitext(replace)[0]
    else:
        folder = os.path.dirname(first)
        stem = os.path.basename(folder) or "aligned"
        out = os.path.join(folder, stem + "_klayout", stem)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lyp = None
    if a.lyp:
        lyp = os.path.abspath(os.path.expanduser(a.lyp))
        if not os.path.exists(lyp):
            sys.exit("no such layer properties file: %s" % lyp)
        config_set(lyp=lyp)
    elif config_get("lyp") and os.path.exists(config_get("lyp")):
        lyp = config_get("lyp")
    add_to = os.path.abspath(os.path.expanduser(a.add_to)) if a.add_to else None
    if add_to and not os.path.exists(add_to):
        sys.exit("no such session: %s" % add_to)

    lib, top, tpl, T, fits, images, placed = place_pictures(a, pics, gds_in)

    if replace:          # only the session's pictures change: no GDS, no new session
        sess, _ = session_pictures(a, placed, images, os.path.splitext(replace)[0] + "_images")
        backup, nv = replace_session_pictures(replace, sess)
        result.update(replaced_in=replace, backup=backup)
        print("pictures of %s replaced by these %d (%d view%s; backup: %s)" % (
            replace, len(sess), nv, "" if nv == 1 else "s", os.path.basename(backup)))
        print("note: if this session is open in KLayout, close it WITHOUT saving the session (saving would put the old "
              "pictures back), then File > Restore Session again")
        write_report(a, out, fits, placed, images, tpl, T, gds_in, top, result)
        print("wrote %s_check.jpg, %s_fit.json" % (out, out))
        return

    # --- GDS: copy of the layout + found marks, picture frames, flake outlines
    top.add(*[gdstk.ellipse(tuple(f["sim"].fwd(f["det_uv"][di][None])[0]), max(0.25, 1.5 * f["sim"].scale),
                            layer=LAYER_FOUND, datatype=k)
              for k, f in enumerate(placed) for di in f["assign"] if di >= 0])
    for k, f in enumerate(placed):
        im = images[f["path"]]
        h, w = im.shape[:2]
        corners = f["sim"].fwd(np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]))
        top.add(gdstk.Polygon(corners, layer=LAYER_FRAME, datatype=k))
        if a.outline_max_um_px and f["sim"].scale <= a.outline_max_um_px:
            mask, _, sub = gold_map(im)
            outs = flake_outlines(im, mask, sub, f["sim"].scale)
            for c in outs:
                top.add(gdstk.Polygon(f["sim"].fwd(to_uv(c, w, h)), layer=LAYER_OUTLINE, datatype=k))
            f["outlines"] = len(outs)
    gds_out = out + ".gds"
    dshapes, dinfo = [], None
    if design is not None:
        df = [f for f in fits if f["path"] == design["picture"]]
        if df and df[0].get("how"):
            dshapes = design_shapes(design, design_to_chip(design, df[0], images[design["picture"]].shape))
            for l, d, poly, nm, col in dshapes:
                top.add(gdstk.Polygon(poly, layer=l, datatype=d))
            ns = sum(1 for x in dshapes if x[0] == LAYER_CONTACT and x[1] == 0)
            nss = sum(1 for x in dshapes if x[0] == LAYER_CONTACT and x[1] == 1)
            print("design \"%s\" placed through %s: %d short region(s), %d sample-sample region(s), %d layer outline(s)" % (
                design.get("name", ""), os.path.basename(design["picture"]), ns, nss,
                sum(1 for x in dshapes if x[0] == LAYER_DESIGN)))
            dinfo = dict(placed=True, name=design.get("name"), picture=design["picture"], shorts=ns, sample_contacts=nss)
        else:
            print("warning: the design's stack picture %s could not be placed, so the design is left out" %
                  os.path.basename(design["picture"]))
            dinfo = dict(placed=False, name=design.get("name"), picture=design["picture"])
    result["design"] = dinfo
    keep_existing(gds_out)
    write_gds_quietly(lib, gds_out)
    own = {LAYER_FOUND: ("marks found", "#ff3030"), LAYER_FRAME: ("picture", "#ffffff"), LAYER_OUTLINE: ("flake outline", "#00e5ff")}
    tabs, covered = (read_lyp(lyp) if lyp else ([("", [])], set()))
    if lyp:
        print("layer properties: %s" % lyp)
    lyr, k0 = [], 0
    for l, d in sorted(set((p.layer, p.datatype) for p in top.get_polygons()) |
                       set((lb.layer, lb.texttype) for lb in top.get_labels())):
        if l in (LAYER_CONTACT, LAYER_DESIGN):
            hit_ = [x for x in dshapes if x[0] == l and x[1] == d]
            lyr.append((l, d, hit_[0][3] if hit_ else "%d/%d" % (l, d), hit_[0][4] if hit_ else "#ffffff"))
        elif l in own:
            nm = "%s: %s" % (own[l][0], placed[d]["name"] if d < len(placed) else d)
            lyr.append((l, d, nm, own[l][1]))
        elif (l, d) not in covered:                 # layers the .lyp does not list would not be shown
            lyr.append((l, d, "%d/%d" % (l, d), PALETTE[k0 % len(PALETTE)]))
            k0 += 1
    tabs = [(n, list(pr)) for n, pr in tabs]
    tabs[0][1].extend(own_props(lyr))

    # --- pictures for the session (copies by default), finest on top
    sess, order = session_pictures(a, placed, images, out + "_images")
    finest = order[-1]
    h, w = images[finest["path"]].shape[:2]
    cr = finest["sim"].fwd(np.array([[-w / 2, -h / 2], [w / 2, h / 2], [w / 2, -h / 2], [-w / 2, h / 2]]))
    pad = 0.1 * (cr[:, 0].max() - cr[:, 0].min())
    vb = (cr[:, 0].min() - pad, cr[:, 1].min() - pad, cr[:, 0].max() + pad, cr[:, 1].max() + pad)
    stem = os.path.basename(out)
    lname = os.path.basename(gds_out)
    title = "%s (aligned)" % stem
    keep_existing(out + ".lys")
    with open(out + ".lys", "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="utf-8"?>\n<session>\n <current-view>0</current-view>\n' +
                 layout_xml(lname, gds_out) + view_xml(lname, title, top.name, sess, vb, tabs_xml(tabs)) + "</session>\n")
    result.update(lys=out + ".lys", gds=gds_out)
    if add_to:
        with open(add_to, encoding="utf-8") as fh:
            used = session_layout_names(fh.read())
        n, k = lname, 2
        while n in used:
            n, k = "%s (%d)" % (lname, k), k + 1
        backup = add_to_session(add_to, layout_xml(n, gds_out), view_xml(n, title, top.name, sess, vb, tabs_xml(tabs)))
        result.update(added_to=add_to, backup=backup)
        print("added as a new tab \"%s\" to %s (backup: %s)" % (title, add_to, os.path.basename(backup)))
        print("note: if this session is open in KLayout, that window does not see the new tab (KLayout does not reload "
              "a session file): close it WITHOUT saving the session (saving would remove the new tab), then "
              "File > Restore Session again")

    write_report(a, out, fits, placed, images, tpl, T, gds_in, top, result)
    print("wrote %s.lys (open in KLayout: File > Restore Session), %s.gds, %s_check.jpg, %s_fit.json" % (out, out, out, out))


def write_report(a, out, fits, placed, images, tpl, T, gds_in, top, result):
    """The check picture (<out>_check.jpg), the fit report (<out>_fit.json) and the per-picture result."""
    tiles = []
    for f in placed:
        v = check_picture(images[f["path"]], f, tpl)
        s = 900.0 / v.shape[1]
        tiles.append(cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    cv2.imwrite(out + "_check.jpg", np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    rep = []
    for f in fits:
        if not f.get("how"):
            rep.append(dict(picture=f["path"], placed=False))
            continue
        sel = f["assign"] >= 0
        rep.append(dict(picture=f["path"], placed=True, how=f["how"], um_per_px=f["sim"].scale,
                        rotation_deg=f["sim"].angle, mirrored=a.mirror, klayout_matrix=f["sim"].matrix(),
                        marks_used=[dict(layout_um=list(map(float, T[ti])),
                                         found_um=list(map(float, f["sim"].fwd(f["det_uv"][f["assign"][ti]][None])[0])))
                                    for ti in np.nonzero(sel)[0]],
                        rms_um=f["rms"], max_um=f["max"], flake_outlines=f.get("outlines", 0)))
    with open(out + "_fit.json", "w", encoding="utf-8") as fh:
        json.dump(dict(layout=os.path.abspath(gds_in), cell=top.name, pictures=rep), fh, indent=1)
    result.update(check_jpg=out + "_check.jpg", fit_json=out + "_fit.json",
                  pictures=[dict(name=os.path.basename(f["path"]), placed=bool(f.get("how")), how=f.get("how"),
                                 um_per_px=f["sim"].scale if f.get("how") else None,
                                 rotation_deg=f["sim"].angle if f.get("how") else None,
                                 n_marks=int(f.get("n", 0)) if f.get("how") else 0,
                                 rms_um=(None if not f.get("how") or not np.isfinite(f["rms"]) else f["rms"]),
                                 warning=f.get("warning"))
                            for f in fits])
    nplaced = sum(1 for f in fits if f.get("how"))
    result["message"] = "%d of %d pictures placed" % (nplaced, len(fits))


if __name__ == "__main__":
    main()
