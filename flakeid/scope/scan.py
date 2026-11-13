"""The chip scan in Python (pymmcore / pymmcore-plus): a port of tools/mm_chip_scan.bsh, without the Micro-Manager GUI.

Same procedure and files as the BeanShell script (see its header): find the chip edges from the start position,
plan the tiles, focus points -> plane / quadratic focus surface, scan the tiles to disk in Micro-Manager's
"separate image files" layout, then (multi mode) image each piece's targets at 100x and, for graphite, the
stacking domains.  The run file, <piece>_chip.json, _scan.pos, scan_params.json, tiles, <piece>_100x/ with its index
and _domains/ are written exactly as the script writes them, so flakeid.live / flakeid.hundred read them unchanged.

Differences: the dialogs and live view are this package's own (flakeid.scope.ui: colour live view in every focus
question, a live viewer to add the pieces); the analysis can run on this PC (Settings.local_analysis: python -m
flakeid.live follows the run folder in the background and writes the target lists); results can be copied to the
share at the end.  The Micro-Manager autofocus manager and position list do not exist without the GUI: autofocus
is this script's own, targets come from a CSV.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .settings import Settings, dump
from flakeid.fieldmosaic import field_size_um, fmt_matrix, format_poly, parse_poly, plan_fields

VERSION = "py-2026-10-27z"
NAN = float("nan")
MANUAL_RETAKE = "Manual focus (retakes)"       # the 100x window's button: a retake focused by hand
IDX100_HEADER = ("order,label,stage_x_um,stage_y_um,z_um,z_drive,z10_um,offset_um,focus,af_ratio,saturated_frac,file,shift_files,"
                 "shift_um,exposure_ms,frames,recentre_um\r\n")



def polygon_area_um2(pts):
    """Area of a polygon [[x, y], ...] (um2, shoelace)."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def targets_for_area(s, area_mm2):
    """100x targets for a piece of area_mm2: s.targets_per_cm2 x the area in cm2, at least targets_min_piece and at
    most targets_max_piece (0: no limit); None when the density is 0 (the analysis's fixed --max-targets)."""
    d = float(getattr(s, "targets_per_cm2", 0) or 0)
    if d <= 0 or not area_mm2:
        return None
    n = max(s.targets_min_piece, round(d * area_mm2 / 100.0))
    hi = int(getattr(s, "targets_max_piece", 0) or 0)
    return int(min(n, hi) if hi > 0 else n)


def hbn_imaged_for_area(s, area_mm2):
    """hBN pictures for a piece of area_mm2 (its targets with the largest best terrace): s.hbn_imaged_per_cm2 x the area
    in cm2, at least targets_min_piece; 0 = every target; s.hbn_max_targets when the area is unknown."""
    d = getattr(s, "hbn_imaged_per_cm2", None)
    if d is not None and float(d) <= 0:
        return 0
    if d is None or not area_mm2:
        return int(getattr(s, "hbn_max_targets", 0) or 0)
    return int(max(s.targets_min_piece, round(float(d) * area_mm2 / 100.0)))

class Stop(Exception):
    """The run was stopped (by the user, or an error the script cannot go on from)."""


EXPORTING = set()          # chips whose run still finishes in the background (export, copy): not re-imaged meanwhile


class Target(tuple):
    """A 100x target: (x, y, z10, thickness_nm, label), unpacked as before, + .poly: its chosen terrace's outline in
    stage um relative to (x, y) ((N, 2) array; None: not in the target list, single picture)."""
    poly = None


MOSAIC100_HEADER = ("main_file,file,k,label,stage_x_um,stage_y_um,z_um,focus,af_ratio,frames,exposure_ms,saturated_frac,"
                    "um_per_px,width,height,grid_i,grid_j\r\n")


class StageLimit(Exception):
    """A move beyond the stage's travel: outside the known limits (stage_limits_um, or learned when the stage stopped
    short at one).  Not an error of the scan: what is out of reach is left out (a chip partly beyond the limits is
    scanned up to them, out-of-reach focus points / tiles / 100x targets skipped; 2026-10-19)."""


def f1(v): return f"{v:.1f}"
def f3(v): return f"{v:.3f}"
def pad3(v): return f"{int(v):03d}"
def safe_name(s): return re.sub(r"[^A-Za-z0-9_.-]", "_", s.strip())
def isnan(v): return v != v


# ------------------------------------------------------------------------------------------------- geometry helpers
def fit_plane(pts):
    """Least-squares plane z = a x + b y + c through [x, y, z] points (>= 3), or None (points on a line)."""
    if len(pts) < 3:
        return None
    P = np.asarray(pts, float)
    mx, my = P[:, 0].mean(), P[:, 1].mean()
    u, v = P[:, 0] - mx, P[:, 1] - my
    C = np.array([[u @ u, u @ v], [u @ v, v @ v]])
    ev = np.linalg.eigvalsh(C)
    if ev[1] <= 0 or ev[0] < 1e-4 * ev[1]:
        return None
    A = np.c_[u, v, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    a, b, c = sol
    return (float(a), float(b), float(c - a * mx - b * my))


def fit_quad(pts):
    """Least-squares quadratic z(u, v) (u, v in mm from the points' mean) through >= 6 points:
    (c0, cu, cv, cuu, cuv, cvv, xmin, xmax, ymin, ymax, mx, my), or None."""
    if len(pts) < 6:
        return None
    P = np.asarray(pts, float)
    mx, my = P[:, 0].mean(), P[:, 1].mean()
    u, v = (P[:, 0] - mx) / 1000.0, (P[:, 1] - my) / 1000.0
    A = np.c_[np.ones(len(P)), u, v, u * u, u * v, v * v]
    if np.linalg.matrix_rank(A) < 6:
        return None
    c, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    return tuple(float(x) for x in c) + (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), float(mx), float(my))


def robust_focus_points(pts, tol, max_tilt=None):
    """Focus points without the ones off the chip's surface (tape over its edge, debris): the plane most points agree
    with (within `tol` um; tried through every 3 points, refitted on the ones that agree) is kept, the others dropped.
    -> (kept, dropped as (x, y, z, off), ambiguous): ambiguous when two different groups agree equally well (4 corner
    points with one wrong: any 3 lie on a plane) - one more point decides.  All points within tol of their plane:
    nothing dropped."""
    from itertools import combinations
    P = [tuple(p) for p in pts]
    n = len(P)

    def offs(pl, idx):
        return {i: P[i][2] - (pl[0] * P[i][0] + pl[1] * P[i][1] + pl[2]) for i in idx}

    if n < 4:
        return P, [], False
    # each point against the plane through the others (a least-squares plane through all spreads one bad point's error
    # over the rest: with 4 points a 40 um step shows as 10 um)
    loo = []
    for i in range(n):
        pl = fit_plane(P[:i] + P[i + 1:])
        if pl is not None:
            loo.append(abs(P[i][2] - (pl[0] * P[i][0] + pl[1] * P[i][1] + pl[2])))
    if not loo or max(loo) <= tol:
        return P, [], False
    groups = {}
    for tri in combinations(range(n), 3):
        pl = fit_plane([P[i] for i in tri])
        if pl is None or (max_tilt is not None and math.hypot(pl[0], pl[1]) * 1000.0 > max_tilt):
            continue                                     # (no chip lies that steeply: max_tilt um/mm)
        inl = [i for i, o in offs(pl, range(n)).items() if abs(o) <= tol]
        if len(inl) >= 3:
            pl2 = fit_plane([P[i] for i in inl])
            if pl2 is not None:
                inl = [i for i, o in offs(pl2, range(n)).items() if abs(o) <= tol]
        if len(inl) >= 3:
            g = frozenset(inl)
            plg = fit_plane([P[i] for i in g])
            if plg is None:
                continue
            r = math.sqrt(sum(o * o for o in offs(plg, g).values()) / len(g))
            groups[g] = (len(g), r, math.hypot(plg[0], plg[1]) * 1000.0)
    if not groups:
        return P, [], True
    m = max(c for c, _, _ in groups.values())
    best = [(r, t, sorted(g)) for g, (c, r, t) in groups.items() if c == m]
    if len(best) > 1:
        # several groups as large: the one on a clearly better plane (2x lower rms); else the clearly flatter one (tape
        # only ever raises points, and a chip lies nearly flat: a group with taped points needs a tilt to agree);
        # else one more point decides
        best.sort()
        if not best[1][0] >= 2.0 * best[0][0] + 0.5:
            best.sort(key=lambda b: b[1])
            if best[1][1] - best[0][1] < 2.0:
                return P, [], True
    keep = best[0][2]
    pl = fit_plane([P[i] for i in keep])
    out = offs(pl, [i for i in range(n) if i not in keep])
    return [P[i] for i in keep], [P[i] + (out[i],) for i in sorted(out)], False

def ray_polygon(sx, sy, dx, dy, poly):
    """Distance along the ray (sx, sy) + t (dx, dy) to where it first meets polygon `poly` (corners [[x, y], ...]),
    or None."""
    best = None
    n = len(poly)
    for k in range(n):
        ax, ay = poly[k]
        bx, by = poly[(k + 1) % n]
        ex, ey = bx - ax, by - ay
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((ax - sx) * ey - (ay - sy) * ex) / den
        u = ((ax - sx) * dy - (ay - sy) * dx) / den
        if t >= 0 and 0 <= u <= 1 and (best is None or t < best):
            best = t
    return best


# ---- scan areas drawn by hand (chip_type "polygon"): polygons [[x, y], ...] in stage um
def point_in_polygon(x, y, poly):
    """Is (x, y) inside polygon `poly` (even-odd rule)?"""
    inside = False
    n = len(poly)
    for k in range(n):
        ax, ay = poly[k][0], poly[k][1]
        bx, by = poly[(k + 1) % n][0], poly[(k + 1) % n][1]
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def segment_meets_box(ax, ay, bx, by, x0, y0, x1, y1):
    """Does the segment a-b touch the box [x0, x1] x [y0, y1]? (Liang-Barsky clipping)"""
    t0, t1 = 0.0, 1.0
    dx, dy = bx - ax, by - ay
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def box_meets_polygon(x0, y0, x1, y1, poly):
    """Does the box [x0, x1] x [y0, y1] overlap polygon `poly` at all (an edge of it crosses the box, or the box lies
    inside it)?"""
    n = len(poly)
    if any(segment_meets_box(poly[k][0], poly[k][1], poly[(k + 1) % n][0], poly[(k + 1) % n][1], x0, y0, x1, y1)
           for k in range(n)):
        return True
    return point_in_polygon((x0 + x1) / 2.0, (y0 + y1) / 2.0, poly)


def polygon_crossing(poly):
    """The first pair of edges (i, j) of polygon `poly` that cross (its points not added in order round the area), or
    None.  Edges k = point k -> point k + 1; neighbouring edges share a point and do not count."""
    n = len(poly)

    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0:
                return i, j
    return None


def polygon_inner_grid(poly, n=41):
    """Points of an n x n grid over the box of polygon `poly` that are inside it, with their distance to its boundary:
    (N, 3) array [x, y, d]."""
    P = np.asarray(poly, float)[:, :2]
    X, Y = np.meshgrid(np.linspace(P[:, 0].min(), P[:, 0].max(), n), np.linspace(P[:, 1].min(), P[:, 1].max(), n))
    X, Y = X.ravel(), Y.ravel()
    inside = np.array([point_in_polygon(x, y, P) for x, y in zip(X, Y)], bool)
    X, Y = X[inside], Y[inside]
    return np.c_[X, Y, polygon_edge_dist(X, Y, P)]


def polygon_edge_dist(X, Y, poly):
    """Distance (um) of the points (X, Y) (arrays) to the nearest edge of polygon `poly`."""
    P = np.asarray(poly, float)[:, :2]
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    d = np.full(X.shape, np.inf)
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        e = b - a
        t = np.clip(((X - a[0]) * e[0] + (Y - a[1]) * e[1]) / max(float(e @ e), 1e-12), 0.0, 1.0)
        d = np.minimum(d, np.hypot(X - a[0] - t * e[0], Y - a[1] - t * e[1]))
    return d


def polygon_interior_point(poly):
    """A point well inside polygon `poly` (the piece's start: substrate reference, first focus): its centroid when
    that is well inside, else the grid point furthest from its edges (a concave polygon)."""
    P = np.asarray(poly, float)[:, :2]
    x, y = P[:, 0], P[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cr.sum() / 2.0
    if abs(a) > 1e-9:
        cx, cy = ((x + np.roll(x, -1)) * cr).sum() / (6 * a), ((y + np.roll(y, -1)) * cr).sum() / (6 * a)
    else:
        cx, cy = x.mean(), y.mean()
    G = polygon_inner_grid(P)
    if not len(G):
        return float(cx), float(cy)
    if point_in_polygon(cx, cy, P) and float(polygon_edge_dist(cx, cy, P)) >= 0.5 * G[:, 2].max():
        return float(cx), float(cy)
    k = int(np.argmax(G[:, 2]))
    return float(G[k, 0]), float(G[k, 1])


def polygon_spread_points(poly, n, inset_um):
    """n points spread over polygon `poly` (the focus points of a scan area drawn by hand): farthest-point sampling of
    its inner grid, at least inset_um in from its edges (less where the polygon is too narrow for that)."""
    G = polygon_inner_grid(poly)
    if not len(G):
        P = np.asarray(poly, float)[:, :2]
        return [tuple(map(float, P.mean(0)))]
    G = G[G[:, 2] >= min(inset_um, 0.6 * G[:, 2].max())]
    cx, cy = polygon_interior_point(poly)
    pick = [int(np.argmax(np.hypot(G[:, 0] - cx, G[:, 1] - cy)))]
    dist = np.hypot(G[:, 0] - G[pick[0], 0], G[:, 1] - G[pick[0], 1])
    while len(pick) < min(n, len(G)):
        k = int(np.argmax(dist))
        pick.append(k)
        dist = np.minimum(dist, np.hypot(G[:, 0] - G[k, 0], G[:, 1] - G[k, 1]))
    return [(float(G[k, 0]), float(G[k, 1])) for k in pick]


def polygon_focus_grid(poly, n, start=None):
    """n focus points spread uniformly over polygon `poly` (each covering an equal share of its area: Lloyd's
    relaxation on a fine grid of its inside, from a farthest-point start), in a short path from the point nearest to
    `start` (each next point the nearest one not visited): [(x, y)]."""
    P = np.asarray(poly, float)[:, :2]
    h = math.sqrt(max(polygon_area_um2(P), 1.0) / 6000.0)     # (a grid of ~6000 points inside)
    X, Y = np.meshgrid(np.arange(P[:, 0].min() + h / 2, P[:, 0].max(), h), np.arange(P[:, 1].min() + h / 2, P[:, 1].max(), h))
    G = np.c_[X.ravel(), Y.ravel()]
    G = G[np.array([point_in_polygon(x, y, P) for x, y in G], bool)]
    if len(G) < n:
        return [tuple(map(float, q)) for q in polygon_spread_points(P, n, 0.0)]
    C = np.asarray(polygon_spread_points(P, n, 0.0), float)
    for _ in range(30):
        lab = np.argmin(((G[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.array([G[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(len(C))])
    C = np.array([G[int(np.argmin(((G - c) ** 2).sum(1)))] for c in C])     # (on the inside, also of a concave polygon)
    sx, sy = start if start is not None else C.mean(0)
    left = list(range(len(C)))
    k = min(left, key=lambda j: (C[j, 0] - sx) ** 2 + (C[j, 1] - sy) ** 2)
    order = []
    while left:
        left.remove(k)
        order.append(k)
        if left:
            k = min(left, key=lambda j: (C[j, 0] - C[k, 0]) ** 2 + (C[j, 1] - C[k, 1]) ** 2)
    return [(float(C[j, 0]), float(C[j, 1])) for j in order]


def _line_axis(P):
    """Points (N, >= 2) nearly on a line (a narrow polygon's focus points): (centre, unit direction), else None."""
    m = P[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov((P[:, :2] - m).T, bias=True))
    if w[1] <= 0 or w[0] < 1e-4 * w[1] or math.sqrt(max(w[0], 0.0)) < 50.0:
        return m, V[:, 1]
    return None


def fit_plane_any(pts):
    """fit_plane, and for points on a line (a narrow polygon) the plane rising along that line, level across it."""
    pl = fit_plane(pts)
    if pl is not None or len(pts) < 2:
        return pl
    P = np.asarray(pts, float)
    ax = _line_axis(P)
    if ax is None:
        return None
    m, d = ax
    t = (P[:, :2] - m) @ d
    if np.ptp(t) <= 0:
        return None
    sl, c0 = np.polyfit(t, P[:, 2], 1)
    return (float(sl * d[0]), float(sl * d[1]), float(c0 - sl * (m @ d)))


def smooth_surface(pts, smoothing=1.0):
    """Focus surface through focus points [(x, y, z)] (um): a thin-plate spline with a plane, coordinates in mm,
    `smoothing` against the points' focus noise; each place follows the points near it.  Points on a line (a narrow
    polygon): the same along the line, level across it.  Returns z(x, y), or None (fewer than 4 points)."""
    if len(pts) < 4:
        return None
    from scipy.interpolate import RBFInterpolator
    P = np.asarray(pts, float)
    ax = _line_axis(P)
    try:
        if ax is not None:
            m, d = ax
            f1d = RBFInterpolator(((P[:, :2] - m) @ d)[:, None] / 1000.0, P[:, 2], kernel="thin_plate_spline",
                                  smoothing=float(smoothing), degree=1)
            return lambda x, y: float(f1d(np.array([[((x - m[0]) * d[0] + (y - m[1]) * d[1]) / 1000.0]]))[0])
        f = RBFInterpolator(P[:, :2] / 1000.0, P[:, 2], kernel="thin_plate_spline", smoothing=float(smoothing), degree=1)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return lambda x, y: float(f(np.array([[x / 1000.0, y / 1000.0]]))[0])


def local_outliers(pts, tol, smoothing=1.0, max_frac=0.25):
    """Focus points that disagree with the smooth surface through the others (leave-one-out) by more than tol um
    (tape, debris, a failed focus): the worst one left out at a time, at most max_frac of them, at least 4 kept.
    Returns (kept, [(x, y, z, off)])."""
    P = [tuple(p) for p in pts]
    out = []
    while len(P) > 4 and len(out) < max_frac * len(pts):
        offs = []
        for i in range(len(P)):
            f = smooth_surface(P[:i] + P[i + 1:], smoothing)
            offs.append(abs(P[i][2] - f(P[i][0], P[i][1])) if f is not None else 0.0)
        i = int(np.argmax(offs))
        if offs[i] <= tol:
            break
        out.append(P[i] + (float(offs[i]),))
        P = P[:i] + P[i + 1:]
    return P, out


def fit_circle(pts):
    """Least-squares circle through [x, y] points (>= 3): (cx, cy, R), or None."""
    P = np.asarray(pts, float)
    mx, my = P[:, 0].mean(), P[:, 1].mean()
    x, y = (P[:, 0] - mx) / 1000.0, (P[:, 1] - my) / 1000.0
    A = np.c_[x, y, np.ones(len(P))]
    try:
        (D, E, F), *_ = np.linalg.lstsq(A, -(x * x + y * y), rcond=None)
    except np.linalg.LinAlgError:
        return None
    r2 = D * D / 4 + E * E / 4 - F
    if r2 <= 0:
        return None
    return (mx - D / 2 * 1000.0, my - E / 2 * 1000.0, math.sqrt(r2) * 1000.0)


def intersect(a, b):
    """Intersection of two edge lines (ex, ey, nx, ny): n.(p - e) = 0."""
    ca, cb = a[2] * a[0] + a[3] * a[1], b[2] * b[0] + b[3] * b[1]
    det = a[2] * b[3] - a[3] * b[2]
    return ((ca * b[3] - a[3] * cb) / det, (a[2] * cb - ca * b[2]) / det)


def idw_offset(x, y, pts):
    """Distance-weighted mean of the third column of pts at (x, y) (1 mm softening)."""
    P = np.asarray(pts, float)
    w = 1.0 / ((P[:, 0] - x) ** 2 + (P[:, 1] - y) ** 2 + 1.0e6)
    return float((w * P[:, 2]).sum() / w.sum())


def offset_model(pts, max_tilt_um_mm=2.0, max_rms_um=3.0, outlier_um=3.0):
    """The 10x -> 100x focus calibration from manual 100x points [(x, y, 100x Z - 10x map Z)]: a plane of the offset
    (the two objectives' focus surfaces a little tilted to each other) when >= 3 points span one, the worst point left
    out while it is > outlier_um off the plane through the others (a focus on dust; at least 3 kept), its tilt <=
    max_tilt_um_mm and its rms <= max_rms_um; else the median (points > outlier_um from it left out).  Returns
    (offset(x, y), description), or (None, '') without points."""
    if not pts:
        return None, ""
    P = np.asarray(pts, float)
    if len(P) >= 3:
        Q = P.copy()
        pl = fit_plane([tuple(q) for q in Q])
        while pl is not None and len(Q) > 3:
            r = np.abs(Q[:, 2] - (pl[0] * Q[:, 0] + pl[1] * Q[:, 1] + pl[2]))
            k = int(np.argmax(r))
            if r[k] <= outlier_um:
                break
            Q = np.delete(Q, k, axis=0)
            pl = fit_plane([tuple(q) for q in Q])
        if pl is not None:
            tilt = math.hypot(pl[0], pl[1]) * 1000.0
            rms = float(np.sqrt(np.mean((Q[:, 2] - (pl[0] * Q[:, 0] + pl[1] * Q[:, 1] + pl[2])) ** 2)))
            if tilt <= max_tilt_um_mm and rms <= max_rms_um:
                a, b, c = pl
                return (lambda x, y: float(a * x + b * y + c)), (f"a plane, {f3(tilt)} um/mm, rms {f3(rms)} um, "
                                                                 f"{len(Q)} of {len(P)} points")
    med = float(np.median(P[:, 2]))
    Q = P[np.abs(P[:, 2] - med) <= outlier_um]
    if len(Q):
        med = float(np.median(Q[:, 2]))
    return (lambda x, y: med), f"the median {f3(med)} um of {len(Q)} of {len(P)} points"


def nearest_dist(x, y, pts):
    if not len(pts):
        return 1e9
    P = np.asarray(pts, float)
    return float(np.hypot(P[:, 0] - x, P[:, 1] - y).min())


def hows_at(x, y, lab, results, tol_um=15.0):
    """How the target at (x, y) labelled lab was focused, from results [(x, y, how, label)]: matched by position, or by
    label within 300 um (the stage moved by hand while focusing).  Labels alone are not enough: those of the
    provisional target lists are not those of the final one."""
    out = []
    for q in results:
        d = math.hypot(q[0] - x, q[1] - y)
        if d < tol_um or (len(q) > 3 and lab and q[3] == lab and d < 300.0):
            out.append(q[2])
    return out


def retake_rows(rows, results):
    """Target rows (dicts with stage_x_um / stage_y_um / label) without a focused image among results, each with
    retake_reason."""
    out = []
    for r in rows:
        try:
            x, y = float(r["stage_x_um"]), float(r["stage_y_um"])
        except (KeyError, TypeError, ValueError):
            continue
        hows = hows_at(x, y, r.get("label"), results)
        if any(h in ("auto", "manual") for h in hows):
            continue
        out.append(dict(r, retake_reason=retake_reason(hows)))
    return out


def retake_reason(hows):
    return "focus not found" if "predicted" in hows else ("skipped" if hows else "not imaged")


def image_name(chip, label, kind, k, prefix=""):
    """File name of a 100x / IR image: <chip>_F<flake>_<thickness>_<kind>[_<re-image>]_<order>.tif, so a folder of them
    sorts by chip, then flake, in a file browser (2026-10-19: Jordan).  label: the target's ("01_F001_15nm");
    anything else (capture mode's "flake_01") is kept as it is."""
    m = re.match(r"^\d+_F(\d+)(?:_(.+))?$", label or "")
    rest = (f"F{int(m.group(1)):03d}" + (f"_{safe_name(m.group(2))}" if m.group(2) else "")) if m else safe_name(label or "flake")
    pre = safe_name(prefix).strip("_")
    return f"{safe_name(chip)}_{rest}_{kind}{'_' + pre if pre else ''}_{int(k):03d}.tif"


def write_rows(path, cols, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(str(tmp), str(path))


def free_index(d, base):
    """First index n such that nothing in d is named <base>_<m>... for any m >= n (indices are never reused)."""
    pat = re.compile("^" + re.escape(base) + r"_(?:run_)?(\d+)(?:[_.].*)?$")
    mx = 0
    if os.path.isdir(d):
        for f in os.listdir(d):
            m = pat.match(f)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def _tree_pairs(src, dst):
    """(source, destination) file pairs of a file or a folder tree."""
    src, dst = Path(src), Path(dst)
    if src.is_file():
        return [(src, dst)]
    out = []
    for root, _dirs, files in os.walk(src):
        for fn in files:
            p = Path(root) / fn
            out.append((p, dst / p.relative_to(src)))
    return out


def share_layout(run_dir, results_dir, run_label, names, dst):
    """(source, destination) pairs for the copy on the share, and {piece: its viewer page in the copy}:

        <run>/                  run file, run log, analysis log
          <piece>/              the viewer's page and tiles (index.html, tiles/), with the 100x images in 100x/
            data/               flake tables, target lists, plan, chip.json, .pos, 100x index, logs, ...

    The viewer is the file people open most, so it sits at the top of the piece's folder (with a "<piece> viewer.html"
    link to it); the local run and results folders keep their layout (the analysis and the tools rely on it)."""
    run_dir, dst = Path(run_dir), Path(dst)
    pairs, viewers = [], {}
    # longest names first: "chip_10_..." must not count as "chip_1"
    order = sorted(names, key=len, reverse=True)

    def piece_of(fname):
        return next((nm for nm in order if fname.startswith(nm + "_")), None)
    for f in sorted(run_dir.iterdir()):
        if f.name.endswith("_tiles") or re.search(r"_tiles_\d+$", f.name):
            continue
        if f.name.startswith(run_label):
            pairs += _tree_pairs(f, dst / f.name)
            continue
        nm = piece_of(f.name)
        if nm is not None:
            pairs += _tree_pairs(f, dst / nm / "data" / f.name)
    if results_dir is not None and Path(results_dir).is_dir():
        res = Path(results_dir)
        for f in sorted(res.iterdir()):
            if f.is_file():
                pairs.append((f, dst / f.name))             # analysis_log.txt
            elif f.name in names:
                for g in sorted(f.iterdir()):
                    if g.is_dir() and g.name.endswith("_viewer"):
                        pairs += _tree_pairs(g, dst / f.name)
                        if (g / "index.html").is_file():
                            viewers[f.name] = dst / f.name / "index.html"
                    else:
                        pairs += _tree_pairs(g, dst / f.name / "data" / g.name)
            else:
                pairs += _tree_pairs(f, dst / f.name)
    return pairs, viewers


def write_copy_done(dst, pieces, rebuild=True, of=None):
    """<run copy>/copy_done.json, written last: the copy is complete.  The vdW Stack Designer builds a run's designer
    files from it only then, and again when this file is newer than its designer run (a re-image updated pieces).
    rebuild: the pieces' pictures changed (a re-image, a copy by hand): "rebuild" = now, the designer builds the whole run
    again; else (a piece copied as its export ended, the IR pictures added at the end) it only adds what is new
    (make_designer_bundle --update).  of: the run's number of pieces while more are still to come ("partial": the
    pieces copied so far of `of`), None = the run is complete."""
    try:
        d = Path(dst) / "copy_done.json"
        old = json.loads(d.read_text()) if d.is_file() else {}
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        old.update(time=now, updated=sorted(set(old.get("updated", [])) | set(pieces)))
        if rebuild:
            old["rebuild"] = now
        else:
            old.setdefault("rebuild", "")             # ("": never; no key at all: a copy before 2026-10-21, rebuilt)
        if of:
            old["partial"] = dict(done=len(old["updated"]), of=int(of))
        else:
            old.pop("partial", None)
        tmp = Path(str(d) + ".tmp")
        tmp.write_text(json.dumps(old, indent=1))
        os.replace(str(tmp), str(d))
    except OSError:
        pass


def write_viewer_link(path, target):
    """A small page that opens the viewer (a clearly named file next to the viewer's index.html)."""
    rel = Path(target).name
    Path(path).write_text(f"""<!doctype html><meta charset="utf-8"><title>{Path(path).stem}</title>
<meta http-equiv="refresh" content="0; url={rel}">
<p>Opening the viewer ... if nothing happens, open <a href="{rel}">{rel}</a>.</p>
""", encoding="utf-8")


DESIGNER_LAST = ("run.json", "bundle.json")      # what the vdW Stack Designer looks for in a run's designer folder


def copy_parallel(pairs, workers=8, progress=None):
    """Copy files with several threads (a viewer is thousands of small files; one at a time over the network share took
    ~15 min for a 2-piece run).  A designer folder's run.json and bundle.json are copied last, after everything else
    (bundle.json the very last): the designer finds a run by its bundle.json, and one found halfway through the copy
    loaded without its pictures (2026-10-13).  When they replace older ones, those are removed first: during the copy
    the run is gone from the designer rather than an old bundle shown with new pictures.  Returns the bytes copied.
    progress(bytes done, bytes in all, files done, files in all): called after each file (from the copy threads)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    pairs = [(Path(a), Path(b)) for a, b in pairs]
    total = 0
    if progress is not None:
        for a, _ in pairs:
            try:
                total += a.stat().st_size
            except OSError:
                pass
    lock, st = threading.Lock(), dict(b=0, n=0)
    for d in {q.parent for _, q in pairs}:
        d.mkdir(parents=True, exist_ok=True)
    last = [pq for pq in pairs if pq[1].name in DESIGNER_LAST and pq[1].parent.name == "designer"]
    rest = [pq for pq in pairs if pq not in last]
    for pq in sorted(last, key=lambda pq: -DESIGNER_LAST.index(pq[1].name)):          # (the old bundle.json first)
        try:
            if pq[1].is_file():
                pq[1].unlink()
        except OSError:
            pass

    def one(pq):
        shutil.copy2(pq[0], pq[1])
        sz = pq[0].stat().st_size
        if progress is not None:
            with lock:
                st["b"] += sz
                st["n"] += 1
                b, k = st["b"], st["n"]
            progress(b, total, k, len(pairs))
        return sz
    with ThreadPoolExecutor(workers) as ex:
        nb = sum(ex.map(one, rest))
    for pq in sorted(last, key=lambda pq: DESIGNER_LAST.index(pq[1].name)):     # (run.json, then bundle.json: the very last)
        nb += one(pq)
    return nb


class TileWriter:
    """Writes tiles (TIFF, then metadata.txt: the analysis takes a tile once its metadata.txt is there) in a thread,
    while the scan moves to the next tile.  A failed write is raised by the next put() / flush()."""

    def __init__(self, depth=8):
        import queue
        import threading
        self.q = queue.Queue(maxsize=depth)
        self.err = None
        self.th = threading.Thread(target=self._run, daemon=True, name="tile writer")
        self.th.start()

    @staticmethod
    def write(job):
        import tifffile
        tif, img, mdp, md = job
        Path(tif).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(tif, img)
        Path(mdp).write_text(md)

    def _run(self):
        while True:
            job = self.q.get()
            try:
                if job is not None and self.err is None:
                    self.write(job)
            except Exception as e:
                self.err = e
            finally:
                self.q.task_done()

    def put(self, job):
        if self.err is not None:
            raise RuntimeError(f"writing a tile failed: {self.err}")
        self.q.put(job)

    def flush(self):
        self.q.join()
        if self.err is not None:
            e, self.err = self.err, None
            raise RuntimeError(f"writing a tile failed: {e}")


# ------------------------------------------------------------------------------------------------- the scanner
class Scanner:
    def __init__(self, core, s: Settings, ui, log_path=None):
        self.core, self.s, self.ui = core, s, ui
        self.logf = None
        self.logbuf = []
        self.stop_all = False
        self.z_shift = 0.0                               # (the focus moved during a pause: added to every Z sent; pause_now)
        self.last_af = {}                                # objective is 100x -> (x, y, z) of the last focus found (pause_now)
        self.last_pic = None                             # the last tile / focused 100x picture: frame, x, y, z (pause check)
        self.contact = {}                                # who runs the job (contact_of): the pause check's message
        self.n_snaps = 0
        self.move_box = None
        self.zref = NAN
        self.REF = None
        self.fa = self.fb = 0.0
        self.fc = 0.0
        self.FQ = None
        self.FS = None                          # a polygon piece's smooth focus surface z(x, y) (smooth_surface), else None
        self.lines = self.side_type = self.arcs = None
        self.polygon = None                     # the piece's scan area drawn by hand (chip_type "polygon"), else None
        self.analysis = None                    # local flakeid.live process
        self.nose = None                        # coded nosepiece reader (flakeid.scope.nosepiece)
        self.M = np.array(s.M, float)
        # where the time goes: wall time per phase (set_phase) and time inside each kind of operation (op)
        self.phase_t, self.cur_phase, self.t_phase, self.t_run0 = {}, None, time.time(), time.time()
        self.ops, self.opn = {}, {}
        self.writer = None                      # background tile writer (TileWriter)
        self.last_xy = (NAN, NAN)
        self.z_cmd, self.t_moved = NAN, 0.0       # last Z sent (set_z); when the last move ended (time.perf_counter)
        self.plus_ok = [False, False]           # per axis: the last move was in + with the backlash taken up (approach_to)
        self.recent_offs = []                 # 100x - 10x focus offsets in the order measured (whole run)
        self.sub100 = []                        # 100x substrate colours of this piece (re-centring)
        self.focus_jumps = []                   # focus jumps found during the 100x imaging (image_flake)
        self.z_limits = {}                      # focus drive short of where it was sent: direction -> {z, n, sent} (z_limit_hit)
        self.mosaic_s = {"100x": 0.0, "IR": 0.0}   # time spent on terrace mosaics this run (hundred_mosaic_budget_min)
        self.mosaic_over = set()                # ... budgets used up (said once)
        self.last_results = []                  # (x, y, how) of every target of the last image_targets()
        self.manifest = None                    # multi mode: the run file (<run>.json)
        self.mono_view = False                  # the live view shows a monochrome camera (the domain camera)
        self.results_dest = None                # where this run's results are copied (recorded in the run file)
        if log_path:
            self.open_log(log_path)

    # ---- log
    def say(self, msg):
        msg = getattr(self, "say_prefix", "") + msg          # (a run finishing in the background: its name first)
        print(msg, flush=True)
        try:
            self.ui.log(msg)
        except Exception:
            pass
        line = time.strftime("%H:%M:%S ") + msg + "\n"
        if self.logf:
            self.logf.write(line); self.logf.flush()
        else:
            self.logbuf.append(line)

    def open_log(self, path, append=False):
        if self.logf:
            self.logf.close()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.logf = open(path, "a" if append else "w", encoding="utf-8")
        self.logf.write("".join(self.logbuf)); self.logf.flush()
        self.logbuf = []

    def fail(self, msg):
        self.say("ERROR: " + msg)
        return Stop(msg)

    # ---- timing
    def set_phase(self, name):
        now = time.time()
        if self.cur_phase is not None:
            self.phase_t[self.cur_phase] = self.phase_t.get(self.cur_phase, 0.0) + now - self.t_phase
        self.cur_phase, self.t_phase = name, now

    def op(self, name, t0):
        self.ops[name] = self.ops.get(name, 0.0) + time.time() - t0
        self.opn[name] = self.opn.get(name, 0) + 1

    def report_times(self):
        self.set_phase(self.cur_phase)
        tot = time.time() - self.t_run0
        self.say("")
        if self.z_limits:
            self.report_z_limits()
            self.say("")
        self.say(f"==== time by phase ({tot / 60:.1f} min in all)")
        for k, v in sorted(self.phase_t.items(), key=lambda kv: -kv[1]):
            self.say(f"  {k:42s} {v / 60:6.1f} min  {100 * v / max(tot, 1e-9):5.1f} %")
        self.say("  time inside operations (overlapping: autofocus includes its Z moves and frames):")
        for k, v in sorted(self.ops.items(), key=lambda kv: -kv[1]):
            n = self.opn.get(k, 1)
            self.say(f"  {k:42s} {v / 60:6.1f} min  {n:6d} x {1000 * v / max(n, 1):6.0f} ms")

    # ---- devices
    def setup_devices(self):
        c = self.core
        self.xy, self.zdev, self.cam = c.getXYStageDevice(), c.getFocusDevice(), c.getCameraDevice()
        self.has_z = bool(self.zdev)
        self.W, self.H = int(c.getImageWidth()), int(c.getImageHeight())
        self.X0, self.Y0 = c.getXPosition(self.xy), c.getYPosition(self.xy)
        if hasattr(c, "setTimeoutMs"):            # long moves (between pieces) take more than Micro-Manager's default 5 s
            try:
                c.setTimeoutMs(int(max(c.getTimeoutMs(), self.s.stage_timeout_s * 1000)))
            except Exception as e:
                self.say(f"  (could not set the device timeout: {e})")
        self.z_scale = self.s.z_scale
        self.FOVX = abs(self.M[0, 0]) * self.W + abs(self.M[0, 1]) * self.H
        self.FOVY = abs(self.M[1, 0]) * self.W + abs(self.M[1, 1]) * self.H
        # focus-drive step size: set it so that Z is in real um; else convert with z_scale
        if self.has_z and self.s.z_stepsize_um > 0:
            if not c.hasProperty(self.zdev, "StepSize_um"):
                raise self.fail(f"{self.zdev} has no StepSize_um property; set z_stepsize_um = 0 and z_scale by hand")
            old = c.getProperty(self.zdev, "StepSize_um")
            zr = c.getPosition(self.zdev)
            if abs(float(old) - self.s.z_stepsize_um) > 1e-7:
                try:
                    c.setProperty(self.zdev, "StepSize_um", repr(self.s.z_stepsize_um)); c.waitForDevice(self.zdev)
                except Exception as e:
                    self.say(f"  ({self.zdev} StepSize_um cannot be set: {e})")
            new = c.getProperty(self.zdev, "StepSize_um")
            self.z_scale = float(new) / self.s.z_stepsize_um
            self.say(f"  {self.zdev} StepSize_um: {old} -> {new} um per step; " +
                     ("Z in real um" if abs(self.z_scale - 1) < 1e-6 else f"converting in software, z_scale {f3(self.z_scale)} drive units per um")
                     + f"; Z read {f3(zr)}, now {f3(c.getPosition(self.zdev))}")
        self.Z0 = self.z_read() if self.has_z else 0.0

    # (z_shift: how far the focus moved while the scan was paused, e.g. the coarse knob, which the drive does not see:
    # every Z the scan works with stays in its first frame, the drive is sent z + z_shift)
    def z_read(self): return self.core.getPosition(self.zdev) / self.z_scale - self.z_shift
    def z_write(self, z): self.core.setPosition(self.zdev, (z + self.z_shift) * self.z_scale)

    def set_z(self, z):
        ref = self.Z0 if isnan(self.zref) else self.zref
        if abs(z - ref) > self.s.max_dz_um:
            return False
        t_op = time.time()
        self.z_write(z)
        self.core.waitForDevice(self.zdev)
        self.op("focus moves (Z)", t_op)
        self.z_cmd = z                               # (where Z was sent: the tiles record it instead of reading it back)
        if self.s.z_settle_ms > 0:
            time.sleep(self.s.z_settle_ms / 1000.0)
        self.t_moved = time.perf_counter()           # (frame_timestamps: frames must start exposing after this)
        return True

    def check_z_reach(self, where):
        """Read the focus drive back: short of the last Z sent by more than z_reach_tol_um, it is at a limit of its travel
        (z_limit_hit).  Returns the Z read."""
        if not self.has_z or isnan(self.z_cmd):
            return self.z_cmd
        zr = self.z_read()
        if abs(zr - self.z_cmd) > self.s.z_reach_tol_um:
            self.z_limit_hit(self.z_cmd, zr, where)
        return zr

    def z_limit_hit(self, sent, read, where):
        """The focus drive stopped at `read` when sent to `sent`: logged every time, and the first time in each
        direction a message to the operator (the pictures beyond it cannot be in focus)."""
        d = "lower" if sent < read else "upper"
        L = self.z_limits.get(d)
        first = L is None
        if first:
            L = self.z_limits[d] = {"z": read, "n": 0, "sent": sent}
        L["n"] += 1
        L["z"] = min(L["z"], read) if d == "lower" else max(L["z"], read)
        L["sent"] = min(L["sent"], sent) if d == "lower" else max(L["sent"], sent)
        self.say(f"    WARNING: focus drive at its {d} limit: sent to Z {f3(sent)}, reads {f3(read)} um ({where})")
        if not first:
            return
        what = (f"The focus drive stopped at Z {f1(read)} um when sent to {f1(sent)} um ({where}): it is at the {d} limit "
                "of its travel, so whatever needs focus beyond it is imaged out of focus.  Check the Oasis focus limits, "
                "or bring the sample's focus nearer the middle of the drive's travel (coarse focus) for the next scan.")
        try:
            from .notify import send
            send(self.s, "operator", f"{self.s.name or 'scan'}: focus drive at its {d} limit", what, log=self.say,
                 run=getattr(self, "slack_run", None))
        except Exception as e:
            self.say(f"  (no message sent: {e})")

    def report_z_limits(self):
        for d, L in sorted(self.z_limits.items()):
            self.say(f"  FOCUS DRIVE LIMIT ({d}): {L['n']} time(s), stopped at {f3(L['z'])} um (sent as far as {f3(L['sent'])} um).  "
                     "Pictures needing focus beyond it are out of focus: check the Oasis focus limits, or start the next scan with "
                     "the sample's focus nearer the middle of the drive's travel.")

    def approach_to(self, x, y, a=None):
        """Move onto a 100x / stacking-domain target always from the same direction: first xy_approach_um short of it
        in -x and -y, then a short +x / +y move onto it (the stage's backlash and settling are then the same every
        time).  2026-10-15, 1190 100x shots against their 10x pictures: misses > 40 um followed long -y moves (10 % of
        moves > 1 mm in -y, 0-0.7 % of moves > 1 mm in +y), ~100 um in -y as if the stage stopped short; the median y
        miss was -8 um after -y moves and -2 um after +y moves.  An axis on which the move is already at least
        xy_approach_clear_um in + needs no pre-move (it arrives from - and its backlash is taken up by the move itself):
        no extra move when both are, and the pre-move only on the other axis otherwise.  a: the pre-move (default
        xy_approach_um)."""
        a = float(self.s.xy_approach_um or 0.0) if a is None else float(a)
        if a > 0:
            clr, (lx, ly) = float(self.s.xy_approach_clear_um or 0.0), self.last_xy

            def direct(d, plus):                       # (this axis arrives from - with its backlash taken up)
                return clr > 0 and not isnan(d) and (d >= clr or (d >= 0 and plus))
            px = x if direct(x - lx, self.plus_ok[0]) else x - a
            py = y if direct(y - ly, self.plus_ok[1]) else y - a
            if (px, py) != (x, y):
                self.move_to(px, py)
        self.move_to(x, y)

    def move_allowed(self, x, y):
        """May the stage go to (x, y)?  Within move_box when one is set (the 100x targets' box; a scan area drawn by
        hand: its polygon's box), else within max_travel_um of the start."""
        if self.move_box is not None:
            b = self.move_box
            return b[0] <= x <= b[1] and b[2] <= y <= b[3]
        return math.hypot(x - self.X0, y - self.Y0) <= self.s.max_travel_um + 1.0

    def move_to(self, x, y, z_pre=None, check=True):
        """Move the stage (checked against the allowed area and the readback).  z_pre: a focus position sent right after
        the XY command, so the Z move happens during the XY move (the tile scan's backlash approach).  check=False: no
        readback (tiles, tile_readback_every): the position is taken as sent; a move that raised is always checked."""
        zp_sent = False
        if not self.move_allowed(x, y):
            if self.move_box is not None:
                b = self.move_box
                raise self.fail(f"move to {f1(x)}, {f1(y)} is outside the allowed box ({f1(b[0])} .. {f1(b[1])}, "
                                f"{f1(b[2])} .. {f1(b[3])})")
            raise self.fail(f"move to {f1(x)}, {f1(y)} is beyond max_travel_um from the start")
        if not self.reachable(x, y):
            raise StageLimit(f"{f1(x)}, {f1(y)} is beyond the stage's travel {self.box_text()}")
        if self.s.verbose:
            self.say(f"    move to {f1(x)}, {f1(y)}")
        c = self.core
        t_op = time.time()
        for attempt in range(2):
            try:
                c.setXYPosition(self.xy, x, y)
                if z_pre is not None and attempt == 0:
                    # the focus drive's backlash approach during the XY move (separate axes of the same controller)
                    self.z_write(z_pre)
                    z_pre = None
                    zp_sent = True
                c.waitForDevice(self.xy)
                if not check:
                    rx, ry = x, y
                    break
            except Exception as e:
                check = True
                # a stage timeout ("Wait for device ... timed out") or a serial hiccup: give the stage time to finish,
                # then check where it is; one more try if it is not there
                self.say(f"    stage move to {f1(x)}, {f1(y)}: {str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__}; waiting for the stage")
                t_end = time.time() + self.s.stage_timeout_s
                while time.time() < t_end:
                    try:
                        if not (hasattr(c, "deviceBusy") and c.deviceBusy(self.xy)):
                            break
                    except Exception:
                        pass
                    time.sleep(0.5)
            try:
                rx, ry = c.getXPosition(self.xy), c.getYPosition(self.xy)
            except Exception as e:
                rx, ry = NAN, NAN
                self.say(f"    stage position not readable: {e}")
            if math.hypot(rx - x, ry - y) <= self.s.arrive_tol_um:
                break
            if attempt == 0:
                self.say(f"    stage not at {f1(x)}, {f1(y)} (reports {f1(rx)}, {f1(ry)}): moving again")
                time.sleep(1.0)
        if not math.hypot(rx - x, ry - y) <= self.s.arrive_tol_um:
            # stopped short on one axis, the way it was going, both tries: the stage's limit (it stops there)
            lim = []
            if not isnan(rx) and not isnan(ry):
                for ax, (sent, got, last) in enumerate(((x, rx, self.last_xy[0]), (y, ry, self.last_xy[1]))):
                    if abs(got - sent) > self.s.arrive_tol_um and not isnan(last) and (sent - last) * (sent - got) > 0 \
                            and abs(got - last) < abs(sent - last) + self.s.arrive_tol_um:
                        lim.append((ax, 1 if sent > got else 0, got))
            if lim and self.s.stage_limit_learn:
                for ax, side, got in lim:
                    self.learn_limit(ax, side, got)
                self.last_xy = (rx, ry)
                raise StageLimit(f"the stage stopped at {f1(rx)}, {f1(ry)} on its way to {f1(x)}, {f1(y)}: its travel ends there "
                                 f"(remembered: {self.box_text()})")
            raise self.fail(f"stage did not arrive: sent {f1(x)}, {f1(y)} but it reports {f1(rx)}, {f1(ry)} (device {self.xy})")
        # per axis: was the last move a + move that took the backlash up (>= xy_approach_clear_um, or any + move after
        # one)?  approach_to needs no pre-move on such an axis for a further + move
        clr = float(self.s.xy_approach_clear_um or 0.0)
        for ax, d in enumerate((x - self.last_xy[0], y - self.last_xy[1])):
            if isnan(d):
                self.plus_ok[ax] = False
            elif abs(d) >= 0.05:
                self.plus_ok[ax] = d > 0 and (d >= clr or self.plus_ok[ax])
        self.last_xy = (rx, ry)
        if zp_sent:
            c.waitForDevice(self.zdev)
        self.t_moved = time.perf_counter()
        self.op("stage moves (XY)", t_op)

    # ---- the stage's travel (stage_limits_um, and limits learned when the stage stopped short at one)
    def stage_box(self):
        """[x min, x max, y min, y max] the stage can reach (None: not known), the settings' and the learned ones."""
        if getattr(self, "_box", None) is None:
            b = [None] * 4
            for src in (self.s.stage_limits_um, (self._learned_limits() or {}).get(str(getattr(self, "xy", "")))):
                if src and len(src) == 4:
                    for i, v in enumerate(src):
                        if v is None:
                            continue
                        v = float(v)
                        b[i] = v if b[i] is None else (max(b[i], v) if i in (0, 2) else min(b[i], v))
            self._box = b
        return self._box

    def _learned_limits(self):
        try:
            f = Path(self.s.local_root) / "stage_limits.json"
            return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        except (OSError, ValueError):
            return {}

    def learn_limit(self, ax, side, value):
        """The stage stopped at value on axis ax (0 x, 1 y) going up (side 1) or down (0): its limit there, a little
        inside (stage_limit_margin_um); kept in <local_root>/stage_limits.json for the next runs."""
        b = self.stage_box()
        i = 2 * ax + side
        v = float(value) - self.s.stage_limit_margin_um if side == 1 else float(value) + self.s.stage_limit_margin_um
        b[i] = v if b[i] is None else (min(b[i], v) if side == 1 else max(b[i], v))
        self.say(f"  stage travel ends at {'xy'[ax]} = {f1(value)} um ({'max' if side else 'min'}): what lies beyond is left out")
        try:
            f = Path(self.s.local_root) / "stage_limits.json"
            L = self._learned_limits()
            L[str(self.xy)] = b
            f.write_text(json.dumps(L, indent=1), encoding="utf-8")
        except OSError:
            pass

    def reachable(self, x, y):
        b = self.stage_box()
        return not ((b[0] is not None and x < b[0]) or (b[1] is not None and x > b[1]) or
                    (b[2] is not None and y < b[2]) or (b[3] is not None and y > b[3]))

    def clamp_reach(self, x, y):
        b = self.stage_box()
        if b[0] is not None: x = max(x, b[0])
        if b[1] is not None: x = min(x, b[1])
        if b[2] is not None: y = max(y, b[2])
        if b[3] is not None: y = min(y, b[3])
        return x, y

    def box_text(self):
        b = self.stage_box()
        g = lambda v: "?" if v is None else f1(v)                # noqa: E731
        return f"(x {g(b[0])} .. {g(b[1])}, y {g(b[2])} .. {g(b[3])} um)"

    def stage_self_test(self):
        c = self.core
        x0, y0 = c.getXPosition(self.xy), c.getYPosition(self.xy)
        for dx, dy in ((0, 0), (self.s.selftest_um, 0), (0, self.s.selftest_um), (0, 0)):
            tx, ty = x0 + dx, y0 + dy
            c.setXYPosition(self.xy, tx, ty); c.waitForDevice(self.xy)
            rx, ry = c.getXPosition(self.xy), c.getYPosition(self.xy)
            if abs(rx - tx) > self.s.arrive_tol_um or abs(ry - ty) > self.s.arrive_tol_um:
                m = (f"stage self-test failed: sent {f1(tx)}, {f1(ty)} but the stage reports {f1(rx)}, {f1(ry)}.  The stage "
                     "readout and moves disagree: switch the stage controller off and on (and check the stage settings) before scanning.")
                try:
                    c.setXYPosition(self.xy, x0, y0)
                except Exception:
                    pass
                self.ui.message("Chip scan: stage problem", m)
                raise self.fail(m)
        self.say(f"  stage self-test passed (readout follows +{f1(self.s.selftest_um)} um moves in X and Y)")

    @staticmethod
    def same_value(a, b):
        try:
            return abs(float(a) - float(b)) < 1e-3
        except ValueError:
            return a.strip().lower() == b.strip().lower()

    def apply_camera(self, exposure):
        self.stream_off()
        c, cam = self.core, self.core.getCameraDevice()
        problems = []
        for p in self.s.camera_props:
            name, want, opt = p[0], p[1], len(p) > 2
            if not c.hasProperty(cam, name):
                if not opt:
                    problems.append(f"{name}: not a property of {cam}")
                continue
            if not self.same_value(c.getProperty(cam, name), want):
                try:
                    c.setProperty(cam, name, want)
                except Exception as e:
                    if not opt:
                        problems.append(f"{name}: could not set {want} ({e})")
        c.setExposure(exposure)
        c.waitForDevice(cam)
        rep = []
        for p in self.s.camera_props:
            name, want = p[0], p[1]
            if not c.hasProperty(cam, name):
                continue
            now = c.getProperty(cam, name)
            line = f"  {name} = {now}"
            if not self.same_value(now, want):
                line += f"   <-- wanted {want}"
                if len(p) <= 2:
                    problems.append(f"{name} is {now}, wanted {want}")
            rep.append(line)
        for extra in ("Sensor data bit depth", "Exposure", "Frame rate"):
            if c.hasProperty(cam, extra):
                rep.append(f"  ({extra} = {c.getProperty(cam, extra)})")
        ex = c.getExposure()
        rep.append(f"  exposure = {f3(ex)} ms")
        if abs(ex - exposure) > 1e-3:
            problems.append(f"exposure is {f3(ex)} ms, wanted {f3(exposure)}")
        self.say(f"  camera {cam}:\n" + "\n".join(rep))
        if problems:
            raise self.fail("camera settings not as requested:\n  " + "\n  ".join(problems))

    def set_props(self, cam, props):
        self.stream_off()
        for p in props:
            nm, want, opt = p[0], p[1], len(p) > 2
            if not self.core.hasProperty(cam, nm):
                if not opt:
                    raise self.fail(f"{nm} is not a property of {cam}")
                continue
            try:
                self.core.setProperty(cam, nm, want)
            except Exception as e:
                if not opt:
                    raise self.fail(f"could not set {nm} = {want} on {cam}: {e}")
            self.say(f"    {cam} {nm} = {self.core.getProperty(cam, nm)}")

    def stop_streaming(self):
        try:
            if self.core.isSequenceRunning():
                self.stop_sequence()
                self.say("    (stopped a running sequence acquisition)")
        except Exception:
            pass

    def stop_sequence(self, timeout=10.0):
        """Stop the camera's stream and wait until it says it has stopped (the Raptor's driver takes a moment, and a
        snap, a new start or a property change before then fails).  True when stopped."""
        c = self.core
        t0 = time.time()
        try:
            c.stopSequenceAcquisition()
        except Exception:
            pass
        try:
            while c.isSequenceRunning() and time.time() - t0 < timeout:
                time.sleep(0.02)
            if c.isSequenceRunning():
                self.say(f"    WARNING: the camera {c.getCameraDevice()} is still live {timeout:g} s after the stop")
                return False
        except Exception:
            pass
        return True

    def wake_camera(self, seconds=1.5):
        """The selected camera live for a moment, then stopped (the Raptor gave empty frames until it had been live once).
        Returns the frames seen."""
        c = self.core
        n, t0 = 0, time.time()
        try:
            if c.isSequenceRunning():
                self.stop_sequence()
            c.startContinuousSequenceAcquisition(0)
            while time.time() - t0 < seconds:
                k = c.getRemainingImageCount()
                if k:
                    n += k
                    c.clearCircularBuffer()
                else:
                    time.sleep(0.01)
        except Exception as e:
            self.say(f"    (camera not woken: {e})")
        finally:
            self.stop_sequence()
        return n

    def alert(self, what="", image=None):
        """The operator is needed: a sound (operator_sound; Windows' exclamation sound, nothing elsewhere) and, with
        notify_* set, a Slack / email message saying what for (flakeid/scope/notify.py)."""
        if self.s.operator_sound:
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass
        if what:
            try:
                from .notify import send
                send(self.s, "operator", f"{self.s.name or 'scan'}: you are needed", what, log=self.say, image=image,
                     run=getattr(self, "slack_run", None))
            except Exception as e:
                self.say(f"  (no message sent: {e})")

    # ---- objective (coded nosepiece)
    def open_nosepiece(self):
        s = self.s
        if self.nose is not None or not s.nosepiece_port:
            return
        try:
            from .nosepiece import Nosepiece
            self.nose = Nosepiece(s.nosepiece_port)
            pos = self.nose.position()
            self.say(f"  nosepiece {self.nose.unit} on {s.nosepiece_port}: position {pos} ({self.objective_name(pos)})")
            if not s.objective_positions:
                self.say("  (objective_positions not set: the objective is recorded but not checked)")
        except Exception as e:
            self.nose = None
            self.say(f"  WARNING: nosepiece not read ({e}); the objective is not checked")

    def close_nosepiece(self):
        """Free the nosepiece's serial port: the next job (start screen) opens it again, and a port still held by the
        previous job cannot be opened (Windows), which left the objective windows without the position (2026-10-04)."""
        n, self.nose = self.nose, None
        if n is not None:
            try:
                n.close()
            except Exception:
                pass

    def objective_name(self, pos):
        for k, v in (self.s.objective_positions or {}).items():
            if v == pos:
                return k
        return "unknown" if pos is None else f"position {pos}"

    def objective(self):
        """(position, name) of the objective in, or (None, '') without a nosepiece reader."""
        if self.nose is None:
            return None, ""
        pos = self.nose.position()
        return pos, self.objective_name(pos)

    # ---- pause and resume (Avram, 2026-10-27): someone takes a few pictures in between; the chips stay where they are
    def stop_check(self):
        """Stop pressed (ui.stop_requested): Stop raised.  Called on every camera frame and at every pause point, so a
        stop takes effect within seconds in every stage (2026-10-27: during the 10x focus points nothing checked it
        while an autofocus ran)."""
        if getattr(self.ui, "stop_requested", False):
            self.stop_all = True
            raise Stop("stopped by the user")

    def pause_point(self):
        """A safe point (between two tiles, focus points, edge searches, 100x flakes, IR pictures; while waiting for a
        target list): Stop takes effect here, and the scan pauses here if Pause was pressed (ui.pause_requested) and goes
        on when it is resumed (Stop raised if it is stopped instead)."""
        self.stop_check()
        if getattr(self.ui, "pause_requested", False):
            self.ui.pause_requested = False
            self.pause_now()

    def _pause_picture(self, x, y, z, hundred):
        """The check's picture when the scan has none of its own yet (paused before its first picture with this objective):
        a frame here, at the focus found before the pause (z; NaN: none found, then the frame shows the place only and the
        focus is not checked at Resume).  None if the camera gives none."""
        try:
            if not isnan(z) and not self.set_z(z):
                self.z_write(z)
            img = self.frame()
        except Stop:
            raise
        except Exception as e:
            self.say(f"    (no picture taken before the pause: {e})")
            return None
        return dict(frame=img, x=float(x), y=float(y), z=float(z), hundred=hundred, what="the picture taken when the scan paused", at_pause=True)

    def _pause_notice(self, what, text):
        """A Slack / email notice (notify_*) to whoever runs the scan: paused, focus moved (not "you are needed")."""
        try:
            from .notify import send
            send(self.s, "operator", f"{self.s.name or 'scan'}: {what}", text, log=self.say, run=getattr(self, "slack_run", None))
        except Exception as e:
            self.say(f"  (no message sent: {e})")

    def _pause_refine(self, zm, zref, hundred):
        """The pause check: the person focused at the reference (the drive reading zm; zref the reading that was in focus
        there); a short autofocus around it refines it, in the offset frame (the hand focus taken as zref: the scan's Z
        limits, max_dz_um, are about zref, and a coarse-knob change can be far beyond them), kept only if it agrees
        (within 3 um at 10x, 1 um at 100x).  Returns the new in-focus reading."""
        d0 = zm - zref
        self.z_shift += d0
        try:
            za = self._pause_focus(zref, hundred) if self.has_z else NAN
        finally:
            self.z_shift -= d0
        if not isnan(za) and abs(za - zref) <= (1.0 if hundred else 3.0):
            self.say(f"    focused by hand at Z {f3(zm)} um, refined by the autofocus by {za - zref:+.2f} um")
            return zm + (za - zref)
        self.say(f"    focused by hand at Z {f3(zm)} um" + ("" if isnan(za) else f" (the autofocus found {za - zref:+.1f} um from it, too far: "
                                                                     "the hand focus kept)"))
        return zm

    def _pause_focus(self, zc, hundred):
        """Autofocus here around zc (the objective in: 10x af10, 100x af100_first; then the wide af_long): Z, or NaN if
        none is found."""
        if not self.has_z:
            return NAN
        s = self.s
        for p in ((s.af100_first if hundred else s.af10), s.af_long):
            try:
                r = self.autofocus(zc, p)
            except Stop:
                raise
            except Exception as e:
                self.say(f"    (autofocus failed: {type(e).__name__}: {e})")
                continue
            if r[2]:
                return float(r[0])
        return NAN

    def pause_now(self):
        """Paused: where the scan is (stage, Z, objective, exposure) is noted and the focus there measured (autofocus, the
        reference), then the pause window (live view, Save picture) until Resume.  Resume puts it all back: the
        objective (asked for if it was changed), the camera settings, the stage, Z; the focus is measured again at the
        same place and, if it moved (the coarse knob, which the drive does not see), every Z from then on is shifted by
        that much (z_shift); without an autofocus result it is focused by hand."""
        s, c = self.s, self.core
        t0, phase = time.time(), self.cur_phase
        status = getattr(self.ui, "_last_status", "")
        self.set_phase("paused")
        self.stream_off()
        hundred = getattr(self, "view_scale", 1.0) < 1.0
        x, y = c.getXPosition(self.xy), c.getYPosition(self.xy)
        z = self.z_read() if self.has_z else NAN
        exp = c.getExposure()
        pos, objn = self.objective()
        objn = objn if pos is not None else ("100x" if hundred else "10x")
        self.say(f"  PAUSED ({phase}) at stage {f1(x)}, {f1(y)} um, Z {f3(z)} um, objective {objn}, exposure {f1(exp)} ms")
        self.ui.progress(self, f"{s.name}: paused ({phase})")
        zref, rx, ry = NAN, x, y                          # (the focus reference and where it was measured)
        # (light, 2026-10-27: Python closed on Pause on the microscope): with a picture of the scan's own (the check at
        # Resume compares with it) nothing is focused or moved here, the window opens at once; only without one (paused
        # before the first picture with this objective) an autofocus here is the reference
        has_ref = self.last_pic is not None and self.last_pic.get("hundred") == hundred
        pause_pic = None                                  # (without one: a picture taken here is the check's picture)
        if has_ref:
            self.say(f"    focus reference: the scan's last picture ({self.last_pic.get('what')}), checked at Resume")
        elif self.has_z and s.pause_refocus:
            zref = self._pause_focus(z, hundred)
            last = self.last_af.get(hundred)
            moved = False
            if isnan(zref) and last is not None:
                # (nothing to focus on here, e.g. bare substrate at 100x: where the last focus was found with this objective)
                rx, ry = last[0], last[1]
                self.approach_to(rx, ry)
                moved = True
                zref = self._pause_focus(last[2], hundred)
            # (no focus found at all: no reference, so no correction; the scan's own Z need not be in focus, e.g. just
            # after the change to 100x, at 10x heights)
            self.say(f"    focus before the pause: Z {f3(zref)} um at {f1(rx)}, {f1(ry)}" if not isnan(zref) else
                     "    no focus found before the pause: the focus cannot be checked at Resume")
            pause_pic = self._pause_picture(rx, ry, zref, hundred)
            if moved:
                self.approach_to(x, y)
            if not self.set_z(z):
                self.z_write(z)
        if not has_ref and pause_pic is None:
            pause_pic = self._pause_picture(x, y, NAN, hundred)
        import threading                                  # (the notice does not hold the pause window up)
        threading.Thread(target=self._pause_notice, daemon=True, name="pause notice",
                         args=("paused", f"The scan was paused ({phase}) to take pictures in between; it goes on with Resume "
                               "in the scan program.")).start()
        vs0, on0 = getattr(self, "view_scale", 1.0), getattr(self, "objective_now", None)
        ans = self.ui.pause_window(self, dict(phase=phase, x=x, y=y, z=z, objective=objn, exposure=exp, hundred=hundred,
                                              reference=not isnan(zref)))
        self.view_scale, self.objective_now = vs0, on0     # (the pause window's live view followed the person's objective)
        if ans == "stop":
            self.stop_all = True
            raise Stop("stopped by the user (while paused)")
        self.say(f"  returning to the scan after {(time.time() - t0) / 60:.1f} min ...")
        self.ui.progress(self, f"{s.name}: returning to the scan ...")
        # the objective back: waited for with the nosepiece reader, else asked for
        if pos is not None and self.nose is not None and self.nose.position() != pos:
            if self.wait_objective(objn, "The scan was paused with it: put it back to resume.") is False:
                self.stop_all = True
                raise Stop("stopped by the user (objective not put back after the pause)")
        elif self.nose is None:
            a = self.ui.confirm("Resume: the scan's objective", f"Put the {objn} objective back in (the one the scan was using), "
                                "then press OK.", ("OK, the " + objn + " objective is in", "Stop the scan"))
            if a == 1:
                self.stop_all = True
                raise Stop("stopped by the user (objective not put back after the pause)")
            self.ui.mock_hook(self, "mockObjective", 100 if hundred else 10)
        # the camera settings; then the check: the stage at the scan's last picture (its objective), the focus drive at the
        # reading that was in focus there; the person focuses (coarse, fine) and ticks focus and position good
        self.stream_off()
        self.apply_camera(exp)
        ref = self.last_pic if (self.last_pic is not None and self.last_pic.get("hundred") == hundred) else pause_pic
        if ref is not None:
            cx, cy, cz = ref["x"], ref["y"], ref["z"]
        else:
            cx, cy, cz = rx, ry, zref
        self.approach_to(cx, cy)
        if self.has_z and not isnan(cz):
            if not self.set_z(cz):
                self.z_write(cz)
        chk = self.ui.pause_check(self, dict(ref=ref, hundred=hundred, objective=objn, x=cx, y=cy, z=cz, phase=phase,
                                             contact=getattr(self, "contact", {}) or {}))
        if chk == "stop":
            self.stop_all = True
            raise Stop("stopped by the user (when returning to the scan)")
        self.say(f"    checked by hand at {ref['what'] if ref is not None else 'where the scan paused'}: focus and position good")
        if self.has_z and not isnan(z) and not isnan(cz):
            zn = self._pause_refine(self.z_read(), cz, hundred) if s.pause_refocus else self.z_read()
            d = zn - cz
            tol = s.pause_refocus_tol_um[1 if hundred else 0]
            if abs(d) > tol:
                self.z_shift += d
                self.say(f"    the focus moved by {d:+.1f} um during the pause: every focus height from now on is offset by it "
                         f"(in all {self.z_shift:+.1f} um)")
            else:
                self.say(f"    focus unchanged ({d:+.2f} um)")
        elif self.has_z:
            self.say("    WARNING: no focus reference (no picture yet and none found before the pause): the focus offset is not "
                     "checked; if the coarse knob was turned, the scan's own autofocus has to find it again")
        self.approach_to(x, y)
        if self.has_z and not isnan(z):
            if not self.set_z(z):
                self.z_write(z)
        self.set_phase(phase)
        self.ui.progress(self, status or f"{s.name}: resumed")
        self.say(f"  RESUMED ({phase})")

    def wait_objective(self, name, msg):
        """Make sure objective `name` is in: True (it is / you went on), False (cancelled), None (not known: no
        nosepiece reader or no position for that name; the caller asks as before)."""
        want = (self.s.objective_positions or {}).get(name)
        if self.nose is None or not want:
            return None
        if self.nose.position() == want:
            return True
        self.say(f"  waiting for the {name} objective (nosepiece position {want}) ...")
        self.alert(f"Put in the {name} objective (nosepiece position {want}).  {msg}")
        ok = self.ui.wait_objective(self, name, want, msg)
        pos = self.nose.position()
        self.say(f"  objective: position {pos} ({self.objective_name(pos)})" + ("" if ok else " (cancelled)"))
        return ok

    # ---- frames
    def frame(self, after_move=True):
        """A frame taken after the last move: (H, W) uint16 (RAW16 Bayer or mono) or (H, W, 3) uint8 (RGB32).
        With stream_frames the camera runs continuously: the buffer is cleared and the second new frame is used (the
        first may have been exposing while the stage settled); a snap takes ~0.2 s with the XC10, a streamed frame a
        couple of frame periods.  Falls back to snaps if the stream does not deliver.  after_move=False: another frame
        at the same place (nothing moved since the last frame): the first new frame of the stream."""
        self.stop_check()
        self.n_snaps += 1
        t_op = time.time()
        try:
            img = self._frame(after_move)
            fl = getattr(self, "frame_flip", "")
            if fl and img is not None:                   # the domain camera's picture turned to the scan camera's way
                img = np.ascontiguousarray(img[::-1 if "y" in fl else slice(None), ::-1 if "x" in fl else slice(None)])
            pv = getattr(self.ui, "preview", None)
            if pv is not None and img is not None:       # (the main window's camera panel: a reference, shown later)
                try:
                    x, y = self.last_xy
                    pv(self, img, f"{self.cur_phase or ''}  ·  {getattr(self.s, 'name', '')}"
                       + ("" if isnan(x) else f"  ·  stage {x:.0f}, {y:.0f} um") + ("" if isnan(self.z_cmd) else f"  ·  Z {self.z_cmd:.1f}"))
                except Exception:
                    pass
            return img
        finally:
            self.op("camera frames", t_op)

    CAMERA_RETRIES = 2              # reconnects of the camera after a refused picture, before the job stops
    CAMERA_RECONNECTS_MAX = 10      # ... and at most this many in a job

    def _frame(self, after_move=True):
        """_frame_once, and when the camera refuses the picture (the XC10, 2026-10-26/28: Error in device "16501874": (Error
        message unavailable) (25) on every snap, its properties still answering) the camera reconnected
        (reconnect_camera: the software way of unplugging it) and the picture taken again; the job stops with what to do
        when that does not help."""
        try:
            return self._frame_once(after_move)
        except Stop:
            raise
        except Exception as e:
            err = e
        cam = self.core.getCameraDevice() or getattr(self, "_cam_spec", {}).get("label", "")
        if not getattr(self, "_cam_props_logged", False):         # (once a job: what the camera says about itself)
            self._cam_props_logged = True
            self.say(f"    camera {cam} gave no picture: {err}\n    its properties now: " + self.camera_props_text(cam))
        for k in range(self.CAMERA_RETRIES):
            n = self.cam_reconnects = getattr(self, "cam_reconnects", 0) + 1
            if n > self.CAMERA_RECONNECTS_MAX:
                break
            self.say(f"    camera {cam} gave no picture ({err}); reconnecting it ({n} this job) ...")
            try:
                self.reconnect_camera()
                img = self._frame_once(True)
                self.say(f"    ... camera {cam} reconnected, it gives pictures again")
                return img
            except Stop:
                raise
            except Exception as e:
                err = e
        self.alert()                                            # (the sound; the job's end message says why)
        raise self.fail(f"the camera {cam} gives no pictures ({err}), and reconnecting it did not help.  Close this program "
                        "and every other program that can use the camera (Stream Motion, XCAP, XIMEA CamTool, xiCOP), unplug "
                        "the camera's cable from the PC for 10 s, plug it back in and start again.  If it still fails, "
                        "restart the PC.")

    def camera_props_text(self, cam):
        c, out = self.core, []
        try:
            for p in c.getDevicePropertyNames(cam):
                try:
                    out.append(f"{p} = {c.getProperty(cam, p)}")
                except Exception as e:
                    out.append(f"{p}: ({e})")
        except Exception as e:
            out.append(f"({e})")
        return "; ".join(out) or "(none)"

    def reconnect_camera(self, timeout=60.0):
        """Unload the camera in use from Micro-Manager and load it again (its driver closes and opens it again: the
        software way of unplugging it), then its settings again: the scan camera's camera_props and exposure
        (apply_camera), another camera's properties as they were.  In a thread: a driver that does not answer stops the
        job after `timeout` s instead of hanging it."""
        import threading
        c = self.core
        cam = c.getCameraDevice()
        spec = getattr(self, "_cam_spec", None)
        if cam and (spec is None or spec["label"] != cam):
            pre, props = [], []
            for p in c.getDevicePropertyNames(cam):
                try:
                    if c.isPropertyPreInit(cam, p):
                        pre.append((p, c.getProperty(cam, p)))
                    elif not c.isPropertyReadOnly(cam, p):
                        props.append((p, c.getProperty(cam, p)))
                except Exception:
                    pass
            try:
                parent = c.getParentLabel(cam)
            except Exception:
                parent = ""
            spec = self._cam_spec = dict(label=cam, lib=c.getDeviceLibrary(cam), dev=c.getDeviceName(cam), parent=parent,
                                         pre=pre, props=props, exposure=c.getExposure())
        if spec is None:
            raise RuntimeError("no camera selected in Micro-Manager")
        cam = spec["label"]
        box = {}

        def work():
            try:
                try:
                    c.stopSequenceAcquisition()
                except Exception:
                    pass
                if cam in c.getLoadedDevices():
                    c.unloadDevice(cam)
                time.sleep(1.0)
                c.loadDevice(cam, spec["lib"], spec["dev"])
                if spec["parent"]:
                    c.setParentLabel(cam, spec["parent"])
                for p, v in spec["pre"]:
                    c.setProperty(cam, p, v)
                c.initializeDevice(cam)
                c.setCameraDevice(cam)
            except BaseException as e:
                box["e"] = e
        th = threading.Thread(target=work, daemon=True, name="camera reconnect")
        th.start()
        th.join(timeout)
        if th.is_alive():
            raise self.fail(f"the camera {cam} did not answer within {timeout:g} s while it was being reconnected.  Close this "
                            "program, unplug the camera's cable from the PC for 10 s, plug it back in and start again.  If it "
                            "still fails, restart the PC.")
        if "e" in box:
            raise box["e"]
        self.stream_failed, self._ts = False, None              # (the stream is tried again: it may work now)
        if cam == self.cam and self.s.set_camera:
            self.apply_camera(spec["exposure"])
        else:
            for p, v in spec["props"]:
                try:
                    c.setProperty(cam, p, v)
                except Exception:
                    pass
            c.setExposure(spec["exposure"])
        c.waitForDevice(cam)

    def _frame_once(self, after_move=True):
        c = self.core
        if self.s.stream_frames and not getattr(self, "stream_failed", False):
            try:
                if self.s.frame_timestamps and not getattr(self, "ts_failed", False):
                    img = self._frame_timed(after_move)
                    if img is not None:
                        return img
                if not c.isSequenceRunning():
                    c.startContinuousSequenceAcquisition(0)
                    after_move = True
                c.clearCircularBuffer()
                t_end = time.time() + 2.0
                # after a move the second new frame (the first may have been exposing during it); the next frames of a
                # burst at the same place (after_move=False: hundred_frames, domain_frames) the first new one
                while c.getRemainingImageCount() < (2 if after_move else 1):
                    if time.time() > t_end:
                        raise RuntimeError("no frames from the stream within 2 s")
                    time.sleep(0.002)
                return self.to_array(c.getLastImage())
            except Exception as e:
                self.stream_failed = True
                self.say(f"    (camera stream not usable: {e}; taking snaps from here on)")
                self.stream_off()
                try:
                    c.stopSequenceAcquisition()                 # (also when the start failed half way)
                except Exception:
                    pass
        c.snapImage()
        return self.to_array(c.getImage())

    @staticmethod
    def _md_elapsed(md):
        """A frame's ElapsedTime-ms (ms since the stream started, when Micro-Manager put it into its buffer) from its
        metadata (pymmcore-plus: dict-like; pymmcore: Metadata), or None."""
        for get in (lambda: md["ElapsedTime-ms"], lambda: md.GetSingleTag("ElapsedTime-ms").GetValue()):
            try:
                return float(get())
            except Exception:
                pass
        return None

    def _frame_timed(self, after_move=True):
        """frame_timestamps: the newest frame from the stream whose exposure started after the last move ended
        (self.t_moved), from its time stamp: buffer time - exposure - frame_readout_ms.  The buffer's clock is tied to
        time.perf_counter by the smallest (seen - stamped) difference over the frames seen (the poll comes a little
        after the frame; 5 ms margin), after 10 frames of a new stream.  None (and the old way from here on) when the
        camera / pymmcore version gives no time stamps."""
        c, ts = self.core, getattr(self, "_ts", None)
        if not c.isSequenceRunning():
            c.startContinuousSequenceAcquisition(0)
            ts = None
        if ts is None:
            ts = self._ts = dict(off=None, n=0, last=-1.0)
        lat = (c.getExposure() + float(self.s.frame_readout_ms)) / 1000.0
        need = (self.t_moved if after_move else -1e30) + 0.005
        t_end = time.perf_counter() + 2.0
        while True:
            if c.getRemainingImageCount() > 0:
                try:
                    img, md = c.getLastImageMD()                 # (pymmcore-plus: (image, metadata))
                except Exception as e:
                    img, md = None, None
                    why = f"{type(e).__name__}: {e}"
                seen = time.perf_counter()
                el = self._md_elapsed(md) if md is not None else None
                if el is None:
                    self.ts_failed = True
                    self.say("    (frame_timestamps: the camera's frames carry no ElapsedTime-ms" + (f" ({why})" if img is None else "")
                             + "; frames as before)")
                    return None
                c.clearCircularBuffer()
                if el < ts["last"]:                              # (a new stream: its clock starts again)
                    ts.update(off=None, n=0, last=-1.0)
                off = seen - el / 1000.0
                ts["off"] = off if ts["off"] is None else min(ts["off"], off)
                ts["n"] += 1
                if ts["n"] >= 10 and el > ts["last"] and el / 1000.0 + ts["off"] - lat >= need:
                    ts["last"] = el
                    return self.to_array(img)
            if time.perf_counter() > t_end:
                raise RuntimeError("no frames from the stream within 2 s")
            time.sleep(0.001)

    def restore_exposure(self, ms):
        """The exposure back to `ms` after a window whose live view changed it (the polygon window's exposure box)."""
        c = self.core
        try:
            if abs(float(c.getExposure()) - float(ms)) > 1e-6:
                self.stream_off()
                c.setExposure(float(ms))
                c.waitForDevice(c.getCameraDevice())
                self.say(f"  exposure back to the scan's {f3(ms)} ms")
        except Exception as e:
            self.say(f"  (exposure not set back to {f3(ms)} ms: {e})")

    def stream_off(self):
        """Stop the continuous stream (before exposure / property / camera changes; frame() restarts it), waiting until
        the camera has stopped."""
        try:
            if self.core.isSequenceRunning():
                self.stop_sequence()
        except Exception:
            pass

    def to_array(self, img):
        a = np.asarray(img)
        c = self.core
        if a.ndim == 1:
            a = a.reshape(int(c.getImageHeight()), int(c.getImageWidth()))
        if a.dtype == np.uint32 or (a.ndim == 3 and a.shape[2] == 4):
            b = a.view(np.uint8).reshape(a.shape[0], a.shape[1], 4) if a.ndim == 2 else a
            return b[..., [2, 1, 0]].copy()
        return a

    def full_scale(self):
        c, cam = self.core, self.core.getCameraDevice()
        if int(c.getBytesPerPixel()) == 4:
            return 255.0
        if c.hasProperty(cam, "Image data bit depth"):
            try:
                return 2.0 ** float(c.getProperty(cam, "Image data bit depth")) - 1
            except ValueError:
                pass
        return 2.0 ** float(c.getImageBitDepth()) - 1

    def blocks(self, img=None):
        """Sampled 2x2 blocks of a frame: (N, 5) array of x, y (pixel centre), R, G, B (dark subtracted)."""
        a = self.frame() if img is None else img
        S, W, H = self.s.sample, self.W, self.H
        nx, ny = (W - 2) // S, (H - 2) // S
        xs = (np.arange(nx) * S + S // 2) & ~1
        ys = (np.arange(ny) * S + S // 2) & ~1
        X, Y = np.meshgrid(xs, ys)
        if a.ndim == 2:
            f = a.astype(np.float64)
            r = f[Y, X] - self.DARK
            g = 0.5 * (f[Y, X + 1] + f[Y + 1, X]) - self.DARK
            b = f[Y + 1, X + 1] - self.DARK
        else:
            r, g, b = (a[Y, X, k].astype(np.float64) for k in range(3))
        return np.stack([X.ravel() + 1.0, Y.ravel() + 1.0, r.ravel(), g.ravel(), b.ravel()], 1)

    def bad_frame(self, a):
        """XC10 glitch frames: random values, or the raw data one pixel off (scrambled colours).  RAW16 only."""
        if a.ndim != 2 or a.dtype != np.uint16 or getattr(self, "mono_view", False):
            return None
        W, H = self.W, self.H
        ys = np.arange(8, H - 8, 16) & ~1
        xs = np.arange(8, W - 10, 8) & ~1
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        f = a.astype(np.float64)
        ga, gb = f[Y, X + 1], f[Y + 1, X]
        d = np.abs(f[Y, X + 3] - f[Y, X + 1]).sum() / max(ga.sum(), 1.0)
        if d > 0.3:
            return f"noise frame (neighbour difference {f3(d)} of the level)"
        bx = np.minimum(7, (X * 8) // W)
        by = np.minimum(7, (np.arange(8, H - 8, 16) * 8) // H)[:, None] * np.ones_like(X)
        for lab, idx in (("x", bx), ("y", by)):
            for k in range(8):
                m = idx == k
                if m.any():
                    a1, b1 = ga[m].mean(), gb[m].mean()
                    if abs(a1 - b1) > 0.15 * max(a1, b1, 1.0):
                        return f"scrambled colours (green sites {f1(a1)} / {f1(b1)})"
        return None

    def snap_good(self, what, after_move=True):
        img = None
        for k in range(4):
            img = self.frame(after_move)
            b = self.bad_frame(img)
            if b is None:
                return img
            self.say(f"    camera glitch on {what}: {b}" + ("; snapping again" if k < 3 else "; kept (check this image)"))
        return img

    def frame_level(self):
        fs = self.full_scale()
        bl = self.blocks()
        sat = ((bl[:, 2] + self.DARK >= fs - 2) | (bl[:, 3] + self.DARK >= fs - 2) | (bl[:, 4] + self.DARK >= fs - 2)).mean()
        return float(np.median(bl[:, 3]) / (fs - self.DARK)), float(sat)

    def check_exposure(self, auto):
        s, c = self.s, self.core
        lv = self.frame_level()
        for _ in range(8 if auto else 0):
            if lv[1] < 0.01 and abs(lv[0] / s.target_level - 1) < 0.15:
                break
            ex = c.getExposure()
            f = 0.1 if (lv[0] > 0.9 or lv[1] > 0.3) else s.target_level / max(lv[0], 0.005)
            nex = max(s.exposure_range[0], min(s.exposure_range[1], ex * f))
            if abs(nex - ex) < 1e-6:
                break
            self.stream_off()
            c.setExposure(nex); c.waitForDevice(c.getCameraDevice())
            self.say(f"    exposure {f3(ex)} -> {f3(nex)} ms (median {f3(lv[0] * 100)} % of full scale, {f3(lv[1] * 100)} % saturated)")
            lv = self.frame_level()
        self.say(f"  frame level: median {f3(lv[0] * 100)} % of full scale ({int(self.full_scale())}), {f3(lv[1] * 100)} % saturated, "
                 f"exposure {f3(c.getExposure())} ms")
        if lv[1] > 0.05 or lv[0] > 0.85:
            raise self.fail(f"the image is overexposed (median {f3(lv[0] * 100)} % of full scale, {f3(lv[1] * 100)} % saturated) at exposure "
                            f"{f3(c.getExposure())} ms.  Check the exposure and the lamp, or set auto_exposure.")
        if lv[0] < 0.02:
            raise self.fail(f"the image is (nearly) black (median {f3(lv[0] * 100)} % of full scale): lamp off or light path closed?")

    # ---- substrate
    def reference(self, bl):
        """Substrate colour (R/G, B/G, G): mode of the chroma histogram (3x3 smoothed), refined on the blocks near it."""
        s = self.s
        NB, BW = 200, 0.02
        g = bl[:, 3]
        ok = g > 1
        if s.substrate_colour is not None:
            rc, bc = s.substrate_colour
        else:
            r, b = bl[ok, 2] / g[ok], bl[ok, 4] / g[ok]
            m = (r >= 0) & (b >= 0) & (r < NB * BW) & (b < NB * BW)
            cnt = np.zeros((NB, NB))
            np.add.at(cnt, ((r[m] / BW).astype(int), (b[m] / BW).astype(int)), 1)
            sm = cnt.copy()
            for dr in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if dr or db:
                        sm += np.roll(np.roll(np.pad(cnt, 1), dr, 0), db, 1)[1:-1, 1:-1]
            sm[cnt == 0] = -1
            ir, ib = np.unravel_index(int(np.argmax(sm)), sm.shape)
            rc, bc = (ir + 0.5) * BW, (ib + 0.5) * BW
        r, b = bl[:, 2] / np.maximum(g, 1e-9), bl[:, 4] / np.maximum(g, 1e-9)
        near = ok & (np.abs(r / rc - 1) <= s.chroma_tol / 2) & (np.abs(b / bc - 1) <= s.chroma_tol / 2)
        if not near.any():
            return (rc, bc, 1.0)
        cen = near & (np.abs(bl[:, 0] - self.W / 2.0) <= self.W / 4.0) & (np.abs(bl[:, 1] - self.H / 2.0) <= self.H / 4.0)
        gref = np.median(g[cen]) if cen.sum() >= 10 else np.median(g[near])
        return (float(np.median(r[near])), float(np.median(b[near])), float(gref))

    def is_substrate(self, bl):
        R = self.REF
        g = bl[:, 3]
        ok = (g >= self.s.bright_min * R[2]) & (g <= self.s.bright_max * R[2])
        gs = np.maximum(g, 1e-9)
        return ok & (np.abs(bl[:, 2] / gs / R[0] - 1) < self.s.chroma_tol) & (np.abs(bl[:, 4] / gs / R[1] - 1) < self.s.chroma_tol)

    def substrate_fraction(self, bl):
        return float(self.is_substrate(bl).mean())

    def substrate_rgb(self):
        """Substrate level (R, G, B counts above dark) from the reference, for the live view's white balance."""
        if self.REF is None:
            return None
        R = self.REF
        return (R[0] * R[2], R[2], R[1] * R[2])

    # ---- autofocus
    def sharpness(self, roi):
        import cv2
        a = self.frame()
        W, H = self.W, self.H
        rw, rh = int(W * roi) & ~1, int(H * roi) & ~1
        x0, y0 = ((W - rw) // 2) & ~1, ((H - rh) // 2) & ~1
        c = a[y0:y0 + rh, x0:x0 + rw].astype(np.float32)
        if c.ndim == 3:
            c = c.mean(axis=2)
        if getattr(self, "mono_view", False):
            # a monochrome camera (the IR one): no mosaic to bin away; its faint, low-contrast pictures: 1 px smoothing
            # against the noise and the squared gradient (edges weigh more than noise; tests on its real pictures 2026-10-19)
            c = cv2.GaussianBlur(c / max(float(c.mean()), 1.0), (0, 0), 1.0)
            gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
            return float((gx * gx + gy * gy).mean())
        c = c.reshape(rh // 2, 2, rw // 2, 2).mean(axis=(1, 3))              # 2x2 bin removes the colour mosaic
        mean = float(c.mean())
        gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.sqrt(gx * gx + gy * gy).mean()) / max(mean, 1.0)

    def sweep(self, zs, roi):
        ms = np.full(len(zs), -1.0)
        if self.set_z(zs[0] - self.s.af_backlash_um):
            self.check_z_reach("autofocus sweep")
        for i, z in enumerate(zs):
            if self.set_z(z):
                ms[i] = self.sharpness(roi)
        if len(zs) > 1 and self.z_cmd == zs[-1]:
            self.check_z_reach("autofocus sweep")       # (the sweep's two ends read back: the drive at a limit)
        self.despike(zs, ms, roi)
        bi = int(np.argmax(ms))
        L = self.z_limits.get("lower")
        if L is not None and ms[bi] > 0 and zs[bi] <= float(L["z"]) + self.s.z_reach_tol_um:
            # (the sharpest place of the sweep at or below the drive's lower limit: below it every frame is taken at the
            # limit, the same picture; the best focus within reach is at the limit: focus_limit_check)
            self._af_limit = (float(zs[bi]), float(L["z"]))
        return bi, ms

    def despike(self, zs, ms, roi):
        for i in range(len(zs)):
            if ms[i] <= 0:
                continue
            nb = max([ms[j] for j in (i - 1, i + 1) if 0 <= j < len(zs)] + [-1])
            if nb <= 0 or ms[i] < self.s.spike_factor * nb:
                continue
            self.set_z(zs[i] - self.s.af_backlash_um); self.set_z(zs[i])
            again = self.sharpness(roi)
            if self.s.verbose:
                self.say(f"      sharpness spike at Z {f3(zs[i])} ({f3(ms[i])} vs neighbours {f3(nb)}): re-measured {f3(again)}")
            ms[i] = again

    def autofocus(self, zc, p, fine=True, extend=None, limit_check=True):
        """Autofocus around zc with p = (coarse range, coarse step, fine range, fine step, roi).  (z, ratio, ok).
        fine=False: the coarse sweep only, the peak refined by a parabola through it and its neighbours (the live view's
        Autofocus button: the scan focuses finely itself later).  fine="fit": the same when the coarse peak is clear
        (inside the sweep, log sharpness curving down around it), else the fine sweep (af100_fit).  extend: how often
        the sweep may move on when the peak is at an end (default af_extend).  limit_check: a best focus at the focus
        drive's lower limit stops the scan (focus_limit_check; off for the live view's own Autofocus button)."""
        t0 = time.time()
        try:
            self._af_limit = None
            r = self._autofocus(zc, p, fine, t0, s_ext=self.s.af_extend if extend is None else int(extend))
            if limit_check:
                self.focus_limit_check(float(r[0]) if r[2] else None)
            if r[2]:                                  # (the last focus found with each objective: a pause's reference)
                try:
                    self.last_af[getattr(self, "view_scale", 1.0) < 1.0] = (self.core.getXPosition(self.xy), self.core.getYPosition(self.xy), float(r[0]))
                except Exception:
                    pass
            return r
        finally:
            self.op("autofocus", t0)

    def focus_limit_check(self, z=None):
        """After an autofocus: is the best focus within reach at the focus drive's lower limit?  Either a sweep's sharpest
        place was at or below the limit (sweep: _af_limit; also when the autofocus found no clear focus: below the limit
        every frame is the same, blurred, picture), or the best focus z it found and sent the drive to reads back higher
        than z (by more than z_reach_tol_um).  Then the pictures would be out of focus: the scan stops and a message is
        sent (2026-10-27).  A search that only passes the limit on its way, its best focus above it, is fine; a move to z
        refused by max_dz_um is not counted."""
        s = self.s
        if not self.has_z or not getattr(s, "focus_limit_stop", True):
            return
        lim = getattr(self, "_af_limit", None)
        if lim is not None:
            z, zr = lim
        else:
            if z is None or isnan(self.z_cmd) or abs(self.z_cmd - z) > 1e-6:
                return
            zr = self.z_read()
            if zr - z <= s.z_reach_tol_um:
                return
        what = (f"The best focus is at the focus drive's lower limit: the autofocus wanted Z {f1(z)} um, the drive stops at "
                f"{f1(zr)} um ({self.cur_phase}).  Every picture from here would be out of focus, so the scan stopped.  "
                "Bring the sample's focus nearer the middle of the drive's travel (coarse focus) or check the Oasis focus "
                "limits, then start the scan again.")
        self.say("  FOCUS DRIVE AT ITS LOWER LIMIT: " + what)
        try:
            from .notify import send
            send(s, "operator", f"{s.name or 'scan'}: stopped, focus drive at its lower limit", what, log=self.say,
                 run=getattr(self, "slack_run", None))
        except Exception as e:
            self.say(f"  (no message sent: {e})")
        self.stop_all = True
        raise Stop(f"the best focus is at the focus drive's lower limit (Z {f1(zr)} um)")

    def _autofocus(self, zc, p, fine, t0, s_ext=2):
        s = self.s
        lo, hi = zc - p[0], zc + p[0]
        best, ratio = zc, 0.0
        for ext in range(s_ext + 1):
            n = int(round((hi - lo) / p[1])) + 1
            zs = lo + np.arange(n) * p[1]
            bi, ms = self.sweep(zs, p[4])
            valid = ms[ms >= 0]
            if len(valid) < 3:
                break
            ratio = ms[bi] / max(float(np.sort(valid)[len(valid) // 2]), 1e-9)
            best = zs[bi]
            if ratio < s.af_min_ratio:
                break
            if bi == 0 and ext < s_ext:
                hi = lo + p[1]; lo = hi - 2 * p[0]; continue
            if bi == n - 1 and ext < s_ext:
                lo = hi - p[1]; hi = lo + 2 * p[0]; continue
            break
        if ratio < s.af_min_ratio:
            if s.verbose:
                self.say(f"      autofocus: no focus signal (peak/median {f3(ratio)})")
            self.set_z(zc)
            return (zc, ratio, False)
        if fine is not True:
            z, clear = float(best), False
            i = int(np.argmin(np.abs(zs - best)))
            if 0 < i < len(zs) - 1 and min(ms[i - 1], ms[i], ms[i + 1]) > 0:
                y0, y1, y2 = np.log(ms[i - 1]), np.log(ms[i]), np.log(ms[i + 1])
                den = y0 - 2 * y1 + y2
                if den < 0:
                    z, clear = float(zs[i] + 0.5 * p[1] * (y0 - y2) / den), True
            if not fine or clear:
                self.set_z(z - s.af_backlash_um)
                self.set_z(z)
                if s.verbose:
                    self.say(f"      {'coarse' if not fine else 'one-sweep'} autofocus: Z {f3(z)} (peak/median {f3(ratio)}, {time.time() - t0:.1f} s)")
                return (z, ratio, True)
        nf = int(round(2 * p[2] / p[3])) + 1
        zf = best - p[2] + np.arange(nf) * p[3]
        fi, mf = self.sweep(zf, p[4])
        if fi <= 1 or fi >= nf - 2:
            zc2 = zf[fi]
            zf = zc2 - p[2] + np.arange(nf) * p[3]
            fi, mf = self.sweep(zf, p[4])
            if s.verbose:
                self.say(f"      fine sweep re-centred on Z {f3(zc2)}")
        z = float(zf[fi])
        pos = mf > 0
        if pos.sum() >= 3:
            lm = np.log(mf[pos])
            thr = lm.min() + 0.5 * (math.log(mf[fi]) - lm.min())
            sel = pos.copy()
            sel[pos] = lm >= thr
            if sel.sum() >= 3:
                u = zf[sel] - zf[fi]
                c2, c1, _ = np.polyfit(u, np.log(mf[sel]), 2)
                if c2 < 0:
                    z = float(max(zf[sel].min(), min(zf[sel].max(), zf[fi] - c1 / (2 * c2))))
        self.set_z(z - s.af_backlash_um)
        self.set_z(z)
        if s.verbose:
            self.say(f"      autofocus: Z {f3(z)} (peak/median {f3(ratio)}, {time.time() - t0:.1f} s)")
        return (z, ratio, True)

    def autofocus_fit(self, zc, p, min_rise=1.03, extend=2):
        """One sweep around a good prediction (p = (half range, step, roi)), focus from a log-parabola through its upper
        half: (z, rise, ok).  Valid when the sharpest point is inside the sweep and min_rise above both its ends; at an
        end the sweep moves on (extend times).  For cameras whose focus changes the picture little over a sweep (the
        IR camera: depth of field of several um): peak / median, as autofocus() tests, is about 1.07 over +-4 um there
        even when the focus is inside, so 3 of 4 IR flakes needed the wide search before (graphite_20261019_run_1)."""
        t0 = time.time()
        s = self.s
        try:
            lo, n = zc - p[0], int(round(2 * p[0] / p[1])) + 1
            for e in range(extend + 1):
                zs = lo + np.arange(n) * p[1]
                bi, ms = self.sweep(zs, p[2])
                if bi == 0 and e < extend:
                    lo -= (n - 2) * p[1]; continue
                if bi == n - 1 and e < extend:
                    lo += (n - 2) * p[1]; continue
                break
            ends = max(ms[0], ms[-1])
            rise = float(ms[bi] / ends) if ends > 0 else 0.0
            z, ok = float(zs[bi]), False
            if 0 < bi < n - 1 and rise >= min_rise and (ms > 0).sum() >= 3:
                pos = ms > 0
                lm = np.log(ms[pos])
                sel = pos.copy()
                sel[pos] = lm >= lm.min() + 0.5 * (math.log(ms[bi]) - lm.min())
                if sel.sum() >= 3:
                    c2, c1, _ = np.polyfit(zs[sel] - zs[bi], np.log(ms[sel]), 2)
                    if c2 < 0:
                        z = float(max(zs[sel].min(), min(zs[sel].max(), zs[bi] - c1 / (2 * c2))))
                ok = True
            if not ok:
                if s.verbose:
                    self.say(f"      autofocus (one sweep): no clear peak (peak / ends {f3(rise)}, at {bi + 1} of {n})")
                self.set_z(zc)
                return (zc, rise, False)
            self.set_z(z - s.af_backlash_um)
            self.set_z(z)
            if s.verbose:
                self.say(f"      autofocus (one sweep): Z {f3(z)} (peak / ends {f3(rise)}, {time.time() - t0:.1f} s)")
            return (z, rise, True)
        finally:
            self.op("autofocus", t0)

    def focus_spot(self, k, qx, qy, ccx, ccy):
        """A focus point's field on bare substrate (focus_min_substrate of it): else moved towards the chip's centre
        (cx, cy) a fifth of the way, up to focus_moves times.  -> (x, y) or None (skipped)."""
        s = self.s
        for m in range(s.focus_moves + 1):
            if m:
                qx, qy = qx + (ccx - qx) * 0.2, qy + (ccy - qy) * 0.2
                try:
                    self.move_to(qx, qy)
                except StageLimit:
                    return None
            fr = self.substrate_fraction(self.blocks())
            if fr >= s.focus_min_substrate:
                if m:
                    self.say(f"  focus point {k + 1}: little bare substrate where planned (tape, debris?): moved {m} step"
                             f"{'s' if m > 1 else ''} towards the centre, to {f1(qx)}, {f1(qy)}")
                return qx, qy
        self.say(f"  focus point {k + 1}: no bare substrate near it (substrate {f3(fr)} of the field): point skipped")
        return None

    def focus_spot_local(self, k, qx, qy):
        """A polygon's focus point on bare substrate: the field there, else the nearest of the 8 fields around it
        (half a field away) inside the polygon.  -> (x, y) or None (skipped)."""
        s = self.s
        hx, hy = 0.5 * self.FOVX, 0.5 * self.FOVY
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            x, y = qx + dx * hx, qy + dy * hy
            if (dx or dy) and not point_in_polygon(x, y, self.polygon):
                continue
            if dx or dy:
                try:
                    self.move_to(x, y)
                except StageLimit:
                    continue
            fr = self.substrate_fraction(self.blocks())
            if fr >= s.focus_min_substrate:
                if dx or dy:
                    self.say(f"  focus point {k + 1}: little bare substrate where planned (tape, debris?): the field beside it, "
                             f"{f1(x)}, {f1(y)}")
                return x, y
        self.say(f"  focus point {k + 1}: no bare substrate there or beside it: point skipped")
        return None

    def ask_on_fail(self, n_focused):
        m = self.s.af_manual_fallback
        return m == "always" or (m == "first" and n_focused == 0)

    def ask_focus(self, title, what, timeout=None, image=None, **kw):
        """0 = use, 1 = skip, 2 = stop, 3 = no answer within timeout (the predicted Z is used).  image: a picture of
        the flake to look for (shown beside the live view); kw: button (its text), cross (centring cross)."""
        if image is None and not kw:
            return self.ui.ask_focus(self, title, what, timeout)
        return self.ui.ask_focus(self, title, what, timeout, image=image, **kw)

    def target_picture(self, P, lab):
        """The 10x picture of target `lab` written by the analysis (<piece>_targets_crops/<label>.jpg), or None."""
        d = getattr(self, "crops_dir", None) or Path(P["out_dir"]) / f"{self.s.name}_targets_crops"
        f = Path(d) / f"{lab}.jpg"
        return str(f) if f.is_file() else None

    def predict_z(self, x, y, pts, chip_tilt, z_default):
        """Z at (x, y): nearest focused point + a tilt (from the points once they span plane_min_spread_um in both
        directions, else the 10x chip's), capped at max_tilt."""
        if not pts:
            return z_default
        P = np.asarray(pts, float)
        nn = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))
        sa = sb = 0.0
        spread = False
        if len(P) >= 3:
            C = np.cov(P[:, :2].T, bias=True)
            spread = math.sqrt(max(np.linalg.eigvalsh(C)[0], 0)) * 2 >= self.s.plane_min_spread_um
        pf = fit_plane(pts) if spread else None
        if pf is not None:
            sa, sb = pf[0], pf[1]
        elif chip_tilt is not None:
            sa, sb = chip_tilt
        t = math.hypot(sa, sb) * 1000.0
        if t > self.s.max_tilt:
            sa *= self.s.max_tilt / t; sb *= self.s.max_tilt / t
        return float(P[nn, 2] + sa * (x - P[nn, 0]) + sb * (y - P[nn, 1]))

    def z_surf(self, x, y):
        if self.FS is not None:
            return self.FS(x, y)
        if self.FQ is None:
            return self.fa * x + self.fb * y + self.fc
        Q = self.FQ
        xc, yc = max(Q[6], min(Q[7], x)), max(Q[8], min(Q[9], y))
        u, v = (xc - Q[10]) / 1000.0, (yc - Q[11]) / 1000.0
        return Q[0] + Q[1] * u + Q[2] * v + Q[3] * u * u + Q[4] * u * v + Q[5] * v * v + self.fa * (x - xc) + self.fb * (y - yc)

    def z_at(self, x, y):
        return max(self.Z0 - self.s.max_dz_um, min(self.Z0 + self.s.max_dz_um, self.z_surf(x, y)))

    # ---- edges
    def neighbour_limit(self, sx, sy, dx, dy):
        """How far the walk from (sx, sy) along (dx, dy) may go before it reaches another chip of this run: the entry
        into its outline (a piece scanned already), else its marked position (a piece not scanned yet; only one roughly
        in this direction) less half a field.  -> (distance um, its name) or (None, None)."""
        best, who = None, None
        here = getattr(self.s, "name", None)
        half = 0.5 * (abs(dx) * self.FOVX + abs(dy) * self.FOVY)
        for p in self.pieces or ():
            if p.get("name") == here:
                continue
            ol = p.get("outline")
            if ol and len(ol) >= 3:
                t = ray_polygon(sx, sy, dx, dy, ol)
                if t is not None and t > 0:
                    lim = t
                else:
                    continue
            else:
                bx, by = p["c"][0] - sx, p["c"][1] - sy
                along = bx * dx + by * dy
                if along <= 0 or along < math.hypot(bx, by) / math.sqrt(2.0):     # (not ahead within 45 degrees)
                    continue
                lim = along - half
            if lim > 0 and (best is None or lim < best):
                best, who = lim, p.get("name")
        return best, who

    def find_edge(self, sx, sy, dx, dy, limit=None, max_um=None, any_um=None):
        """Walk from (sx, sy) along (dx, dy) until off the chip: distance to the edge (um), or -1 (none within
        max_travel_um), or -2 (the next chip reached at `limit` um with no gap before it: touching chips).  With a
        limit, a narrower dark gap (narrow_gap_um) before it counts as the edge: the last one, with substrate (the next
        chip) after it.  self.edge_how: "gap", "narrow gap", "stage limit", "neighbour".  max_um: no edge within this
        distance -> -1 (a 1 cm chip's probe window).  any_um: how long a stretch without substrate (not dark: tape) is the
        edge (default edge_any_um); self.edge_how "no substrate" when the edge was such a stretch."""
        s, M, W, H = self.s, self.M, self.W, self.H
        self.pause_point()
        self.edge_how = "gap"
        ext = abs(dx) * self.FOVX + abs(dy) * self.FOVY
        step = s.step_frac * ext
        nb = int((s.max_travel_um + ext) / s.bin_um) + 2
        gap_bins = int(math.ceil(s.gap_um / s.bin_um))
        any_bins = int(math.ceil((any_um or s.edge_any_um) / s.bin_um))
        sub, tot, drk = np.zeros(nb), np.zeros(nb), np.zeros(nb)
        k = 0
        self.edge_limited = False
        while True:
            px, py = sx + dx * step * k, sy + dy * step * k
            if math.hypot(px - self.X0, py - self.Y0) > s.max_travel_um:
                break
            if max_um is not None and k * step - ext / 2.0 > max_um:
                return -1.0
            if limit is not None and k * step - ext / 2.0 > limit:
                # the next chip reached without the usual gap: a narrower dark one before it, else the user
                return self._narrow_edge(sub, tot, drk, limit)
            try:
                self.move_to(px, py)
            except StageLimit:
                # the chip goes on beyond the stage's travel: its "edge" here is the last field reached (its far side)
                self.edge_limited = True
                return max(0.0, (k - 1) * step) + ext / 2.0 if k > 0 else 0.0
            bl = self.blocks()
            ox, oy = bl[:, 0] - W / 2.0, bl[:, 1] - H / 2.0
            tx, ty = px + M[0, 0] * ox + M[0, 1] * oy, py + M[1, 0] * ox + M[1, 1] * oy
            ss = (tx - sx) * dx + (ty - sy) * dy
            is_sub = self.is_substrate(bl)
            b = np.floor(ss / s.bin_um).astype(int) + 1
            ok = (b >= 0) & (b < nb)
            np.add.at(tot, b[ok], 1.0)
            np.add.at(sub, b[ok & is_sub], 1.0)
            np.add.at(drk, b[ok & (bl[:, 3] < s.dark_level * self.REF[2])], 1.0)
            if s.verbose:
                self.say(f"    frame {k} at {f1(px)}, {f1(py)}: substrate fraction {f3(is_sub.mean())}, brightness "
                         f"{f3(np.median(bl[:, 3]) / self.REF[2])} x substrate")
            run = run_any = 0
            for bb in range(1, nb):
                if tot[bb] == 0:
                    break
                no_sub = sub[bb] / tot[bb] < s.gap_fraction
                gap_bin = no_sub and (not s.surround_dark or drk[bb] / tot[bb] >= 0.5)
                run = run + 1 if gap_bin else 0
                run_any = run_any + 1 if no_sub else 0
                if run >= gap_bins or run_any >= any_bins:
                    if limit is not None and (bb - run + 1) * s.bin_um > limit + s.bin_um:
                        continue                         # (beyond the next chip's limit: not this chip's edge)
                    if run < gap_bins:
                        run = run_any
                        self.edge_how = "no substrate"
                        self.say(f"    (edge: {f1(any_um or s.edge_any_um)} um without substrate; the surround on this side is not dark)")
                    first = bb - run + 1
                    if first <= 1:
                        return 0.0
                    fa, fb = sub[first - 1] / tot[first - 1], sub[first] / tot[first]
                    t = min(max((fa - 0.5) / max(fa - fb, 1e-6), 0.0), 1.0)
                    if fa < 0.5:
                        t = 0.0
                    return (first - 2 + 0.5 + t) * s.bin_um
            k += 1
        return -1.0

    def _narrow_edge(self, sub, tot, drk, limit):
        """find_edge's search before the next chip: the last run of at least narrow_gap_um without substrate (dark if
        surround_dark) followed by substrate again, within `limit` um.  -> distance (um) or -2."""
        s = self.s
        nbin = max(1, int(math.ceil(s.narrow_gap_um / s.bin_um)))
        last = int(limit / s.bin_um) + 1
        frac = np.where(tot > 0, sub / np.maximum(tot, 1), np.nan)
        dark = np.where(tot > 0, drk / np.maximum(tot, 1), 0.0)
        gap = (frac < max(s.gap_fraction, 0.15)) & (~s.surround_dark | (dark >= 0.5))
        found, bb = None, 2
        while bb <= min(last, len(tot) - 1):
            if gap[bb]:
                e = bb
                while e + 1 < len(tot) and gap[e + 1]:
                    e += 1
                after = frac[e + 1] if e + 1 < len(tot) else np.nan
                if e - bb + 1 >= nbin and (not np.isfinite(after) or after >= 0.5 or e + 1 > last):
                    found = bb
                bb = e + 1
            else:
                bb += 1
        if found is None:
            self.edge_how = "neighbour"
            return -2.0
        self.edge_how = "narrow gap"
        fa, fb = frac[found - 1], frac[found]
        t = min(max((fa - 0.5) / max(fa - fb, 1e-6), 0.0), 1.0) if np.isfinite(fa) and fa >= 0.5 else 0.0
        return (found - 2 + 0.5 + t) * s.bin_um

    def ask_edge(self, name, sx, sy, dx, dy, limit, who):
        """No gap between this chip and the next one (touching): the user puts the crosshair on this chip's edge.
        -> distance from (sx, sy) along (dx, dy) in um."""
        s = self.s
        if not s.edge_ask:
            raise self.fail(f"the {name} edge runs into the next chip ({who}) with no gap between them: move the chips apart, "
                            "or set edge_ask to be asked for the edge")
        self.say(f"  {name}: the next chip ({who}) is reached with no gap between them: asking for the edge")
        try:
            self.move_to(sx + dx * max(limit - 0.5 * (abs(dx) * self.FOVX + abs(dy) * self.FOVY), 0.0),
                         sy + dy * max(limit - 0.5 * (abs(dx) * self.FOVX + abs(dy) * self.FOVY), 0.0))
        except StageLimit:
            pass
        self.ui.mock_hook(self, "mockEdge", sx, sy, dx, dy)
        try:
            from .notify import send as _notify
            _notify(s, "operator", f"{s.name}: you are needed", f"The {name} edge of {s.name} touches the next chip ({who}): "
                    "put the crosshair on the edge at the microscope.", log=self.say)
        except Exception:
            pass
        r = self.ask_focus(f"{s.name}: where is the {name} edge?",
                           f"This chip and the next one ({who}) touch, so the {name} edge of {s.name} could not be found. "
                           f"Move the stage until the cross is on the {name} edge of {s.name} (where its substrate ends), "
                           "then press the button.", button="The edge is at the cross", cross=True)
        if r == 2:
            raise Stop("stopped by the user")
        if r != 0:
            raise self.fail(f"{name} edge not given (chip touches {who})")
        c = self.core
        x, y = c.getXPosition(self.xy), c.getYPosition(self.xy)
        d = (x - sx) * dx + (y - sy) * dy
        if d <= 0:
            raise self.fail(f"the {name} edge given is not on the {name} side of the start")
        self.say(f"  {name} edge given by hand: {f1(d)} um from the start")
        return d

    def inside_chip(self, x, y, grow):
        for e in range(4):
            L = self.lines[e]
            if L[2] * (x - L[0]) + L[3] * (y - L[1]) > grow:
                return False
            if self.side_type[e] == 1 and math.hypot(x - self.arcs[e][0], y - self.arcs[e][1]) > self.arcs[e][2] + grow:
                return False
        return True

    def tile_needed(self, tx, ty):
        FX, FY, mg = self.FOVX, self.FOVY, self.s.margin_um
        if self.polygon is not None:                # (a scan area drawn by hand: every field that overlaps it)
            return box_meets_polygon(tx - FX / 2, ty - FY / 2, tx + FX / 2, ty + FY / 2, self.polygon)
        for e in range(4):
            L = self.lines[e]
            best = min(L[2] * (tx + sx * FX / 2 - L[0]) + L[3] * (ty + sy * FY / 2 - L[1]) for sx in (-1, 1) for sy in (-1, 1))
            if best > mg:
                return False
            if self.side_type[e] == 1:
                a = self.arcs[e]
                qx = max(tx - FX / 2, min(tx + FX / 2, a[0])); qy = max(ty - FY / 2, min(ty + FY / 2, a[1]))
                if math.hypot(qx - a[0], qy - a[1]) > a[2] + mg:
                    return False
        return True

    # ---- files
    def write_manifest(self, path, pieces, state=None):
        """The run file; state None: the state written last (chip notes saved from the notes window's thread)."""
        import threading
        lk = self.__dict__.setdefault("_manifest_lock", threading.Lock())
        with lk:
            if state is None:
                state = getattr(self, "manifest_state", "scanning")
            self.manifest_state = state
            self._write_manifest(path, pieces, state)

    def _write_manifest(self, path, pieces, state):
        d = {"script_version": VERSION, "state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
             **({"results_copied_to": self.results_dest} if self.results_dest else {}),
             "pieces": [{"name": p["name"], "center": [p["c"][0], p["c"][1]], "z": p["c"][2], "material": p["material"],
                         "oxide_nm": p["oxide"], "state": p["state"], "tiles": p["name"] + "_tiles", "targets": p["name"] + "_targets.csv",
                         **({"notes": p["notes"]} if p.get("notes") else {}),
                         **({"area_mm2": p["area_mm2"]} if p.get("area_mm2") else {}),
                         **({"max_targets": p["max_targets"]} if p.get("max_targets") else {}),
                         **({"hbn_max_targets": p["hbn_max_targets"]} if "hbn_max_targets" in p else {}),
                         **({"mount": p["mount"]} if p.get("mount") else {}),
                         **({"polygon": p["polygon"]} if p.get("polygon") else {})}
                        for p in pieces]}
        if getattr(self, "mounts", None):               # (mounted chips: one stage frame per block, for the AFM)
            d["chip_type"] = self.s.chip_type
            d["mounts"] = [{"mount": m, "corners": cs, "pieces": [p["name"] for p in pieces if p.get("mount") == m]}
                           for m, cs in sorted(self.mounts.items())]
        elif self.s.chip_type != "large":
            d["chip_type"] = self.s.chip_type
        # via a .tmp + rename (the analysis never reads half a file); a rename refused for a moment on Windows (virus
        # scanner, indexer) is retried, then the file is written directly; the run file must never stop the scan
        txt, tmp = json.dumps(d, indent=1), Path(str(path) + ".tmp")
        for k in range(10):
            try:
                tmp.write_text(txt)
                os.replace(str(tmp), str(path))
                return
            except OSError:
                time.sleep(0.3)
        try:
            Path(path).write_text(txt)
        except OSError as e:
            self.say(f"  (run file not updated: {e})")

    def save_tile(self, d, index, label, col, row, background=False, cached=False):
        """Snap a tile and write it (TIFF + metadata.txt, Micro-Manager layout).  background: written by the tile writer
        thread while the stage moves on; cached: the camera properties read once per piece and the position from the
        last move (each property read and position query is a round trip to the driver / controller)."""
        img = self.snap_good(f"tile {label}")
        t_op = time.time()
        c, cam = self.core, self.core.getCameraDevice()
        tif = f"img_channel000_position{pad3(index)}_time000000000_z000.tif"
        if cached and not any(isnan(v) for v in self.last_xy):
            xy = self.last_xy
        else:
            xy = (c.getXPosition(self.xy), c.getYPosition(self.xy))
        if not self.has_z:
            zr = 0.0
        elif cached == "sent" and not isnan(self.z_cmd):      # (tile_readback_every: Z as sent, read back on the checked tiles)
            zr = self.z_cmd
        else:
            zr = self.z_read()
            if cached and not isnan(self.z_cmd) and abs(zr - self.z_cmd) > self.s.z_reach_tol_um:
                self.z_limit_hit(self.z_cmd, zr, f"tile {label}")
            elif cached and not isnan(self.z_cmd) and abs(zr - self.z_cmd) > 0.5:
                self.say(f"    WARNING: the focus drive reads {f3(zr)} um, it was sent to {f3(self.z_cmd)} um")
        self.last_pic = dict(frame=img, x=float(xy[0]), y=float(xy[1]), z=float(zr), hundred=False, what=f"10x tile {label}")
        fm = {"Label": label, "GridCol": int(col), "GridRow": int(row), "XPositionUm": xy[0], "YPositionUm": xy[1],
              "Objective": getattr(self, "tile_objective", ""),
              "ZPositionUm": zr, "ZPositionDrive": zr * self.z_scale if self.has_z else 0,
              "Exposure-ms": c.getExposure(), "Camera": cam, "Width": self.W, "Height": self.H}
        props = getattr(self, "_props", None) if cached else None
        if props is None:
            props = {}
            try:
                for nm in c.getDevicePropertyNames(cam):
                    try:
                        props[f"{cam}-{nm}"] = c.getProperty(cam, nm)
                    except Exception:
                        pass
            except Exception:
                pass
            if cached:
                self._props = props
        fm.update(props)
        md = {"Summary": {"Source": f"flakeid.scope {VERSION}", "Width": self.W, "Height": self.H,
                          "PixelType": "GRAY16" if img.ndim == 2 else "RGB32", "Z_SCALE": self.z_scale},
              f"Metadata-Default/{tif}": fm}
        job = (str(Path(d) / tif), np.array(img, copy=True), str(Path(d) / "metadata.txt"), json.dumps(md, indent=1))
        if background:
            if self.writer is None:
                self.writer = TileWriter()
            self.writer.put(job)
        else:
            TileWriter.write(job)
        self.op("file saving (tiles, in the scan thread)", t_op)

    def save_image(self, path, img, info):
        import tifffile
        tifffile.imwrite(str(path), img, description="\n".join(f"{k}={v}" for k, v in info.items()))

    # ================================================================================================= modes
    def run(self):
        s, c = self.s, self.core
        self.set_phase("start-up and setup dialogs")
        if s.ask_mode:                          # the start screen (program started without --mode)
            r = self.ui.ask_mode(self)
            if r is None:
                raise Stop("cancelled")
            s.mode = r["mode"]
            if s.mode not in ("flat", "offsets", "scalecal"):   # who gets this job's messages (Slack / email), before anything
                m = self.ui.ask_messages(self)            # else (a popup; closed: as last time)
                from .ui import apply_messages, remembered_messages, contact_of
                m = m if m is not None else remembered_messages(s)
                apply_messages(s, m)
                self.contact = contact_of(m)
                self.ui.busy_banner(self, self.contact)  # (the big "Scan in progress" window, until the job ends)
            if s.mode in ("multi", "chip"):              # what is on the stage, its focus inset, 100x pictures per cm2
                from .ui import apply_chips, remember_chips, remembered_chips
                ch = self.ui.ask_chips(self) if s.mode == "multi" else None
                if ch == "cancel":                      # (drawn areas: the targets per cm2 must be typed; window closed)
                    self.say("  the Chips window was closed: drawn areas need the targets per cm2 typed for each job; cancelled")
                    raise Stop("cancelled")
                if ch is not None:
                    remember_chips(s, ch)
                else:                                   # (no window: as last time; a single chip scan: larger pieces)
                    ch = remembered_chips(s) if s.mode == "multi" else dict(remembered_chips(s), chip_type="large")
                apply_chips(s, ch)
                self.mount_plan = list(ch.get("mounts") or []) if s.chip_type == "mounted_1cm" else []
                self.say(f"  chips: {s.chip_type.replace('_', ' ')}" + (f", {len(self.mount_plan)} block(s) of "
                         f"{' + '.join(map(str, self.mount_plan))} chips" if self.mount_plan else "") +
                         (f", focus points {s.chip_focus_inset_mm:g} mm in from the edges" if s.chip_type in ("loose_1cm", "mounted_1cm") else "") +
                         f"; 100x: {s.targets_per_cm2:g} targets per cm2 of each piece, hBN: "
                         + (f"{float(s.hbn_imaged_per_cm2):g} per cm2 of them imaged" if s.hbn_imaged_per_cm2 else "all imaged"))
            if s.mode == "flat":
                s.flat_objective, s.flat_oxide_nm, s.flat_frames = r["objective"], r["oxide"], r["frames"]
                s.flat_dark_frames = (s.flat_dark_frames or 20) if r["dark"] else 0
                s.flat_focus = r.get("focus", "fieldstop")
                s.flat_measured_oxide_nm = r.get("measured_oxide") or 0.0
                s.flat_gui = True
            if s.mode == "retarget":
                s.retarget_piece, s.retarget_reloaded, s.retarget_only_failed = r["piece"], r["reloaded"], r["only_failed"]
                s.retarget_manual_points = int(r.get("manual_points", s.retarget_manual_points))
                s.retarget_update_only = bool(r.get("update_only", False))
            if s.mode == "device":
                s.device_name, s.oxide_nm = r["device_name"], r["oxide"]
                if r.get("stack_dir"):                  # (a fabrication step of a stack folder: devicephotos.py)
                    s.device_stack_dir, s.device_step = r["stack_dir"], r["step"]
                    s.device_step_description = r.get("description", "")
            if s.mode == "offsets":
                s.offsets_reference = r.get("reference") or s.offsets_reference
        if s.mode == "retarget" and s.retarget_update_only:
            return self.run_update()
        self.setup_devices()
        self.say(f"chip scan (flakeid.scope {VERSION}): start at {f1(self.X0)}, {f1(self.Y0)} um, field {f1(self.FOVX)} x {f1(self.FOVY)} um")
        self.say(f"  z_scale {self.z_scale} drive units per um (Z in real um from here on)")
        self.say(f"  devices: XY stage '{self.xy}', focus '{self.zdev}', camera '{self.cam}' {self.W} x {self.H} px, "
                 f"{c.getBytesPerPixel()} bytes/px, Z {f1(self.Z0)}")
        if s.mode == "flat" and s.flat_objective == "domain":
            # IR flat field: nothing with the scan camera (its exposure check fails when the IR light path is in place)
            self.say("  mode flat, IR (domain camera): the scan camera is not used")
            self.DARK = 0.0                              # (the scan camera's black level does not apply to the IR camera)
            self.open_nosepiece()
            if self.wait_objective("domain", "Put in the IR objective (the window closes by itself when it is in).") is False:
                raise self.fail("cancelled: not on the IR objective")
            self.view_scale = 1.0 / 9.54
            from .domainflat import run_domain_flat
            return run_domain_flat(self)
        if s.mode == "device":                           # (its own objectives, exposures and focus: devicephotos.py)
            from .devicephotos import run_device_photos
            return run_device_photos(self)
        if s.mode == "offsets":                          # objective offsets tool (objoffsets.py)
            from .objoffsets import run_objective_offsets
            return run_objective_offsets(self)
        if s.mode == "scalecal":                         # scale bar calibration (scalecal.py)
            from .scalecal import run_scale_calibration
            return run_scale_calibration(self)
        hundred = s.mode in ("targets", "capture", "domaintest") or (s.mode == "flat" and s.flat_objective == "100x")
        exposure = s.target_exposure_ms if hundred else s.exposure_ms
        if s.set_camera:
            self.apply_camera(exposure)
        self.DARK = s.dark * (self.full_scale() + 1) / 4096.0
        auto = s.target_auto_exposure if s.mode in ("targets", "capture") else (False if s.mode == "flat" else s.auto_exposure)
        self.check_exposure(auto)
        if s.focus_mode not in ("auto", "manual"):
            raise self.fail('focus_mode must be "auto" or "manual"')
        self.say(f"  mode {s.mode}, focus {s.focus_mode}")
        self.open_nosepiece()
        if not hundred:
            if self.wait_objective("10x", "Put in the 10x objective for the scan (the window closes by itself when it is in).") is False:
                raise self.fail("cancelled: not on the 10x objective")
        else:
            if self.wait_objective("100x", "Put in the 100x objective (the window closes by itself when it is in).") is False:
                raise self.fail("cancelled: not on the 100x objective")
            self.view_scale = 1.0 / 9.54
        self.stage_self_test()
        self.domain_list = []                              # (also for the 100x imaging of the flat / capture / retarget modes)
        self.mosaic_s, self.mosaic_over = {"100x": 0.0, "IR": 0.0}, set()
        self.domain_min_nm = (s.domain_min_layers - 0.5) * 0.335
        if s.mode == "flat":
            return self.run_flat()
        if s.mode == "capture":
            return self.run_capture()
        if s.mode == "retarget":
            return self.run_retarget()
        if s.mode == "domaintest":
            from .domaintest import run_domain_test
            return run_domain_test(self)
        pieces = []
        self.manifest = None
        if s.mode == "multi":
            pieces = self.setup_multi()
        elif s.mode == "chip":
            inf = self.ui.ask_piece_info(self, "Chip scan: material and oxide", f"Piece {s.name}", s.material, s.oxide_nm)
            if inf is None:
                raise self.fail("cancelled")
            Path(s.out_dir).mkdir(parents=True, exist_ok=True)
            self.open_log(Path(s.out_dir) / f"{s.name}_log.txt")
            pieces = [dict(name=s.name, c=(self.X0, self.Y0, self.Z0), state="planned", material=inf[0], oxide=inf[1])]
        self.pieces, self.piece_focus = pieces, []
        if pieces:                                      # (Slack bot: the run's thread)
            try:
                from .notify import start_run
                self.slack_run = start_run(s, f"Scan {getattr(self, 'run_label', None) or s.name} started",
                          f"{len(pieces)} chip{'s' if len(pieces) > 1 else ''}: " + ", ".join(f"{p['name']} ({p['material']}, {p['oxide']:g} nm)" for p in pieces)
                          + ".  Messages about this run come in this thread.", log=self.say)
            except Exception as e:
                self.say(f"  (no Slack thread: {e})")
        self.ask_chip_notes(pieces)                     # (a window beside the scan: it goes on meanwhile)
        for pi, pc in enumerate(pieces):
            if self.stop_all:
                break
            self.scan_piece(pi, pc)
        jobs = self.targets_jobs()
        self.image_targets(jobs)
        self.image_domains()
        if s.mode == "targets":
            self.move_to(self.X0, self.Y0); self.set_z(self.Z0)
        if self.manifest:
            self.write_manifest(self.manifest, self.pieces, "stopped" if self.stop_all else "done")
        self.say("done")
        self.stream_off()
        self.close_nosepiece()
        self.finish_analysis()

    # ---- multi-piece setup
    def setup_multi(self):
        s = self.s
        self.mounts = {}                                # mounted 1 cm chips: block -> its 3 reference corners
        polys = []                                      # scan areas drawn by hand: each piece's polygon [[x, y], ...]
        if s.chip_type == "polygon":
            if s.piece_polygons:
                areas = []
                for i, pg in enumerate(s.piece_polygons):
                    mat, ox = (s.piece_materials[i] if s.piece_materials and i < len(s.piece_materials) else (s.material, s.oxide_nm))
                    areas.append(dict(points=[(float(q[0]), float(q[1]), float(q[2]) if len(q) > 2 else self.z_read()) for q in pg],
                                      material=str(mat).lower(), oxide=float(ox)))
            else:
                exp0 = self.core.getExposure()
                try:
                    areas = self.ui.ask_polygons(self)
                finally:                                # (the window's exposure box is for the view only)
                    self.restore_exposure(exp0)
                if not areas:
                    raise self.fail("cancelled")
                if areas[0].get("focus_mode") in ("auto", "manual"):     # (the polygon window's choice: its 10x focus points)
                    s.polygon_focus_mode = areas[0]["focus_mode"]
                    self.say(f"  10x focus points: {'autofocus' if s.polygon_focus_mode == 'auto' else 'focused by hand'} "
                             f"(100x: {'autofocus' if s.focus_mode == 'auto' else 'focused by hand'})")
            centres = []
            for i, a in enumerate(areas):
                if len(a["points"]) < 3:
                    raise self.fail(f"scan area {i + 1} has fewer than 3 points")
                pg = [[float(q[0]), float(q[1])] for q in a["points"]]
                cx, cy = polygon_interior_point(pg)
                cz = float(np.median([q[2] for q in a["points"]]))
                centres.append((cx, cy, cz, a["material"], a["oxide"]))
                polys.append(pg)
                self.say(f"  scan area {i + 1}: polygon of {len(pg)} points, {polygon_area_um2(pg) / 1e6:.2f} mm2, start {f1(cx)}, "
                         f"{f1(cy)} um, Z {f1(cz)}, {a['material']} on {f1(a['oxide'])} nm oxide")
        elif s.piece_centers:
            centres = []
            for i, pcn in enumerate(s.piece_centers):
                mat, ox = (s.piece_materials[i] if s.piece_materials and i < len(s.piece_materials) else (s.material, s.oxide_nm))
                centres.append((pcn[0], pcn[1], pcn[2] if len(pcn) > 2 else self.z_read(), str(mat).lower(), float(ox)))
            if s.chip_type == "mounted_1cm":
                mts = list(s.piece_mounts or [1] * len(centres))
                centres = [c + (int(mts[i]),) for i, c in enumerate(centres)]
                self.mounts = {int(k): [[float(v) for v in q] for q in cs] for k, cs in (s.mount_corners or {}).items()}
        elif s.chip_type == "mounted_1cm" and getattr(self, "mount_plan", None):
            r = self.ui.ask_mounted(self, self.mount_plan)
            if not r:
                raise self.fail("cancelled")
            centres = r["pieces"]
            self.mounts = {int(m): [[float(v) for v in q] for q in cs] for m, cs in r["corners"].items()}
        else:
            centres = self.ui.ask_pieces(self)
            if not centres:
                raise self.fail("cancelled")
        # images go to a local folder (fast; the network share only gets the finished results); the folder chosen here
        # is where the results (viewers, records) are copied at the end
        run_dir = s.run_dir or s.local_root
        if not s.copy_results_to and s.ask_results_folder:
            from .ui import remember_results_folder, remembered_results_folder
            dest = self.ui.ask_dir(f"Where to save the results of this scan ({len(centres)} pieces)", remembered_results_folder(s))
            if not dest:
                raise self.fail("no folder chosen for the results")
            s.copy_results_to = dest
            remember_results_folder(s, dest)             # (the next scan's question opens here)
        # every question that can be answered now is asked now, so the run then goes on without anyone at the microscope
        # until the objective change: the stacking-domain question (asked after the 100x images before)
        if s.domain_imaging == "ask" and any(str(cn[3]).lower() == "graphite" for cn in centres):
            a = self.ui.confirm("Rhombohedral domains", "Do you want to image rhombohedral (stacking) domains after the 100x images?\n\n"
                                f"The graphite flakes of {s.domain_min_layers} layers or more are imaged again with the IR camera "
                                "(you will be asked to change to the IR objective, camera and filter then).", ("Yes", "No"))
            s.domain_imaging = "yes" if a == 0 else "no"
            self.say(f"  stacking domains: {'imaged after the 100x images' if a == 0 else 'not imaged'} (asked at the start)")
        today = datetime.date.today().strftime("%Y%m%d")
        name = s.run_name
        if not name:
            name = self.ui.ask_text("Multi-piece scan: name", "Name for this scan (letters, digits, - and _).\nLeave empty for today's date "
                                    f"({today}).  The pieces become <name>_1, <name>_2, ...", "")
            if name is None:
                raise self.fail("cancelled")
        base = safe_name(name) or today
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        n0 = free_index(run_dir, base)
        pieces = [dict(name=f"{base}_{n0 + i}", c=tuple(cn[:3]), state="planned", material=cn[3], oxide=cn[4],
                       **({"mount": int(cn[5])} if len(cn) > 5 else {})) for i, cn in enumerate(centres)]
        for p, pg in zip(pieces, polys):
            p["polygon"] = pg
        for m, cs in sorted(self.mounts.items()):
            self.say(f"  block {m}: chips {', '.join(p['name'] for p in pieces if p.get('mount') == m)}; reference corners "
                     + "; ".join(f"{f1(q[0])}, {f1(q[1])}" for q in cs) + " um")
        self.run_dir = run_dir
        self.run_label = f"{base}_run_{n0}"
        self.results_dest = str(Path(s.copy_results_to) / self.run_label) if s.copy_results_to else None
        self.open_log(Path(run_dir) / f"{base}_run_{n0}_log.txt")
        self.manifest = str(Path(run_dir) / f"{base}_run_{n0}.json")
        self.write_manifest(self.manifest, pieces, "scanning")
        self.say(f"  {len(pieces)} pieces: {pieces[0]['name']}" + (f" .. {pieces[-1]['name']}" if len(pieces) > 1 else "") +
                 f" in {run_dir}  (run file {Path(self.manifest).name})")
        if s.copy_results_to:
            self.say(f"  results (viewers, target lists, logs) are copied to {Path(s.copy_results_to) / self.run_label} at the end")
        s.run_scan = True
        s.out_dir = run_dir
        self.say(f"  the scan needs the operator again for the 100x objective once the {len(pieces)} pieces are scanned "
                 "(the expected time is logged after each piece)")
        self.start_analysis()
        self.start_piece_uploads()
        return pieces

    # ---- local analysis (python -m flakeid.live on this PC)
    def start_analysis(self):
        s = self.s
        if not s.local_analysis or self.ui.headless_analysis_off:
            return
        out = Path(s.results_dir or Path(s.local_root) / "results") / self.run_label
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "flakeid.live", str(self.run_dir), "-o", str(out), "--workers", str(s.analysis_workers),
               "--run", Path(self.manifest).name, "--min-targets", str(s.min_targets)] + ([] if s.designer_on_pc else ["--no-designer"])
        # hBN targets: imaging order and ranking (settings target_order / target_rank; unknown values: the analysis's
        # defaults, size order and score ranking)
        if getattr(s, "target_mode", "") == "avram_new1":           # (which flakes: by their clean area, not the score)
            cmd += ["--target-mode", "avram_new1", "--min-clean-um2", f"{float(s.target_min_clean_um2):g}"]
            ct = tuple(s.target_clean_thickness_nm or ())
            if len(ct) == 2:
                cmd += ["--clean-thickness", f"{float(ct[0]):g}", f"{float(ct[1]):g}"]
            self.say(f"  100x targets (avram new1): every clean area (one colour of the right thickness"
                     + (f", {float(ct[0]):g}-{float(ct[1]):g} nm" if len(ct) == 2 else ", the material's thickness window")
                     + f") of at least {float(s.target_min_clean_um2):g} um2, largest first")
        if getattr(s, "target_order", "") in ("size", "path"):
            cmd += ["--target-order", s.target_order]
        if getattr(s, "hbn_max_targets", None) is not None:
            cmd += ["--hbn-max-targets", str(int(s.hbn_max_targets))]
        if getattr(s, "graphite_rank", "") in ("model", "tiers"):
            cmd += ["--graphite-rank", s.graphite_rank]
        if getattr(s, "graphite_max_targets", None) is not None:
            cmd += ["--graphite-max-targets", str(int(s.graphite_max_targets))]
        if getattr(s, "target_rank", "") in ("score", "size"):
            cmd += ["--target-rank", s.target_rank]
            if s.target_rank != "score":
                self.say(f"  100x targets of hBN pieces ranked by '{s.target_rank}' (trial: logit(score) + 2 ln(best terrace d)); "
                         "<piece>_targets_alt.csv lists what the score ranking would have chosen instead")
        if s.flat10:
            cmd += ["--flat", s.flat10]
        for f in s.flat100:
            cmd += ["--flat100", f]
        if s.keep_tiles:
            cmd.append("--keep-tiles")
        if s.flat_domain:
            if Path(s.flat_domain).is_file():
                cmd += ["--flat-domain", s.flat_domain]
            else:
                self.say(f"  WARNING: flat_domain {s.flat_domain} not found: domain images exported without it")
        logp = out / "analysis_log.txt"
        self.analysis_out = out
        try:
            env = dict(os.environ, PYTHONUTF8="1")             # text files as UTF-8 (Windows defaults to cp1252)
            self.analysis = subprocess.Popen(cmd, stdout=open(logp, "a", encoding="utf-8"), stderr=subprocess.STDOUT,
                                             cwd=str(Path(__file__).resolve().parents[2]), env=env)
            self.say(f"  analysis on this PC: {' '.join(cmd[2:])}  (log {logp})")
            import threading
            threading.Thread(target=self._follow_analysis, args=(logp,), daemon=True).start()
        except Exception as e:
            self.say(f"  WARNING: could not start the analysis ({e}); start it by hand: python -m flakeid.live {self.run_dir} ...")

    def _follow_analysis(self, logp):
        """Show the analysis log in the run window as it is written (block progress, target lists, errors), and warn
        if the analysis process ends with an error while the run goes on."""
        pos = 0
        proc = self.analysis
        keep = ("====", "block (", "targets", "ERROR", "Error", "Traceback", "WARNING", "oxide fixed", "scan complete", "100x", "File \"")
        A = self.ana_state = dict(waiting=False)
        while True:
            try:
                with open(logp, encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
                    pos = fh.tell()
            except OSError:
                chunk = ""
            for line in chunk.splitlines():
                if self._ana_line(A, line):
                    continue                             # (a progress line: the bar, not the log)
                if any(k in line for k in keep) or line.startswith(("    ", "  File", "PermissionError", "OSError")):
                    try:
                        self.ui.log("    [analysis] " + line.rstrip())
                    except Exception:
                        pass
            if A["waiting"]:
                self._ana_show(A)
            if proc.poll() is not None:
                if proc.returncode not in (0, None) and self.analysis is proc:
                    msg = (f"The analysis on this PC stopped with an error (exit {proc.returncode}); no more target lists will be "
                           f"written.  See {logp}.")
                    self.say("  WARNING: " + msg)
                    try:
                        self.ui.message("Analysis stopped", msg)
                    except Exception:
                        pass
                return
            time.sleep(3)

    # the progress bar while the run waits for the analysis at the end: what the export has written so far (one JPEG per
    # 100x picture, one PNG per IR picture in each chip's viewer folder) against what was taken (the image index files);
    # the log's progress lines come in batches from parallel worker processes, so they cannot say which chip is done
    def _ana_line(self, A, line):
        """True for an analysis progress line ("  100x: writing images:  44% (7/16), ..."): kept out of the run log."""
        return bool(re.match(r"^\s*(.+?):\s+(\d+)% \((\d+)/(\d+)\)", line))

    def _ana_counts(self):
        """[(chip, 100x taken, exported, IR taken, exported)] from the run folder and the analysis's results folder."""
        rd = Path(getattr(self, "run_dir", None) or self.s.out_dir)
        res = Path(getattr(self, "analysis_out", None) or rd)
        out = []

        def stems(idx):
            try:
                with open(idx, newline="") as fh:
                    return {Path(r["file"]).stem for r in csv.DictReader(fh) if r.get("file")}
            except (OSError, KeyError, ValueError):
                return set()
        for p in self.pieces or ():
            nm = p["name"]
            want = stems(rd / f"{nm}_100x" / f"{nm}_100x_index.csv")
            wantD = set()
            for f in (rd / f"{nm}_domains").glob("*_domains_index.csv") if (rd / f"{nm}_domains").is_dir() else ():
                wantD |= stems(f)
            if not want and not wantD:
                continue
            v = res / nm / f"{nm}_tiles_viewer" / "100x"
            have = {f.stem for f in v.glob("*.jpg")} if v.is_dir() else set()
            haveD = {f.stem for f in (v / "domains").glob("*.png")} if (v / "domains").is_dir() else set()
            out.append((nm, len(want), len(want & have), len(wantD), len(wantD & haveD)))
        return out

    def _ana_show(self, A):
        C = self._ana_counts()
        tot = sum(c[1] + c[3] for c in C)
        done = sum(c[2] + c[4] for c in C)
        if not tot:
            frac, text = None, "waiting for the analysis ..."
        else:
            frac = done / tot
            now = time.time()
            A.setdefault("t0", (now, done))
            t0, d0 = A["t0"]
            left = ""
            if done > d0 and now - t0 > 20:                  # (the rate since the wait began)
                sec = (tot - done) * (now - t0) / (done - d0)
                left = f", about {int(sec // 60)} min {int(sec % 60):02d} s left" if sec >= 60 else f", about {int(sec)} s left"
            # the 100x pictures are exported chip by chip as each chip's 100x imaging ends; the IR pictures after all of them
            cur = next(((k, c) for k, c in enumerate(C) if c[2] < c[1]), None)
            curD = next(((k, c) for k, c in enumerate(C) if c[4] < c[3]), None)
            if cur is not None:
                k, c = cur
                text = f"analysis: chip {k + 1} of {len(C)} ({c[0]}): 100x pictures {c[2]} of {c[1]} exported{left}"
            elif curD is not None:
                k, c = curD
                text = f"analysis: IR pictures of chip {k + 1} of {len(C)} ({c[0]}): {c[4]} of {c[3]} exported{left}"
            else:
                text = f"analysis: all {len(C)} chips exported; finishing ..."
        if getattr(self, "export_bg", False):
            text = f"previous run {self.run_label}: " + text.replace("analysis: ", "")
        try:
            self.ui.analysis_progress(self, frac, text)
        except Exception:
            pass

    def finish_analysis(self):
        """The end of a run: wait for the analysis (100x export), then copy the results to the share.  With
        export_in_background (and the analysis still exporting) both go on in the background: the start screen is back
        at once (the microscope is free), the progress line stays, and the Slack bot says so now and when it is done."""
        s = self.s
        self.set_phase("waiting for the analysis (100x export)")
        import threading
        bg = (s.export_in_background and self.analysis is not None and self.analysis.poll() is None
              and getattr(self, "run_label", None) is not None)
        if bg:
            self.report_times()
            self.export_bg = True
            self._lower_priority(self.analysis)
            label = self.run_label
            C = self._ana_counts()
            left = sum(c[1] - c[2] + c[3] - c[4] for c in C)
            self.say(f"  the microscope is free: set up the next job.  The 100x export ({left} pictures to go) and the copy to the "
                     "share go on in the background (lower priority); the line under the status shows how far it is.")
            try:
                from .notify import send
                send(s, "scanned", f"{label}: scanning finished", "The microscope is free: set up the next job.  The 100x export "
                     f"({left} pictures to go) and the copy to the share go on in the background; another message when they are done.",
                     log=self.say, run=getattr(self, "slack_run", None))
            except Exception:
                pass
            EXPORTING.update(p["name"] for p in self.pieces or ())
            self.upload_thread = threading.Thread(target=self._finish_in_background, name=f"finish {label}")
            self.upload_thread.start()
            return
        if self.analysis is not None:
            self.say(f"  waiting for the analysis on this PC to finish (100x export; up to {s.analysis_wait_h:g} h) ...")
            self.ui.progress(self, "waiting for the analysis (100x export) ...")
            A = getattr(self, "ana_state", None)
            if A is not None:                            # (the progress bar: _follow_analysis updates it every 3 s)
                A["waiting"] = True
                self._ana_show(A)
            try:
                self.analysis.wait(timeout=s.analysis_wait_h * 3600)
                if A is not None:
                    A["waiting"] = False
                    self.ui.analysis_progress(self, "done", "")
                self.say(f"  analysis finished (exit {self.analysis.returncode}); results in {self.analysis_out}")
            except subprocess.TimeoutExpired:
                self.say("  the analysis is still running; results not copied (copy them later from " + str(self.analysis_out) + ")")
                self.report_times()
                return
        self.report_times()
        # the copy to the share runs in the background (a non-daemon thread: the program does not exit before it is
        # done), so the setup of the next job can start at once
        self.upload_thread = threading.Thread(target=self.upload, name="upload results")
        self.upload_thread.start()

    def _finish_in_background(self):
        """The rest of a run after the microscope work: the export, the copy to the share, the last messages."""
        s, label = self.s, self.run_label
        self.say_prefix = f"[{label}] "
        t0 = time.time()
        A = getattr(self, "ana_state", None)
        if A is not None:
            A["waiting"] = True
            self._ana_show(A)
        kind, text = "finished", ""
        try:
            self.analysis.wait(timeout=s.analysis_wait_h * 3600)
            ok = self.analysis.returncode == 0
            self.say(f"  analysis finished (exit {self.analysis.returncode}) after {(time.time() - t0) / 60:.0f} more min; results in {self.analysis_out}")
            self.upload()
            where = self.results_dest or self.analysis_out
            if ok:
                text = f"The 100x export is done ({(time.time() - t0) / 60:.0f} min after the scanning) and the results are copied.  Results: {where}"
            else:
                kind, text = "failed", f"The analysis stopped with an error (exit {self.analysis.returncode}): see analysis_log.txt in {self.analysis_out}."
        except subprocess.TimeoutExpired:
            kind, text = "failed", (f"The export is still running after {s.analysis_wait_h:g} h: results not copied (copy them later from "
                                    f"{self.analysis_out}).")
            self.say("  " + text)
        except Exception as e:
            kind, text = "failed", f"Finishing the run failed: {type(e).__name__}: {e}"
            self.say("  ERROR: " + text)
        finally:
            if A is not None:
                A["waiting"] = False
                try:
                    self.ui.analysis_progress(self, "done", "")
                except Exception:
                    pass
        EXPORTING.difference_update(p["name"] for p in self.pieces or ())
        try:
            from .notify import send, end_run
            send(s, kind, f"{label}: export {'finished' if kind == 'finished' else 'failed'}", text, log=self.say,
                 run=getattr(self, "slack_run", None))
            end_run(getattr(self, "slack_run", None))
        except Exception:
            pass

    def _lower_priority(self, proc):
        """The background export below normal priority (its worker processes started from now on inherit it): the next
        scan's live analysis comes first."""
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(int(proc._handle), 0x00004000)     # BELOW_NORMAL_PRIORITY_CLASS
            else:
                os.setpriority(os.PRIO_PROCESS, proc.pid, 10)
        except Exception as e:
            self.say(f"  (export priority not lowered: {e})")

    def upload(self):
        """Copy this run's results to copy_results_to/<name>_run_<n>/, one folder per piece with its viewer on top
        (share_layout); the raw 10x tiles are not copied.  Files the piece copies during the run (upload_piece) already
        sent and unchanged are not copied again."""
        s = self.s
        self.stop_piece_uploads()
        if not s.copy_results_to or getattr(self, "run_label", None) is None:
            return
        dst = Path(s.copy_results_to) / self.run_label
        rd = Path(self.run_dir)
        names = [p["name"] for p in self.pieces]
        self.say(f"  copying the results to {dst} in the background ...")
        t0 = time.time()
        res = None
        # a window with the copy's progress, the time left and the outcome (ui.upload_progress; 2026-10-27)
        last = [0.0]
        info = dict(run=self.run_label, dst=str(dst), state="copying", done=0, total=0, files=0, n_files=0, elapsed=0.0)

        def show(**kw):
            info.update(kw, elapsed=time.time() - t0)
            try:
                self.ui.upload_progress(self, dict(info))
            except Exception:
                pass

        def prog(b, tb, k, n):
            if time.time() - last[0] >= 0.5 or k == n:
                last[0] = time.time()
                show(done=b, total=tb, files=k, n_files=n)
        show()
        try:
            dst.mkdir(parents=True, exist_ok=True)
            res = getattr(self, "analysis_out", None)
            pairs, viewers = share_layout(rd, res, self.run_label, names, dst)
            todo = self._changed(pairs)
            tb = 0
            for a_, _ in todo:
                try:
                    tb += Path(a_).stat().st_size
                except OSError:
                    pass
            show(n_files=len(todo), total=tb)
            nb = copy_parallel(todo, progress=prog)
            for nm, vfile in viewers.items():
                write_viewer_link(dst / nm / f"{nm} viewer.html", vfile)
            write_copy_done(dst, names, rebuild=False)
            self.uploaded_to = dst                       # (chip notes saved after this are copied there by themselves)
            self.say(f"  results copied to {dst} ({len(todo)} of {len(pairs)} files, {nb / 1e6:.0f} MB, {time.time() - t0:.0f} s)")
            show(state="done", done=nb, total=nb, files=len(todo), n_files=len(todo))
        except Exception as e:
            self.say(f"  WARNING: copying the results to {dst} failed: {e}; they are in {rd}" + (f" and {res}" if res else ""))
            show(state="failed", error=f"{type(e).__name__}: {e}", local=str(res or rd))

    # ---- each piece to the share as soon as its 100x export ends (2026-10-21): the designer builds the first pieces
    # while the next ones are imaged, instead of the whole run after the copy at the end
    def start_piece_uploads(self):
        s = self.s
        if not (s.upload_each_chip and s.copy_results_to and getattr(self, "run_label", None)
                and getattr(self, "analysis_out", None)):
            return
        import threading
        self.pieces_uploaded, self.piece_uploads_stop = set(), threading.Event()
        self.piece_uploader = threading.Thread(target=self._piece_uploads, daemon=True, name="piece uploads")
        self.piece_uploader.start()

    def stop_piece_uploads(self):
        """The piece copies end (one in progress first): the copy at the end takes over."""
        t = getattr(self, "piece_uploader", None)
        if t is not None:
            self.piece_uploads_stop.set()
            t.join()
            self.piece_uploader = None

    def _piece_uploads(self):
        EXPORT_DONE = "export_done.json"                # (flakeid.live.EXPORT_DONE: not imported into the scan program)
        ev = self.piece_uploads_stop
        while not ev.wait(15):
            for p in list(self.pieces or ()):
                nm = p["name"]
                if ev.is_set():
                    return
                if nm not in self.pieces_uploaded and (Path(self.analysis_out) / nm / EXPORT_DONE).is_file():
                    self.pieces_uploaded.add(nm)
                    self.upload_piece(nm)

    def upload_piece(self, nm):
        """The piece's folder (viewer with its 100x pictures, data) and the run's own files (run file, logs) copied to
        the run's folder on the share, copy_done.json marked partial: the designer adds the piece to the run."""
        s = self.s
        dst = Path(s.copy_results_to) / self.run_label
        names = [p["name"] for p in self.pieces]
        t0 = time.time()
        try:
            dst.mkdir(parents=True, exist_ok=True)
            pairs, viewers = share_layout(Path(self.run_dir), self.analysis_out, self.run_label, names, dst)
            todo = self._changed([(a, b) for a, b in pairs if Path(b).parent == dst or dst / nm in Path(b).parents])
            nb = copy_parallel(todo, workers=4)
            if nm in viewers:
                write_viewer_link(dst / nm / f"{nm} viewer.html", viewers[nm])
            write_copy_done(dst, [nm], rebuild=False, of=len(names))
            self.say(f"  {nm} copied to {dst} ({len(todo)} files, {nb / 1e6:.0f} MB, {time.time() - t0:.0f} s): the designer adds it now")
        except Exception as e:
            self.say(f"  WARNING: copying {nm} to {dst} failed ({type(e).__name__}: {e}); it is copied with the rest at the end")

    # ---- chip notes (exfoliation details), asked beside the scan
    CHIP_NOTE_FIELDS = (("batch", "Crystal / batch"), ("exfoliation", "Exfoliation (tape, temperature, method, date)"),
                        ("notes", "Other comments"))

    def ask_chip_notes(self, pieces):
        """A window (beside the scan, which goes on) for each chip's exfoliation details, one chip at a time, each
        filled in from the chip before (the first from the last chip noted on this PC)."""
        s = self.s
        if not s.ask_chip_notes or not pieces or not self.ui.can_ask:
            return
        last = None
        try:
            f = Path(s.local_root) / "last_chip_notes.json"
            last = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
        except (OSError, ValueError):
            last = None
        chips = [dict(name=p["name"], material=p["material"], oxide=p["oxide"]) for p in pieces]
        self.ui.chip_notes(self, chips, self.CHIP_NOTE_FIELDS, last, self.save_chip_notes)

    def save_chip_notes(self, name, notes):
        """Chip `name`'s notes (dict of CHIP_NOTE_FIELDS): <run folder>/<chip>_notes.json (copied with the chip's data;
        also to the share if the results are copied already), the run file, and the next run's first suggestion."""
        s = self.s
        rec = dict({k: str(notes.get(k, "")).strip() for k, _ in self.CHIP_NOTE_FIELDS}, chip=name,
                   saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        pc = next((p for p in (self.pieces or ()) if p["name"] == name), None)
        if pc is not None:
            pc["notes"] = {k: rec[k] for k, _ in self.CHIP_NOTE_FIELDS}
        d = Path(getattr(self, "run_dir", None) or s.out_dir)
        try:
            (d / f"{name}_notes.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            (Path(s.local_root) / "last_chip_notes.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        except OSError as e:
            self.say(f"  (notes of {name} not saved: {e})")
            return
        if self.manifest and self.pieces:
            self.write_manifest(self.manifest, self.pieces)
        up = getattr(self, "uploaded_to", None)
        if up is not None:
            try:
                (Path(up) / name / "data").mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(d / f"{name}_notes.json"), str(Path(up) / name / "data" / f"{name}_notes.json"))
                shutil.copy2(str(self.manifest), str(Path(up) / Path(self.manifest).name))
            except Exception as e:
                self.say(f"  (notes of {name} not copied to {up}: {e})")
        self.say(f"  notes of {name}: " + "; ".join(f"{lab.split(' (')[0].lower()}: {rec[k]}" for k, lab in self.CHIP_NOTE_FIELDS if rec[k]))

    # ---- one piece: edges, plan, focus surface, tiles
    def scan_piece(self, pi, pc):
        s = self.s
        if pi == 0:
            self.t_pieces0 = time.time()
        self.set_phase("chip locating (edges, plan)")
        s.name = pc["name"]
        self.material, self.oxide = pc["material"], pc["oxide"]
        if s.mode == "multi":
            cc = pc["c"]
            self.say("")
            self.say(f"==== piece {pi + 1} of {len(self.pieces)}: {s.name}, centre {f1(cc[0])}, {f1(cc[1])} um, Z {f1(cc[2])}, "
                     f"{self.material} on {f1(self.oxide)} nm oxide")
            self.X0, self.Y0, self.Z0 = cc
            self.move_to(self.X0, self.Y0)
            if self.has_z:
                self.set_z(self.Z0 - s.af_backlash_um); self.set_z(self.Z0)
            pc["state"] = "scanning"
            self.write_manifest(self.manifest, self.pieces, "scanning")
        self.fa = self.fb = 0.0
        self.fc = self.Z0
        self.FQ = self.FS = None
        try:
            if s.mode == "multi":
                self.check_exposure(s.auto_exposure)
            try:
                self.plan_and_scan(pc)
            finally:
                if pc.get("polygon"):
                    self.move_box = None                # (a polygon's box: only for its own scan)
            pc["state"] = "scanned"
        except Stop as e:
            if s.mode != "multi" or self.stop_all:
                raise
            self.say(f"  piece {s.name} FAILED: {e}  (going on with the next piece)")
            pc["state"] = "failed"
        self.piece_focus.append(((self.fa, self.fb, self.fc), self.FQ))
        self.__dict__.setdefault("piece_surface", {})[pi] = self.FS
        if self.manifest:
            self.write_manifest(self.manifest, self.pieces, "scans_done" if pi == len(self.pieces) - 1 else "scanning")
            left, el = len(self.pieces) - pi - 1, time.time() - self.t_pieces0
            if left > 0 and not self.stop_all:
                self.say(f"  the 100x objective change is expected at about {time.strftime('%H:%M', time.localtime(time.time() + el / (pi + 1) * left))} "
                         f"({left} more piece{'s' if left > 1 else ''} to scan, {el / (pi + 1) / 60:.0f} min per piece so far)")

    def start_reference(self):
        """The substrate reference (self.REF) at the start position, from the neighbouring fields when the start frame
        has little bare substrate; returns the start frame's blocks."""
        s = self.s
        start = self.blocks()
        self.REF = self.reference(start)
        fstart = self.substrate_fraction(start)
        R = self.REF
        self.say(f"  reference substrate R/G {f3(R[0])}, B/G {f3(R[1])}, G {f1(R[2])} counts; substrate fraction of the start frame {f3(fstart)}")
        if fstart < 0.5:
            self.say("  start frame has little bare substrate; checking the 8 neighbouring fields")
            best_f, best_ref, best_pos = fstart, self.REF, None
            for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                nx, ny = self.X0 + ox * self.FOVX, self.Y0 + oy * self.FOVY
                if not self.reachable(nx, ny):
                    continue
                try:
                    self.move_to(nx, ny)
                except StageLimit:
                    continue
                bl = self.blocks()
                self.REF = self.reference(bl)
                f = self.substrate_fraction(bl)
                if s.verbose:
                    self.say(f"    field {f1(nx)}, {f1(ny)}: substrate fraction {f3(f)}")
                if f > best_f:
                    best_f, best_ref, best_pos = f, self.REF, (nx, ny)
            self.REF = best_ref
            self.move_to(self.X0, self.Y0)
            fstart = self.substrate_fraction(self.blocks())
            if best_pos:
                self.say(f"  reference from {f1(best_pos[0])}, {f1(best_pos[1])}: R/G {f3(self.REF[0])}, B/G {f3(self.REF[1])}, G {f1(self.REF[2])} "
                         f"(substrate fraction there {f3(best_f)}, at the start {f3(fstart)})")
            if best_f < s.start_min_substrate:
                raise self.fail(f"no bare substrate found around the start (best fraction {f3(best_f)}): move to a cleaner part of the chip, "
                                "or set substrate_colour and retry")
        return start

    def find_outline_polygon(self, pc):
        """A scan area drawn by hand (chip_type "polygon"): the piece's polygon pc["polygon"] is the area, as marked;
        no edge search.  Only the substrate reference is taken, at the start (a point well inside the polygon).
        Returns what find_outline returns (corners: the polygon's points; no edges)."""
        s = self.s
        start = self.start_reference()
        P = [[float(q[0]), float(q[1])] for q in pc["polygon"]]
        A = np.asarray(P)
        sizeX, sizeY = float(np.ptp(A[:, 0])), float(np.ptp(A[:, 1]))
        self.lines = self.side_type = self.arcs = None
        self.clipped = []
        self.say(f"  scan area drawn by hand: polygon of {len(P)} points, {f3(sizeX / 1000)} x {f3(sizeY / 1000)} mm box, "
                 f"{polygon_area_um2(P) / 1e6:.2f} mm2 (no edge search: every field that overlaps it is scanned)")
        return dict(start=start, sizeX=sizeX, sizeY=sizeY, cx=self.X0, cy=self.Y0, small=max(sizeX, sizeY) < s.small_piece_mm * 1000.0,
                    corners=P, rot=0.0, pts=[], n_bad=0, names=(), dirs=(), L=[], polygon=P)

    def find_outline_1cm(self):
        """A 1 cm chip (diced by hand: each side within chip_size_tol of chip_nominal_mm) around the marked centre
        (self.X0, self.Y0, within chip_centre_err_mm of the true one): an axis-aligned square, each side placed by one
        short probe in the window where that edge can be (find_edge from the window's inner end, at most to its outer
        end); a side whose probe finds no edge there (tape over the edge, glue, the slide) takes the window's outer end,
        so the largest chip is still covered.  No rotation fit and no extra edge points: tape and neighbouring chips
        outside the window cannot move a side.  Returns what find_outline returns."""
        s = self.s
        start = self.start_reference()
        half = s.chip_nominal_mm * 500.0
        err = s.chip_centre_err_mm * 1000.0
        inner = max(half * (1 - s.chip_size_tol) - err, 0.6 * max(self.FOVX, self.FOVY))
        outer = half * (1 + s.chip_size_tol) + err
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        names = ("+X", "-X", "+Y", "-Y")
        self.say(f"  1 cm chip ({s.chip_type.replace('_', ' ')}): each edge looked for {f1(inner / 1000)}-{f1(outer / 1000)} mm from the "
                 f"marked centre (nominal {f1(s.chip_nominal_mm)} mm +-{round(s.chip_size_tol * 100)} %, centre within "
                 f"{f1(s.chip_centre_err_mm)} mm)")
        dist, clipped, how = [], [False] * 4, []
        for e in range(4):
            dx, dy = dirs[e]
            lim, who = self.neighbour_limit(self.X0, self.Y0, dx, dy)
            hi = outer if lim is None else min(outer, lim)
            self.edge_limited = False
            d = self.find_edge(self.X0 + dx * inner, self.Y0 + dy * inner, dx, dy, limit=None if lim is None else max(lim - inner, 0.0),
                               max_um=max(hi - inner, 0.0), any_um=s.chip_edge_any_um) if hi > inner else -1.0
            if d == -2:                                 # (touching the next chip: the edge at it)
                dist.append(hi)
                how.append(f"at the next chip ({who})")
            elif d >= 0:
                dist.append(inner + d)
                how.append(("found" if self.edge_how != "no substrate" else "found (tape or glue: no substrate from here on)")
                           if d > 0 else "at the window's inner end (smaller chip, or tape there)")
            else:
                dist.append(hi)
                how.append("not found: the window's outer end" + (f" (the next chip, {who})" if lim is not None and lim < outer else ""))
            clipped[e] = self.edge_limited
            self.say(f"  edge {names[e]}: {f1(dist[e])} um from the centre, {how[e]}")
        self.move_to(self.X0, self.Y0)
        self.clipped = [names[e] for e in range(4) if clipped[e]]
        sizeX, sizeY = dist[0] + dist[1], dist[2] + dist[3]
        cx, cy = self.X0 + (dist[0] - dist[1]) / 2.0, self.Y0 + (dist[2] - dist[3]) / 2.0
        L = [(self.X0 + dist[0], self.Y0, 1.0, 0.0), (self.X0 - dist[1], self.Y0, -1.0, 0.0),
             (self.X0, self.Y0 + dist[2], 0.0, 1.0), (self.X0, self.Y0 - dist[3], 0.0, -1.0)]
        self.lines, self.side_type, self.arcs = L, [0] * 4, [None] * 4
        corners = [intersect(L[0], L[2]), intersect(L[1], L[2]), intersect(L[1], L[3]), intersect(L[0], L[3])]
        pts = [(self.X0 + dirs[e][0] * dist[e], self.Y0 + dirs[e][1] * dist[e]) for e in range(4)]
        n_found = sum(1 for h in how if h.startswith("found"))
        self.say(f"  chip about {f3(sizeX / 1000)} x {f3(sizeY / 1000)} mm ({n_found} of 4 edges found; no rotation fitted)")
        return dict(start=start, sizeX=sizeX, sizeY=sizeY, cx=cx, cy=cy, small=True, corners=corners, rot=0.0, pts=pts,
                    n_bad=0, names=names, dirs=dirs, L=L)

    def find_outline(self):
        """From the start position (self.X0, self.Y0, on the piece): the substrate reference, then the piece's four
        edges (one walk per direction from the start, then more points along each edge), its corners and rotation.
        Sets self.REF, self.lines / side_type / arcs; returns a dict of what the plan needs."""
        s = self.s
        start = self.start_reference()
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        names = ("+X", "-X", "+Y", "-Y")
        dist0, clipped = [], [False] * 4
        for e in range(4):
            lim, who = self.neighbour_limit(self.X0, self.Y0, *dirs[e])
            d = self.find_edge(self.X0, self.Y0, *dirs[e], limit=lim)
            if d == -2:
                d = self.ask_edge(names[e], self.X0, self.Y0, *dirs[e], lim, who)
                self.edge_limited = False
            elif d >= 0 and self.edge_how == "narrow gap":
                self.say(f"  {names[e]}: a narrow gap before the next chip ({who}) taken as the edge")
            if d < 0:
                self.move_to(self.X0, self.Y0)
                raise self.fail(f"no edge found towards {names[e]} within max_travel_um")
            dist0.append(d)
            clipped[e] = self.edge_limited
            if clipped[e]:
                self.say(f"  {names[e]}: the chip goes on beyond the stage's travel: scanned up to it, {f1(d)} um from the start "
                         "(move the chip towards the stage's centre to scan the rest)")
            else:
                self.say(f"  edge {names[e]}: {f1(d)} um from the start")
        self.clipped = [names[e] for e in range(4) if clipped[e]]
        sizeX, sizeY = dist0[0] + dist0[1], dist0[2] + dist0[3]
        cx, cy = self.X0 + (dist0[0] - dist0[1]) / 2.0, self.Y0 + (dist0[2] - dist0[3]) / 2.0
        self.lines, self.side_type, self.arcs = [None] * 4, [0] * 4, [None] * 4
        n_bad = n_arcs = 0
        small = max(sizeX, sizeY) < s.small_piece_mm * 1000.0
        n_edge = min(s.edge_points, s.small_edge_points) if small else s.edge_points
        if small:
            self.say(f"  small piece ({f1(sizeX / 1000)} x {f1(sizeY / 1000)} mm, under {s.small_piece_mm:g} mm): {n_edge} points per edge, "
                     f"{s.small_focus_points if s.focus_mode == 'auto' else s.focus_points} focus points")
        offs = (-s.edge_spread, -s.edge_spread / 2, s.edge_spread / 2, s.edge_spread) if n_edge >= 5 else (-0.75 * s.edge_spread, 0.75 * s.edge_spread)
        pts = []
        for e in range(4):
            edx, edy = dirs[e]
            ex, ey = self.X0 + edx * dist0[e], self.Y0 + edy * dist0[e]
            pts.append((ex, ey))
            enx, eny = float(edx), float(edy)
            if s.edge_points >= 2 and not clipped[e]:
                acc = [(ex, ey)]
                for o in offs:
                    span = o * (sizeY if edx != 0 else sizeX)
                    lead = max(dist0[e] - s.lead * (self.FOVX if edx != 0 else self.FOVY), 0.0)
                    osx = self.X0 + edx * lead if edx != 0 else cx + span
                    osy = cy + span if edx != 0 else self.Y0 + edy * lead
                    try:
                        self.move_to(osx, osy)
                        if self.substrate_fraction(self.blocks()) < 0.5:
                            osx = self.X0 if edx != 0 else cx + span
                            osy = cy + span if edx != 0 else self.Y0
                            self.move_to(osx, osy)
                            if self.substrate_fraction(self.blocks()) < 0.5:
                                self.say(f"  ({names[e]} edge: no bare substrate at {f1(osx)}, {f1(osy)}, point skipped)")
                                continue
                    except StageLimit:
                        self.say(f"  ({names[e]} edge: {f1(osx)}, {f1(osy)} is beyond the stage's travel, point skipped)")
                        continue
                    olim, _ = self.neighbour_limit(osx, osy, edx, edy)
                    dd = self.find_edge(osx, osy, edx, edy, limit=olim)
                    if dd == -2:
                        self.say(f"  ({names[e]} edge: the next chip reached with no gap at {f1(osx)}, {f1(osy)}, point skipped)")
                        continue
                    if dd > 0 and not self.edge_limited:
                        pe = (osx + edx * dd, osy + edy * dd)
                        acc.append(pe); pts.append(pe)
                if len(acc) >= 2:
                    A = np.asarray(acc)
                    mx, my = A[:, 0].mean(), A[:, 1].mean()
                    ax, ay = A[:, 0] - mx, A[:, 1] - my
                    th = 0.5 * math.atan2(2 * (ax @ ay), (ax @ ax) - (ay @ ay))
                    enx, eny = -math.sin(th), math.cos(th)
                    if enx * edx + eny * edy < 0:
                        enx, eny = -enx, -eny
                    ex, ey = mx, my
                    worst = float(np.abs(enx * (A[:, 0] - ex) + eny * (A[:, 1] - ey)).max())
                    if worst > s.edge_tol_um:
                        arc = fit_circle(acc) if len(acc) >= 3 else None
                        arc_ok = False
                        aw = 0.0
                        if arc is not None:
                            aw = float(np.abs(np.hypot(A[:, 0] - arc[0], A[:, 1] - arc[1]) - arc[2]).max())
                            inward = (arc[0] - mx) * edx + (arc[1] - my) * edy < 0
                            arc_ok = aw <= s.edge_tol_um and inward and s.arc_r_range[0] <= arc[2] <= s.arc_r_range[1]
                            if not arc_ok:
                                self.say(f"  {names[e]} edge: not straight ({f1(worst)} um off a line), not an arc either "
                                         f"(radius {f1(arc[2] / 1000)} mm, {f1(aw)} um off)")
                        enx, eny = float(edx), float(edy)
                        k = int(np.argmax(edx * A[:, 0] + edy * A[:, 1]))
                        ex, ey = A[k]
                        if arc_ok:
                            self.side_type[e], self.arcs[e] = 1, arc
                            n_arcs += 1
                            self.say(f"  {names[e]} edge is round: arc of radius {f1(arc[2] / 1000)} mm (points within {f1(aw)} um)")
                        else:
                            self.side_type[e] = 2
                            n_bad += 1
                            self.say(f"  {names[e]} edge points are not on one line ({f1(worst)} um off); using a bounding edge there (a few extra tiles)")
            self.lines[e] = (float(ex), float(ey), enx, eny)
        L = self.lines
        corners = [intersect(L[0], L[2]), intersect(L[1], L[2]), intersect(L[1], L[3]), intersect(L[0], L[3])]
        rs = next((e for e in range(4) if self.side_type[e] == 0), 0)
        rot = math.degrees(math.atan2(dirs[rs][0] * L[rs][3] - dirs[rs][1] * L[rs][2], dirs[rs][0] * L[rs][2] + dirs[rs][1] * L[rs][3]))
        self.say(f"  chip about {f3(sizeX / 1000)} x {f3(sizeY / 1000)} mm, rotation {f1(rot)} deg")
        angs = [math.degrees(math.atan2(dirs[e][0] * L[e][3] - dirs[e][1] * L[e][2], dirs[e][0] * L[e][2] + dirs[e][1] * L[e][3]))
                for e in range(4) if self.side_type[e] == 0]
        if n_arcs:
            self.say(f"  {n_arcs} round edge(s) (wafer rim)")
        if s.edge_points >= 2 and angs and max(angs) - min(angs) > 4.0:
            self.say(f"  WARNING: the four edges disagree by {f1(max(angs) - min(angs))} deg (chip rotated by more than ~12 deg, or not "
                     "rectangular); check the plan, or rotate the chip closer to the stage axes and rerun.")
        elif n_bad:
            self.say(f"  note: {n_bad} edge(s) not straight (broken corner or chipped edge) and replaced by bounding edges.")
        elif abs(rot) > 12.0:
            self.say(f"  WARNING: chip rotated by {f1(rot)} deg; the edge search is reliable to ~12 deg. Check the plan.")

        return dict(start=start, sizeX=sizeX, sizeY=sizeY, cx=cx, cy=cy, small=small, corners=corners, rot=rot, pts=pts,
                    n_bad=n_bad, names=names, dirs=dirs, L=L)

    def plan_and_scan(self, pc):
        s = self.s
        one_cm = s.chip_type in ("loose_1cm", "mounted_1cm")
        self.polygon = None
        if pc.get("polygon"):
            O = self.find_outline_polygon(pc)
        else:
            O = self.find_outline_1cm() if one_cm else self.find_outline()
        self.polygon = O.get("polygon")
        if self.polygon is not None:
            A = np.asarray(self.polygon, float)
            mg = 2000.0                                 # (fields over the edges, focus searches)
            self.move_box = (A[:, 0].min() - mg, A[:, 0].max() + mg, A[:, 1].min() - mg, A[:, 1].max() + mg)
        pc["outline"] = [[float(v) for v in q] for q in O["corners"]]       # (the next pieces' edge walks stop at it)
        pc["area_mm2"] = round(polygon_area_um2(pc["outline"]) / 1e6, 2)
        mt = targets_for_area(s, pc["area_mm2"])
        if getattr(s, "target_mode", "") == "avram_new1":     # (every clean area: no limit on the number)
            mt = None
            self.say(f"  piece area {pc['area_mm2'] / 100:.2f} cm2: every clean area of at least {float(s.target_min_clean_um2):g} um2 "
                     "imaged at 100x (avram new1: no limit)")
        elif pc.get("material") == "hbn":                       # (hBN pictures in proportion to the area too)
            pc["hbn_max_targets"] = hbn_imaged_for_area(s, pc["area_mm2"])
        if mt:
            pc["max_targets"] = mt
            hm = int(pc.get("hbn_max_targets", 0) or 0) if pc.get("material") == "hbn" else 0
            gm = int(getattr(s, "graphite_max_targets", 0) or 0) if pc.get("material") == "graphite" else 0
            self.say(f"  piece area {pc['area_mm2'] / 100:.2f} cm2: up to {mt} 100x targets ({f1(s.targets_per_cm2)} per cm2)"
                     + (f", the {hm} with the largest best terrace imaged ({f1(s.hbn_imaged_per_cm2)} per cm2)"
                        if 0 < hm < mt and getattr(s, "hbn_imaged_per_cm2", None) else
                        f", the {hm} with the largest best terrace imaged" if 0 < hm < mt else "")
                     + (f", the {gm} most likely kept at 100x imaged" if 0 < gm < mt else ""))
        if getattr(self, "manifest", None) and getattr(self, "pieces", None):
            # (the piece's area and target numbers in the run file before its first tile: the analysis takes the piece's
            # entry as it starts on it; 20261027_run_6 got its fixed 80 instead of 660, the file written only after)
            self.write_manifest(self.manifest, self.pieces)
        sizeX, sizeY, cx, cy, small = O["sizeX"], O["sizeY"], O["cx"], O["cy"], O["small"]
        corners, rot, pts, names, L = O["corners"], O["rot"], O["pts"], O["names"], O["L"]

        # tile grid
        stepX, stepY = self.FOVX * (1 - s.overlap), self.FOVY * (1 - s.overlap)
        C = np.asarray(corners)
        minx, maxx = C[:, 0].min() - s.margin_um, C[:, 0].max() + s.margin_um
        miny, maxy = C[:, 1].min() - s.margin_um, C[:, 1].max() + s.margin_um
        ncol = int(math.ceil(max(maxx - minx - self.FOVX, 0) / stepX)) + 1
        nrow = int(math.ceil(max(maxy - miny - self.FOVY, 0) / stepY)) + 1
        gx0, gy0 = (minx + maxx) / 2 - (ncol - 1) * stepX / 2, (miny + maxy) / 2 - (nrow - 1) * stepY / 2
        # a chip that goes on beyond the stage's travel on one side: the grid starts at the limit there (its last column /
        # row centred on the limit), so everything within reach is scanned
        cl, bx = getattr(self, "clipped", []) or [], self.stage_box()
        if ("+X" in cl) != ("-X" in cl):
            gx0 = bx[1] - (ncol - 1) * stepX if "+X" in cl and bx[1] is not None else bx[0] if "-X" in cl and bx[0] is not None else gx0
            ncol += 1                                   # (one column more: the far side stays covered)
            if "+X" in cl:
                gx0 -= stepX
        if ("+Y" in cl) != ("-Y" in cl):
            gy0 = bx[3] - (nrow - 1) * stepY if "+Y" in cl and bx[3] is not None else bx[2] if "-Y" in cl and bx[2] is not None else gy0
            nrow += 1
            if "+Y" in cl:
                gy0 -= stepY
        tiles = [[gx0 + c_ * stepX, gy0 + r * stepY, c_, r] for r in range(nrow) for c_ in range(ncol)
                 if self.tile_needed(gx0 + c_ * stepX, gy0 + r * stepY)]
        n_all = len(tiles)
        tiles = [t for t in tiles if self.reachable(t[0], t[1])]
        if len(tiles) < n_all:
            self.say(f"  {n_all - len(tiles)} tiles beyond the stage's travel {self.box_text()} left out")
        if not tiles:
            raise self.fail("no part of this chip is within the stage's travel")
        self.say(f"  scan: {len(tiles)} tiles (grid {ncol} x {nrow}, overlap {round(s.overlap * 100)} %)"
                 + ("" if self.polygon is not None else f", {self.n_snaps} frames used to find the edges"))

        # focus surface
        self.set_phase("10x focus points")
        self.fa = self.fb = 0.0
        self.fc = self.Z0
        focus_rms = -1.0
        fpts = []
        self.FQ = self.FS = None
        # (a polygon: its window's choice for these 10x points (polygon_focus_mode); the 100x keeps focus_mode)
        fmode = (s.polygon_focus_mode or s.focus_mode) if self.polygon is not None else s.focus_mode
        nfp = (min(s.focus_points_auto, s.small_focus_points) if small else s.focus_points_auto) if fmode == "auto" else s.focus_points
        if self.polygon is not None:                    # (a scan area drawn by hand: polygon_focus_per_cm2 of its area)
            nfp = max(4, int(math.ceil(s.polygon_focus_per_cm2 * polygon_area_um2(self.polygon) / 1e8 - 1e-9)))
        fin = s.focus_inset                             # (1 cm chips: chip_focus_inset_mm in from the edges)
        if one_cm:
            fin = min(max(s.chip_focus_inset_mm * 1000.0 / max(min(sizeX, sizeY), 1.0), 0.02), 0.45)
            self.say(f"  focus points {f1(s.chip_focus_inset_mm)} mm in from the edges")
        if nfp >= 3 and self.has_z:
            uv = ([(1, 1), (0, 1), (0, 0), (1, 0), (0.5, 0.5), (0.5, 1), (0, 0.5), (0.5, 0), (1, 0.5)] if nfp >= 9 else
                  [(1, 1), (0, 1), (0, 0), (1, 0)] if nfp >= 4 else [(1, 1), (0, 1), (1, 0)])
            if self.polygon is not None:
                # (a scan area drawn by hand: the points spread uniformly over the polygon, in a short path)
                ccx, ccy = O["cx"], O["cy"]
                fq = polygon_focus_grid(self.polygon, nfp, start=(self.X0, self.Y0))
                self.say(f"  {len(fq)} focus points spread over the polygon ({s.polygon_focus_per_cm2:g} per cm2), "
                         + ("autofocus" if fmode == "auto" else "focused by hand"))
            else:
                ccx, ccy = C[:, 0].mean(), C[:, 1].mean()
                fq = []
                for u0, v0 in uv:
                    uu, vv = fin + (1 - 2 * fin) * u0, fin + (1 - 2 * fin) * v0
                    fq.append(((1 - uu) * (1 - vv) * C[2, 0] + uu * (1 - vv) * C[3, 0] + uu * vv * C[0, 0] + (1 - uu) * vv * C[1, 0],
                               (1 - uu) * (1 - vv) * C[2, 1] + uu * (1 - vv) * C[3, 1] + uu * vv * C[0, 1] + (1 - uu) * vv * C[1, 1]))
            for k, (qx, qy) in enumerate(fq):
                self.pause_point()
                qx, qy = self.clamp_reach(qx, qy)          # (a chip partly beyond the stage's travel: focused within it)
                self.move_to(qx, qy)
                if fpts:
                    self.set_z(self.predict_z(qx, qy, fpts, None, self.Z0))
                if fmode == "auto":
                    # on bare substrate (not tape over the chip's edge, debris, a large flake): else towards the centre
                    q2 = self.focus_spot_local(k, qx, qy) if self.polygon is not None else self.focus_spot(k, qx, qy, ccx, ccy)
                    if q2 is None:
                        continue
                    qx, qy = q2
                zguess = self.z_read()
                if fmode == "auto":
                    ok = False
                    for tr in range(s.af_retries + 1):
                        if ok:
                            break
                        if tr > 0:
                            if self.polygon is not None:  # (half a field towards the middle: the points stay spread)
                                dd = math.hypot(ccx - qx, ccy - qy) or 1.0
                                st = min(0.5 * min(self.FOVX, self.FOVY), dd)
                                qx, qy = qx + (ccx - qx) / dd * st, qy + (ccy - qy) / dd * st
                            else:
                                qx, qy = qx + (ccx - qx) * 0.12, qy + (ccy - qy) * 0.12
                            self.move_to(qx, qy)
                        ok = self.autofocus(zguess, s.af10)[2]
                    if not ok:
                        self.say(f"  focus point {k + 1}: autofocus failed, point skipped")
                        continue
                else:
                    ans = self.ask_focus(f"{s.name} focus surface", f"Focus point {k + 1} of {len(fq)} at {f1(qx)}, {f1(qy)} um.")
                    if ans == 2:
                        break
                    if ans in (1, 3):
                        continue
                zf = self.z_read()
                if fmode == "auto" and len(fpts) >= 3 and self.polygon is None:
                    # far off the plane through the others (tape is tens of um thick): once more, further in
                    # (a polygon: its points are checked against their neighbours once all are in, local_outliers)
                    pl0 = fit_plane(fpts)
                    if pl0 is not None:
                        off = zf - (pl0[0] * qx + pl0[1] * qy + pl0[2])
                        if abs(off) > s.focus_outlier_um:
                            self.say(f"  focus point {k + 1}: Z {f1(zf)} is {f1(off)} um off the other points (tape, debris?): "
                                     "again further in")
                            q2 = self.focus_spot(k, qx + (ccx - qx) * 0.25, qy + (ccy - qy) * 0.25, ccx, ccy)
                            if q2 is not None:
                                self.move_to(*q2)
                                self.set_z(pl0[0] * q2[0] + pl0[1] * q2[1] + pl0[2])
                                r2 = self.autofocus(self.z_read(), s.af10)
                                if r2[2]:
                                    z2 = self.z_read()
                                    off2 = z2 - (pl0[0] * q2[0] + pl0[1] * q2[1] + pl0[2])
                                    if abs(off2) < abs(off):
                                        qx, qy, zf = q2[0], q2[1], z2
                                        self.say(f"  focus point {k + 1}: moved to {f1(qx)}, {f1(qy)}: {f1(off2)} um off")
                fpts.append((qx, qy, zf))
                self.say(f"  focus point {k + 1}: {f1(qx)}, {f1(qy)} -> Z {f1(zf)}")
                pl = fit_plane_any(fpts) if self.polygon is not None else fit_plane(fpts)
                if pl is not None:
                    self.fa, self.fb, self.fc = pl
            if fmode == "auto" and self.polygon is not None and len(fpts) >= 5:
                # (against the smooth surface through the others: a bend of the sample is not an outlier)
                fpts, dropped = local_outliers(fpts, s.focus_outlier_um, s.polygon_focus_smoothing)
                for d in dropped:
                    self.say(f"  focus point at {f1(d[0])}, {f1(d[1])} (Z {f1(d[2])}) left out: {f1(d[3])} um off the surface "
                             "through its neighbours (tape, debris?)")
                if dropped:
                    pl = fit_plane_any(fpts)
                    if pl is not None:
                        self.fa, self.fb, self.fc = pl
            elif fmode == "auto" and len(fpts) >= 4:
                fpts, dropped, unsure = robust_focus_points(fpts, s.focus_outlier_um, s.max_tilt)
                if unsure:
                    # (corner points disagree and which one is wrong cannot be told: one more point, at the centre)
                    self.say("  focus points disagree by more than focus_outlier_um and the odd one cannot be told: "
                             "one more point, inside the chip")
                    # (off both diagonals: a centre point lies on them and cannot tell a corner from the opposite one)
                    uu, vv = 0.5, fin + (1 - 2 * fin) * 0.3
                    ex = (1 - uu) * (1 - vv) * C[2, 0] + uu * (1 - vv) * C[3, 0] + uu * vv * C[0, 0] + (1 - uu) * vv * C[1, 0]
                    ey = (1 - uu) * (1 - vv) * C[2, 1] + uu * (1 - vv) * C[3, 1] + uu * vv * C[0, 1] + (1 - uu) * vv * C[1, 1]
                    q2 = self.focus_spot(len(fpts), ex, ey, ccx, ccy)
                    if q2 is not None:
                        self.move_to(*q2)
                        self.set_z(self.predict_z(q2[0], q2[1], fpts, None, self.Z0))
                        if self.autofocus(self.z_read(), s.af10)[2]:
                            fpts.append((q2[0], q2[1], self.z_read()))
                            self.say(f"  focus point (extra): {f1(q2[0])}, {f1(q2[1])} -> Z {f1(fpts[-1][2])}")
                    fpts, dropped, unsure = robust_focus_points(fpts, s.focus_outlier_um, s.max_tilt)
                    if unsure:
                        self.say("  WARNING: the focus points still disagree and the odd one cannot be told: all kept")
                for d in dropped:
                    self.say(f"  focus point at {f1(d[0])}, {f1(d[1])} (Z {f1(d[2])}) left out: {f1(d[3])} um off the plane "
                             "through the others (tape over the chip, debris?)")
                if dropped:
                    pl = fit_plane(fpts)
                    if pl is not None:
                        self.fa, self.fb, self.fc = pl
            if len(fpts) >= 3:
                P = np.asarray(fpts)
                dz = P[:, 2] - (self.fa * P[:, 0] + self.fb * P[:, 1] + self.fc)
                focus_rms = float(np.sqrt((dz ** 2).mean()))
                tilt = math.hypot(self.fa, self.fb) * 1000
                self.say(f"  focus plane: tilt {f3(tilt)} um/mm (dZ/dX {f3(self.fa * 1000)}, dZ/dY {f3(self.fb * 1000)} um/mm)"
                         + (f", rms {f3(focus_rms)} um" if len(fpts) > 3 else ""))
                if self.polygon is not None:
                    # (a polygon: the tiles follow a smooth surface through all the points, not the plane)
                    self.FS = smooth_surface(fpts, s.polygon_focus_smoothing)
                    if self.FS is not None:
                        rs = float(np.sqrt(np.mean([(f[2] - self.FS(f[0], f[1])) ** 2 for f in fpts])))
                        self.say(f"  focus surface: smooth through {len(fpts)} points (thin-plate spline, smoothing "
                                 f"{s.polygon_focus_smoothing:g}), {f3(rs)} um rms at the points (the plane: {f3(focus_rms)} um)")
                        focus_rms = rs
                zz = [self.z_surf(t[0], t[1]) for t in tiles]
                self.say(f"  tile Z from {f1(min(zz))} to {f1(max(zz))} um (start {f1(self.Z0)})")
                bad = tilt > s.max_tilt or (len(fpts) > 3 and focus_rms > s.max_focus_rms)
                clipped = min(zz) < self.Z0 - s.max_dz_um or max(zz) > self.Z0 + s.max_dz_um
                if bad or clipped:
                    msg = (f"Focus plane looks suspicious: tilt {f1(tilt)} um/mm" + (f", rms {f1(focus_rms)} um" if len(fpts) > 3 else "")
                           + ("; tile Z would exceed the max_dz_um limit (clipped)" if clipped else "") + ".")
                    self.say("  WARNING: " + msg)
                    if self.ui.confirm(f"{s.name} focus plane", msg + "\n\nUse it anyway? (No = one Z for all tiles)", ("Yes", "No")) != 0:
                        self.fa = self.fb = 0.0
                        self.fc = self.Z0
                        self.FS = None
                        fpts = []
                        self.say("  using one Z for all tiles")
                if s.focus_surface == "quadratic" and len(fpts) >= 6 and self.FS is None:
                    q = fit_quad(fpts)
                    if q is not None:
                        self.FQ = q
                        P = np.asarray(fpts)
                        rq = math.sqrt(np.mean([(f[2] - self.z_surf(f[0], f[1])) ** 2 for f in fpts]))
                        bow = max(abs(self.z_surf(gx, gy) - (self.fa * gx + self.fb * gy + self.fc))
                                  for gx in np.linspace(q[6], q[7], 11) for gy in np.linspace(q[8], q[9], 11))
                        if bow > s.max_bow_um:
                            self.FQ = None
                            self.say(f"  quadratic focus surface not used: it bends {f1(bow)} um away from the plane (max_bow_um)")
                        else:
                            self.say(f"  focus surface: quadratic through {len(fpts)} points, rms {f3(rq)} um (plane {f3(focus_rms)} um); "
                                     f"bow up to {f3(bow)} um from the plane")
                if self.FQ is not None:
                    zz = [self.z_at(t[0], t[1]) for t in tiles]
                    self.say(f"  tile Z from {f1(min(zz))} to {f1(max(zz))} um with the quadratic surface")
            else:
                if fpts:
                    self.fc = fpts[-1][2]
                self.fa = self.fb = 0.0
                self.say(f"  fewer than 3 focus points: one Z ({f1(self.fc)}) for all tiles")
        elif nfp >= 3:
            self.say("  no focus device: tiles get no Z")
        pc["n_focus10"] = len(fpts)                     # (the piece's 10x focus points: avram new1's manual 100x points / 3)
        for t in tiles:
            t.append(self.z_at(t[0], t[1]))

        # output: .pos, chip.json, plan
        out = Path(s.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pos = out / f"{s.name}_scan.pos"
        arr = []
        for t in tiles:
            dp = [{"Device": {"type": "STRING", "scalar": self.xy}, "Position_um": {"type": "DOUBLE", "array": [t[0], t[1]]}}]
            if self.zdev:
                dp.append({"Device": {"type": "STRING", "scalar": self.zdev}, "Position_um": {"type": "DOUBLE", "array": [t[4] * self.z_scale]}})
            arr.append({"DefaultXYStage": {"type": "STRING", "scalar": self.xy}, "DefaultZStage": {"type": "STRING", "scalar": self.zdev or ""},
                        "DevicePositions": {"type": "PROPERTY_MAP", "array": dp}, "GridCol": {"type": "INTEGER", "scalar": int(t[2])},
                        "GridRow": {"type": "INTEGER", "scalar": int(t[3])}, "Label": {"type": "STRING", "scalar": f"Pos-{pad3(t[2])}_{pad3(t[3])}"},
                        "Properties": {"type": "PROPERTY_MAP", "scalar": {"Source": {"type": "STRING", "scalar": "flakeid.scope"}}}})
        pos.write_text(json.dumps({"encoding": "UTF-8", "format": "Micro-Manager Property Map", "major_version": 2, "minor_version": 0,
                                   "map": {"StagePositions": {"type": "PROPERTY_MAP", "array": arr}}}, indent=1))
        cam = self.core.getCameraDevice()
        js = {"material": self.material, "oxide_nm": self.oxide, "z_scale": self.z_scale, "start": [self.X0, self.Y0, self.Z0],
              "reference_rg_bg_g": list(self.REF), "size_um": [sizeX, sizeY], "rotation_deg": rot}
        if self.polygon is not None:
            # (drawn by hand: no corners, edges or sides; a re-image of the piece then finds it by reference points, not
            # by its edges)
            js["scan_polygon"] = [list(q) for q in self.polygon]
        else:
            js.update({"corners": [list(q) for q in corners], "edge_points": [list(q) for q in pts],
                       "edges": [{"dir": names[e], "point": [L[e][0], L[e][1]], "normal": [L[e][2], L[e][3]]} for e in range(4)],
                       "sides": [dict({"type": ("line", "arc", "bounding")[self.side_type[e]], "point": [L[e][0], L[e][1]],
                                       "normal": [L[e][2], L[e][3]]}, **({"arc": list(self.arcs[e])} if self.side_type[e] == 1 else {}))
                                 for e in range(4)]})
        js.update({"focus_points": [list(f) for f in fpts], "focus_plane": [self.fa, self.fb, self.fc], "focus_rms_um": focus_rms})
        if self.FQ is not None:
            Q = self.FQ
            js["focus_surface"] = {"type": "quadratic", "coef_um_mm": list(Q[:6]), "centre": [Q[10], Q[11]], "box": list(Q[6:10])}
        elif self.FS is not None:                       # (rebuilt from its points: smooth_surface(points, smoothing))
            js["focus_surface"] = {"type": "smooth", "kernel": "thin_plate_spline", "smoothing": s.polygon_focus_smoothing,
                                   "points": [list(f) for f in fpts]}
        js["camera"] = {"exposure_ms": self.core.getExposure()}
        for p, k in (("Data format", "data_format"), ("Device output data bit depth", "bit_depth")):
            if self.core.hasProperty(cam, p):
                js["camera"][k] = self.core.getProperty(cam, p)
        js.update({"M": self.M.tolist(), "field_um": [self.FOVX, self.FOVY], "overlap": s.overlap, "margin_um": s.margin_um, "n_tiles": len(tiles)})
        (out / f"{s.name}_chip.json").write_text(json.dumps(js, indent=1))
        self.say(f"  wrote {pos} and {s.name}_chip.json")
        try:
            self.draw_plan(tiles, corners, pts, fpts, sizeX, sizeY, rot, cx, cy, minx, maxx, miny, maxy)
        except Exception as e:
            self.say(f"  (plan image not drawn: {e})")
        self.move_to(self.X0, self.Y0)
        if self.has_z:
            self.z_write(self.z_at(self.X0, self.Y0)); self.core.waitForDevice(self.zdev)

        if s.run_scan:
            scan_dir = out / f"{s.name}_tiles"
            n = 2
            while scan_dir.exists():
                scan_dir = out / f"{s.name}_tiles_{n}"; n += 1
            scan_dir.mkdir(parents=True)
            sp = {"piece": s.name, "material": self.material, "oxide_nm": self.oxide, "exposure_ms": self.core.getExposure(),
                  "script_version": VERSION}
            if self.manifest:
                sp["run_file"] = Path(self.manifest).name
            (scan_dir / "scan_params.json").write_text(json.dumps(sp))
            self.tile_objective = self.objective()[1]
            self.say(f"  scanning {len(tiles)} tiles into {scan_dir} ({self.material} on {f1(self.oxide)} nm)"
                     + (f", objective {self.tile_objective}" if self.tile_objective else "") + " ...")
            t0 = time.time()
            self.set_phase("10x tiles")
            self._props = None
            ref = self.Z0 if isnan(self.zref) else self.zref
            nrb = max(1, int(s.tile_readback_every or 1))
            for i, t in enumerate(tiles):
                self.ui.check_stop(self)
                self.pause_point()
                zpre = t[4] - s.af_backlash_um
                # the position read back on every nrb-th tile, the first and the last (the others as sent: a stage that
                # stops arriving is still caught within nrb tiles; positions agree with the readback to ~0.1 um)
                chk = i % nrb == 0 or i == len(tiles) - 1
                if self.has_z and s.overlap_z_move and abs(zpre - ref) <= s.max_dz_um:
                    self.move_to(t[0], t[1], z_pre=zpre, check=chk)        # backlash approach during the XY move
                    self.set_z(t[4])
                else:
                    self.move_to(t[0], t[1], check=chk)
                    if self.has_z:
                        self.set_z(zpre); self.set_z(t[4])
                lbl = f"Pos-{pad3(t[2])}_{pad3(t[3])}"
                self.save_tile(scan_dir / lbl, i, lbl, t[2], t[3], background=s.background_save, cached=True if chk else "sent")
                if (i + 1) % 25 == 0 or i == len(tiles) - 1:
                    el = time.time() - t0
                    self.say(f"    {i + 1} / {len(tiles)} tiles, {f1(el)} s, about {f1(el / (i + 1) * (len(tiles) - i - 1))} s left")
                    self.ui.progress(self, f"{s.name}: tile {i + 1} / {len(tiles)}")
            if self.writer is not None:
                t_w = time.time()
                self.writer.flush()                     # (raises if a write failed)
                self.op("waiting for the tile writer", t_w)
            self.move_to(self.X0, self.Y0)
            if self.has_z:
                self.set_z(self.z_at(self.X0, self.Y0))
            self.say(f"  scan finished: {scan_dir}  ({f1((time.time() - t0) / max(len(tiles), 1))} s per tile)")
        else:
            self.say("  run_scan = false: check the plan, then run again with run_scan = true")

    def draw_plan(self, tiles, corners, pts, fpts, sizeX, sizeY, rot, cx, cy, minx, maxx, miny, maxy):
        import cv2
        s = self.s
        FX, FY = self.FOVX, self.FOVY
        S = 700.0 / max(maxx - minx + FX, maxy - miny + FY)
        pw, ph = int((maxx - minx + FX) * S) + 40, int((maxy - miny + FY) * S) + 60
        ox, oy = minx - FX / 2, miny - FY / 2 - 20.0 / S
        im = np.full((ph, pw, 3), 255, np.uint8)
        P = lambda x, y: (int((x - ox) * S) + 20, int((y - oy) * S) + 20)                      # noqa: E731
        for t in tiles:
            cv2.rectangle(im, P(t[0] - FX / 2, t[1] - FY / 2), P(t[0] + FX / 2, t[1] + FY / 2), (150, 150, 150), 1)
        if self.polygon is not None:                    # (a scan area drawn by hand: its polygon and its points)
            cv2.polylines(im, [np.array([P(q[0], q[1]) for q in self.polygon], np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
            for k, q in enumerate(self.polygon):
                cv2.circle(im, P(q[0], q[1]), 4, (0, 0, 255), -1)
                cv2.putText(im, str(k + 1), (P(q[0], q[1])[0] + 6, P(q[0], q[1])[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        for i in range(4 if self.polygon is None else 0):
            a, b = corners[i], corners[(i + 1) % 4]
            prev = None
            for k in range(61):
                qx, qy = a[0] + (b[0] - a[0]) * k / 60, a[1] + (b[1] - a[1]) * k / 60
                for _ in range(60):
                    if self.inside_chip(qx, qy, 1.0):
                        break
                    qx += (cx - qx) * 0.01; qy += (cy - qy) * 0.01
                q = P(qx, qy)
                if prev:
                    cv2.line(im, prev, q, (0, 0, 255), 2, cv2.LINE_AA)
                prev = q
        for p in pts:
            cv2.circle(im, P(*p), 4, (255, 0, 0), -1)
        for f in fpts:
            q = P(f[0], f[1])
            cv2.rectangle(im, (q[0] - 6, q[1] - 6), (q[0] + 6, q[1] + 6), (0, 120, 230), 1)
            cv2.putText(im, f"Z {f1(f[2])}", (q[0] + 9, q[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 230), 1, cv2.LINE_AA)
        cv2.circle(im, P(self.X0, self.Y0), 6, (0, 160, 0), -1)
        what = (f"area drawn by hand {polygon_area_um2(self.polygon) / 1e6:.2f} mm2" if self.polygon is not None
                else f"chip {f3(sizeX / 1000)} x {f3(sizeY / 1000)} mm, {f1(rot)} deg")
        cv2.putText(im, f"{s.name}: {len(tiles)} tiles, {what}   (stage X ->, Y down; green = start)",
                    (20, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        path = Path(s.out_dir) / f"{s.name}_plan.png"
        cv2.imwrite(str(path), im)
        try:                                            # (Slack bot: the plan in the run's thread)
            from .notify import send
            send(s, "progress", f"{s.name}: planned", f"{len(tiles)} tiles, {what}"
                 + (f"; beyond the stage's travel: {', '.join(self.clipped)}" if getattr(self, "clipped", None) else ""), image=str(path), log=self.say,
                 run=getattr(self, "slack_run", None))
        except Exception:
            pass
        if s.show_plan:                              # (written with the results either way; shown only on request)
            self.ui.show_image(self, f"{s.name} scan plan", str(path))

    # ---- 100x
    def targets_jobs(self):
        s = self.s
        self.set_phase("objective change / prompts")
        jobs = []
        self.exp_checked = s.mode != "multi"
        if s.mode == "targets":
            if not s.targets_csv:
                raise self.fail("targets mode needs targets_csv (there is no Micro-Manager position list without the GUI)")
            jobs.append(dict(csv=s.targets_csv, chip=s.chip_json, name=s.name, dir=s.out_dir, piece=-1))
        if s.mode == "multi" and not self.stop_all:
            scanned = [i for i, p in enumerate(self.pieces) if p["state"] == "scanned"]
            ready = sum(1 for i in scanned if (Path(self.run_dir) / f"{self.pieces[i]['name']}_targets.csv").exists())
            for i in scanned:
                nm = self.pieces[i]["name"]
                jobs.append(dict(csv=str(Path(self.run_dir) / f"{nm}_targets.csv"), chip=str(Path(self.run_dir) / f"{nm}_chip.json"),
                                 name=nm, dir=self.run_dir, piece=i))
            self.say("")
            self.say(f"==== scans finished: {len(scanned)} of {len(self.pieces)} pieces scanned; target lists ready for {ready}")
            if scanned and s.objective_prompt and (s.objective_positions or {}).get("100x") and self.nose is not None:
                ok = self.wait_objective("100x", f"All pieces are scanned ({len(scanned)} of {len(self.pieces)}).  Target lists ready: "
                                         f"{ready} of {len(scanned)}.\n\nSwitch to the 100x objective now, without changing the focus.  "
                                         "The 100x images start as soon as it is in (the scan waits for target lists that are not ready yet).")
                if ok is False:
                    self.say("  100x part cancelled")
                    jobs = []
                else:
                    self.view_scale = 1.0 / 9.54
            elif scanned and s.objective_prompt:
                self.alert(f"All pieces are scanned ({len(scanned)} of {len(self.pieces)}): switch to the 100x objective (without changing "
                           "the focus) and press OK in the scan program.")
                a = self.ui.confirm("Multi-piece scan: change the objective",
                                    f"All pieces are scanned ({len(scanned)} of {len(self.pieces)}).  Target lists ready: {ready} of {len(scanned)}.\n\n"
                                    "Switch to the 100x objective now, without changing the focus.\nPress OK when the 100x objective is in place: "
                                    "the 100x images start then\n(the scan waits for target lists that are not ready yet).", ("OK", "Cancel"))
                if a != 0:
                    self.say("  100x part cancelled")
                    jobs = []
                else:
                    self.ui.mock_hook(self, "mockObjective", 100)
                    self.view_scale = 1.0 / 9.54            # live view stage jog: 100x pixels
            if jobs:
                self.write_manifest(self.manifest, self.pieces, "imaging")
                self.stream_off()
                self.core.setExposure(s.target_exposure_ms)
        return jobs

    def wait_for_file(self, path, timeout_s, what):
        path = Path(path)
        t0, last = time.time(), 0.0
        h = None
        if not path.exists():
            h = self.ui.control(self, "Waiting for the analysis", f"Waiting for {what}:\n{path}\n\n(the analysis on this PC writes it when the piece is analysed)",
                                ("Skip this piece", "Stop"))
        try:
            while True:
                self.pause_point()
                if path.exists():
                    n = path.stat().st_size
                    time.sleep(2)
                    if path.exists() and path.stat().st_size == n:
                        return True
                    continue
                v = h.value() if h else None
                if v == "Skip this piece":
                    return False
                if v == "Stop":
                    self.stop_all = True
                    return False
                el = time.time() - t0
                if el > timeout_s:
                    self.say(f"    gave up waiting for {path} after {f1(el)} s")
                    return False
                if self.analysis is not None and self.analysis.poll() is not None:
                    self.say(f"    the analysis on this PC has ended (exit {self.analysis.returncode}) without writing {path.name}; "
                             f"see {self.analysis_out / 'analysis_log.txt'}")
                    self.analysis = None
                if time.time() - last > 60:
                    self.say(f"    waiting for {what} ({f1(el)} s)")
                    last = time.time()
                time.sleep(1)
        finally:
            if h:
                h.close()

    def read_targets(self, path, fi):
        """[(x, y, z10, thickness_nm, label)] from a target list (flakeid.live: <piece>_targets[_partial].csv)."""
        out = []
        with open(path, newline="") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                if "stage_x_um" not in r or "stage_y_um" not in r:
                    raise self.fail(f"{path} has no stage_x_um / stage_y_um columns")
                x, y = float(r["stage_x_um"]), float(r["stage_y_um"])
                z10 = float(r["z10_um"]) if (r.get("z10_um") or "").strip() else NAN
                if isnan(z10) and fi >= 0:
                    z10 = self.z_surf(x, y)
                try:
                    th = float(r["thickness_nm"]) if (r.get("thickness_nm") or "").strip() else NAN
                except ValueError:
                    th = NAN
                t = Target((x, y, z10, th, r.get("label") or f"T{pad3(i + 1)}"))
                pg = parse_poly(r.get("terrace_poly_um"))         # (lists before 2026-10-19: none, single picture)
                if pg is not None:
                    t.poly = pg - np.array([x, y])
                out.append(t)
        return out

    def image_targets(self, jobs):
        """100x images of each piece's targets.  Multi mode: while the analysis is still finishing a piece, the flakes
        of its provisional list (<piece>_targets_partial.csv: flakes it is already sure about, rewritten as blocks
        complete) are imaged; then the rest of the final list (<piece>_targets.csv), skipping flakes already imaged."""
        s, c = self.s, self.core
        self.stop_100x = False
        self.last_results = []
        for ji, job in enumerate(jobs):
            if self.stop_all or self.stop_100x:
                break
            if not self.has_z:
                raise self.fail("no focus device")
            s.name = job["name"]
            out_dir, chip_json, fi = job["dir"], job["chip"], job["piece"]
            if s.mode == "multi":
                self.say("")
                self.say(f"==== 100x, piece {s.name} ({ji + 1} of {len(jobs)})")
                self.X0, self.Y0 = c.getXPosition(self.xy), c.getYPosition(self.xy)
                (self.fa, self.fb, self.fc), self.FQ = self.piece_focus[fi]
                self.FS = getattr(self, "piece_surface", {}).get(fi)
                self.material, self.oxide = self.pieces[fi]["material"], self.pieces[fi]["oxide"]
            else:
                self.material = s.material
            img_dir = Path(out_dir) / f"{s.name}_100x"
            img_dir.mkdir(parents=True, exist_ok=True)
            self.sub100 = []                         # (each piece its own substrate: oxide and material differ)
            # the start offset: the latest measured (the median of the last three, any piece), else the setting
            off0 = float(np.median(self.recent_offs[-3:])) if self.recent_offs else s.focus_offset_um
            seeds = [tuple(q) for q in job.get("seed_focus") or []]         # 100x focus points known already (re-image)
            P = dict(k=0, n_ok=0, n_img=0, offs=[], focused=list(seeds), prefix=job.get("file_prefix", ""), seg_offs=[], seg_focused=list(seeds), off0=off0, jumps=[],
                     piece_i=fi,
                     results=[], fail_run=0, last_rej=NAN, stop=False, imaged=[],
                     idx=open(img_dir / f"{s.name}_100x_index.csv", "w", newline=""), fs=self.full_scale(), cam=c.getCameraDevice(),
                     chip_tilt=tuple(job["chip_tilt"]) if job.get("chip_tilt") else None, out_dir=out_dir, img_dir=img_dir, box=None)
            P["idx"].write(IDX100_HEADER)
            ctl = self.ui.control(self, f"100x imaging: {s.name}", f"100x imaging of {s.name}.\n\n"
                                  "\"Skip the rest of this piece\" finishes the current flake and goes on with the next piece;\n"
                                  "\"Stop 100x imaging\" ends the 100x part for all pieces;\n"
                                  f"\"{MANUAL_RETAKE}\" (during the retakes of the flakes not focused, at the end of the piece): "
                                  "the current one, if the autofocus does not find it, or the next one is focused by hand.",
                                  ("Skip the rest of this piece", "Stop 100x imaging"), once=(MANUAL_RETAKE,))
            P["ctl"] = ctl
            try:
                if s.mode != "multi":
                    batch = self.read_targets(job["csv"], fi)
                    self.say(f"  {len(batch)} targets from {job['csv']}")
                    if not batch:
                        raise self.fail("no targets")
                    self._setup_piece_focus(P, chip_json, batch)
                    self.image_batch(P, batch, "")
                    if s.pattern_frames > 0 and not (P["stop"] or self.stop_100x):
                        self.capture_pattern(P, batch[0])
                else:
                    self._image_piece_multi(P, job, fi, chip_json)
                self.retake_failed(P)
            except Exception as e:
                if s.mode != "multi" or self.stop_all:
                    raise
                import traceback
                self.say(f"  ERROR during the 100x imaging of {s.name}: {type(e).__name__}: {e}; going on with the next piece")
                self.say(traceback.format_exc().rstrip())
            finally:
                P["idx"].close()
                if P.get("mos_idx"):
                    P["mos_idx"].close()
                if ctl:
                    ctl.close()
            offs, focused = P["offs"], P["focused"]
            if offs:
                ov = np.sort([o[2] for o in offs])
                med = float(np.median(ov))
                if not self.focus_jumps:
                    self.say(f"  focus offset 100x - 10x: median {f3(med)} um ({len(offs)} flakes, range {f3(ov[0])} .. {f3(ov[-1])}).  "
                             f"Next time: focus_offset_um = {f3(med)}")
                else:
                    self.say(f"  focus offset 100x - 10x: median {f3(med)} um ({len(offs)} flakes, range {f3(ov[0])} .. {f3(ov[-1])}); "
                             f"latest {f3(float(np.median(self.recent_offs[-3:])))} um (the next piece starts from it)")
                s.focus_offset_um = med
                po = fit_plane(offs)
                if po is not None and len(offs) >= 4 and not P["jumps"]:
                    self.say(f"  offset varies by {f3(po[0] * 1000)} um/mm in X, {f3(po[1] * 1000)} um/mm in Y across the piece (10x map tilt error)")
            if P["jumps"]:
                self.say(f"  FOCUS JUMPS on {s.name}: {len(P['jumps'])} (" + ", ".join(f"{j['time']} at {j['label']}: {f3(j['jump'])} um" for j in P["jumps"])
                         + f"), {f3(sum(j['jump'] for j in P['jumps']))} um in all")
            self.write_retake_list(job, P)
            pz = fit_plane(focused)
            if pz is not None and len(focused) >= 4:
                self.say(f"  focus plane through the focused flakes: dZ/dX {f3(pz[0] * 1000)}, dZ/dY {f3(pz[1] * 1000)} um/mm")
            self.say(f"  {P['n_img']} images ({P['n_ok']} focused) in {img_dir}")
            self.move_box = None
            if s.mode == "multi" and fi >= 0 and self.manifest:
                # the analysis puts this piece's 100x images into its viewer now, while the next piece is imaged
                self.pieces[fi]["state"] = "imaged"
                self.write_manifest(self.manifest, self.pieces, "imaging")
            if P["stop"]:
                self.stop_100x = True
            self.last_results.extend(P["results"])
        if self.focus_jumps:
            self.say("")
            self.say(f"  FOCUS JUMPS during the 100x imaging: {len(self.focus_jumps)}, {f3(sum(j['jump'] for j in self.focus_jumps))} um in all "
                     f"({', '.join(j['piece'] + ' ' + j['time'] for j in self.focus_jumps)}).  Each was a clear focus found twice, and the "
                     "flakes after it were focused from there.  The focus should not move by itself: check the focus drive (its coupling "
                     "on the focus knob slipping?) and the coarse-focus tension.")

    def retake_failed(self, P):
        """retake_failed: the flakes of this piece that were not focused (imaged at the predicted Z), once more at the end
        of the piece, when a flake was focused after them (the prediction knows more: after a focus jump, from the flakes
        focused since; 20261014_run_1: every flake after a slip of the focus drive).  Not after "skip" / "stop"."""
        s, ctl = self.s, P.get("ctl")
        failed = P.get("failed") or []
        if not s.retake_failed or not failed or P["stop"] or self.stop_100x or self.stop_all or s.focus_mode != "auto":
            return
        if ctl and ctl.value() in ("Skip the rest of this piece", "Stop 100x imaging"):
            return
        done = [f for f in failed if any(h in ("auto", "manual") for h in hows_at(f[0][0], f[0][1], f[0][4], P["results"]))]
        todo = [f for f in failed if f not in done and P["n_ok"] > f[2]]
        if len(todo) < len(failed) - len(done):
            self.say(f"  {len(failed) - len(done) - len(todo)} flake(s) not focused and no flake focused after them: not retaken")
        if not todo:
            return
        self.say(f"  retaking the {len(todo)} flake(s) of {s.name} not focused (the focus is known better now; a wider search; "
                 f"\"{MANUAL_RETAKE}\" in the 100x window to focus one by hand)")
        P["fail_run"], P["long_tried"], P["last_rej"] = 0, False, NAN
        if ctl is not None and hasattr(ctl, "enable"):
            ctl.take(MANUAL_RETAKE)                       # (a press before the retakes does not count)
            ctl.enable(MANUAL_RETAKE, True)
        try:
            self._retakes(P, ctl, todo)
        finally:
            if ctl is not None and hasattr(ctl, "enable"):
                ctl.enable(MANUAL_RETAKE, False)
        n = sum(1 for t, _, _ in todo if any(h in ("auto", "manual") for h in hows_at(t[0], t[1], t[4], P["results"])))
        self.say(f"  retakes: {n} of {len(todo)} focused")

    def _retakes(self, P, ctl, todo):
        for t, k, _ in todo:
            v = ctl.value() if ctl else None
            if P["stop"] or self.stop_100x or v in ("Skip the rest of this piece", "Stop 100x imaging"):
                break
            try:
                self.image_flake(P, t, " (retake)", retake_k=k)
            except Exception as e:
                if self.stop_all:
                    raise
                self.say(f"    ERROR on the retake of {t[4]}: {type(e).__name__}: {e}")

    def write_retake_list(self, job, P):
        """<piece>_retake.csv beside its target list: the targets without a focused 100x image (focus not found, skipped
        or not imaged), with the target list's columns + retake_reason.  A re-image run with "only the failed targets"
        images these (re-image runs keep the list up to date themselves).  Targets are matched by position: the labels
        of the provisional lists are not those of the final one."""
        if job.get("retake") or not job.get("csv") or not Path(job["csv"]).is_file():
            return
        try:
            with open(job["csv"], newline="") as fh:
                rd = csv.DictReader(fh)
                rows, cols = list(rd), list(rd.fieldnames or [])
            out = retake_rows(rows, P["results"])
            path = Path(job["dir"]) / f"{job['name']}_retake.csv"
            write_rows(path, cols + ["retake_reason"], out)
            self.say(f"  retake list: {len(out)} of {len(rows)} targets without a focused image -> {path}" if out else
                     f"  retake list: every target has a focused image ({path} empty)")
        except Exception as e:
            self.say(f"  (retake list not written: {type(e).__name__}: {e})")

    def _setup_piece_focus(self, P, chip_json, batch):
        s = self.s
        if P["chip_tilt"] is None and chip_json and Path(chip_json).exists():
            try:
                fp = json.loads(Path(chip_json).read_text()).get("focus_plane")
                if fp:
                    P["chip_tilt"] = (fp[0], fp[1])
                    self.say(f"  chip tilt from {chip_json}: dZ/dX {f3(fp[0] * 1000)}, dZ/dY {f3(fp[1] * 1000)} um/mm")
            except Exception as e:
                self.say(f"  (could not read {chip_json}: {e})")
        T = np.array([t[:3] for t in batch], float)
        b = P["box"] or (self.X0, self.X0, self.Y0, self.Y0)
        P["box"] = (min(b[0], T[:, 0].min()), max(b[1], T[:, 0].max()), min(b[2], T[:, 1].min()), max(b[3], T[:, 1].max()))
        mg = s.target_box_margin_um
        self.move_box = (P["box"][0] - mg, P["box"][1] + mg, P["box"][2] - mg, P["box"][3] + mg)
        n10 = int(np.isfinite(T[:, 2]).sum())
        self.say(f"  moves allowed within X {f1(self.move_box[0])} .. {f1(self.move_box[1])}, Y {f1(self.move_box[2])} .. {f1(self.move_box[3])}")
        self.say(f"  10x focus known for {n10} of {len(batch)} targets" + (
            f"; focus offset 100x - 10x: " + ("to be measured at the first focused flake" if isnan(s.focus_offset_um) else f"{f3(s.focus_offset_um)} um")
            if n10 else ""))

    def _image_piece_multi(self, P, job, fi, chip_json):
        """Provisional list while the analysis finishes the piece, then the rest of the final list."""
        s = self.s
        final, partial = Path(job["csv"]), Path(str(job["csv"]).replace("_targets.csv", "_targets_partial.csv"))
        t0, last_say, part_mtime = time.time(), 0.0, None
        ctl = P["ctl"]

        def new_only(batch):
            return [t for t in batch if not any(math.hypot(t[0] - q[0], t[1] - q[1]) < 15.0 for q in P["imaged"])]
        while True:
            self.pause_point()
            v = ctl.value() if ctl else None
            if v == "Skip the rest of this piece":
                self.say(f"  rest of {s.name} skipped")
                return
            if v == "Stop 100x imaging":
                self.say("  100x imaging stopped")
                self.stop_100x = True
                return
            if final.exists():
                n = final.stat().st_size
                time.sleep(1.0)
                if final.exists() and final.stat().st_size == n:
                    allt = self.read_targets(final, fi)
                    batch = new_only(allt)
                    cap_set = ("hbn_max_targets" if self.material == "hbn" else
                               "graphite_max_targets" if self.material == "graphite" else "")
                    cap = getattr(s, cap_set, 0) if cap_set else 0
                    if cap_set == "hbn_max_targets" and P.get("piece_i") is not None and self.pieces:
                        cap = self.pieces[P["piece_i"]].get("hbn_max_targets", cap)      # (the piece's own: its area)
                    if cap and P["imaged"] and getattr(s, "target_mode", "") != "avram_new1":
                        # (hBN cut by size, graphite by the kept-at-100x model: flakes imaged from a provisional list
                        # but not in the final one count too, so the piece gets no more pictures than the final list
                        # has; its last in imaging order left out)
                        room = max(0, len(allt) - len(P["imaged"]))
                        if len(batch) > room:
                            self.say(f"  {len(batch) - room} of the final list's last flakes left out: "
                                     f"{len(P['imaged'])} already imaged, {len(allt)} in all ({cap_set})")
                            batch = batch[:room]
                    self.say(f"  {len(allt)} targets in the final list {final.name}" + (f"; {len(allt) - len(batch)} already imaged from the provisional list"
                                                                                       if len(batch) < len(allt) else ""))
                    if not allt and not P["imaged"]:
                        self.say(f"  no flakes selected on {s.name}: nothing to image")
                    if batch:
                        self._setup_piece_focus(P, chip_json, batch)
                        self.image_batch(P, batch, "")
                    return
                continue
            if partial.exists() and partial.stat().st_mtime != part_mtime:
                part_mtime = partial.stat().st_mtime
                time.sleep(0.5)                                  # (written via a .tmp + rename: complete when it appears)
                try:
                    batch = new_only(self.read_targets(partial, fi))
                except Exception as e:
                    self.say(f"  (provisional list not readable yet: {e})")
                    batch = []
                if batch:
                    self.say(f"  provisional list {partial.name}: {len(batch)} new flakes to image while the analysis finishes {s.name}")
                    self._setup_piece_focus(P, chip_json, batch)
                    if not self.image_batch(P, batch, " (provisional)"):
                        return
                continue
            self.set_phase("100x: waiting for target lists")
            el = time.time() - t0
            if el > s.targets_wait_s:
                self.say(f"    gave up waiting for {final} after {f1(el)} s" + (f" ({len(P['imaged'])} flakes imaged from the provisional list)"
                                                                               if P["imaged"] else ""))
                return
            if self.analysis is not None and self.analysis.poll() is not None:
                self.say(f"    the analysis on this PC has ended (exit {self.analysis.returncode}) without writing {final.name}; "
                         f"see {self.analysis_out / 'analysis_log.txt'}")
                self.analysis = None
            if time.time() - last_say > 60:
                self.say(f"    waiting for the target list of {s.name} ({f1(el)} s)")
                self.ui.progress(self, f"{s.name}: waiting for the target list ({int(el)} s)")
                last_say = time.time()
            time.sleep(1.0)

    def predicted_offset(self, P, x, y):
        """The 100x - 10x focus offset at (x, y): the manual 100x points' calibration (P["off_model"], offset_model) where
        there is one, corrected by the flakes focused since (a distance-weighted mean of their differences from it);
        else the distance-weighted mean of the flakes focused so far, or the start offset."""
        m = P.get("off_model")
        if m is not None:
            res = [(px, py, o - m(px, py)) for px, py, o in P["seg_offs"]]
            return m(x, y) + (idw_offset(x, y, res) if res else 0.0)
        return idw_offset(x, y, P["seg_offs"]) if P["seg_offs"] else P["off0"]

    def manual_focus_points(self, P, batch):
        """avram new1: at the start of a piece's 100x imaging, int(its 10x focus points / hundred_manual_focus_divisor) of its targets (`batch`, the
        first list it has), spread over the piece (farthest-point choice, visited in a short path), focused by hand.
        Each one's 100x focus is then known to the prediction of every flake, as a flake already focused (its 100x -
        10x offset).  The flakes themselves are imaged later as usual.  Skip: that point; Stop, or no answer within
        focus_dialog_timeout_s: no more points."""
        s = self.s
        P["manual_done"] = True
        div = int(getattr(s, "hundred_manual_focus_divisor", 0) or 0)
        n10 = int(self.pieces[P["piece_i"]].get("n_focus10", 0)) if P.get("piece_i") is not None and self.pieces else 0
        n = min(n10 // div if div > 0 else 0, len(batch))
        if n <= 0 or not self.ui.can_ask:
            if div > 0 and self.ui.can_ask:
                self.say(f"  no manual 100x focus points ({n10} 10x focus points / {div})")
            return
        self.set_phase("100x manual focus points")
        T = np.array([[t[0], t[1]] for t in batch], float)
        pick = [int(np.argmax(np.hypot(*(T - T.mean(0)).T)))]
        dist = np.hypot(*(T - T[pick[0]]).T)
        while len(pick) < n:
            k = int(np.argmax(dist))
            pick.append(k)
            dist = np.minimum(dist, np.hypot(*(T - T[k]).T))
        c = self.core
        here = np.array([c.getXPosition(self.xy), c.getYPosition(self.xy)])
        left, order = list(pick), []
        while left:                                     # (a short path: the nearest point next)
            k = min(left, key=lambda i: float(np.hypot(*(T[i] - (here if not order else T[order[-1]])))))
            left.remove(k)
            order.append(k)
        self.say(f"  {len(order)} manual 100x focus points ({n10} 10x focus points / {div}), spread over {s.name}")
        n_ok, manual = 0, []
        for j, i in enumerate(order):
            x, y, z10, th, lab = batch[i]
            try:
                self.approach_to(x, y)
            except StageLimit as e:
                self.say(f"    focus point {j + 1}: beyond the stage's travel ({e}): skipped")
                continue
            off_known = self.predicted_offset(P, x, y)
            zg = z10 + off_known if not isnan(z10) and not isnan(off_known) else self.predict_z(x, y, P["seg_focused"], P["chip_tilt"], self.z_read())
            self.set_z(zg - s.af_backlash_um); self.set_z(zg)
            a = self.ask_focus(f"{s.name}: 100x focus point {j + 1} of {len(order)}",
                               f"Focus by hand on this flake ({lab}), then press \"Use this focus\".  The {len(order)} points are spread "
                               "over the piece; every flake's 100x focus is predicted from them.  Skip: this point; Stop: no more "
                               "points (the 100x imaging goes on with autofocus).",
                               timeout=s.focus_dialog_timeout_s, image=self.target_picture(P, lab), button="Use this focus")
            if a == 2:
                self.say(f"    manual focus points stopped after {n_ok}")
                break
            if a == 3:
                self.say(f"    focus point {j + 1}: no answer within {f1(s.focus_dialog_timeout_s)} s: no more manual points")
                break
            if a != 0:
                self.say(f"    focus point {j + 1} ({lab}): skipped")
                continue
            zz = self.z_read()
            P["focused"].append((x, y, zz))
            P["seg_focused"].append((x, y, zz))
            n_ok += 1
            msg = f"    manual focus point {j + 1}: {lab} at {f1(x)}, {f1(y)} um, Z {f3(zz)}"
            if not isnan(z10):
                off = zz - z10
                P["offs"].append((x, y, off))
                P["seg_offs"].append((x, y, off))
                self.recent_offs.append(off)
                manual.append((x, y, off))
                msg += f", offset (100x - 10x) {f3(off)} um"
            self.say(msg)
        # the 10x -> 100x calibration: every flake's 100x start = the 10x map there (its z10, from the tiles of the 10x
        # focus surface) + this offset, then its autofocus
        m, how = offset_model(manual)
        if m is not None:
            P["off_model"] = m
            self.say(f"  10x -> 100x focus calibration from the manual points: {how}; every flake starts at the 10x map + it, "
                     "then autofocus")
        self.set_phase("100x imaging")

    def image_batch(self, P, batch, tag):
        """Image a list of targets; False if the 100x part was stopped / the piece skipped."""
        if (self.s.mode == "multi" and getattr(self.s, "target_mode", "") == "avram_new1" and not P.get("manual_done")
                and batch):                             # (avram new1 only)
            self.manual_focus_points(P, batch)
        ctl = P["ctl"]
        # how many in all, for the window's top line ("100x flake 43 of 120"): the flakes so far + this list; a
        # provisional list's total can still grow with the final list ("so far")
        P["total"], P["total_more"] = P["k"] + len(batch), "provisional" in tag
        for t in batch:
            if P["stop"] or self.stop_100x:
                return False
            v = ctl.value() if ctl else None
            if v == "Skip the rest of this piece":
                self.say(f"  rest of {self.s.name} skipped")
                return False
            if v == "Stop 100x imaging":
                self.say("  100x imaging stopped")
                self.stop_100x = True
                return False
            self.pause_point()
            P["k"] += 1
            try:
                ok = self.image_flake(P, t, tag)
                P["errors_run"] = 0
            except Exception as e:
                # (a stage or camera error on one flake must not end the 100x imaging of everything)
                if self.stop_all:
                    raise
                P["errors_run"] = P.get("errors_run", 0) + 1
                self.say(f"    ERROR on {t[4]}: {type(e).__name__}: {e}; flake skipped")
                if P["errors_run"] >= 3:
                    self.say(f"  3 flakes in a row failed: rest of {self.s.name} skipped")
                    return False
                continue
            if not ok:
                return False
        return True

    def run_capture(self):
        """Capture mode: raw 100x images of flakes picked in the live view, where the stage is (no target list), for
        developing the 100x processing: the run's frames for each flake (mean, every single frame, shifted frames) and
        pattern_frames (default 24) around it for the optics pattern.  Nothing is deleted.  -> <local_root>/<name>/"""
        s, c = self.s, self.core
        name = s.name if s.name and s.name != "chip" else "capture_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
        out = Path(s.local_root) / safe_name(name)
        img_dir = out / f"{safe_name(name)}_100x"
        img_dir.mkdir(parents=True, exist_ok=True)
        self.open_log(out / f"{safe_name(name)}_log.txt")
        s.name, s.hundred_save_frames = safe_name(name), True
        npat = s.pattern_frames if s.pattern_frames > 0 else 24
        self.say(f"capture: {img_dir} (single frames saved; {npat} optics-pattern frames per flake)")
        self.material = s.material
        P = dict(k=0, n_ok=0, n_img=0, offs=[], focused=[], fail_run=0, last_rej=NAN, stop=False, imaged=[],
                 idx=open(img_dir / f"{s.name}_100x_index.csv", "w", newline=""), fs=self.full_scale(), cam=c.getCameraDevice(),
                 chip_tilt=None, out_dir=str(out), img_dir=img_dir, box=None, ctl=None, seg_offs=[], seg_focused=[], off0=NAN,
                 results=[], jumps=[], prefix="")
        P["idx"].write(IDX100_HEADER)
        self.move_box = None
        try:
            while True:
                a = self.ask_focus("Capture: pick a flake", f"Flake {P['k'] + 1}: move the stage to the flake (arrow buttons or the "
                                   "stage control) and focus, then press \"Use this focus\".  \"Stop\" ends the capture.")
                if a == 2 or a is None:
                    break
                if a != 0:
                    continue
                x, y, z = c.getXPosition(self.xy), c.getYPosition(self.xy), self.z_read()
                self.X0, self.Y0, self.zref = x, y, z           # (moves for the shifts and pattern frames are around here)
                P["focused"] = P["seg_focused"] = [(x, y, z)]    # narrow autofocus around the user's focus
                label = f"flake_{P['k'] + 1:02d}"
                t = (x, y, NAN, NAN, label)
                P["k"] += 1
                self.set_phase("100x imaging")
                self.image_flake(P, t, "")
                if npat:
                    s.pattern_frames = npat
                    self.capture_pattern(P, t)
                    os.replace(str(img_dir / "pattern"), str(img_dir / f"pattern_{label}"))
                if self.ui.confirm("Capture", f"{label} captured ({P['n_img']} flake(s) so far) in\n{img_dir}\n\nCapture another flake?",
                                   ("Another flake", "Done")) != 0:
                    break
        finally:
            P["idx"].close()
            self.stream_off()
        self.say(f"capture done: {P['n_img']} flake(s) in {img_dir}")
        self.report_times()

    # ---- re-imaging the targets of a piece scanned before
    def run_retarget(self):
        """Find a piece scanned before (shifted on the stage, or taken off and put back) from its flakes, convert its
        target list to the new position and image the targets at 100x, then add the images to its viewer."""
        from . import retarget as RT
        s, c = self.s, self.core
        root = Path(s.local_root)
        piece = s.retarget_piece
        if piece in EXPORTING:
            raise self.fail(f"{piece} is still being exported from its scan (the line under the status shows how far): "
                            "re-image it when that is done")
        tcsv, cjson = root / f"{piece}_targets.csv", root / f"{piece}_chip.json"
        if not piece or not tcsv.is_file():
            raise self.fail(f"no target list {tcsv}")
        manual = int(s.retarget_manual_points or 0) > 0       # the user centres a few flakes; else found automatically
        res_root = Path(s.results_dir) if s.results_dir else root / "results"
        fls = sorted(res_root.glob(f"**/{piece}/{piece}_tiles_live_flakes.csv"), key=lambda p: p.stat().st_mtime)
        if not fls and not manual:
            raise self.fail(f"no flake list of {piece} under {res_root} (the analysis results are needed to find the chip again)")
        fcsv = fls[-1] if fls else None
        chip = json.loads(cjson.read_text()) if cjson.is_file() else {}
        n = 1
        while (root / f"{piece}_retake{n}_targets.csv").exists():
            n += 1
        name = f"{piece}_retake{n}"
        self.open_log(root / f"{name}_log.txt")
        self.say(f"re-imaging the targets of {piece} ({'chip put back' if s.retarget_reloaded else 'chip still on the stage'}) as {name}")
        self.say(f"  targets {tcsv}" + (f", flakes {fcsv}" if fcsv else ""))
        # where the run's results were copied (the share): asked now, while you are here; updated at the end
        label = self.run_label_of(piece, fcsv)
        dest = self.results_dest_for(label) if label else None
        F = []
        if fcsv:
            with open(fcsv, newline="") as fh:
                F = [r for r in csv.DictReader(fh) if r.get("stage_x_um") not in (None, "") and float(r.get("area_um2") or 0) >= 150]
        with open(tcsv, newline="") as fh:
            rd = csv.DictReader(fh)
            T, tcols = list(rd), list(rd.fieldnames or [])
        flk = {r["flake_id"]: (float(r["stage_x_um"]), float(r["stage_y_um"])) for r in F}
        pts = np.array(list(flk.values()), float).reshape(-1, 2)
        # the 10x -> 100x objective offset in the target list (its positions are for the 100x objective)
        d = [(float(t["stage_x_um"]) - flk[t["flake_id"]][0], float(t["stage_y_um"]) - flk[t["flake_id"]][1]) for t in T if t.get("flake_id") in flk]
        off = np.median(np.array(d), axis=0) if d else np.array([24.0, -18.3])
        if not d or np.hypot(*off) > 300:
            off = np.array([24.0, -18.3])
        if not len(pts):                                  # (no flake list: the targets stand in for the flakes)
            pts = np.array([[float(t["stage_x_um"]), float(t["stage_y_um"])] for t in T], float) - off
        self.say(f"  {len(pts)} flakes of 150 um^2 or more to match; target offset {f1(off[0])}, {f1(off[1])} um")
        # first estimate: find the piece's edges again (as the scan does) and compare its corners with the first scan's
        corners_old = chip.get("corners")
        c_old = np.mean(np.array(corners_old, float), axis=0) if corners_old else np.median(pts, axis=0)
        if s.retarget_reloaded and self.ui.can_ask:
            a = self.ask_focus(f"Re-image {piece}: find the chip", "Put the chip back the same way round as before, move onto it at 10x "
                               "(bare substrate near the middle is best) and focus, then press \"Use this focus\".")
            if a != 0:
                raise self.fail("cancelled")
        elif not s.retarget_reloaded:
            self.move_to(*c_old)
            self.autofocus(self.z_read(), s.af10)
        self.X0, self.Y0 = c.getXPosition(self.xy), c.getYPosition(self.xy)
        self.Z0 = self.z_read() if self.has_z else 0.0
        self.REF = self.reference(self.blocks())
        Rm, tv = np.eye(2), np.zeros(2)
        radius, angles = 200.0, [0.0]
        outline = None
        if corners_old and s.retarget_find_edges and (s.retarget_reloaded or not manual):
            self.set_phase("finding the piece's edges")
            try:
                outline = self.find_outline()
            except Stop as e:
                if self.stop_all:
                    raise
                self.say(f"  (the edges were not found: {e}; going on from {'here' if s.retarget_reloaded else 'the old position'})")
        if outline:
            Co, Cn = np.array(corners_old, float), np.array(outline["corners"], float)
            pair = [int(np.argmin(np.hypot(*((Co - Co.mean(0)) - (q - Cn.mean(0))).T))) for q in Cn]     # pair them by position
            Co = Co[pair]
            # corners on straight sides only (a broken corner's bounding edge moves its corners): corner k = sides CS[k]
            CS = ((0, 2), (1, 2), (1, 3), (0, 3))
            sides_old = [str(d.get("type", "line")) for d in chip.get("sides") or []]
            good = [all(self.side_type[e] == 0 for e in CS[k]) and (len(sides_old) != 4 or all(sides_old[e] == "line" for e in CS[pair[k]]))
                    for k in range(4)]
            use = [k for k in range(4) if good[k]] if sum(good) >= 2 else list(range(4))
            Rm, tv, crms = RT.rigid_fit(Co[use], Cn[use])
            c_new = Rm @ c_old + tv
            ang0 = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
            self.say(f"  piece found from its edges ({len(use)} corners{'' if len(use) == 4 else ' on straight sides'}): centre was "
                     f"{f1(c_old[0])}, {f1(c_old[1])}, now {f1(c_new[0])}, {f1(c_new[1])} um (moved {f1(c_new[0] - c_old[0])}, "
                     f"{f1(c_new[1] - c_old[1])}), rotated {ang0:+.2f} deg" + (f"; corners agree to {f1(crms)} um rms" if len(use) > 2 else ""))
            if abs(ang0) > 15:
                Rm, tv = np.eye(2), c_new - c_old
                radius, angles = 1200.0, [float(a_) for a_ in np.arange(-8, 8.01, 1.0)]
            elif crms > 150:
                radius, angles = 800.0, [float(a_) for a_ in np.arange(-2, 2.01, 0.5)]
            else:
                radius, angles = min(max(300.0, 3.0 * crms), 800.0), [-0.5, 0.0, 0.5]
        elif s.retarget_reloaded:
            tv = np.array([self.X0, self.Y0]) - c_old
            radius, angles = 1200.0, [float(a_) for a_ in np.arange(-8, 8.01, 1.0)]
            self.say(f"  chip centre was {f1(c_old[0])}, {f1(c_old[1])}; assuming it is about here, {f1(self.X0)}, {f1(self.Y0)}")
        mos, zpts, src, rms = None, [], [], NAN
        if manual:
            self.say(f"  positions: from {int(s.retarget_manual_points)} flakes centred by hand at 100x" + (" (starting from the edges found)" if outline else ""))
        else:
            # the first scan's 10x mosaic (the viewer's full-resolution tiles): what each spot should look like
            mos = None
            for vd in [*sorted(fcsv.parent.glob("*_viewer")), fcsv.parent, fcsv.parent.parent]:
                if (vd / "tiles" / "image").is_dir():
                    try:
                        with open(fcsv, newline="") as fh:
                            mos = RT.Mosaic(vd, list(csv.DictReader(fh)))
                        self.say(f"  first scan's mosaic: {mos.dir} ({f3(mos.um_px)} um/px, mapping residual {f1(mos.residual_um)} um)")
                        break
                    except Exception as e:
                        self.say(f"  (mosaic in {vd} not usable: {e})")
            if mos is None:
                self.say("  WARNING: no viewer mosaic of the first scan found next to " + str(fcsv.parent) + ": positions from the edges only")
                if not outline:
                    raise self.fail("the chip cannot be found again: neither its edges nor the first scan's viewer mosaic")
            # 10x flat field (vignetting, dust in the light path), as the analysis used
            fl10 = None
            if s.flat10 and Path(s.flat10).is_file():
                try:
                    from ..mm import FlatField
                    from .ui import half_flat
                    fl10 = half_flat(FlatField.load(s.flat10))
                except Exception as e:
                    self.say(f"  (10x flat {s.flat10} not usable: {e})")
            # every spot matched on its own against the estimate from the edges; then the spots that agree
            fov = (self.FOVX, self.FOVY)
            spots = RT.choose_spots(pts, fov, n_outer=max(1, s.retarget_spots - 1)) if mos else []
            zpts, frames = [], []
            Mf, cf = 2.0 * np.asarray(self.M, float), np.array([self.W / 4.0, self.H / 4.0])     # half-resolution frames
            from .ui import bayer_half
            R0, t0 = Rm.copy(), tv.copy()
            for k, q in enumerate(spots):
                self.ui.check_stop(self)
                self.ui.progress(self, f"finding the chip: spot {k + 1} / {len(spots)}")
                guess = R0 @ q + t0
                self.move_to(*guess)
                zr = self.autofocus(self.z_read(), s.af10)
                if zr[2]:
                    zpts.append((guess[0], guess[1], zr[0]))
                img = self.frame()
                if img.ndim == 2:
                    rgb = bayer_half(img)
                    rgb = (rgb - fl10.dark) / fl10.flat if fl10 is not None and fl10.flat.shape == rgb.shape else rgb - self.DARK
                else:
                    rgb = img[..., :3].astype(np.float32)
                g = np.array([c.getXPosition(self.xy), c.getYPosition(self.xy)])
                F = RT.grey(rgb)
                res = RT.register(F, mos, g, Mf, cf, R0, t0, radius, angles)
                frames.append((F, g, res))
                if res is None:
                    self.say(f"    spot {k + 1} at {f1(g[0])}, {f1(g[1])}: outside the first scan's mosaic")
                    continue
                self.say(f"    spot {k + 1} at {f1(g[0])}, {f1(g[1])}: correlation {f3(res['peak'])} (next best {f3(res['second'])}), "
                         f"shift {f1(res['shift_um'][0])}, {f1(res['shift_um'][1])} um" + (f", rotation {res['ang']:+.1f} deg" if len(angles) > 1 else ""))
            # consensus: the pair of spots whose transform most others agree with (within 25 um), then all of those
            cand = [(k, r) for k, (F, g, r) in enumerate(frames) if r is not None and r["peak"] >= s.retarget_min_peak and r["second"] <= 0.85 * r["peak"]]
            cs = [(np.mean(r["src"], axis=0), np.mean(r["dst"], axis=0)) for k, r in cand]
            best = []
            if len(cand) == 1:
                best = [0]
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    Ra, ta, _ = RT.rigid_fit(np.array([cs[a][0], cs[b][0]]), np.array([cs[a][1], cs[b][1]]))
                    if abs(np.hypot(*(cs[a][0] - cs[b][0])) - np.hypot(*(cs[a][1] - cs[b][1]))) > 25:
                        continue                                   # the two disagree about the distance between them
                    inl = [i for i in range(len(cand)) if np.hypot(*(Ra @ cs[i][0] + ta - cs[i][1])) < 25]
                    key = (len(inl), sum(cand[i][1]["peak"] for i in inl))
                    if not best or key > (len(best), sum(cand[i][1]["peak"] for i in best)):
                        best = inl
            if len(best) <= 1 and len(cand) > 1:                   # no two spots agree: the strongest alone
                best = [max(range(len(cand)), key=lambda i: cand[i][1]["peak"])]
            src, dst = [], []
            if best:
                for i in best:
                    src += cand[i][1]["src"]; dst += cand[i][1]["dst"]
                if len(best) == 1:                                 # one spot: its shift with the rotation from the edges (or its own)
                    Rm = cand[best[0]][1]["R"]
                    tv = np.mean(np.array(dst) - np.array(src) @ Rm.T, axis=0)
                else:
                    Rm, tv, _ = RT.rigid_fit(np.array(src), np.array(dst))
                left = [cand[i][0] + 1 for i in range(len(cand)) if i not in best]
                self.say(f"  spots that agree: {', '.join(str(cand[i][0] + 1) for i in best)}" + (f"; left out: {', '.join(map(str, left))}" if left else ""))
                # second pass: every spot frame again, narrowly around the consensus (no stage moves)
                src2, dst2, used = [], [], []
                for k, (F, g, r0) in enumerate(frames):
                    r = RT.register(F, mos, g, Mf, cf, Rm, tv, 60.0)
                    if r is None:
                        continue
                    ok = r["peak"] >= s.retarget_min_peak and r["second"] <= 0.85 * r["peak"]
                    try:
                        RT.picture(r, root / f"{name}_spot{k + 1}.jpg")
                    except Exception:
                        pass
                    self.say(f"    spot {k + 1} refined: correlation {f3(r['peak'])} (next best {f3(r['second'])}), shift {f1(r['shift_um'][0])}, "
                             f"{f1(r['shift_um'][1])} um" + ("" if ok else "  -> not used"))
                    if ok:
                        src2 += r["src"]; dst2 += r["dst"]; used.append(k + 1)
                rms = NAN
                if len(used) >= 2:
                    src, dst = src2, dst2
                    Rm, tv, rms = RT.rigid_fit(np.array(src), np.array(dst))
                elif used:
                    src, dst = src2, dst2
                    tv = np.mean(np.array(dst) - np.array(src) @ Rm.T, axis=0)
            nspots = len(src) // 9
            if not src:
                rms = NAN
                if outline:
                    Rm, tv = R0, t0
                    self.say("  WARNING: no spot matched the first scan's mosaic: going on with the position from the edges alone "
                             "(the 100x check and re-centring correct it)")
                else:
                    raise self.fail("the chip was not found: no spot matched the first scan's mosaic (is it the right piece, the same way "
                                    "round? see the _spot pictures next to the log)")
            rot_deg = math.degrees(math.atan2(Rm[1, 0], Rm[0, 0]))
            cshift = Rm @ np.median(pts, axis=0) + tv - np.median(pts, axis=0)
            self.say(f"  chip found: moved {f1(cshift[0])}, {f1(cshift[1])} um, rotated {rot_deg:+.2f} deg"
                     + ((f"; {nspots} spots" + (f", residual {f1(rms)} um rms" if not isnan(rms) else "") + ("  WARNING: large residual" if rms > 15 else ""))
                        if src else " (from the edges)"))
        # which targets: all, or those without a focused image so far (the retake list; else the first run's 100x index,
        # matched by position: the provisional lists' labels are not the final list's)
        todo = T
        retake_csv = root / f"{piece}_retake.csv"
        idx100 = root / f"{piece}_100x" / f"{piece}_100x_index.csv"
        if s.retarget_only_failed:
            if retake_csv.is_file():
                with open(retake_csv, newline="") as fh:
                    todo = list(csv.DictReader(fh))
                self.say(f"  only the targets in the retake list {retake_csv}: {len(todo)} of {len(T)}")
            elif idx100.is_file():
                with open(idx100, newline="") as fh:
                    res = [(float(r["stage_x_um"]), float(r["stage_y_um"]), r.get("focus") or "", r.get("label") or "")
                           for r in csv.DictReader(fh) if (r.get("stage_x_um") or "").strip()]
                todo = retake_rows(T, res)
                self.say(f"  only the targets without a focused image last time ({idx100.name}, matched by position): {len(todo)} of {len(T)}")
            else:
                self.say(f"  (neither a retake list nor a 100x index of {piece}: all {len(T)} targets)")
        if not todo:
            raise self.fail("no targets left to image (every target has a focused image)")
        # 100x
        s.material = chip.get("material", s.material)
        if self.wait_objective("100x", "Switch to the 100x objective now, without changing the focus (the window closes by itself).") is None:
            if self.ui.confirm("Re-image: 100x", "Switch to the 100x objective now, without changing the focus.\nPress OK when it is in.", ("OK", "Cancel")) != 0:
                raise self.fail("cancelled")
            self.ui.mock_hook(self, "mockObjective", 100)
        self.view_scale = 1.0 / 9.54
        self.stream_off()
        c.setExposure(s.target_exposure_ms)
        self.crops_dir = root / f"{piece}_targets_crops"
        seeds = []
        if manual:
            fp_old = chip.get("focus_plane")
            R100, t100, seeds, rms = self.align_manual(T, Rm, tv, off, int(s.retarget_manual_points), root / name, piece,
                                                        tuple(fp_old[:2]) if fp_old else None)
            Rm, tv = R100, R100 @ off + t100 - off            # the same for 10x positions (outline, centre)
        # targets in the new coordinates; 10x focus from the spots (by hand: the 100x focus of the centred flakes instead)
        pf = fit_plane(zpts) if len(zpts) >= 3 else None
        zc = float(np.median([z[2] for z in zpts])) if zpts else NAN
        out, orig = [], []
        for t in todo:
            q = np.array([float(t["stage_x_um"]), float(t["stage_y_um"])]) - off
            p10 = Rm @ q + tv
            z10 = NAN if manual else ((pf[0] * p10[0] + pf[1] * p10[1] + pf[2]) if pf else zc)
            row = {k: t.get(k, "") for k in tcols}
            row.update(stage_x_um=f"{p10[0] + off[0]:.2f}", stage_y_um=f"{p10[1] + off[1]:.2f}", z10_um="" if isnan(z10) else f"{z10:.3f}", z_um="")
            pg = parse_poly(t.get("terrace_poly_um"))           # (the terrace outline moves with the chip)
            if pg is not None:
                row["terrace_poly_um"] = format_poly((Rm @ (pg - off).T).T + tv + off)
            out.append(row)
            orig.append(t)
        new_cjson = cjson if cjson.is_file() else None
        pz = fit_plane(seeds) if len(seeds) >= 3 else None
        if chip:
            nc = dict(chip, retake_of=str(cjson), transform=dict(R=Rm.tolist(), t=tv.tolist(), rms_um=None if isnan(rms) else rms, by_hand=manual))
            nc["corners"] = [list(map(float, Rm @ np.array(q, float) + tv)) for q in chip.get("corners") or []]
            if outline:
                nc["corners"], nc["rotation_deg"] = [list(map(float, q)) for q in outline["corners"]], outline["rot"]
            nc.pop("sides", None); nc.pop("edges", None); nc.pop("edge_points", None)
            nc["focus_points"] = [list(z) for z in zpts]
            if pf:
                nc["focus_plane"] = list(pf)
            elif pz:
                nc["focus_plane"] = list(pz)                   # (the centred flakes' 100x focus: its tilt is what is used)
            else:
                nc.pop("focus_plane", None)
            if seeds:
                nc["focus_points_100x"] = [list(map(float, z)) for z in seeds]
            new_cjson = root / f"{name}_chip.json"
            new_cjson.write_text(json.dumps(nc, indent=1))
        # the 10x -> 100x offset checked on the first targets (100x frame against the mosaic; not needed by hand)
        corr = None
        if mos is not None and s.retarget_check100:
            corr = self.retarget_check100(out, mos, Rm, tv, off, 60.0 if src else 150.0, root / name)
            if corr is not None:
                for row in out:
                    row["stage_x_um"] = f"{float(row['stage_x_um']) + corr[0]:.2f}"
                    row["stage_y_um"] = f"{float(row['stage_y_um']) + corr[1]:.2f}"
                    pg = parse_poly(row.get("terrace_poly_um"))
                    if pg is not None:
                        row["terrace_poly_um"] = format_poly(pg + np.asarray(corr, float))
        new_csv = root / f"{name}_targets.csv"
        write_rows(new_csv, tcols, out)
        self.say(f"  {len(out)} targets" + (f" ({len(T) - len(out)} with a focused image left out)" if len(out) < len(T) else "") + f" -> {new_csv}")
        if mos is not None and s.retarget_recentre:
            self.recentre = dict(mos=mos, R=Rm, t=tv, off=off + (corr if corr is not None else 0.0), corr=np.zeros(2), hist=[], radius=60.0)
        self.image_targets([dict(csv=str(new_csv), chip=str(new_cjson) if new_cjson else None, name=name, dir=str(root), piece=-1,
                                 retake=True, seed_focus=seeds, chip_tilt=(pz[0], pz[1]) if (pz and not new_cjson) else None,
                                 file_prefix=f"r{name.rsplit('_retake', 1)[1]}_")])
        self.recentre = None
        # the retake list again: the targets still without a focused image, in the first scan's positions
        left = []
        for o, row in zip(orig, out):
            x, y = float(row["stage_x_um"]), float(row["stage_y_um"])
            hows = hows_at(x, y, row.get("label"), self.last_results)
            if not any(h in ("auto", "manual") for h in hows):
                left.append(dict(o, retake_reason=retake_reason(hows)))
        try:
            write_rows(retake_csv, tcols + ["retake_reason"], left)
            self.say(f"  retake list: {len(left)} targets still without a focused image -> {retake_csv}" if left else
                     f"  retake list: every target now has a focused image ({retake_csv} empty)")
        except OSError as e:
            self.say(f"  (retake list not written: {e})")
        self.move_to(*(Rm @ np.median(pts, axis=0) + tv))
        # the images into the piece's viewer (replacing the flakes' earlier ones), the designer run, the results folder
        n_ok = sum(1 for q in self.last_results if q[2] in ("auto", "manual"))
        summary = self.update_results(piece, fcsv, chip, dest)
        if self.ui.can_ask:
            self.ui.message("Re-image", f"Done: {n_ok} of {len(out)} flakes re-imaged with a focused image"
                            + (f" ({len(left)} still without one: {retake_csv.name})" if left else "") + ".\n\n" + summary)
        self.say("done")

    def run_update(self):
        """Start screen, re-image without imaging: the re-images of a piece done so far into its viewer, designer run and
        results folder (update_results), e.g. for re-images made before this was automatic."""
        s = self.s
        root = Path(s.local_root)
        piece = s.retarget_piece
        self.open_log(root / f"{piece}_update_log.txt", append=True)
        self.say(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: the results of {piece} updated from its re-images (no imaging)")
        res_root = Path(s.results_dir) if s.results_dir else root / "results"
        fls = sorted(res_root.glob(f"**/{piece}/{piece}_tiles_live_flakes.csv"), key=lambda p: p.stat().st_mtime)
        cjson = root / f"{piece}_chip.json"
        chip = json.loads(cjson.read_text()) if cjson.is_file() else {}
        fcsv = fls[-1] if fls else None
        label = self.run_label_of(piece, fcsv)
        dest = self.results_dest_for(label) if label else None
        summary = self.update_results(piece, fcsv, chip, dest)
        if self.ui.can_ask:
            self.ui.message("Update results", summary)
        self.say("done")

    def run_label_of(self, piece, fcsv):
        """The name of the run a piece belongs to (e.g. 20261007_run_2): its results folder, else the run file
        (<local_root>/<run>.json) that lists it."""
        if fcsv is not None:
            return fcsv.parent.parent.name
        root = Path(self.s.local_root)
        for m in sorted(root.glob("*_run_*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
            try:
                if any(q.get("name") == piece for q in json.loads(m.read_text()).get("pieces", [])):
                    return m.stem
            except (OSError, ValueError):
                continue
        self.say(f"  (no run file in {root} lists {piece})")
        return None

    def results_dest_for(self, label):
        """Where run `label`'s results were copied (its folder in the results folder, e.g. on the share): from the run
        file (<local_root>/<label>.json, "results_copied_to"), else <copy_results_to>/<label> if it exists, else asked
        (and recorded in the run file).  None: not copied."""
        s = self.s
        man = Path(s.local_root) / f"{label}.json"
        try:
            M = json.loads(man.read_text()) if man.is_file() else {}
        except ValueError:
            M = {}
        for cand in (M.get("results_copied_to"), str(Path(s.copy_results_to) / label) if s.copy_results_to else ""):
            if cand and Path(cand).is_dir():
                self.say(f"  results folder of {label}: {cand}")
                return cand
            if cand:
                self.say(f"  ({cand} is not reachable)")
        if not self.ui.can_ask:
            self.say(f"  no results folder known for {label}: the share copy is not updated")
            return None
        from .ui import remembered_results_folder
        pick = self.ui.ask_dir(f"Where are the results of {label} (the copy that is updated)? Choose the {label} folder "
                               f"on the share, or the folder it is in (e.g. scanresults)", s.copy_results_to or remembered_results_folder(s))
        if not pick:
            self.say(f"  no results folder chosen for {label}: the share copy is not updated")
            return None
        pk = Path(pick)
        dst = pk if pk.name == label else pk / label
        if not dst.is_dir():
            if self.ui.confirm("Results folder", f"{dst} does not exist.\n\nCreate it and copy the results of {label} there?",
                               ("Create it", "Cancel")) != 0:
                self.say(f"  {dst} does not exist: the share copy is not updated")
                return None
        if man.is_file():                                # remembered for the next re-image of this run
            M["results_copied_to"] = str(dst)
            try:
                man.write_text(json.dumps(M, indent=1))
            except OSError:
                pass
        self.say(f"  results folder of {label}: {dst} (chosen; recorded in {man.name})")
        return str(dst)

    def _tool(self, cmd, code):
        """A helper program (flakeid.hundred, the designer bundle): its output read as UTF-8 whatever the console's
        code page (a character it cannot decode must not stop the update)."""
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(code), env=env)

    def update_results(self, piece, fcsv, chip, dest=None):
        """After re-imaging: every re-image of the piece (<piece>_retakeN_100x, in order) into its viewer, each flake's
        newest image replacing its earlier ones (flakeid.hundred --replace, with the re-image's transform back to the
        first scan's positions; this also redoes re-images already exported); the run's designer bundle rebuilt (the
        designer uses each flake's focused image); then the piece's folder and the bundle copied over the run's copy in
        the results folder `dest` (the share).  Returns a summary for the user."""
        s = self.s
        root = Path(s.local_root)
        res_root = Path(s.results_dir) if s.results_dir else root / "results"
        # the viewer: next to the flake list, anywhere in the local results, else the copy in the results folder (the
        # share; there the piece's folder is the viewer): then that copy is updated directly
        looked, cands = [], []
        if fcsv is not None:
            looked.append(str(fcsv.parent))
            cands += sorted(fcsv.parent.glob("*_viewer"))
        looked.append(f"{res_root} (any {piece}_tiles_viewer)")
        try:
            cands += sorted(res_root.glob(f"**/{piece}_tiles_viewer"), key=lambda q: q.stat().st_mtime, reverse=True)
        except OSError:
            pass
        vdir = next((v for v in cands if (v / "index.html").is_file()), None)
        direct = False
        if vdir is not None:
            run_res = fcsv.parent.parent if (fcsv is not None and vdir.parent == fcsv.parent) else vdir.parent.parent
        elif dest and (Path(dest) / piece / "index.html").is_file():
            vdir, run_res, direct = Path(dest) / piece, Path(dest), True
            self.say(f"  no viewer of {piece} on this PC (looked in {'; '.join(looked)}): the copy in the results folder is "
                     f"updated directly: {vdir}")
        else:
            if dest:
                looked.append(str(Path(dest) / piece))
            if fcsv is not None:
                try:
                    self.say(f"  ({fcsv.parent} holds: {', '.join(sorted(q.name for q in fcsv.parent.iterdir())[:30])})")
                except OSError:
                    pass
            self.say(f"  WARNING: no viewer of {piece} found (looked in {'; '.join(looked)}): nothing updated")
            return (f"Nothing was updated: no viewer of {piece} was found.\nLooked in:\n  " + "\n  ".join(looked)
                    + ("" if dest else "\n(and no results folder was chosen, where its copy is)"))
        num = re.compile(rf"{re.escape(piece)}_retake(\d+)_100x$")
        rdirs = sorted((d for d in root.glob(f"{piece}_retake*_100x") if d.is_dir() and num.match(d.name)
                        and next(d.glob("*_100x_index.csv"), None) is not None), key=lambda d: int(num.match(d.name).group(1)))
        if not rdirs:
            self.say(f"  no re-images of {piece} in {root}: nothing to update")
            return f"Nothing to update: no re-images of {piece} in {root}."
        code = Path(__file__).resolve().parents[2]
        material = chip.get("material", s.material)
        fl = self._flat100_for(float(chip.get("oxide_nm") or s.oxide_nm))
        bad, matched = [], []
        for d in rdirs:
            cmd = [sys.executable, "-m", "flakeid.hundred", str(d), "--viewer", str(vdir), "--material", material, "--replace"] + (["--flat", fl] if fl else [])
            rc_ = root / (d.name[:-len("_100x")] + "_chip.json")
            if rc_.is_file():
                cmd += ["--retake-chip", str(rc_)]
            else:
                self.say(f"  (no {rc_.name}: {d.name} is matched by the flake numbers in its labels)")
            self.say(f"  {d.name} into the viewer: " + " ".join(cmd))
            self.ui.progress(self, f"{d.name}: into the viewer ...")
            try:
                r = self._tool(cmd, code)
            except Exception as e:
                bad.append(d.name)
                self.say(f"  WARNING: {d.name} not exported: {type(e).__name__}: {e}")
                continue
            out_ = (r.stdout or "").strip().splitlines()
            for ln in out_[-3:]:
                self.say("    " + ln.strip())
            if r.returncode != 0:
                bad.append(d.name)
                self.say(f"  WARNING: {d.name} not exported ({r.returncode}): {(r.stderr or '')[-600:]}")
            m_ = next((re.search(r"\((\d+) of (\d+) images matched", ln) for ln in reversed(out_) if "images matched" in ln), None)
            if m_:
                matched.append((d.name, int(m_.group(1)), int(m_.group(2))))
                if int(m_.group(1)) == 0 and int(m_.group(2)) > 0:
                    self.say(f"  WARNING: none of the {m_.group(2)} images of {d.name} matched a flake of the viewer")
        if direct:                                       # (the records first: the designer reads the piece's data/)
            self.update_copy_direct(piece, Path(dest), None)
        tool = code / "tools" / "make_designer_bundle.py"
        des_out = (root / f"{run_res.name}_designer_build") if direct else (run_res / "designer")
        if not s.designer_on_pc:
            des_ok = False
            self.say("  designer run: rebuilt by the vdW Stack Designer from the results folder (it sees that the run changed)")
        else:
            self.ui.progress(self, "rebuilding the designer run ...")
            try:
                rb = self._tool([sys.executable, str(tool), str(run_res), "--run", run_res.name, "-o", str(des_out)], code)
                des_ok = rb.returncode == 0
                self.say("  designer run " + (f"rebuilt: {run_res / 'designer'}" if des_ok else
                                              f"not rebuilt ({rb.returncode}): {((rb.stdout or '') + (rb.stderr or ''))[-600:]}"))
            except Exception as e:
                des_ok = False
                self.say(f"  WARNING: the designer run was not rebuilt: {type(e).__name__}: {e}")
        where = None
        if direct:
            where = self.update_copy_direct(piece, Path(dest), des_out if des_ok else None, records=False)
        elif dest:
            where = self.copy_piece_results(piece, run_res, dest, des_ok)
        mt = "; ".join(f"{n_} of {t_} images matched" for _, n_, t_ in matched)
        return (f"Viewer: {len(rdirs) - len(bad)} of {len(rdirs)} re-images added" + (f" ({mt})" if mt else "") + "; each flake shows its newest image.\n"
                + ("Designer run rebuilt.\n" if des_ok else ("The designer run is rebuilt by the vdW Stack Designer (it sees the change).\n"
                   if not s.designer_on_pc else "WARNING: the designer run was not rebuilt (see the log).\n"))
                + (f"Results folder updated: {where}\n(in the vdW Stack Designer: reload the page)" if where else
                   "The results folder (share) was NOT updated" + ("" if dest else " (no folder chosen)") + ": see the log."))

    def update_copy_direct(self, piece, dst, des_built, records=True):
        """The results folder's copy updated in place (no viewer of the piece on this PC): the re-image's records
        (<local_root>/<piece>_*: targets, chip, index, logs) into <piece>/data/, and the designer run built here
        (des_built) put over <run>/designer.  Returns dst, or None."""
        s = self.s
        root, label = Path(s.local_root), dst.name
        try:
            if records:
                pairs, _ = share_layout(root, None, label, [piece], dst)
                todo = self._changed([(a, b) for a, b in pairs if (dst / piece) in Path(b).parents])
                nb = copy_parallel(todo)
                self.say(f"  results folder {dst}: {len(todo)} records of {piece} copied ({nb / 1e6:.1f} MB)")
                write_copy_done(dst, [piece])
                return str(dst)
            gone = self._replace_tree(des_built, dst / "designer") if des_built is not None else None
            self.say(f"  results folder {dst}: " + (f"designer run replaced ({gone} old files removed)" if gone is not None
                                                     else "designer run NOT updated"))
            return str(dst)
        except Exception as e:
            self.say(f"  WARNING: updating {dst} failed: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _changed(pairs):
        """The (source, destination) pairs whose destination is missing or differs (size, older)."""
        out = []
        for a, b in pairs:
            b = Path(b)
            try:
                sa = a.stat()
                if b.is_file():
                    sb = b.stat()
                    if sb.st_size == sa.st_size and sb.st_mtime >= sa.st_mtime - 2:
                        continue
            except OSError:
                pass
            out.append((a, b))
        return out

    def _replace_tree(self, src, dst):
        """dst made the same as src: changed files copied, files src does not have deleted.  Returns the number deleted."""
        src, dst = Path(src), Path(dst)
        copy_parallel(self._changed(_tree_pairs(src, dst)))
        gone = 0
        for q in sorted(dst.rglob("*")):
            if q.is_file() and not (src / q.relative_to(dst)).is_file():
                try:
                    q.unlink()
                    gone += 1
                except OSError:
                    pass
        return gone

    def copy_piece_results(self, piece, run_res, dest, with_designer=True):
        """The piece's folder (viewer, data) and the run's designer bundle copied over the run's copy `dest` in the
        results folder.  Only changed files are copied; files of the old designer bundle that the new one does not have
        are deleted.  Returns dest, or None."""
        s = self.s
        root, label = Path(s.local_root), run_res.name
        dst = Path(dest)
        man = root / f"{label}.json"
        try:
            M = json.loads(man.read_text()) if man.is_file() else {}
        except ValueError:
            M = {}
        names = [q["name"] for q in M.get("pieces", [])] or [d.name for d in run_res.iterdir() if d.is_dir() and d.name != "designer"]
        if piece not in names:
            names.append(piece)
        self.ui.progress(self, f"copying {piece} to {dst} ...")
        t0 = time.time()
        try:
            dst.mkdir(parents=True, exist_ok=True)
            pairs, viewers = share_layout(root, run_res, label, names, dst)
            mine = [dst / piece] + ([dst / "designer"] if with_designer else [])
            todo = []
            for a, b in pairs:
                b = Path(b)
                if not any(m_ == b or m_ in b.parents for m_ in mine):
                    continue
                try:
                    sa = a.stat()
                    if b.is_file():
                        sb = b.stat()
                        if sb.st_size == sa.st_size and sb.st_mtime >= sa.st_mtime - 2:
                            continue                     # (unchanged: most of the viewer's tiles)
                except OSError:
                    pass
                todo.append((a, b))
            nb = copy_parallel(todo)
            gone = 0
            ld, rd = run_res / "designer", dst / "designer"
            if with_designer and ld.is_dir() and rd.is_dir():
                for q in sorted(rd.rglob("*")):
                    if q.is_file() and not (ld / q.relative_to(rd)).is_file():
                        try:
                            q.unlink()
                            gone += 1
                        except OSError:
                            pass
            if piece in viewers and not (dst / piece / f"{piece} viewer.html").is_file():
                write_viewer_link(dst / piece / f"{piece} viewer.html", viewers[piece])
            write_copy_done(dst, [piece])
            self.say(f"  results folder {dst}: {len(todo)} files of {piece}" + (" and the designer run" if with_designer else "")
                     + f" updated ({nb / 1e6:.0f} MB, {time.time() - t0:.0f} s)" + (f"; {gone} old designer files removed" if gone else ""))
            return str(dst)
        except Exception as e:
            self.say(f"  WARNING: copying to {dst} failed: {type(e).__name__}: {e} (the results are in {run_res})")
            return None

    def align_manual(self, T, Rm, tv, off, n_points, stem, piece, tilt):
        """Re-image, by hand: the user centres n_points target flakes at 100x, spread over the piece (the stage first moves
        to where each should be: the first scan's position through the estimate so far, which each centred flake
        improves).  Returns (R, t, [(x, y, z) of the centred flakes], rms): new 100x position = R old + t."""
        from . import retarget as RT
        s, c = self.s, self.core
        old = np.array([[float(r["stage_x_um"]), float(r["stage_y_um"])] for r in T], float)
        R = np.asarray(Rm, float)
        t = np.asarray(tv, float) + off - R @ off
        rms = NAN

        def thick(i):
            try:
                return float(T[i].get("thickness_nm") or 0)
            except ValueError:
                return 0.0
        lab_of = [r.get("label") or f"T{pad3(i + 1)}" for i, r in enumerate(T)]
        cand = [i for i in range(len(T)) if self.target_picture(None, lab_of[i])]
        if len(cand) < n_points:
            cand = list(range(len(T)))
        easy = [i for i in cand if thick(i) >= 5.0]                     # (thicker flakes are easier to recognise at 100x)
        if len(easy) >= max(n_points, 4):
            cand = easy

        def plan(sel, avail):
            """The flakes to centre: those already done, then the most spread out (the farthest pair, then the farthest
            from the line through them, then the farthest from all chosen)."""
            sel = list(sel)
            while len(sel) < n_points:
                rest = [i for i in avail if i not in sel]
                if not rest:
                    break
                if not sel:
                    A = old[rest]
                    D = np.hypot(A[:, None, 0] - A[None, :, 0], A[:, None, 1] - A[None, :, 1])
                    a, b = np.unravel_index(int(np.argmax(D)), D.shape)
                    sel += [rest[a]] + ([rest[b]] if b != a else [])
                    continue
                if len(sel) == 1:
                    sel.append(max(rest, key=lambda i: np.hypot(*(old[i] - old[sel[0]]))))
                elif len(sel) == 2:
                    u = old[sel[1]] - old[sel[0]]
                    un = u / max(np.hypot(*u), 1e-9)
                    sel.append(max(rest, key=lambda i: abs(un[0] * (old[i] - old[sel[0]])[1] - un[1] * (old[i] - old[sel[0]])[0])))
                else:
                    sel.append(max(rest, key=lambda i: min(np.hypot(*(old[i] - old[j])) for j in sel)))
            return sel
        done, skipped = [], set()                                       # done: (index, centred stage position, Z)
        while len(done) < n_points:
            avail = [i for i in cand if i not in skipped]
            nxt = [i for i in plan([d[0] for d in done], avail) if i not in [d[0] for d in done]]
            if not nxt:
                break
            here = np.array([c.getXPosition(self.xy), c.getYPosition(self.xy)])
            i = min(nxt, key=lambda j: np.hypot(*(R @ old[j] + t - here)))
            lab, k = lab_of[i], len(done) + 1
            g = R @ old[i] + t
            self.move_to(*g)
            if done:
                zg = self.predict_z(g[0], g[1], [(d[1][0], d[1][1], d[2]) for d in done], tilt, self.z_read())
            else:
                zg = self.Z0 + (s.focus_offset_um if not isnan(s.focus_offset_um) else 0.0)
                if tilt:
                    zg += tilt[0] * (g[0] - self.X0) + tilt[1] * (g[1] - self.Y0)
            r = self.autofocus(zg, s.af100_first)
            th = thick(i)
            self.say(f"  alignment flake {k} of {n_points}: {lab}" + (f" ({th:g} nm)" if th else "") + f", expected at {f1(g[0])}, {f1(g[1])} um"
                     + ("" if r[2] else " (autofocus failed)"))
            a = self.ask_focus(f"Re-image {piece}: alignment flake {k} of {n_points}",
                               f"Alignment flake {k} of {n_points}: {lab}" + (f" ({th:g} nm)" if th else "") + ".\n\n"
                               "The stage is where this flake should be" + (" (corrected by the flakes centred so far)" if done else
                                                                              " according to the first scan") + ".  Find it with the picture on the right "
                               "(the box is the 100x field) and move the stage (joystick, the arrow buttons or the arrow keys) until the live "
                               "view shows what is in the box, with the box's centre on the cross.  Focus (Z buttons or Autofocus), then press "
                               "\"Use this position\".\n\nIf you cannot find it, \"Skip\" picks another flake.",
                               None, image=self.target_picture(None, lab), button="Use this position", cross=True)
            if a == 2:
                raise self.fail("cancelled")
            if a != 0:
                skipped.add(i)
                self.say(f"    {lab} skipped: another flake instead")
                continue
            new = np.array([c.getXPosition(self.xy), c.getYPosition(self.xy)])
            z = self.z_read()
            done.append((i, new, z))
            self.save_view(Path(f"{stem}_align{k}.jpg"))
            if len(done) == 1:
                t = new - R @ old[i]
            else:
                R, t, rms = RT.rigid_fit(old[[d[0] for d in done]], np.array([d[1] for d in done]))
            self.say(f"    centred at {f1(new[0])}, {f1(new[1])} um, Z {f3(z)}: {f1(new[0] - g[0])}, {f1(new[1] - g[1])} um from where it was expected")
        if not done:
            raise self.fail("no flake was centred for the alignment")
        ang = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        sh = R @ np.median(old, axis=0) + t - np.median(old, axis=0)
        self.say(f"  aligned by hand on {len(done)} flake{'s' if len(done) != 1 else ''}: the piece moved {f1(sh[0])}, {f1(sh[1])} um, rotated "
                 f"{ang:+.3f} deg" + (f"; the flakes agree to {f1(rms)} um rms" if len(done) >= 3 else
                                      ("  (one flake only: the rotation is the estimate's)" if len(done) == 1 else "")))
        if len(done) >= 3 and rms > 8.0:
            self.say(f"  WARNING: the centred flakes disagree by {f1(rms)} um rms: one of them may not have been the right flake "
                     f"(pictures {Path(str(stem)).name}_align*.jpg); the positions may be off by about that much")
        return R, t, [(float(d[1][0]), float(d[1][1]), float(d[2])) for d in done], rms

    def save_view(self, path):
        """The current camera frame as a JPEG (as the live view shows it)."""
        try:
            from PIL import Image
            from .ui import to_display
            img, _ = to_display(self.frame(), self.DARK, None, self.s.display_level, False, 0.5, None)
            Image.fromarray(img).save(str(path), quality=88)
        except Exception as e:
            self.say(f"    (picture {path} not saved: {type(e).__name__}: {e})")

    def _crop_grey(self, pic):
        """A target's 10x picture as grey 0..1 for matching, with the marks live.write_target_crops draws (the 100x
        field's box and a cross, both centred) painted out: they are not on the sample."""
        import cv2
        bgr = cv2.imread(str(pic))
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        fw, fh = self.W / 9.54, self.H / 9.54                       # the 100x field in 10x pixels (same camera)
        marks = np.zeros((h, w), np.uint8)
        cv2.rectangle(marks, (int(w / 2 - fw / 2), int(h / 2 - fh / 2)), (int(w / 2 + fw / 2), int(h / 2 + fh / 2)), 1, 6)
        cv2.drawMarker(marks, (w // 2, h // 2), 1, cv2.MARKER_CROSS, 18, 5)
        bgr = cv2.inpaint(bgr, marks, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    def _match_here(self, crop):
        """The current frame (100x) against a target's 10x picture: retarget.match_crop's result, or None."""
        import cv2
        from . import retarget as RT
        from .ui import bayer_half
        img = self.frame()
        if img.ndim == 2:
            rgb, kk = bayer_half(img) - self.DARK, 2.0 / 9.54
        else:
            rgb, kk = img[..., :3].astype(np.float32), 1.0 / 9.54
        small = cv2.resize(np.ascontiguousarray(rgb), None, fx=kk, fy=kk, interpolation=cv2.INTER_AREA)
        m = RT.match_crop(RT.grey(small), crop, um_px=float(np.sqrt(abs(np.linalg.det(np.asarray(self.M, float))))))
        # colours for the bare-substrate test: the frame's commonest colour (and how much of it) and the centre's
        from flakeid.hundred import _mode_rgb
        h, w = rgb.shape[:2]
        q = max(1.0, float(np.percentile(rgb, 99)) / 1000.0)            # (any scale: _mode_rgb bins absolute values)
        fm, frac = _mode_rgb(rgb[::2, ::2] / q, step=4)
        cm, _ = _mode_rgb(rgb[h // 2 - h // 10:h // 2 + h // 10, w // 2 - h // 10:w // 2 + h // 10] / q, step=4)
        return m, dict(mode=fm * q, frac=frac, centre=cm * q)

    def centre_on_target(self, P, lab, xy):
        """After focusing at 100x: the frame against the target's 10x picture (<piece>_targets_crops/<label>.jpg, which
        the analysis writes with the target list); when the flake is off-centre the stage is moved onto it before the
        image is taken.  No confident match (e.g. only bare substrate in the frame): a ring of 6 frames around the
        target (recentre_search) is searched for it.  The correction feeds the next targets' start (recentre_carry).
        2026-10-14: 214 of 1574 100x frames had bare substrate at the centre; misses of 10-75 um in all directions.
        The ring search only runs when the frame centre is bare substrate (a frame filled by a large flake matches its
        10x picture weakly, yet is centred).  Returns the move made ("dx:dy" um), "none" (centred), "unsure" (no
        confident match, the centre is not bare substrate: imaged as it is), "not found", or "" (no picture)."""
        s, c = self.s, self.core
        pic = self.target_picture(P, lab)
        crop = self._crop_grey(pic) if pic else None
        if crop is None:
            return ""
        M = np.asarray(self.M, float)
        cur = np.array([c.getXPosition(self.xy), c.getYPosition(self.xy)])

        def good(m):
            return m is not None and m["peak"] >= s.recentre_min_peak and m["second"] <= s.recentre_max_second * m["peak"]
        m, col = self._match_here(crop)
        base = cur
        # the run's bare substrate at 100x: the commonest colour of frames that are mostly one colour (most frames show
        # mostly substrate; a frame filled by a large flake is outvoted)
        if col["frac"] >= 0.5:
            self.sub100.append(col["mode"])
            del self.sub100[:-25]
        ref = np.median(np.array(self.sub100), axis=0) if len(self.sub100) >= 3 else (col["mode"] if col["frac"] >= 0.7 else None)
        centre_bare = ref is not None and float(np.abs(col["centre"] / np.maximum(ref, 1e-6) - 1.0).max()) < 0.03
        if not good(m) and s.recentre_search and centre_bare:
            fw, fh = self.FOVX / 9.54, self.FOVY / 9.54                  # the 100x field (um)
            best = None
            for a in range(0, 360, 60):
                p = cur + 0.8 * np.array([fw * math.cos(math.radians(a)), fh * math.sin(math.radians(a))])
                self.approach_to(*p)
                m2, _ = self._match_here(crop)
                if good(m2) and (best is None or m2["peak"] > best[1]["peak"]):
                    best = (p, m2)
            if best is None:
                self.approach_to(*cur)
                self.say(f"    re-centring: no confident match against the 10x picture" +
                         (f" (correlation {f3(m['peak'])}, next best {f3(m['second'])})" if m else "") + ", nor around it: imaged here")
                return "not found"
            base, m = best
            self.say(f"    re-centring: found {f1(np.hypot(*(base - cur)))} um away (ring search)")
        elif not good(m):
            self.say("    re-centring: no confident match" + (f" (correlation {f3(m['peak'])}, next best {f3(m['second'])})" if m else "")
                     + (": imaged here" if not centre_bare else ": imaged here (no search: recentre_search off)"))
            return "unsure" if not centre_bare else "not found"
        target = base + M @ m["d"]
        sh = target - cur
        if np.hypot(*sh) > s.recentre_max_um:
            self.approach_to(*cur)
            self.say(f"    re-centring: the match is {f1(np.hypot(*sh))} um away (more than {f1(s.recentre_max_um)}): not used")
            return "not found"
        # the whole correction of this target (the carried one it started with + this move), for the next ones
        P.setdefault("rc_hist", []).append(tuple(target - np.asarray(xy, float) + P.get("rc_carried", np.zeros(2))))
        if np.hypot(*sh) < s.recentre_min_um:
            if not np.allclose(base, cur):
                self.approach_to(*cur)
            return "none"
        self.approach_to(*target)
        self.say(f"    re-centred by {f1(sh[0])}, {f1(sh[1])} um (correlation {f3(m['peak'])}, next best {f3(m['second'])})")
        return f"{f1(sh[0])}:{f1(sh[1])}"

    def recentre_target(self, rc):
        """Retarget mode, before a 100x image: this frame against the first scan's mosaic; the stage moved onto the flake
        if it is off.  The correction is carried on to the next targets (median of the last five)."""
        import cv2
        from . import retarget as RT
        from .ui import bayer_half
        s, c = self.s, self.core
        img = self.frame()
        rgb = bayer_half(img) - self.DARK if img.ndim == 2 else img[..., :3].astype(np.float32)
        kk = 2.0 / 9.54
        small = cv2.resize(np.ascontiguousarray(rgb), None, fx=kk, fy=kk, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        cur = np.array([c.getXPosition(self.xy), c.getYPosition(self.xy)])
        res = RT.register(RT.grey(small), rc["mos"], cur - rc["off"], np.asarray(self.M, float), np.array([w / 2.0, h / 2.0]),
                          rc["R"], rc["t"], rc["radius"], sig=(1.0, 8.0), crop=0.02)
        if res is None:
            self.say("    re-centring: outside the first scan's mosaic")
            return
        sh = res["shift_um"]
        if not (res["peak"] >= s.retarget_min_peak100 and res["second"] <= 0.85 * res["peak"]):
            self.say(f"    not re-centred: correlation {f3(res['peak'])} (next best {f3(res['second'])})")
            return
        if np.hypot(*sh) > 1.0:
            self.approach_to(*(cur + sh))
        rc["hist"].append(rc["corr"] + sh)
        rc["corr"] = np.median(np.array(rc["hist"][-5:]), axis=0)
        self.say(f"    re-centred by {f1(sh[0])}, {f1(sh[1])} um (correlation {f3(res['peak'])}, next best {f3(res['second'])})")

    def retarget_check100(self, rows, mos, Rm, tv, off, radius, stem):
        """Where the first targets really are at 100x: a 100x frame at each (focused), shrunk to the 10x scale, against
        the first scan's mosaic.  The stage correction for all targets (median of the confident ones), or None."""
        import cv2
        from . import retarget as RT
        from .ui import bayer_half
        s, c = self.s, self.core
        M = np.asarray(self.M, float)
        found = []
        for k, t in enumerate(rows[:3]):
            self.ui.check_stop(self)
            self.ui.progress(self, f"checking the 100x position on target {k + 1}")
            p = np.array([float(t["stage_x_um"]), float(t["stage_y_um"])])
            self.approach_to(*p)
            z10 = float(t["z10_um"]) if t.get("z10_um") else NAN
            zg = z10 + (s.focus_offset_um if not isnan(s.focus_offset_um) else 0.0) if not isnan(z10) else self.z_read()
            fr = self.autofocus(zg, s.af100_first)
            img = self.frame()
            rgb = bayer_half(img) - self.DARK if img.ndim == 2 else img[..., :3].astype(np.float32)
            kk = 2.0 / 9.54                                         # half-resolution 100x pixels -> 10x pixels (same camera)
            small = cv2.resize(np.ascontiguousarray(rgb), None, fx=kk, fy=kk, interpolation=cv2.INTER_AREA)
            h, w = small.shape[:2]
            g = np.array([c.getXPosition(self.xy), c.getYPosition(self.xy)]) - off       # as a 10x stage position
            res = RT.register(RT.grey(small), mos, g, M, np.array([w / 2.0, h / 2.0]), Rm, tv, radius, sig=(1.0, 8.0), crop=0.02)
            if res is None:
                self.say(f"    100x check on {t['label']}: outside the mosaic")
                continue
            try:
                RT.picture(res, Path(str(stem) + f"_check100_{k + 1}.jpg"))
            except Exception:
                pass
            ok = res["peak"] >= s.retarget_min_peak100 and res["second"] <= 0.85 * res["peak"]
            sh = res["shift_um"]
            self.say(f"    100x check on {t['label']} ({'focused' if fr[2] else 'focus failed'}): correlation {f3(res['peak'])} "
                     f"(next best {f3(res['second'])}), the flake is {f1(sh[0])}, {f1(sh[1])} um from where it was expected"
                     + ("" if ok else "  -> not used"))
            if ok:
                found.append((sh, bool(fr[2])))
                fs_ = [f_ for f_, foc in found if foc]
                if len(fs_) >= 2 and np.hypot(*(fs_[-1] - fs_[-2])) < 6.0:
                    break
        if found and any(foc for _, foc in found):
            found = [f_ for f_, foc in found if foc]                  # focused frames are the more reliable
        else:
            found = [f_ for f_, _ in found]
        if not found:
            self.say("  100x check: no confident match; the targets keep the 10x positions")
            return None
        corr = np.median(np.array(found), axis=0)
        if len(found) >= 2 and max(np.hypot(*(f_ - corr)) for f_ in found) > 10.0:
            self.say(f"  100x check: the corrections disagree ({', '.join(f'{f1(f_[0])}, {f1(f_[1])}' for f_ in found)}); not applied")
            return None
        self.say(f"  100x check: all targets moved by {f1(corr[0])}, {f1(corr[1])} um (10x -> 100x offset now "
                 f"{f1(off[0] + corr[0])}, {f1(off[1] + corr[1])} um)")
        return corr

    def capture_pattern(self, P, t):
        """Frames around target t (pattern_frames, within pattern_radius_um), at the Z the flakes' focus predicts (autofocus
        on bare substrate would focus on the dust in the light path): the optics pattern (dust in the light path, camera
        specks) is what stays at the same pixel in all of them.  Saved in <name>_100x/pattern/."""
        s, c = self.s, self.core
        d = P["img_dir"] / "pattern"
        d.mkdir(exist_ok=True)
        n = int(s.pattern_frames)
        self.say(f"  optics-pattern frames: {n} around {t[4]} (up to {f1(s.pattern_radius_um)} um away) -> {d}")
        box, self.move_box = self.move_box, None
        rng = np.random.default_rng(1)
        with open(d / "pattern_index.csv", "w", newline="") as fh:
            fh.write("n,file,stage_x_um,stage_y_um,z_um,focus,exposure_ms\r\n")
            for k in range(n):
                self.ui.check_stop(self)
                a, r = rng.uniform(0, 2 * np.pi), s.pattern_radius_um * np.sqrt(rng.uniform(0.15, 1.0))
                x, y = t[0] + r * np.cos(a), t[1] + r * np.sin(a)
                self.move_to(x, y)
                zg = self.predict_z(x, y, P["focused"], P["chip_tilt"], self.z_read())
                self.set_z(zg - s.af_backlash_um); self.set_z(zg)
                rr = (zg, 0.0, False)
                im = self.snap_good(f"pattern {k + 1}")
                fn = f"p{k + 1:03d}.tif"
                self.save_image(d / fn, im, {"stage_x_um": x, "stage_y_um": y, "z_um": self.z_read(), "focus": "auto" if rr[2] else "predicted",
                                             "exposure_ms": c.getExposure(), "bayer": "RGGB (R at top-left)"})
                fh.write(f"{k + 1},{fn},{x},{y},{self.z_read()},{'auto' if rr[2] else 'predicted'},{c.getExposure()}\r\n")
                self.ui.progress(self, f"optics-pattern frame {k + 1} / {n}")
        self.move_box = box
        self.move_to(t[0], t[1])

    def _recentre(self, P, rc, lab, x, y):
        """The 100x frame onto the flake: against the first scan's mosaic (retarget mode; returns "") or against the
        target's 10x picture (centre_on_target's result)."""
        if rc:
            self.recentre_target(rc)
            return ""
        try:
            return self.centre_on_target(P, lab, (x, y))
        except Exception as e:                       # (never stops the imaging: the target is imaged where it is)
            self.say(f"    (re-centring failed: {type(e).__name__}: {e})")
            return ""

    def image_flake(self, P, t, tag, retake_k=None):
        """Focus and image one target; False to stop.  retake_k: a retake at the end of the piece (retake_failed) of
        the flake imaged as number retake_k + 1: no long search, no focus question; its image and index row replace the
        first ones."""
        s, c = self.s, self.core
        retake = retake_k is not None
        self.set_phase("100x imaging")
        x, y, z10, th, lab = t
        rc = getattr(self, "recentre", None)
        if rc:                                       # retarget mode: what the targets before were off by
            x, y = x + rc["corr"][0], y + rc["corr"][1]
        P["rc_carried"] = np.zeros(2)
        if not rc and s.hundred_recentre and s.recentre_carry > 0 and len(P.get("rc_hist") or ()) >= 3:
            # what the targets before were off by (median of the last recentre_carry corrections)
            P["rc_carried"] = np.median(np.array(P["rc_hist"][-int(s.recentre_carry):]), axis=0)
            x, y = x + P["rc_carried"][0], y + P["rc_carried"][1]
        k = P["k"] - 1 if not retake else retake_k
        fs, cam, img_dir, out_dir, chip_tilt = P["fs"], P["cam"], P["img_dir"], P["out_dir"], P["chip_tilt"]
        idx = P["idx"]
        of = f" of {P['total']}" + (" so far" if P.get("total_more") else "") if P.get("total") and not retake else ""
        self.say(f"  [{k + 1}{of}{tag}] {lab} at {f1(x)}, {f1(y)}")
        self.ui.progress(self, f"{s.name}: 100x flake {k + 1}{of}{tag}")      # (the window's top line, while it is focused)
        want100 = (s.objective_positions or {}).get("100x")
        if self.nose is not None and want100 and self.nose.position() != want100:
            self.say("    the objective is no longer 100x")
            if self.wait_objective("100x", "The objective changed during the 100x imaging.  Put the 100x objective back "
                                   "in (the window closes by itself) to go on.") is False:
                self.stop_100x = True
                return False
        try:
            self.approach_to(x, y)
        except StageLimit as e:                      # (a chip partly beyond the stage's travel, a target near its edge)
            self.say(f"    beyond the stage's travel ({e}): skipped")
            P["results"].append((t[0], t[1], "", lab))
            return not P["stop"]
        if s.mode == "multi" and not self.exp_checked:
            self.check_exposure(s.target_auto_exposure)
            self.exp_checked = True
        # the prediction uses only the flakes focused since the last focus jump (P["seg_..."]): after the focus has moved,
        # the flakes before it would pull the prediction back to where the focus was
        off_known = self.predicted_offset(P, x, y)
        zg = z10 + off_known if not isnan(z10) and not isnan(off_known) else self.predict_z(x, y, P["seg_focused"], chip_tilt, self.z_read())
        if s.verbose:
            self.say(f"    start Z {f3(zg)}" + (f" (10x focus {f3(z10)}" + (", offset not known yet)" if isnan(off_known) else f" + offset {f3(off_known)})")
                                                if not isnan(z10) else ""))
        known = bool(P["seg_focused"]) or (not isnan(z10) and not isnan(off_known))
        self.zref = zg if known else NAN
        how, ratio, rej, jump = "", 0.0, NAN, NAN
        tol = s.af_max_dev_um[0]
        wide = P["fail_run"] >= 2
        recentred, pre = "", False
        if s.focus_mode == "auto" and s.recentre_before_focus and known and not wide and (rc or s.hundred_recentre):
            # re-centred first, at the predicted Z (within ~1.3 um for 95 % of flakes: sharp enough to match at the 10x
            # scale), so that the autofocus sees the flake rather than bare substrate (no focus signal, a wide retry)
            self.set_z(zg - s.af_backlash_um); self.set_z(zg)
            recentred, pre = self._recentre(P, rc, lab, x, y), True
        ctl = P.get("ctl")
        manual_first = retake and ctl is not None and hasattr(ctl, "take") and ctl.take(MANUAL_RETAKE)   # (pressed before)
        if manual_first:
            pass
        elif s.focus_mode == "auto":
            if wide and known:
                self.say(f"    the last {P['fail_run']} flakes were not focused: wide search")
            r = self.autofocus(zg, s.af100 if (known and not wide) else s.af100_first, fine="fit" if (known and not wide and s.af100_fit) else True)
            if not r[2] and known and not wide:
                r = self.autofocus(zg, s.af100_first)
            if not r[2] and retake:                   # (a retake tries harder: a much wider search)
                self.say(f"    retake: wider search +-{f1(s.af_retake[0])} um (up to +-{f1(s.af_retake[0] * (1 + s.af_retake_extend))}) "
                         f"around {f3(zg)}")
                r = self.autofocus(zg, s.af_retake, extend=s.af_retake_extend)
            # a focus question only on the piece's first flakes (af_manual_fallback "first"), and not again on this piece
            # once one went unanswered (focus_dialog_repeat)
            asking = (not retake and self.ask_on_fail(len(P["focused"])) and self.ui.can_ask
                      and (s.focus_dialog_repeat or not P.get("no_ask")))
            if not r[2] and not retake and s.af_long_after > 0 and not P.get("long_tried") and (P["fail_run"] + 1 >= s.af_long_after or asking):
                # the usual searches failed again (or a question would be asked): has the focus drive slipped (20261014_run_1:
                # 40-60 um)?  One long coarse search per run of failures; a focus far away is confirmed below (a focus jump)
                P["long_tried"] = True
                self.say(f"    {P['fail_run'] + 1} flake(s) in a row not focused: long search +-{f1(s.af_long[0])} um around {f3(zg)} "
                         "(has the focus drive slipped?)")
                r = self.autofocus(zg, s.af_long, extend=0)
            tol = s.af_max_dev_um[1 if (len(P["seg_focused"]) >= 5 and not wide) else 0] + s.af_dev_per_mm * min(nearest_dist(x, y, P["seg_focused"]), 1e5) / 1000
            tol = min(tol, s.af_max_dev_cap)
            if r[2] and known and abs(r[0] - zg) > tol:
                # a focus far from the prediction: either the focus has moved (the Z drive slipped; seen in 20261007_run_2)
                # or the sweep caught something else.  Kept when nothing is found near the prediction and the same focus
                # is found again; then the prediction is reset to it (a focus jump, logged)
                dev, far = r[0] - zg, r
                if not isnan(P["last_rej"]) and abs(dev - P["last_rej"]) <= s.af_agree_um:
                    self.say(f"    autofocus result {f3(r[0])} is {f3(dev)} um from the prediction, as at the previous flake ({f3(P['last_rej'])})")
                    jump = dev
                else:
                    self.say(f"    autofocus result {f3(r[0])} is {f3(dev)} um from the prediction (limit {f1(tol)}): checking around the prediction")
                    r2 = self.autofocus(zg, (tol, s.af100[1] / 2, s.af100[2], s.af100[3], s.af100[4]))
                    if r2[2] and abs(r2[0] - zg) <= tol:
                        self.say("    focus found near the prediction: the far result is not used")
                        r = r2
                    else:
                        r3 = r2 if (r2[2] and abs(r2[0] - far[0]) <= s.af_agree_um) else self.autofocus(far[0], s.af100)
                        if r3[2] and abs(r3[0] - far[0]) <= s.af_agree_um:
                            self.say(f"    the same focus found again ({f3(far[0])}, {f3(r3[0])})")
                            r, jump = r3, r3[0] - zg
                        else:
                            self.say("    not found again: the far result is not used")
                            rej = dev
                            r = (zg, max(far[1], r2[1], r3[1]), False)
            ratio = r[1]
            if r[2]:
                how = "auto"
            elif asking:
                self.set_z(zg)
                self.alert(image=self.target_picture(P, lab), what=f"Autofocus failed on {lab} ({s.name}, 100x): focus by hand within {int(s.focus_dialog_timeout_s)} s, or the "
                           "scan goes on at the predicted focus.")
                a = self.ask_focus(f"{s.name} 100x", f"Autofocus failed on {lab}. Check the stage is on this flake (picture), then focus by hand.",
                                   s.focus_dialog_timeout_s, image=self.target_picture(P, lab))
                if a == 0:
                    how = "manual"
                elif a == 2:
                    P["stop"] = True
                elif a == 3:
                    self.set_z(zg)
                    how = "predicted"
                    P["no_ask"] = True
                    self.say(f"    no answer within {int(s.focus_dialog_timeout_s)} s: imaged at the predicted Z" +
                             ("" if s.focus_dialog_repeat else f"; no more focus questions on {s.name} (flakes not focused are "
                                                               "imaged at the predicted Z and retaken at the end of the piece)"))
            else:
                self.set_z(zg)
                how = "predicted"
                self.say(f"    autofocus failed: imaged at the predicted Z {f3(zg)} (focus \"predicted\" in the index)")
                self.check_z_reach(f"100x {lab}")
        else:
            self.set_z(zg)
            a = self.ask_focus(f"{s.name} 100x", f"Flake {k + 1}: {lab}", image=self.target_picture(P, lab))
            if a == 0:
                how = "manual"
            elif a == 2:
                P["stop"] = True
        if retake and how != "auto" and (manual_first or (ctl is not None and hasattr(ctl, "take") and ctl.take(MANUAL_RETAKE))):
            # "Manual focus (retakes)": this retake focused by hand
            self.set_z(zg)
            self.say(f"    {lab}: focused by hand ({MANUAL_RETAKE})")
            a = self.ask_focus(f"{s.name} 100x: retake by hand", f"Retake of {lab}: focus it by hand with the fine focus knob "
                               "(check the stage is on this flake: picture), then press \"Use this focus\".  \"Skip\": imaged at "
                               "the predicted focus; \"Stop\": no more flakes on this piece.", image=self.target_picture(P, lab))
            if a == 0:
                how = "manual"
            elif a == 2:
                P["stop"] = True
            elif not how:
                self.set_z(zg)
                how = "predicted"
        if how in ("auto", "manual"):
            P["fail_run"], P["last_rej"], P["long_tried"] = 0, NAN, False
        else:
            P["fail_run"], P["last_rej"] = P["fail_run"] + 1, rej
        P["results"].append((t[0], t[1], how, lab))
        if how == "predicted" and not retake:
            P.setdefault("failed", []).append((t, k, P["n_ok"]))      # (retake_failed: with the focused count then)
        if not how:
            self.say("    skipped")
            return not P["stop"]
        if not pre:
            if rc or s.hundred_recentre:
                recentred = self._recentre(P, rc, lab, x, y)
        elif recentred == "unsure" and how != "predicted":
            recentred = self._recentre(P, rc, lab, x, y)          # (no confident match out of focus: once more, focused)
        zz = self.z_read()
        off = NAN
        if how == "manual" and known and abs(zz - zg) > tol:
            jump = zz - zg                                   # focused by hand far from the prediction: the focus has moved too
        if how != "predicted":
            if not isnan(jump):
                P["seg_offs"], P["seg_focused"] = [], []
            P["focused"].append((x, y, zz))
            P["seg_focused"].append((x, y, zz))
            P["n_ok"] += 1
            if not isnan(z10):
                off = zz - z10
                P["offs"].append((x, y, off))
                P["seg_offs"].append((x, y, off))
                self.recent_offs.append(off)
                self.say(f"    focus offset here (100x - 10x): {f3(off)} um")
            if not isnan(jump):
                j = dict(time=time.strftime("%H:%M:%S"), piece=s.name, label=lab, jump=jump, off_before=off_known, off_after=off)
                P["jumps"].append(j)
                self.focus_jumps.append(j)
                self.say(f"    FOCUS JUMP: the focus is {f3(jump)} um from the prediction" +
                         (f" (100x - 10x offset {f3(off_known)} -> {f3(off)} um)" if not isnan(off) and not isnan(off_known) else "") +
                         "; the next flakes are predicted from this one on (focus jumps so far: " + str(len(self.focus_jumps)) + ")")
        nfr = max(1, int(s.hundred_frames))
        frames_ = [self.snap_good(lab if nfr == 1 else f"{lab} frame {j + 1}", after_move=j == 0) for j in range(nfr)]
        sat = max(float((f_ >= fs - 1).mean()) for f_ in frames_) if frames_[0].ndim == 2 else 0.0
        if nfr == 1:
            im = frames_[0]
        else:                                       # mean of glitch-checked frames, kept 16-bit (raw Bayer)
            im = np.round(np.mean(np.asarray(frames_, np.float32), axis=0)).astype(frames_[0].dtype)
        fn = image_name(s.name, lab, "100x", k + 1, P.get("prefix", ""))
        lines = P.setdefault("idx_lines", {})                    # file -> index row (a retake replaces its row)
        new_file = fn not in lines
        sx0, sy0 = c.getXPosition(self.xy), c.getYPosition(self.xy)
        opos, oname = self.objective()
        info = {"label": lab, "stage_x_um": sx0, "stage_y_um": sy0, "z_um": zz, "z_drive": zz * self.z_scale, "focus": how,
                "objective": oname, "nosepiece_position": "" if opos is None else opos,
                "af_ratio": ratio, "exposure_ms": c.getExposure(), "camera": cam,
                "format": c.getProperty(cam, "Data format") if c.hasProperty(cam, "Data format") else "", "frames": nfr,
                "bit_depth": fs, "bayer": "RGGB (R at top-left)"}
        self.save_image(img_dir / fn, im, info)
        # the pause check's reference: the last focused picture; a predicted one (no autofocus at each flake) when there is
        # no focused 100x picture yet (2026-10-27: a pause at 100x had none, a coarse-knob change went unseen)
        lp = self.last_pic
        if how in ("auto", "manual") or lp is None or not lp.get("hundred") or lp.get("how") == "predicted":
            self.last_pic = dict(frame=im, x=float(sx0), y=float(sy0), z=float(zz), hundred=True, what=f"100x {lab}", how=how)
        if s.hundred_save_frames and nfr > 1:            # (processing development: the single frames as well)
            for j, f_ in enumerate(frames_):
                self.save_image(img_dir / fn.replace(".tif", f"_f{j + 1}.tif"), f_, dict(info, frames=1, frame=j + 1))
        P["n_img"] += 1 if new_file else 0
        shf, shu = [], []
        for j, (dx, dy) in enumerate(s.hundred_shifts_um or ()):
            # every shifted frame reached from -x / -y too (with the default shifts the second one is a move in -x: it
            # goes hundred_shift_approach_um short first)
            try:
                self.approach_to(sx0 + dx, sy0 + dy, a=s.hundred_shift_approach_um)
            except StageLimit:
                self.say("    the shifted frames are beyond the stage's travel: the averaged frame alone")
                shf, shu = [], []
                break
            ims = self.snap_good(f"{lab} shift {j + 1}")
            fns = fn.replace(".tif", f"_s{j + 1}.tif")
            self.save_image(img_dir / fns, ims, {"label": lab, "shift": j + 1, "stage_x_um": c.getXPosition(self.xy),
                                                  "stage_y_um": c.getYPosition(self.xy), "z_um": zz, "exposure_ms": c.getExposure(),
                                                  "camera": cam, "bayer": "RGGB (R at top-left)"})
            shf.append(fns)
            shu.append(f"{f3(c.getXPosition(self.xy) - sx0)}:{f3(c.getYPosition(self.xy) - sy0)}")
        lines[fn] = (f"{k + 1},{lab},{sx0},{sy0},{zz},{zz * self.z_scale},{'' if isnan(z10) else z10},{'' if isnan(off) else off},"
                     f"{how},{f3(ratio)},{sat},{fn},{';'.join(shf)},{';'.join(shu)},{c.getExposure()},{nfr},{recentred}\r\n")
        if new_file:
            idx.write(lines[fn])
        else:                                       # a retake: the index written again with its row replaced
            idx.seek(0); idx.truncate()
            idx.write(IDX100_HEADER + "".join(lines.values()))
        idx.flush()
        self.say(f"    Z {f3(zz)} ({how}), saved {fn}" + (f" (mean of {nfr} frames)" if nfr > 1 else "") + (f" + {len(shf)} shifted" if shf else "")
                 + (f"   WARNING: {f3(sat * 100)} % of pixels saturated" if sat > 0.001 else ""))
        self.ui.progress(self, f"{s.name}: 100x flake {k + 1}{of}{tag}")
        if getattr(t, "poly", None) is not None and s.hundred_mosaic:
            try:
                self.mosaic100(P, t, fn, (sx0, sy0, zz), how, nfr if s.hundred_mosaic_frames <= 0 else int(s.hundred_mosaic_frames))
            except Stop:
                raise
            except Exception as e:                   # (never costs the flake: its main picture is saved)
                if self.stop_all:
                    raise
                self.say(f"    mosaic of {lab} not finished ({type(e).__name__}: {e})")
        if self.material == "graphite" and not isnan(th) and th >= self.domain_min_nm and (s.domain_max_nm <= 0 or th <= s.domain_max_nm):
            self.domain_list = [d for d in self.domain_list if not (d["piece"] == s.name and d["label"] == lab)]     # (a retake)
            # tilt: the chip's (as the 100x mosaic fields), else a plane through the piece's focused 100x points: the IR
            # mosaic fields' focus is predicted with it (the chip's slope is the same for both cameras)
            pf = None if P.get("chip_tilt") else fit_plane([q for q in P.get("focused") or []][-30:])
            tilt = tuple(P["chip_tilt"]) if P.get("chip_tilt") else ((float(pf[0]), float(pf[1])) if pf is not None else None)
            self.domain_list.append(dict(q=(x, y, zz, th), label=lab, piece=s.name, dir=out_dir, how=how, poly=getattr(t, "poly", None),
                                         tilt=tilt))
        # (the target's position in its list, not where the stage went: the re-centring carried from the flakes before
        # moves that; the final list's flakes are compared with these, new_only: 20261027_run_6 imaged 14 flakes twice)
        P["imaged"].append((float(t[0]), float(t[1])))
        return not P["stop"]

    # ---- mosaics of large terraces (flakeid/fieldmosaic.py)
    def mosaic_budget(self, kind):
        """Seconds left of this run's mosaic time (kind "100x" / "IR"); <= 0: used up (said once)."""
        s = self.s
        left = 60.0 * (s.hundred_mosaic_budget_min if kind == "100x" else s.domain_mosaic_budget_min) - self.mosaic_s[kind]
        if left <= 0 and kind not in self.mosaic_over:
            self.mosaic_over.add(kind)
            self.say(f"    {kind} mosaic time budget used up ({f1(self.mosaic_s[kind] / 60)} min, "
                     f"{'hundred' if kind == '100x' else 'domain'}_mosaic_budget_min): the remaining flakes get their single {kind} picture")
        return left

    def plan_mosaic(self, poly, M, wh, lab, kind):
        """The extra fields of a terrace mosaic around the stage position the main picture was taken at: [(i, j, dx, dy)]."""
        s = self.s
        fields, n_all = plan_fields(poly, M, wh, s.hundred_mosaic_overlap, max(1, int(s.hundred_mosaic_max_tiles)))
        if not fields:
            return []
        fw, fh = field_size_um(M, wh)
        ext = np.ptp(poly, axis=0)
        self.say(f"    terrace {f1(ext[0])} x {f1(ext[1])} um is larger than the {kind} field ({f1(fw)} x {f1(fh)} um): mosaic of "
                 f"{len(fields) + 1} pictures" + (f" (needs {n_all}; capped at hundred_mosaic_max_tiles {s.hundred_mosaic_max_tiles})"
                                                  if n_all > len(fields) + 1 else ""))
        return fields

    def _mosaic_ok(self, x, y):
        b = self.move_box
        if b is not None and (x < b[0] or x > b[1] or y < b[2] or y > b[3]):
            return False
        return self.reachable(x, y)

    def mosaic100(self, P, t, fn, main_xyz, how, nfr):
        """After a flake's main 100x picture: the other fields its chosen terrace reaches into (t.poly), each focused from
        the main picture's focus + the tilt and one short sweep (hundred_mosaic_af), nfr frames averaged, no shifted frames;
        saved as <main>_m02.tif ... and listed in <piece>_100x_mosaic.csv (the main picture as k = 1).  Only after a
        focused main picture (a predicted one is retaken at the end of the piece, and its mosaic then); within the run's
        time budget; "Skip" / "Stop" end it after the current field."""
        s, c = self.s, self.core
        lab = t[4]
        if how not in ("auto", "manual"):
            self.say("    (no mosaic: the main picture is not focused)")
            return
        if self.mosaic_budget("100x") <= 0:
            return
        M100 = np.asarray(self.M, float) / 9.54
        sx0, sy0, z0 = main_xyz
        fields = self.plan_mosaic(t.poly, M100, (self.W, self.H), lab, "100x")
        if not fields:
            return
        t0 = time.time()
        if P.get("mos_idx") is None:
            P["mos_idx"] = open(P["img_dir"] / f"{s.name}_100x_mosaic.csv", "w", newline="")
            P["mos_idx"].write(MOSAIC100_HEADER)
        mi, mm = P["mos_idx"], fmt_matrix(M100)
        mi.write(f"{fn},{fn},1,{lab},{sx0},{sy0},{z0},{how},,{max(1, int(s.hundred_frames))},{c.getExposure()},,{mm},{self.W},{self.H},0,0\r\n")
        pts = [(sx0, sy0, z0)]
        fs, ctl, n_ok, n_img = P["fs"], P.get("ctl"), 0, 0
        af = tuple(s.hundred_mosaic_af or ())
        try:
            for kk, (gi, gj, dx, dy) in enumerate(fields):
                v = ctl.value() if ctl else None
                if self.stop_all or P["stop"] or self.stop_100x or v in ("Skip the rest of this piece", "Stop 100x imaging"):
                    self.say(f"    mosaic stopped after {n_img + 1} pictures")
                    break
                self.ui.check_stop(self)
                if self.mosaic_s["100x"] + time.time() - t0 >= 60.0 * s.hundred_mosaic_budget_min:
                    self.say(f"    mosaic stopped after {n_img + 1} pictures: time budget")
                    break
                x, y = sx0 + dx, sy0 + dy
                if not self._mosaic_ok(x, y):
                    self.say(f"    mosaic field {kk + 2} at {f1(x)}, {f1(y)} is beyond the stage's travel / allowed area: left out")
                    continue
                try:
                    self.approach_to(x, y)
                except StageLimit as e:
                    self.say(f"    mosaic field {kk + 2}: {e}; left out")
                    continue
                zg = self.predict_z(x, y, pts, P["chip_tilt"], z0)
                fhow, ratio = "predicted", 0.0
                if s.focus_mode == "auto" and len(af) >= 3:
                    r = self.autofocus_fit(zg, af, s.hundred_mosaic_af_min_rise, extend=1)
                    ratio = r[1]
                    if r[2] and abs(r[0] - zg) <= s.hundred_mosaic_max_dev_um:
                        fhow = "auto"
                if fhow == "predicted":
                    self.set_z(zg - s.af_backlash_um); self.set_z(zg)
                zz = self.z_read()
                if fhow == "auto":
                    pts.append((x, y, zz)); n_ok += 1
                frames_ = [self.snap_good(f"{lab} mosaic {kk + 2}" if j == 0 else f"{lab} mosaic {kk + 2} frame {j + 1}", after_move=j == 0)
                           for j in range(nfr)]
                sat = max(float((f_ >= fs - 1).mean()) for f_ in frames_) if frames_[0].ndim == 2 else 0.0
                im = frames_[0] if nfr == 1 else np.round(np.mean(np.asarray(frames_, np.float32), axis=0)).astype(frames_[0].dtype)
                fm = fn.replace(".tif", f"_m{kk + 2:02d}.tif")
                ax, ay = c.getXPosition(self.xy), c.getYPosition(self.xy)
                self.save_image(P["img_dir"] / fm, im, {"label": lab, "mosaic_of": fn, "mosaic_k": kk + 2, "stage_x_um": ax, "stage_y_um": ay,
                                                        "z_um": zz, "focus": fhow, "af_ratio": ratio, "exposure_ms": c.getExposure(),
                                                        "frames": nfr, "um_per_px": mm, "bit_depth": fs, "bayer": "RGGB (R at top-left)"})
                mi.write(f"{fn},{fm},{kk + 2},{lab},{ax},{ay},{zz},{fhow},{f3(ratio)},{nfr},{c.getExposure()},{sat},{mm},{self.W},{self.H},{gi},{gj}\r\n")
                mi.flush()
                n_img += 1
                if s.verbose:
                    self.say(f"    mosaic {kk + 2}/{len(fields) + 1} at {f1(ax)}, {f1(ay)}: Z {f3(zz)} ({fhow}, {f3(zz - zg)} from the prediction)")
        finally:
            el = time.time() - t0
            self.mosaic_s["100x"] += el
            mi.flush()
            self.say(f"    mosaic: {n_img} more pictures ({n_ok} focused by a sweep) in {f1(el)} s; mosaic time this run "
                     f"{f1(self.mosaic_s['100x'] / 60)} of {f1(s.hundred_mosaic_budget_min)} min")

    def mosaic_ir(self, e, ddir, fn, main_xyz, how, didx, fsD, camD, q):
        """The IR picture's mosaic over the terrace (as mosaic100, with the domain camera): <IR main>_m02.tif ... in
        <piece>_domains_mosaic.csv.  Field size and directions from domain_px_um_matrix (else the 100x camera's at
        0.358 IR px per 100x px); focus: the main IR picture's (+ domain_mosaic_af's sweep if set); its own budget (domain_mosaic_budget_min)."""
        s, c = self.s, self.core
        lab = e["label"]
        if e.get("poly") is None or not s.hundred_mosaic or s.domain_mosaic_budget_min <= 0:
            return
        if how not in ("auto", "manual") or self.mosaic_budget("IR") <= 0:
            return
        MD = np.asarray(s.domain_px_um_matrix, float).reshape(2, 2) if s.domain_px_um_matrix else np.asarray(self.M, float) / 9.54 / 0.358
        sx0, sy0, z0 = main_xyz
        fields = self.plan_mosaic(e["poly"], MD, (self.W, self.H), lab, "IR")
        if not fields:
            return
        t0 = time.time()
        if didx.get("f") is None or didx.get("piece") != e["piece"]:
            if didx.get("f") is not None:
                didx["f"].close()
            didx["f"] = open(ddir / f"{e['piece']}_domains_mosaic.csv", "a" if didx.get("seen", set()) & {e["piece"]} else "w", newline="")
            didx.setdefault("seen", set()).add(e["piece"])
            didx["piece"] = e["piece"]
            if didx["f"].tell() == 0:
                didx["f"].write(MOSAIC100_HEADER)
        mi, mm = didx["f"], fmt_matrix(MD)
        nfr = max(1, s.domain_frames)
        mi.write(f"{fn},{fn},1,{lab},{sx0},{sy0},{z0},{how},,{nfr},{c.getExposure()},,{mm},{self.W},{self.H},0,0\r\n")
        pts, n_img, n_ok = [(sx0, sy0, z0)], 0, 0
        af = tuple(s.domain_mosaic_af or ())
        try:
            for kk, (gi, gj, dx, dy) in enumerate(fields):
                if self.stop_all:
                    break
                self.ui.check_stop(self)
                if self.mosaic_s["IR"] + time.time() - t0 >= 60.0 * s.domain_mosaic_budget_min:
                    self.say(f"    IR mosaic stopped after {n_img + 1} pictures: time budget")
                    break
                x, y = sx0 + dx, sy0 + dy
                if not self._mosaic_ok(x, y):
                    self.say(f"    IR mosaic field {kk + 2} at {f1(x)}, {f1(y)} is beyond the stage's travel / allowed area: left out")
                    continue
                try:
                    self.approach_to(x, y)
                except StageLimit as ex:
                    self.say(f"    IR mosaic field {kk + 2}: {ex}; left out")
                    continue
                zg = self.predict_z(x, y, pts, e.get("tilt"), z0)
                fhow, ratio = "predicted", 0.0
                if s.focus_mode == "auto" and len(af) >= 3:
                    r = self.autofocus_fit(zg, af, s.domain_af_min_rise, extend=1)
                    ratio = r[1]
                    if r[2] and abs(r[0] - zg) <= s.domain_max_dev_um:
                        fhow = "auto"
                if fhow == "predicted":
                    self.set_z(zg - s.af_backlash_um); self.set_z(zg)
                zz = self.z_read()
                if fhow == "auto":
                    pts.append((x, y, zz)); n_ok += 1
                self.stop_streaming()
                frames, sat = [], 0.0
                for j in range(nfr):
                    im = self.frame(after_move=j == 0)
                    if im.ndim == 2:
                        sat = max(sat, float((im >= fsD - 1).mean()))
                    frames.append(im)
                img = frames[0] if len(frames) == 1 else np.mean(np.asarray(frames, np.float32), axis=0).astype(np.float32)
                fm = fn.replace(".tif", f"_m{kk + 2:02d}.tif")
                ax, ay = c.getXPosition(self.xy), c.getYPosition(self.xy)
                self.save_image(ddir / fm, img, {"label": lab, "mosaic_of": fn, "mosaic_k": kk + 2, "thickness_nm": q[3], "stage_x_um": ax,
                                                 "stage_y_um": ay, "z_um": zz, "focus": fhow, "af_ratio": ratio, "exposure_ms": c.getExposure(),
                                                 "frames": nfr, "camera": camD, "bit_depth": fsD, "um_per_px": mm, "configuration": "stacking domains"})
                mi.write(f"{fn},{fm},{kk + 2},{lab},{ax},{ay},{zz},{fhow},{f3(ratio)},{nfr},{c.getExposure()},{sat},{mm},{self.W},{self.H},{gi},{gj}\r\n")
                mi.flush()
                n_img += 1
                if s.verbose:
                    self.say(f"    IR mosaic {kk + 2}/{len(fields) + 1} at {f1(ax)}, {f1(ay)}: Z {f3(zz)} ({fhow}, {f3(zz - zg)} from the prediction)")
        finally:
            el = time.time() - t0
            self.mosaic_s["IR"] += el
            mi.flush()
            self.say(f"    IR mosaic: {n_img} more pictures ({n_ok} focused by a sweep) in {f1(el)} s; IR mosaic time this run "
                     f"{f1(self.mosaic_s['IR'] / 60)} of {f1(s.domain_mosaic_budget_min)} min")

    # ---- stacking domains (graphite)
    def image_domains(self):
        self.set_phase("stacking domains")
        return self._image_domains()

    def setup_domain_camera(self):
        """Select the stacking-domain (IR) camera and set it up as it needs (woken up, domain_camera_props, Frame Average
        1, the exposure).  Returns (scan camera, its exposure, its frame size, the domain camera, its full scale)."""
        s, c = self.s, self.core
        self.stop_streaming()
        cam_scan, exp_scan, wh_scan = c.getCameraDevice(), c.getExposure(), (self.W, self.H)
        if s.domain_camera and s.domain_camera != cam_scan:
            try:
                c.setCameraDevice(s.domain_camera)
            except Exception as e:
                raise self.fail(f"cannot select the domain camera {s.domain_camera}: {e}")
        camD = c.getCameraDevice()
        if s.domain_wake:
            nwk = self.wake_camera()
            self.say(f"  {camD} woken up (live for a moment: {nwk} frames), then its settings")
        self.set_props(camD, s.domain_camera_props)
        try:                                             # (the RaptorEPIX driver's own averaging leaves the camera stuck)
            if c.hasProperty(camD, "Frame Average") and str(c.getProperty(camD, "Frame Average")).strip() not in ("1", "1.0"):
                c.setProperty(camD, "Frame Average", "1")
                self.say(f"    {camD} Frame Average set to 1: the driver's averaging is unreliable; {max(1, s.domain_frames)} frames "
                         "are averaged by the scan (domain_frames)")
        except Exception as e:
            self.say(f"    (Frame Average not checked: {e})")
        self.stream_off()
        c.setExposure(s.domain_exposure_ms); c.waitForDevice(camD)
        self.W, self.H = int(c.getImageWidth()), int(c.getImageHeight())
        fsD = 255.0 if int(c.getBytesPerPixel()) == 4 else 2.0 ** float(c.getImageBitDepth()) - 1
        self.mono_view = True                            # (the live view: a monochrome camera, not a colour mosaic)
        self.frame_flip = (s.domain_flip or "").lower()
        if self.frame_flip:
            self.say(f"  domain camera pictures flipped in {' and '.join(self.frame_flip.upper())} (domain_flip), as the scan camera sees them")
        return cam_scan, exp_scan, wh_scan, camD, fsD

    def restore_scan_camera(self, cam_scan, exp_scan, wh_scan, camD):
        """The scan camera selected again after the domain camera."""
        self.mono_view = False
        self.frame_flip = ""
        if self.s.domain_restore_camera and camD != cam_scan:
            try:
                self.stream_off()
                self.core.setCameraDevice(cam_scan); self.core.setExposure(exp_scan)
                self.W, self.H = wh_scan
                self.say(f"  camera {cam_scan} selected again")
            except Exception as e:
                self.say(f"  (could not select {cam_scan} again: {e})")

    def _image_domains(self):
        s, c = self.s, self.core
        any_graphite = any(p["material"] == "graphite" for p in self.pieces) or (s.mode == "targets" and s.material == "graphite")
        if not any_graphite or s.domain_imaging == "no" or self.stop_all or s.mode not in ("multi", "targets"):
            return
        self.say("")
        if not self.domain_list:
            self.say(f"==== stacking domains: no graphite flake of {s.domain_min_layers}+ layers (>= {f3(self.domain_min_nm)} nm) was imaged at 100x")
            return
        npieces = len({d["piece"] for d in self.domain_list})
        what = (f"{len(self.domain_list)} graphite flake{'s' if len(self.domain_list) > 1 else ''} of {s.domain_min_layers} layers or more"
                + (f" on {npieces} pieces" if npieces > 1 else "") + " were imaged at 100x.")
        go = True
        if s.domain_imaging == "ask":
            self.alert("Stacking-domain imaging: answer the question in the scan program.  " + what)
            go = self.ui.confirm("Rhombohedral domains", "Do you want to image rhombohedral domains?\n\n" + what, ("Yes", "No")) == 0
        if go:
            ir_pos = (s.objective_positions or {}).get("domain")
            self.alert("Change to the IR objective" + (f" (nosepiece position {ir_pos})" if ir_pos else "") + ", camera and filter for the "
                       "stacking-domain images, then press OK in the scan program.")
            go = self.ui.confirm("Rhombohedral domains: change to the IR camera",
                                 "Change the setup to the IR camera:\n   - the IR objective" + (f" (nosepiece position {ir_pos})" if ir_pos else "")
                                 + "\n   - the IR camera" + (f" ({s.domain_camera}, selected by the scan)" if s.domain_camera else "")
                                 + "\n   - the IR filter\n\nDo not move the stage or the focus.  Press OK when everything is in place.",
                                 ("OK", "Cancel")) == 0
            if go:
                self.ui.mock_hook(self, "mockDomain")
                if self.wait_objective("domain", "Put in the IR objective (the window closes by itself when it is in).") is False:
                    go = False
        if not go:
            self.say(f"==== stacking domains: not imaged ({what})")
            return
        self.say(f"==== stacking domains: {what}")
        if self.manifest:
            self.write_manifest(self.manifest, self.pieces, "domains")
        cam_scan, exp_scan, wh_scan, camD, fsD = self.setup_domain_camera()
        off_xy = s.domain_xy_offset_um
        self.say(f"  camera {camD} {self.W} x {self.H}, exposure {f3(c.getExposure())} ms, full scale {int(fsD)}; XY offset {f3(off_xy[0])}, "
                 f"{f3(off_xy[1])} um, focus offset " + ("not known (found at the first flake)" if isnan(s.domain_focus_offset_um) else f"{f3(s.domain_focus_offset_um)} um"))
        Q = np.array([d["q"] for d in self.domain_list])
        mg = s.target_box_margin_um
        self.move_box = (Q[:, 0].min() + off_xy[0] - mg, Q[:, 0].max() + off_xy[0] + mg, Q[:, 1].min() + off_xy[1] - mg, Q[:, 1].max() + off_xy[1] + mg)
        dom_offs = []
        cur, didx, nD, nDok, stopD, ddir, no_askD = "", None, 0, 0, False, None, False
        dmos = {}                                        # the IR mosaic index (mosaic_ir): its file, piece
        try:
            for k, e in enumerate(self.domain_list):
                if stopD:
                    break
                self.pause_point()
                q, lab, how100 = e["q"], e["label"], e["how"]
                if cur != e["piece"]:
                    if didx:
                        didx.close()
                    cur = e["piece"]
                    ddir = Path(e["dir"]) / f"{cur}_domains"
                    ddir.mkdir(parents=True, exist_ok=True)
                    didx = open(ddir / f"{cur}_domains_index.csv", "w", newline="")
                    didx.write("order,label,thickness_nm,stage_x_um,stage_y_um,z_um,z100_um,offset_um,focus,af_ratio,frames,exposure_ms,saturated_frac,file\r\n")
                    self.say(f"  piece {cur} -> {ddir}")
                dx, dy = q[0] + off_xy[0], q[1] + off_xy[1]
                self.say(f"  [{k + 1}/{len(self.domain_list)}] {lab} ({f1(q[3])} nm, {f1(q[3] / 0.335)} layers) at {f1(dx)}, {f1(dy)}")
                self.approach_to(dx, dy)
                offk = idw_offset(dx, dy, dom_offs) if dom_offs else s.domain_focus_offset_um
                known = not isnan(offk)
                zg = q[2] + (offk if known else 0.0)
                self.zref = zg
                how, ratio = "", 0.0
                if s.focus_mode == "auto":
                    if not known:
                        r = self.autofocus(zg, s.domain_search)
                    else:
                        if s.domain_af_fit and how100 != "predicted":     # (one sweep: autofocus_fit)
                            r = self.autofocus_fit(zg, s.domain_af_fit, s.domain_af_min_rise)
                        else:
                            r = self.autofocus(zg, s.domain_af_first if how100 == "predicted" else s.domain_af)
                        if not r[2]:
                            r = self.autofocus(zg, s.domain_af_first)
                        if r[2] and abs(r[0] - zg) > s.domain_max_dev_um and how100 != "predicted":
                            self.say(f"    autofocus result {f3(r[0])} is {f3(r[0] - zg)} um from the prediction: rejected, retrying around it")
                            r = self.autofocus(zg, (s.domain_max_dev_um, s.domain_af[1] / 2, s.domain_af[2], s.domain_af[3], s.domain_af[4]))
                            if r[2] and abs(r[0] - zg) > s.domain_max_dev_um:
                                r = (zg, r[1], False)
                    ratio = r[1]
                    if r[2]:
                        how = "auto"
                    elif self.ask_on_fail(len(dom_offs)) and self.ui.can_ask and (s.focus_dialog_repeat or not no_askD):
                        self.set_z(zg)
                        self.alert(f"Autofocus failed on {lab} (domain camera): focus by hand within {int(s.focus_dialog_timeout_s)} s.")
                        a = self.ask_focus(f"{cur} domains", f"Autofocus failed on {lab} (domain camera). Focus by hand?", s.focus_dialog_timeout_s)
                        if a == 0:
                            how = "manual"
                        elif a == 2:
                            stopD = True
                        elif a == 3:
                            no_askD = True                     # (not asked again: focus_dialog_repeat)
                            if known:
                                self.set_z(zg); how = "predicted"
                    elif known:
                        self.set_z(zg); how = "predicted"
                else:
                    self.set_z(zg)
                    a = self.ask_focus(f"{cur} domains", f"Flake {k + 1} of {len(self.domain_list)}: {lab} (domain camera)")
                    if a == 0:
                        how = "manual"
                    elif a == 2:
                        stopD = True
                if not how:
                    self.say("    skipped")
                    continue
                zz = self.z_read()
                off = zz - q[2]
                if how != "predicted" and how100 != "predicted":
                    dom_offs.append((dx, dy, off))
                    nDok += 1
                    self.say(f"    focus offset here (domain - 100x): {f3(off)} um")
                self.stop_streaming()
                frames, sat = [], 0.0
                for j in range(max(1, s.domain_frames)):
                    im = self.frame(after_move=j == 0)
                    if im.ndim == 2:
                        sat = max(sat, float((im >= fsD - 1).mean()))
                    frames.append(im)
                img = frames[0] if len(frames) == 1 else np.mean(np.asarray(frames, np.float32), axis=0).astype(np.float32)
                fn = image_name(cur, lab, "IR", k + 1)
                self.save_image(ddir / fn, img, {"label": lab, "thickness_nm": q[3], "stage_x_um": c.getXPosition(self.xy),
                                                 "stage_y_um": c.getYPosition(self.xy), "z_um": zz, "z100_um": q[2], "focus": how, "af_ratio": ratio,
                                                 "exposure_ms": c.getExposure(), "frames": max(1, s.domain_frames), "camera": camD,
                                                 "bit_depth": fsD, "configuration": "stacking domains"})
                nD += 1
                didx.write(f"{k + 1},{lab},{q[3]},{c.getXPosition(self.xy)},{c.getYPosition(self.xy)},{zz},{q[2]},{off},{how},{f3(ratio)},"
                           f"{max(1, s.domain_frames)},{c.getExposure()},{sat},{fn}\r\n")
                didx.flush()
                self.say(f"    Z {f3(zz)} ({how}), saved {fn}" + (f"   WARNING: {f3(sat * 100)} % of pixels saturated" if sat > 0.001 else ""))
                try:
                    self.mosaic_ir(e, ddir, fn, (c.getXPosition(self.xy), c.getYPosition(self.xy), zz), how, dmos, fsD, camD, q)
                except Stop:
                    raise
                except Exception as ex:              # (the flake's IR picture is saved)
                    if self.stop_all:
                        raise
                    self.say(f"    IR mosaic of {lab} not finished ({type(ex).__name__}: {ex})")
        finally:
            if didx:
                didx.close()
            if dmos.get("f") is not None:
                dmos["f"].close()
        if dom_offs:
            ov = np.sort([o[2] for o in dom_offs])
            self.say(f"  focus offset domain - 100x: median {f3(float(np.median(ov)))} um ({len(ov)} flakes, range {f3(ov[0])} .. {f3(ov[-1])}).  "
                     f"Next time: domain_focus_offset_um = {f3(float(np.median(ov)))}")
        self.say(f"  {nD} domain images ({nDok} focused)")
        self.move_box, self.zref = None, NAN
        self.restore_scan_camera(cam_scan, exp_scan, wh_scan, camD)
        self.say("  the domain objective, camera and filter are still in: change back to the scan configuration before the next run")

    # ---- flat field
    def run_flat(self):
        s, c = self.s, self.core
        if s.flat_objective not in ("10x", "100x"):
            raise self.fail('flat_objective must be "10x" or "100x"')
        gui = s.flat_gui
        if gui:                                 # started from the start screen: frames go to the local folder
            s.out_dir = str(Path(s.local_root) / "flat_fields")
            s.name = f"flat_{s.flat_objective}" + (f"_{s.flat_oxide_nm:g}nm" if s.flat_objective == "100x" else "") + \
                datetime.datetime.now().strftime("_%Y%m%d_%H%M")
        Path(s.out_dir).mkdir(parents=True, exist_ok=True)
        self.open_log(Path(s.out_dir) / f"{s.name}_log.txt")
        fstop = s.flat_focus == "fieldstop" and self.has_z
        if fstop:
            self.ui.mock_hook(self, "mockFieldStop", True)
        if gui and self.ui.can_ask:
            a = self.ask_focus(f"Flat field ({s.flat_objective})", f"Move to a clean area of a bare chip with {s.flat_oxide_nm:g} nm oxide (no flakes, "
                               "no tape residue; the frames are taken around this point). " +
                               ("Close the field stop (F.S.) on the illuminator until its edge shows in the live view, focus on that edge, "
                                "then press \"Use this focus\"." if fstop else "Focus on the substrate, then press \"Use this focus\"."))
            if a != 0:
                raise self.fail("flat field cancelled")
            self.X0, self.Y0 = c.getXPosition(self.xy), c.getYPosition(self.xy)
            if self.has_z:
                self.zref = self.z_read()
        fs_ = 1.0 / 9.54 if s.flat_objective == "100x" else 1.0
        # positions only need to differ by more than the dust / particles to be removed: 0.6 of a field from the start
        # screen (25 frames fit in ~4 mm at 10x), a full field from the command line as before
        step = s.flat_step_um if s.flat_step_um > 0 else (0.6 if gui else 1.1) * max(self.FOVX * fs_, self.FOVY * fs_)
        flat_dir = Path(s.out_dir) / f"{s.name}_flat_{s.flat_objective}"
        n = 2
        while flat_dir.exists():
            flat_dir = Path(s.out_dir) / f"{s.name}_flat_{s.flat_objective}_{n}"; n += 1
        self.say(f"flat field ({s.flat_objective}, exposure {f3(c.getExposure())} ms): {s.flat_frames} frames {f1(step)} um apart around "
                 f"{f1(self.X0)}, {f1(self.Y0)}, focus {s.flat_focus} -> {flat_dir}")
        plane = self.fieldstop_plane(step) if fstop else None
        if fstop:
            self.move_to(self.X0, self.Y0)
            if plane:
                zc = plane[0] * self.X0 + plane[1] * self.Y0 + plane[2]
                self.set_z(zc - s.af_backlash_um); self.set_z(zc)
        self.REF = self.reference(self.blocks())
        self.say(f"  substrate R/G {f3(self.REF[0])}, B/G {f3(self.REF[1])}, G {f1(self.REF[2])} counts")
        zf = self.z_read() if self.has_z else 0.0
        fsc = self.full_scale()
        n_saved = n_tried = ring = 0
        self.flat_xy = []
        while n_saved < s.flat_frames and ring <= 12:
            pts = [(0, 0)] if ring == 0 else ([(i, -ring) for i in range(-ring, ring + 1)] + [(i, ring) for i in range(-ring, ring + 1)]
                                              + [(-ring, j) for j in range(-ring + 1, ring)] + [(ring, j) for j in range(-ring + 1, ring)])
            for gx, gy in pts:
                if n_saved >= s.flat_frames:
                    break
                px, py = self.X0 + gx * step, self.Y0 + gy * step
                if math.hypot(px - self.X0, py - self.Y0) > s.max_travel_um:
                    continue
                n_tried += 1
                self.move_to(px, py)
                if self.has_z and plane:
                    zf = plane[0] * px + plane[1] * py + plane[2]
                    self.set_z(zf - s.af_backlash_um); self.set_z(zf)
                elif self.has_z and s.flat_focus == "auto":
                    r = self.autofocus(zf, s.af100 if s.flat_objective == "100x" else s.af10)
                    if r[2]:
                        zf = r[0]
                    else:
                        self.set_z(zf)
                        self.say(f"    no focus signal here: kept Z {f3(zf)}")
                elif self.has_z:
                    self.set_z(zf - s.af_backlash_um); self.set_z(zf)
                bl = self.blocks()
                sf = self.substrate_fraction(bl)
                nsat = int(((bl[:, 3] + self.DARK >= fsc - 2) | (bl[:, 2] + self.DARK >= fsc - 2)).sum())
                if sf < s.flat_min_substrate or nsat:
                    self.say(f"    {f1(px)}, {f1(py)}: " + ("saturated pixels" if nsat else f"substrate fraction {f3(sf)}") + ", skipped")
                    continue
                lbl = f"Pos-{pad3(n_saved)}_000"
                self.save_tile(flat_dir / lbl, n_saved, lbl, n_saved, 0)
                self.flat_xy.append((px, py))
                n_saved += 1
                self.say(f"    frame {n_saved}/{s.flat_frames} at {f1(px)}, {f1(py)} (substrate {f3(sf)})")
            ring += 1
        self.move_to(self.X0, self.Y0)
        if self.has_z:
            self.set_z(plane[0] * self.X0 + plane[1] * self.Y0 + plane[2] if plane else zf)
        if n_saved < s.flat_frames:
            self.say(f"  WARNING: only {n_saved} clean positions found in {n_tried} tried: move to a cleaner area or lower flat_min_substrate")
        self.say(f"  {n_saved} flat frames in {flat_dir}")
        dark_dir = None
        if s.flat_dark_frames > 0:
            blocked = self.ui.confirm(f"{s.name}: dark frames", f"Flat frames done ({n_saved}).\n\nNow block all light to the camera (close the lamp "
                                      f"shutter or the light path), then press OK to record {s.flat_dark_frames} dark frames.\nCancel = no dark frames.",
                                      ("OK", "Cancel")) == 0
            if blocked:
                self.ui.mock_hook(self, "mockDark", True)
                lv = self.frame_level()
                if lv[0] > 0.05:
                    self.say(f"  WARNING: the 'dark' frame is {f3(lv[0] * 100)} % of full scale: is the light really blocked?")
                dark_dir = Path(s.out_dir) / f"{s.name}_dark_{s.flat_objective}"
                n = 2
                while dark_dir.exists():
                    dark_dir = Path(s.out_dir) / f"{s.name}_dark_{s.flat_objective}_{n}"; n += 1
                for k in range(s.flat_dark_frames):
                    lbl = f"Pos-{pad3(k)}_000"
                    self.save_tile(dark_dir / lbl, k, lbl, k, 0)
                self.say(f"  {s.flat_dark_frames} dark frames in {dark_dir} (unblock the light again)")
                self.ui.mock_hook(self, "mockDark", False)
            else:
                self.say("  no dark frames")
        if gui and n_saved >= 5:
            self.finish_flat(flat_dir, dark_dir)
        else:
            self.say("  build the flat field:")
            self.say(f"    python -m flakeid.mm flat \"{flat_dir}\"" + (f" --dark \"{dark_dir}\"" if dark_dir else "")
                     + f" -o flat_{s.flat_objective}_{int(round(c.getExposure()))}ms.npz")
        self.say("done")

    def fieldstop_visible(self):
        """Is the closed field stop's edge in the frame (a dark surround and a lit middle)?"""
        im = self.snap_good("field stop").astype(np.float32)
        g = im[..., 1] if im.ndim == 3 else im[0::2, 1::2] - self.DARK
        hi = float(np.percentile(g, 95))
        f = float((g < 0.3 * hi).mean()) if hi > 0 else 0.0
        return 0.03 < f < 0.95

    def edge_sharpness(self):
        """Focus metric for the field-stop edge: the strongest gradients (top 0.3 % of pixels, lightly smoothed against
        noise) over the lit level.  The whole-frame mean of the scan's autofocus is mostly noise when only a thin edge
        carries the signal."""
        import cv2
        a = self.frame().astype(np.float32)
        c = a.mean(axis=2) if a.ndim == 3 else a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2].reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3)) - self.DARK
        c = cv2.GaussianBlur(c, (0, 0), 1.0)
        gm = np.hypot(cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3))
        k = max(50, gm.size // 333)
        return float(np.partition(gm.ravel(), gm.size - k)[-k:].mean()) / max(float(np.percentile(c, 95)), 1.0)

    def fieldstop_focus(self, zc, p):
        """Z of the sharpest field-stop edge near zc (coarse sweep over +-p[0], extended once at an end, then fine), or None."""
        s = self.s
        def sweep(zs):
            self.set_z(zs[0] - s.af_backlash_um)
            return np.array([self.edge_sharpness() if self.set_z(z) is not False else -1.0 for z in zs])
        zs = zc + np.arange(-p[0], p[0] + 1e-9, p[1] / 2)
        m = sweep(zs)
        for _ in range(2):
            i = int(np.argmax(m))
            if 0 < i < len(zs) - 1:
                break
            zs = zs[i] + np.arange(-p[0], p[0] + 1e-9, p[1] / 2)
            m = sweep(zs)
        i = int(np.argmax(m))
        if not (0 < i < len(zs) - 1) or m[i] < 1.15 * max(float(np.min(m[m > 0])) if (m > 0).any() else 1e9, 1e-9):
            return None
        zf = zs[i] + np.arange(-p[2], p[2] + 1e-9, p[3] / 2)
        mf = sweep(zf)
        j = int(np.argmax(mf))
        z = float(zf[j])
        if 0 < j < len(zf) - 1 and min(mf[j - 1], mf[j], mf[j + 1]) > 0:
            y0, y1, y2 = np.log(mf[j - 1]), np.log(mf[j]), np.log(mf[j + 1])
            den = y0 - 2 * y1 + y2
            if den < 0:
                z += 0.5 * (p[3] / 2) * (y0 - y2) / den
        self.set_z(z - s.af_backlash_um); self.set_z(z)
        return z

    def fieldstop_plane(self, step):
        """Focus plane for a flat field: the closed field stop's image on the substrate is sharp only when the surface
        is in focus (bare substrate has nothing else to focus on: autofocus there finds the dust in the light path,
        which comes into focus a few um off).  Focused on its edge at the centre and 4 corners of the frame grid;
        then the user opens the field stop again.  (a, b, c) of z = a x + b y + c, or None (fixed Z)."""
        s = self.s
        af = s.af100 if s.flat_objective in ("100x", "domain") else s.af10
        if not self.fieldstop_visible():
            ok = self.ui.confirm("Field stop", "The field-stop edge is not visible.\n\nClose the field stop (F.S.) on the reflected-light illuminator "
                                 "until its edge shows in the live view, then press OK.\nCancel = focus fixed at the current Z.", ("OK", "Cancel")) == 0
            if not ok or not self.fieldstop_visible():
                self.say("  field-stop edge not visible: focus fixed at the current Z (no tilt correction)")
                self.reopen_fieldstop()
                return None
        R = max(1, int(math.ceil((math.sqrt(s.flat_frames) - 1) / 2)))
        pts = []
        for gx, gy in ((0, 0), (-R, -R), (R, -R), (R, R), (-R, R)):
            self.ui.check_stop(self)
            px, py = self.X0 + gx * step, self.Y0 + gy * step
            self.move_to(px, py)
            pf = fit_plane(pts)
            zg = pf[0] * px + pf[1] * py + pf[2] if pf else (pts[-1][2] if pts else self.z_read())
            zr = self.fieldstop_focus(zg, af)
            if zr is not None:
                pts.append((px, py, zr))
                self.say(f"    field-stop focus at {f1(px)}, {f1(py)}: Z {f3(zr)}")
            elif self.ui.can_ask and self.ask_focus("Field-stop focus", "Autofocus found no field-stop edge here: focus on the edge by hand, "
                                                    "then press \"Use this focus\" (Skip = leave this point out).") == 0:
                pts.append((px, py, self.z_read()))
                self.say(f"    field-stop focus at {f1(px)}, {f1(py)}: Z {f3(pts[-1][2])} (by hand)")
            else:
                self.say(f"    no field-stop focus at {f1(px)}, {f1(py)}: left out")
        pf = fit_plane(pts)
        if pf is None:
            pf = (0.0, 0.0, float(np.median([p[2] for p in pts]))) if pts else None
            self.say("  WARNING: fewer than 3 field-stop focus points: " + ("one Z for all frames" if pf else "focus fixed at the current Z"))
        else:
            res = [p[2] - (pf[0] * p[0] + pf[1] * p[1] + pf[2]) for p in pts]
            rms = float(np.sqrt(np.mean(np.square(res))))
            self.say(f"  focus plane from the field-stop edge: tilt {math.hypot(pf[0], pf[1]) * 1000:.2f} um/mm, {len(pts)} points, "
                     f"residual {rms:.2f} um rms" + ("  WARNING: large; check the edge was sharp at every point" if rms > (0.4 if s.flat_objective in ("100x", "domain") else 3.0) else ""))
        self.reopen_fieldstop()
        return pf

    def reopen_fieldstop(self):
        for k in range(2):
            self.ui.confirm("Field stop", "Open the field stop (F.S.) again to the setting you scan with (normally fully open), then press OK.", ("OK",))
            self.ui.mock_hook(self, "mockFieldStop", False)
            if not self.fieldstop_visible():
                return
            self.say("  the field stop still shows in the frame")
        self.say("  WARNING: the field stop still shows in the frame: the flat field will be wrong at the edges")

    def flat_focus_trend(self, flat_dir, spots):
        """Depth of the camera-fixed spots in each frame against its position: dust in the light path comes into focus
        when the substrate is out of focus, so a trend across the grid means frames went out of focus (chip tilt)."""
        import cv2
        from ..mm import read_dataset, read_frame
        xy = getattr(self, "flat_xy", None) or []
        tiles = read_dataset(str(flat_dir))
        if spots.sum() < 20 or len(xy) != len(tiles) or len(tiles) < 6:
            return []
        d = []
        for t in tiles:
            g = read_frame(t)[..., 1]
            q = g / np.maximum(cv2.GaussianBlur(g, (0, 0), 12), 1e-3)
            d.append(1 - float(q[spots].mean()))
        d = np.array(d) * 100
        P = np.asarray(xy, float) - np.mean(xy, 0)
        P /= max(float(np.abs(P).max()), 1e-9)
        def r2_of(A):
            co = np.linalg.lstsq(A, d, rcond=None)[0]
            return 1 - float(((d - A @ co) ** 2).sum()) / max(float(((d - d.mean()) ** 2).sum()), 1e-12)
        A1 = np.c_[np.ones(len(d)), P]
        r2 = r2_of(A1)                      # a tilt: the spots deepen steadily across the grid
        if len(d) >= 12:                    # or pass through their sharpest point inside it
            r2 = max(r2, r2_of(np.c_[A1, P[:, 0] ** 2, P[:, 1] ** 2, P[:, 0] * P[:, 1]]) - 0.1)
        out = [f"spot depth in the frames: {d.min():.1f} to {d.max():.1f} % (trend across the grid R² {r2:.2f})"]
        if r2 > 0.5 and d.max() - d.min() > 1.0:          # (the old 285 nm flat: R² 0.85, 1.8-5.1 %; the 90 nm one: no trend)
            out.append("WARNING: the spots get deeper across the grid: frames went out of focus (the chip is tilted); these spots are dust "
                       "in the light path that sharpens out of focus. Retake with the field-stop focus.")
        return out

    def finish_flat(self, flat_dir, dark_dir):
        """Build the flat field from the frames just taken, show it next to the one in use, and on request save it in
        the flats folder and make it the one the scans use (scan_settings.json updated; old files kept)."""
        import cv2
        from ..mm import FlatField, build_flatfield
        s, c = self.s, self.core
        self.ui.progress(self, "building the flat field ...")
        self.say("  building the flat field ...")
        ff = build_flatfield(str(flat_dir), str(dark_dir) if dark_dir else None)
        meas = s.flat_measured_oxide_nm if s.flat_measured_oxide_nm > 0 else None
        ff.info.update(objective=s.flat_objective, oxide_nm=meas or (s.flat_oxide_nm if s.flat_objective == "100x" else None),
                       oxide_nominal_nm=s.flat_oxide_nm, oxide_measured_nm=meas,
                       made=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), program=f"flakeid.scope {VERSION}")
        exp = int(round(c.getExposure()))
        stamp = datetime.datetime.now().strftime("%Y%m%d")
        name = (f"flat_100x_{s.flat_oxide_nm:g}nm_{stamp}.npz" if s.flat_objective == "100x" else f"flat_10x_{exp}ms_{stamp}.npz")
        flats = Path(s.flats_dir) if s.flats_dir else Path(__file__).resolve().parents[2] / "flats"
        flats.mkdir(parents=True, exist_ok=True)
        dst = flats / name
        n = 2
        while dst.exists():
            dst = flats / name.replace(".npz", f"_{n}.npz"); n += 1
        # what it shows: vignetting depth, dust on the optics (small dips the median over positions keeps), consistency
        g = ff.flat[..., 1].astype(np.float32)
        smooth = cv2.GaussianBlur(g, (0, 0), 25)
        loc = g / np.maximum(cv2.GaussianBlur(g, (0, 0), 12), 1e-3)          # small dark spots against their surroundings
        dips = (loc < 0.98).astype(np.uint8)
        b_ = 40
        dips[:b_] = 0; dips[-b_:] = 0; dips[:, :b_] = 0; dips[:, -b_:] = 0     # (the blur's edge effects)
        nd, _, st, _ = cv2.connectedComponentsWithStats(dips)
        dust = int(sum(1 for i in range(1, nd) if st[i, cv2.CC_STAT_AREA] >= 12))
        _, lab_ = cv2.connectedComponents(dips)
        spots = np.isin(lab_, [i for i in range(1, nd) if st[i, cv2.CC_STAT_AREA] >= 12])
        deepest = float(1 - loc[b_:-b_, b_:-b_].min())
        lines = [f"{ff.info.get('n_frames')} frames" + (", dark frames subtracted" if dark_dir else ", no dark frames"),
                 f"vignetting: the corners at {100 * float(smooth.min()):.0f} % of the centre",
                 (f"spots fixed to the camera (dust in the light path, divided out of every image): {dust}, deepest {100 * deepest:.1f} %"
                  + (" (many: clean the optics if they are new)" if dust > 20 else "") if dust else "no dust spots in the light path"),
                 f"frame-to-frame consistency: {100 * float(ff.info.get('frame_spread') or 0):.1f} % spread (lower is better; above ~3 %: dirty area or drifting lamp)"]
        try:
            lines += self.flat_focus_trend(flat_dir, spots)
        except Exception as e:
            lines.append(f"(focus check across the grid not possible: {e})")
        # against the flat field in use for this objective / oxide
        cur = s.flat10 if s.flat_objective == "10x" else self._flat100_for(s.flat_oxide_nm)
        change = None
        if cur and Path(cur).is_file():
            try:
                old = FlatField.load(cur).flat[..., 1]
                if old.shape == g.shape:
                    q = cv2.GaussianBlur(g, (0, 0), 3) / np.maximum(cv2.GaussianBlur(old, (0, 0), 3), 1e-3)
                    q /= float(np.median(q))
                    change = (float(np.std(q) * 100), float(np.abs(q - 1).max() * 100))
                    lines.append(f"compared with the one in use ({Path(cur).name}): {change[0]:.1f} % rms difference, up to {change[1]:.1f} %")
            except Exception as e:
                lines.append(f"(the flat field in use could not be compared: {e})")
        if meas:
            lines += self.record_oxide_calibration(ff, meas)
        for ln in lines:
            self.say("  " + ln)
        # preview: the flat field (green), and the change from the one in use
        vis = cv2.applyColorMap(np.clip((g - g.min()) / max(float(g.max() - g.min()), 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        vis = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2), interpolation=cv2.INTER_AREA)
        if change is not None:
            d = np.clip(128 + (q - 1) / 0.02 * 127, 0, 255).astype(np.uint8)
            dv = cv2.resize(cv2.applyColorMap(d, cv2.COLORMAP_TWILIGHT_SHIFTED), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_AREA)
            vis = np.hstack([vis, np.full((vis.shape[0], 8, 3), 255, np.uint8), dv])
        prev = Path(flat_dir).parent / f"{Path(flat_dir).name}_preview.jpg"
        cv2.imwrite(str(prev), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.ui.show_image(self, "New flat field" + (" (left) and the change from the one in use (right, ±2 %)" if change else ""), str(prev))
        ff.save(str(dst))
        self.say(f"  saved {dst}")
        a = self.ui.confirm("Flat field", "New flat field (" + s.flat_objective + (f", {s.flat_oxide_nm:g} nm" if s.flat_objective == "100x" else "") +
                            f"):\n\n" + "\n".join("• " + ln for ln in lines) + f"\n\nSaved as {dst}.\n\nUse it for the scans from now on?",
                            ("Use it", "Keep the current one"))
        if a == 0:
            self.adopt_flat(dst)
        else:
            self.say("  not adopted: the scans keep the current flat field (the new file stays in the flats folder)")

    def record_oxide_calibration(self, ff, meas):
        """A flat field of a piece whose oxide was measured (Filmetrics): the nominal -> measured value for the analysis
        and the bare-substrate colour reference, both in ~/.flakeid (kept across program updates), and a check of the
        optical model (the oxide its colours alone imply).  Returns lines for the summary."""
        from ..thickness import LOCAL_CAL, ContrastModel, camera_from_profile, load_profile
        s = self.s
        out = []
        try:
            LOCAL_CAL.mkdir(parents=True, exist_ok=True)
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            mf = LOCAL_CAL / "oxide_measured.json"
            table = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else {}
            key = f"{s.flat_oxide_nm:g}"
            prev = table.get(key, {}).get("measured_nm")
            table[key] = dict(measured_nm=round(meas, 1), date=date, objective=s.flat_objective, source="Filmetrics, with a flat field")
            mf.write_text(json.dumps(table, indent=1), encoding="utf-8")
            out.append(f"oxide calibration: chips entered as {key} nm are analysed as {meas:g} nm from now on" + (f" (was {prev:g} nm)" if prev else ""))
            rf = LOCAL_CAL / "substrate_refs.json"
            refs = json.loads(rf.read_text(encoding="utf-8")).get("refs", []) if rf.is_file() else []
            refs = [r for r in refs if not (r.get("objective") == s.flat_objective and abs(float(r.get("oxide_nm", 0)) - meas) <= 15)]
            refs.append(dict(objective=s.flat_objective, oxide_nm=round(meas, 1), substrate_counts=ff.info.get("substrate_counts"),
                             exposure_ms=ff.info.get("exposure_ms"), bit_depth=ff.info.get("bit_depth", 12), date=date,
                             source="flat field of a Filmetrics-measured piece", nominal_nm=s.flat_oxide_nm))
            rf.write_text(json.dumps(dict(note="bare-substrate references recorded on this computer (flakeid.scope flat field)", refs=refs), indent=1), encoding="utf-8")
            out.append(f"substrate colour reference for {s.flat_objective} at {meas:g} nm recorded")
        except Exception as e:
            out.append(f"WARNING: the oxide calibration could not be recorded ({e})")
            return out
        # model check: the oxide the bare substrate's colour implies (R/G, B/G against the optical model, near the nominal)
        try:
            prof = load_profile("bx51_xc10_mm_raw16")
            cam = camera_from_profile(prof)
            sub = np.asarray(ff.info.get("substrate_counts"), float)
            m = np.log([sub[0] / sub[1], sub[2] / sub[1]])
            na = 0.9 if s.flat_objective == "100x" else float(prof.get("na", 0.30))
            best, bl = None, 1e9
            for ox in np.arange(0.75 * s.flat_oxide_nm, 1.25 * s.flat_oxide_nm, 0.25):
                r = ContrastModel("hbn", float(ox), na=na, camera=cam).substrate_reflectance()
                l_ = float(np.sum((np.log([r[0] / r[1], r[2] / r[1]]) - m) ** 2))
                if l_ < bl:
                    best, bl = float(ox), l_
            if best is not None:
                d = best - meas
                out.append(f"model check: the substrate colour alone implies {best:.1f} nm against {meas:g} nm measured ({d:+.1f} nm, "
                           f"{100 * d / meas:+.1f} %)" + ("; more than 5 %: the camera's colour response in the profile is off" if abs(d / meas) > 0.05 else ""))
        except Exception as e:
            out.append(f"(model check not possible: {e})")
        return out

    def _flat100_for(self, oxide):
        best, bd = None, 1e9
        for e in self.s.flat100 or ():
            k, _, v = str(e).partition("=")
            try:
                d = abs(float(k) - oxide)
            except ValueError:
                continue
            if v and d < bd:
                best, bd = v, d
        return best if bd <= 40 else None

    def adopt_flat(self, path):
        """Make `path` the flat field of its objective / oxide in the settings file (a dated backup is kept)."""
        s = self.s
        sf = Path(s.settings_file) if s.settings_file else None
        if sf is None or not sf.is_file():
            key = "flat10" if s.flat_objective == "10x" else "flat100"
            self.say(f"  no settings file to update: set {key} to {path} in your scan settings")
            return
        try:
            cfg = json.loads(sf.read_text(encoding="utf-8"))
            bak = sf.with_name(sf.stem + datetime.datetime.now().strftime("_before_flat_%Y%m%d_%H%M") + sf.suffix)
            bak.write_text(sf.read_text(encoding="utf-8"), encoding="utf-8")
            p = str(path).replace("\\", "/")
            if s.flat_objective == "10x":
                old = cfg.get("flat10")
                cfg["flat10"] = p
            else:
                lst = list(cfg.get("flat100") or [])
                keep = []
                old = None
                for e in lst:
                    k, _, v = str(e).partition("=")
                    try:
                        same = abs(float(k) - s.flat_oxide_nm) <= 40
                    except ValueError:
                        same = False
                    if same:
                        old = v
                    else:
                        keep.append(e)
                cfg["flat100"] = keep + [f"{s.flat_oxide_nm:g}={p}"]
            tmp = sf.with_name(sf.name + ".tmp")
            tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            os.replace(str(tmp), str(sf))
            self.say(f"  adopted: {sf.name} now uses {p}" + (f" (was {old}; backup {bak.name})" if old else f" (backup {bak.name})"))
            self.ui.message("Flat field", f"The scans now use the new flat field:\n{p}\n\nThe old file is kept, and the settings before the change are in {bak.name}.")
        except Exception as e:
            self.say(f"  WARNING: {sf} could not be updated ({e}); set the flat field there by hand: {path}")
