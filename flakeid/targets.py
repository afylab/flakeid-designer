"""Flake table -> Micro-Manager position list (.pos) for high-magnification imaging.

    python -m flakeid.targets scan_flakes.csv -o targets.pos --top 30
    python -m flakeid.targets g34_flakes.csv -o g34.pos --min-score 0.5 \\
        --ref 1520 3310 53012.4 12877.0 --ref 8120 2204 59630.1 11760.3 --ref 4410 7880 56025.8 17410.9

Coordinates
-----------
* Tables from Micro-Manager tile scans already hold stage coordinates (the
  stage position that centres each flake's best terrace); they are used as-is.
* For any other scan (e.g. a Stream stitched image) give reference points:
  ``--ref X_PX Y_PX STAGE_X STAGE_Y`` pairs a pixel in the scan image (the
  zoomable viewer shows it under the cursor) with the stage position (read in
  Micro-Manager) that centres the same feature.  Two or more points fit a
  similarity (shift, rotation, scale) by least squares; with two, give
  --mirror if the image is mirrored relative to the stage, with three or more
  it is detected and the residual reported (use 3-4 points spread over the
  chip).  --affine fits all six parameters instead.
* ``--ref`` may also be used with a Micro-Manager table to re-align after the
  chip was removed and replaced (1 point: shift only).
* ``--offset DX DY``: parcentricity offset of the high-magnification
  objective relative to the scan objective (stage um), measured once.

Focus
-----
Without --z / --focus the list carries XY only (focus is left alone; use
Micro-Manager autofocus).  ``--z Z`` sets one Z; ``--focus X Y Z`` (>= 3,
stage coordinates) fits a focus plane that is evaluated at every target.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Selection and ordering
# ---------------------------------------------------------------------------

def _f(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_flakes(path) -> list:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --target-rank size (trial, 2026-10-22): the candidate pool ranked by logit(score) + SIZE_RANK_W * ln(best terrace d um)
# instead of the score alone (21 hBN pieces: kept 119 of the 123 known keeps of the score ranking and brought in ~320
# never-imaged flakes, mostly larger ones; keep rate per 100x shot 3 % at 30-40 um best terrace vs 13-16 % from 50 um)
SIZE_RANK_W = 2.0
RANKS = ("score", "size")


def rank_value(r, rank="score"):
    """Priority of a flake-table row (higher first): its score, or for rank 'size' logit(score) + SIZE_RANK_W * ln(d)
    with d the best terrace's inscribed diameter (um, at least 1)."""
    s = _f(r.get("score"), -1)
    if rank != "size":
        return s
    p = min(max(s, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p)) + SIZE_RANK_W * math.log(max(_f(r.get("best_terrace_d_um"), 0), 1.0))


def select(rows, top=None, min_score=None, thickness=None, min_d=None, ids=None, exclude_flags=(), rank="score"):
    """The rows passing the filters, best first (rank: 'score', or 'size' = rank_value's score-and-size priority)."""
    out = []
    for r in rows:
        if ids is not None:
            if int(r["flake_id"]) in ids:
                out.append(r)
            continue
        if min_score is not None and not _f(r.get("score"), -1) >= min_score:
            continue
        t = _f(r.get("best_thickness_nm"))
        if thickness is not None and not (thickness[0] <= t <= thickness[1]):
            continue
        if min_d is not None and not _f(r.get("best_terrace_d_um"), 0) >= min_d:
            continue
        flags = set(filter(None, (r.get("quality") or "").split(";")))
        if flags & set(exclude_flags):
            continue
        out.append(r)
    out.sort(key=lambda r: -rank_value(r, rank))
    if ids is not None:
        order = {i: k for k, i in enumerate(ids)}
        out.sort(key=lambda r: order[int(r["flake_id"])])
    return out[:top] if top else out


def order_path(xy: np.ndarray, start: int = 0) -> list:
    """Short stage path: nearest neighbour from `start`, then 2-opt."""
    n = len(xy)
    if n <= 2:
        return ([start] + [i for i in range(n) if i != start]) if n else []
    left = set(range(n)) - {start}
    path = [start]
    while left:
        last = xy[path[-1]]
        nxt = min(left, key=lambda j: float(np.hypot(*(xy[j] - last))))
        path.append(nxt)
        left.remove(nxt)
    d = lambda i, j: float(np.hypot(*(xy[i] - xy[j])))
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = path[i - 1], path[i]
                c, e = path[j], path[j + 1] if j + 1 < n else None
                old = d(a, b) + (d(c, e) if e is not None else 0)
                new = d(a, c) + (d(b, e) if e is not None else 0)
                if new < old - 1e-9:
                    path[i:j + 1] = reversed(path[i:j + 1])
                    improved = True
    return path


SIZE_CHUNK = 10


def order_by_size(xy: np.ndarray, d, chunk: int = SIZE_CHUNK) -> list:
    """Largest best terrace first (a 100x run cut short has imaged the biggest flakes), with a short stage path within
    groups: the targets sorted by best-terrace diameter d (largest first; equal d: the given order) are taken `chunk`
    at a time, and each group is visited along order_path from its target nearest to where the previous group ended
    (the first group from its largest).  chunk 1 = strictly by size.  (Any value works as d: order 'model' passes the
    kept-at-100x probability.)"""
    n = len(xy)
    dd = np.nan_to_num(np.asarray(d, float), nan=0.0)
    srt = sorted(range(n), key=lambda i: (-dd[i], i))
    chunk = max(1, int(chunk or 1))
    out = []
    for k in range(0, n, chunk):
        grp = srt[k:k + chunk]
        sub = xy[grp]
        start = 0 if not out else int(np.argmin(np.hypot(*(sub - xy[out[-1]]).T)))
        out += [grp[j] for j in order_path(sub, start)]
    return out


def path_length(xy: np.ndarray, idx) -> float:
    """Stage travel (um) visiting xy[idx] in that order."""
    P = np.asarray(xy, float)[list(idx)]
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def _similarity(src, dst, mirror):
    s = src.copy()
    if mirror:
        s[:, 1] = -s[:, 1]
    zs, zd = s[:, 0] + 1j * s[:, 1], dst[:, 0] + 1j * dst[:, 1]
    ms, md = zs.mean(), zd.mean()
    a = np.vdot(zs - ms, zd - md) / np.vdot(zs - ms, zs - ms)
    A = np.array([[a.real, -a.imag], [a.imag, a.real]])
    if mirror:
        A = A @ np.diag([1.0, -1.0])
    t = dst.mean(0) - A @ src.mean(0)
    return A, t


def fit_transform(src: np.ndarray, dst: np.ndarray, mirror: bool | None = None, affine: bool = False):
    """dst ~ A @ src + t.

    1 point: shift only.  >= 2 points: similarity (shift, rotation, one scale)
    by least squares; with >= 3 points a mirror image is detected
    automatically (``mirror=None``), with 2 it must be given.  ``affine``
    (>= 3 points) fits all 6 parameters, which is only better than a
    similarity when the points are far apart and accurately centred.
    Returns (A, t, rms residual in um, or None when there is no redundancy).
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n == 1:
        return np.eye(2), dst[0] - src[0], None
    if affine and n >= 3:
        X = np.hstack([src, np.ones((n, 1))])
        sol, *_ = np.linalg.lstsq(X, dst, rcond=None)
        A, t = sol[:2].T, sol[2]
        dof = n - 3
    else:
        opts = [bool(mirror)] if (mirror is not None or n == 2) else [False, True]
        fits = []
        for m in opts:
            A, t = _similarity(src, dst, m)
            fits.append((float(((dst - (src @ A.T + t)) ** 2).sum()), A, t))
        _, A, t = min(fits, key=lambda f: f[0])
        dof = n - 2
    res = dst - (src @ A.T + t)
    return A, t, (float(np.sqrt((res ** 2).sum(1).mean())) if dof > 0 else None)


def describe(A) -> str:
    sx, sy = np.hypot(*A[:, 0]), np.hypot(*A[:, 1])
    rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    det = np.linalg.det(A)
    return f"scale {sx:.4f}/{sy:.4f} um/px, rotation {rot:+.2f} deg{', mirrored' if det < 0 else ''}"


def focus_plane(points):
    p = np.asarray(points, float)
    X = np.hstack([p[:, :2], np.ones((len(p), 1))])
    coef, *_ = np.linalg.lstsq(X, p[:, 2], rcond=None)
    res = p[:, 2] - X @ coef
    return coef, float(np.sqrt((res ** 2).mean()))


# ---------------------------------------------------------------------------
# .pos writing (Micro-Manager 2 "Property Map" format)
# ---------------------------------------------------------------------------

def _s(v):
    return {"type": "STRING", "scalar": str(v)}


def position_entry(label, xy, z=None, xy_stage="OasisXYStage", z_stage="OasisZFocus", props=None):
    devs = [{"Device": _s(xy_stage), "Position_um": {"type": "DOUBLE", "array": [float(xy[0]), float(xy[1])]}}]
    if z is not None:
        devs.append({"Device": _s(z_stage), "Position_um": {"type": "DOUBLE", "array": [float(z)]}})
    return {
        "DefaultXYStage": _s(xy_stage),
        "DefaultZStage": _s(z_stage),
        "DevicePositions": {"type": "PROPERTY_MAP", "array": devs},
        "GridCol": {"type": "INTEGER", "scalar": 0},
        "GridRow": {"type": "INTEGER", "scalar": 0},
        "Label": _s(label),
        "Properties": {"type": "PROPERTY_MAP", "scalar": {k: _s(v) for k, v in (props or {}).items()}},
    }


def write_pos(path, entries):
    doc = {"encoding": "UTF-8", "format": "Micro-Manager Property Map", "major_version": 2, "minor_version": 0,
           "map": {"StagePositions": {"type": "PROPERTY_MAP", "array": entries}}}
    Path(path).write_text(json.dumps(doc, indent=2))


def read_pos(path) -> list:
    """(label, x, y, z or None) of every position in a .pos file."""
    d = json.loads(Path(path).read_text())
    out = []
    for e in d["map"]["StagePositions"]["array"]:
        xy_dev = e["DefaultXYStage"]["scalar"]
        x = y = z = None
        for dp in e["DevicePositions"]["array"]:
            v = dp["Position_um"]["array"]
            if dp["Device"]["scalar"] == xy_dev:
                x, y = v
            else:
                z = v[0]
        out.append((e["Label"]["scalar"], x, y, z))
    return out


# ---------------------------------------------------------------------------

def focus_map_10x(scan_json, z_scale: float = 12.0, log=print):
    """10x focus Z (real um) as a function of stage (x, y), from the tile Z of a Micro-Manager scan.

    scan_json: <scan>_flakes.json (or *_stagemap.json).  Multi-D tiles record the focus-drive
    reading: divided by z_scale (drive units per real um; BX51 + Oasis: 12).  Returns f or None."""
    import json as _json
    d = _json.loads(Path(scan_json).read_text())
    sm = d.get("stage_map") or d.get("metadata", {}).get("stage_map")
    if not sm or not sm.get("z") or any(v is None for v in sm["z"]):
        return None
    P, zz = np.array(sm["P"], float), np.array(sm["z"], float)
    if sm.get("z_units", "drive") != "um":
        zz = zz / z_scale
    A = np.c_[P, np.ones(len(P))]
    coef, *_ = np.linalg.lstsq(A, zz, rcond=None)
    res = zz - A @ coef
    log(f"10x focus map from {len(zz)} tiles of {Path(scan_json).name}: tilt {np.hypot(coef[0], coef[1]) * 1000:.2f} um/mm, "
        f"range {zz.min():.1f} .. {zz.max():.1f} um, rms off the plane {np.sqrt((res ** 2).mean()):.2f} um"
        + ("" if sm.get("z_units") == "um" else f" (drive units / {z_scale:g})"))
    # interpolate between the tiles: their Z follow the scan's focus surface, which is quadratic for bowed
    # pieces (mm_chip_scan.bsh FOCUS_SURFACE); nearest tile outside them
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    try:
        lin = LinearNDInterpolator(P, zz)
    except Exception:
        lin = None
    near = NearestNDInterpolator(P, zz)

    def f(x, y):
        v = float(lin(x, y)) if lin is not None else float("nan")
        return v if v == v else float(near(x, y))
    return f


def make_targets(rows, refs=None, mirror=None, affine=False, offset=(0.0, 0.0), z=None, focus=None,
                 order="path", xy_stage="OasisXYStage", z_stage="OasisZFocus", log=print, focus10=None, chunk=SIZE_CHUNK):
    """Target records (dicts) with stage coordinates, in acquisition order: order 'path' (shortest stage path),
    'score' (as given), 'size' (largest best terrace first, a short path within groups of `chunk`: order_by_size) or
    'model' (the same by the rows' p_keep100, flakeid.target_model)."""
    have_stage = all(r.get("stage_x_um") not in (None, "", "None") for r in rows)
    px = np.array([[_f(r["target_x_px"]), _f(r["target_y_px"])] for r in rows])
    if refs:
        refs = np.asarray(refs, float)
        if have_stage and len(refs) >= 1:
            # re-alignment of a stage-referenced table: refs are (old stage x, y, new stage x, y)
            src = np.array([[_f(r["stage_x_um"]), _f(r["stage_y_um"])] for r in rows])
            A, t, rms = fit_transform(refs[:, :2], refs[:, 2:], mirror=False)
            if len(refs) >= 2:
                log(f"re-alignment: {describe(A).replace(' um/px', '')}")
            log(f"re-alignment from {len(refs)} point(s): shift {np.round(t + (A - np.eye(2)) @ src.mean(0), 1)} um"
                + (f", rms {rms:.1f} um" if rms is not None else ""))
            xy = src @ A.T + t
        else:
            A, t, rms = fit_transform(refs[:, :2], refs[:, 2:], mirror=mirror, affine=affine)
            log(f"image -> stage from {len(refs)} reference point(s): {describe(A)}"
                + (f"; residual rms {rms:.1f} um" if rms is not None else
                   " (no residual: use >= 3 points to check the fit)"))
            rot = math.degrees(math.atan2(A[1, 0], A[0, 0])) % 90
            if min(rot, 90 - rot) > 5:
                log("warning: the rotation is far from a multiple of 90 deg; check the reference points "
                    "(with 2 points, also whether --mirror is needed)")
            xy = px @ A.T + t
    elif have_stage:
        xy = np.array([[_f(r["stage_x_um"]), _f(r["stage_y_um"])] for r in rows])
    else:
        raise ValueError("the table has no stage coordinates: give reference points with --ref X_PX Y_PX STAGE_X STAGE_Y")
    xy10 = xy.copy()                                    # where the flakes are at 10x (before the objective offset)
    xy = xy + np.asarray(offset, float)

    zs = [None] * len(rows)
    if focus is not None:
        coef, rms = focus_plane(focus)
        tilt = np.hypot(coef[0], coef[1]) * 1000
        log(f"focus plane from {len(focus)} points: tilt {tilt:.2f} um/mm, rms {rms:.2f} um")
        zs = [float(coef[0] * x + coef[1] * y + coef[2]) for x, y in xy]
    elif z is not None:
        zs = [float(z)] * len(rows)

    if order == "size":
        idx = order_by_size(xy, [_f(r.get("best_terrace_d_um"), 0.0) for r in rows], chunk)
    elif order == "model":
        idx = order_by_size(xy, [_f(r.get("p_keep100"), 0.0) for r in rows], chunk)
    elif order == "clean":                              # (live --target-mode avram_new1: largest clean area first)
        idx = order_by_size(xy, [_f(r.get("clean_area_um2"), 0.0) for r in rows], chunk)
    elif order == "path":
        idx = order_path(xy)
    else:
        idx = list(range(len(rows)))
    out = []
    for n, i in enumerate(idx, 1):
        r = rows[i]
        t = _f(r.get("best_thickness_nm"))
        nl = r.get("layers")
        label = f"{n:02d}_F{int(r['flake_id']):03d}_" + (f"{nl}L" if nl else f"{t:.0f}nm")
        out.append(dict(order=n, label=label, flake_id=int(r["flake_id"]), score=_f(r.get("score")),
                        thickness_nm=t, terrace_d_um=_f(r.get("best_terrace_d_um")), area_um2=_f(r.get("area_um2")),
                        x_px=px[i, 0], y_px=px[i, 1], stage_x_um=float(xy[i, 0]), stage_y_um=float(xy[i, 1]),
                        z_um=zs[i], z10_um=(round(focus10(*xy10[i]), 3) if focus10 else None),
                        quality=r.get("quality", ""), **({"layers": r["layers"]} if "layers" in r else {}),
                        **({"clean_area_um2": _f(r["clean_area_um2"])} if "clean_area_um2" in r else {}),
                        **({"p_keep100": _f(r["p_keep100"])} if "p_keep100" in r else {})))
    travel = sum(math.hypot(out[k]["stage_x_um"] - out[k - 1]["stage_x_um"],
                            out[k]["stage_y_um"] - out[k - 1]["stage_y_um"]) for k in range(1, len(out)))
    log(f"{len(out)} targets, stage travel {travel / 1000:.1f} mm ({order} order"
        + (f", largest best terrace first in groups of {chunk}, path order within" if order == "size" else "")
        + (f", most likely kept at 100x first in groups of {chunk}, path order within" if order == "model" else "")
        + (f", largest clean area first in groups of {chunk}, path order within" if order == "clean" else "") + ")")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m flakeid.targets",
                                description="Write a Micro-Manager .pos target list from a flake table.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("csv", help="<scan>_flakes.csv from find_flakes")
    p.add_argument("-o", "--out", help="output .pos (default: <csv stem>_targets.pos)")
    g = p.add_argument_group("selection")
    g.add_argument("--top", type=int, help="best N flakes by score")
    g.add_argument("--min-score", type=float)
    g.add_argument("--thickness", nargs=2, type=float, metavar=("MIN", "MAX"), help="nm")
    g.add_argument("--min-d", type=float, help="min inscribed diameter of the best terrace (um)")
    g.add_argument("--ids", help="comma-separated flake ids (overrides the other filters)")
    g.add_argument("--exclude-flags", default="saturated", help="comma-separated quality flags to skip (default: saturated)")
    c = p.add_argument_group("coordinates")
    c.add_argument("--ref", nargs=4, type=float, action="append", metavar=("X_PX", "Y_PX", "STAGE_X", "STAGE_Y"),
                   help="reference point (repeat); with a stage-referenced table: OLD_X OLD_Y NEW_X NEW_Y (stage um)")
    c.add_argument("--mirror", action="store_true", default=None,
                   help="image is mirrored relative to the stage (needed for 2-point fits only; detected with >= 3)")
    c.add_argument("--affine", action="store_true", help="full affine fit (>= 3 points) instead of a similarity")
    c.add_argument("--offset", nargs=2, type=float, default=(0.0, 0.0), metavar=("DX", "DY"),
                   help="objective offset added to every target (stage um)")
    c.add_argument("--z", type=float, help="fixed Z for every target (um)")
    c.add_argument("--focus", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"),
                   help="focus point in stage um (repeat >= 3 times for a focus plane)")
    c.add_argument("--order", choices=("path", "score", "size"), default="path",
                   help="path: short stage route (default); score: best first; size: largest best terrace first, a short "
                        f"route within groups of {SIZE_CHUNK}")
    c.add_argument("--scan-json", help="<scan>_flakes.json with the tile Z (10x focus map); default: next to the CSV")
    c.add_argument("--z10-scale", type=float, default=12.0,
                   help="focus-drive units per real um in Multi-D tile Z (BX51 + Oasis: 12; 1 if the step size is right)")
    c.add_argument("--xy-stage", default="OasisXYStage")
    c.add_argument("--z-stage", default="OasisZFocus")
    a = p.parse_args(argv)
    if a.focus is not None and len(a.focus) < 3:
        p.error("--focus needs at least 3 points")

    rows = read_flakes(a.csv)
    ids = [int(x) for x in a.ids.split(",")] if a.ids else None
    sel = select(rows, top=a.top, min_score=a.min_score, thickness=a.thickness, min_d=a.min_d, ids=ids,
                 exclude_flags=[f for f in a.exclude_flags.split(",") if f])
    if not sel:
        p.error("no flakes pass the selection")
    sj = Path(a.scan_json) if a.scan_json else Path(a.csv).with_name(Path(a.csv).name.replace("_flakes.csv", "_flakes.json"))
    f10 = focus_map_10x(sj, a.z10_scale) if sj.exists() else None
    if f10 is None:
        print("(no 10x focus map: the target list gets no z10_um column)")
    tg = make_targets(sel, refs=a.ref, mirror=a.mirror, affine=a.affine, offset=a.offset, z=a.z, focus=a.focus, focus10=f10,
                      order=a.order, xy_stage=a.xy_stage, z_stage=a.z_stage)
    out = Path(a.out) if a.out else Path(a.csv).with_name(Path(a.csv).stem.replace("_flakes", "") + "_targets.pos")
    write_pos(out, [position_entry(t["label"], (t["stage_x_um"], t["stage_y_um"]), t["z_um"], a.xy_stage, a.z_stage,
                                   props={"flake_id": t["flake_id"], "thickness_nm": f"{t['thickness_nm']:.1f}",
                                          "score": f"{t['score']:.3f}", "source": Path(a.csv).name})
                    for t in tg])
    tcsv = out.with_suffix(".csv")
    with open(tcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tg[0]))
        w.writeheader()
        for t in tg:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items()})
    print(f"wrote {out}\nwrote {tcsv}")


if __name__ == "__main__":
    main()
