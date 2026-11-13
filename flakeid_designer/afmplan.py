"""AFM lists for a chip (Library › a chip's AFM…): the flakes' scan sites in AFM stage coordinates, and the AFM images
taken there, imported and analysed.

Plan: the page sends the scan sites (one per flake: a point on its target terrace's edge to the bare substrate, so the
scan has the step to measure) and three reference flakes, in the chip's own micrometres (the designer's chip map:
origin_um + cut-out pixel x px_um), with the AFM stage X, Y (and Z) the user read at each reference flake.
Since 2026-10-30 a single chip's list is fitted in the 100x stage's own micrometres (frame "100x stage um": each point's
st = the stage position its flake's 100x picture was taken at + its offset from the frame's centre through the 100x
camera's pixel -> stage matrix; no 10x coordinates, no 10x -> 100x offset); a block's list (its corners: 10x stage um)
and a list with a flake without a 100x position (an imported flake) use chip as before.  The
microscope -> AFM transform is a similarity (rotation, one scale, shift; a mirror image found with 3 points:
flakeid.targets.fit_transform, as tools/afm_pgm.py), Z a plane through the references (or one value).  Written to
<data>/afm/<plan>/: plan.json, <chip>_afm_sites.pgm (NanoScope programmed-move point list, in site order) and a csv.

Import: the AFM files of the run (Bruker NanoScope .spm), matched in order to the sites (the n-th file is the n-th
point of the list; a first site can be given if the run started later), each masked, levelled and its flat levels'
heights measured (flakeid.afm.analyse_auto), and a picture of the levelled height written (<n>_<file>.png; the height
itself, reduced, in <n>_<file>.npz for the tip offset).

Tip offset (optional): the AFM's tip is not exactly where its optical crosshair points, so every point of a list is
scanned shifted by the same amount.  After the first point's image is imported, tip_offset() finds where that scan
really was: the height image's edges matched to the site's 100x picture (all 8 ways the AFM image's axes can lie on the
stage's: turned by 90 degrees, mirrored; the one used before is preferred when about as good).  The shift, in AFM
stage units, is taken off every point (apply_tip): the list is written again.
"""
from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import numpy as np


def n2(v):
    """2 decimals, no trailing zeros (as the NanoScope point lists: -86662.5, -105646.88, -16944)."""
    r = str(round(float(v), 2))
    return r[:-2] if r.endswith(".0") else r


def plan_id(scan):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(scan)).strip("_")[:60] + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")


def make_plan(b, out_dir):
    """b: dict(scan, run, sites=[{n, flake, terrace, chip:[x, y], px:[x, y], t_nm}], refs=[{flake, chip:[x, y], afm:[x, y, z?]}],
    zoff, scan_um, by, id?).  Writes the plan and its lists; returns the plan (with the sites' AFM coordinates).
    A block's list (mounted 1 cm chips, mount: {m, pieces}): chip = 10x stage um for every site and reference (a
    corner of the block, or a flake), each site with its chip's chip -> stage rotation c2s (for the tip offset).
    frame "100x stage um" (a single chip, 2026-10-30): every reference and site has st (100x stage um), fitted from that
    (each site with its picture -> stage rotation c2s, for the tip offset)."""
    from flakeid.targets import fit_transform
    refs = [r for r in b.get("refs") or [] if r.get("afm") and r["afm"][0] not in (None, "") and r["afm"][1] not in (None, "")]
    if len(refs) < 3:
        raise ValueError(f"the AFM X and Y of 3 reference points (flakes or a block's corners) are needed (have {len(refs)})")
    key = "st" if b.get("frame") == "100x stage um" else "chip"
    if key == "st" and not all(q.get("st") for q in list(refs) + list(b.get("sites") or [])):
        raise ValueError("a point has no 100x stage position: make the list again")
    src = np.array([r[key] for r in refs], float)
    dst = np.array([[float(r["afm"][0]), float(r["afm"][1])] for r in refs], float)
    if np.linalg.matrix_rank(np.c_[src - src.mean(0)]) < 2 or np.linalg.matrix_rank(np.c_[dst - dst.mean(0)]) < 2:
        raise ValueError("the reference points (or the AFM positions entered) lie on one line: choose others")
    A, t, rms = fit_transform(src, dst)
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    mirror = bool(np.linalg.det(A) < 0)
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    pred = src @ A.T + t
    res = [float(np.hypot(*(p - d))) for p, d in zip(pred, dst)]
    zp = [(float(r["afm"][0]), float(r["afm"][1]), float(r["afm"][2])) for r in refs if len(r["afm"]) > 2 and r["afm"][2] not in (None, "")]
    if len(zp) >= 3:
        Z = np.array(zp)
        c, *_ = np.linalg.lstsq(np.c_[Z[:, :2], np.ones(len(Z))], Z[:, 2], rcond=None)
        zf, zhow = (lambda x, y: float(c[0] * x + c[1] * y + c[2])), f"a plane through the {len(zp)} references"     # noqa: E731
    elif zp:
        z0 = float(np.mean([p[2] for p in zp]))
        zf, zhow = (lambda x, y: z0), f"{len(zp)} reference(s): one value"                                   # noqa: E731
    else:
        zf, zhow = None, "none entered (the list has no Z: the AFM keeps its own)"
    zoff = float(b.get("zoff") or 0.0)
    sites = []
    for s in b.get("sites") or []:
        x, y = A @ np.array(s[key], float) + t
        sites.append(dict(s, afm=[round(float(x), 2), round(float(y), 2)] + ([round(zf(x, y) + zoff, 2)] if zf else [])))
    pid = b.get("id") or plan_id(b.get("scan") or "chip")
    d = Path(out_dir) / pid
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(b.get("scan") or "chip"))
    old = {}
    try:
        old = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    tip = b["tip"] if "tip" in b else old.get("tip")      # (a tip offset: kept when the list is made again)
    for s in sites:
        s["afm0"] = list(s["afm"])                           # (where the crosshair goes; afm: with the tip offset taken off)
    _offset_sites(sites, tip)
    tf = dict(scale=round(scale, 6), rotation_deg=round(rot, 3), mirror=mirror, rms_um=None if rms is None else round(float(rms), 3),
              ref_residual_um=[round(v, 3) for v in res], z=zhow)
    if b.get("preview"):
        # (2026-11-03s) the points as the list would send them (the 3D check before Generate AFM file): nothing written
        return dict(preview=True, sites=[dict(n=s.get("n"), flake=s.get("flake"), afm=s["afm"]) for s in sites],
                    refs=[dict(flake=r.get("flake"), corner=r.get("corner"), afm=[float(v) for v in r["afm"][:2]]
                               + ([float(r["afm"][2])] if len(r["afm"]) > 2 and r["afm"][2] not in (None, "") else [])) for r in refs],
                    has_z=zf is not None, zoff=zoff, tip=(tip or {}).get("offset"), transform=tf)
    d.mkdir(parents=True, exist_ok=True)
    pgm = write_lists(d, stem, sites)
    # (2026-11-03e, Library › AFM runs: a list made from a draft keeps its name, where it was redone from and its scan
    # mark; the draft's own fields go)
    keep = {k: old[k] for k in ("name", "from", "scanned", "afm_folder") if old.get(k) is not None}
    if b.get("name"):
        keep["name"] = str(b["name"])[:80]
    plan = dict(id=pid, scan=b.get("scan"), run=b.get("run"), by=b.get("by"), created=old.get("created") or int(datetime.datetime.now().timestamp() * 1000),
                updated=int(datetime.datetime.now().timestamp() * 1000), scan_um=float(b.get("scan_um") or 20.0), zoff=zoff,
                frame="10x stage um" if b.get("mount") else ("100x stage um" if key == "st" else "chip um"),
                refs=b.get("refs"), sites=sites, results=old.get("results") or {}, tip=tip, mount=b.get("mount") or old.get("mount"),
                transform=dict(A=A.round(8).tolist(), t=t.round(4).tolist(), scale=round(scale, 6), rotation_deg=round(rot, 3), mirror=mirror,
                               rms_um=None if rms is None else round(float(rms), 3), ref_residual_um=[round(v, 3) for v in res], z=zhow),
                pgm=pgm.name, csv=f"{stem}_afm_sites.csv", folder=str(d), **keep)
    (d / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


def new_draft(b, out_dir, src=None):
    """(2026-11-03e, Library › AFM runs ＋ / Redo) a named AFM list not made yet: <out_dir>/<id>/plan.json with draft=True,
    no points.  b: dict(run, scan, name, by).  src: the list it redoes (its flakes are offered, its reference flakes and
    their AFM positions copied: the chip is usually still where it was on the AFM).  Returns the draft."""
    scan = str(b.get("scan") or (src or {}).get("scan") or "chip")
    pid = plan_id(scan)
    d = Path(out_dir) / pid
    k = 2
    while d.exists():                                   # (two in the same second)
        d, pid = Path(out_dir) / f"{plan_id(scan)}_{k}", f"{plan_id(scan)}_{k}"
        k += 1
    d.mkdir(parents=True)
    now = int(datetime.datetime.now().timestamp() * 1000)
    plan = dict(id=pid, draft=True, name=str(b.get("name") or "List")[:80], scan=scan, run=b.get("run") or (src or {}).get("run"),
                by=b.get("by"), created=now, updated=now, sites=[], results={}, folder=str(d))
    if src:
        plan.update({"from": src.get("id"), "flakes": [s["flake"] for s in src.get("sites") or [] if s.get("flake")],
                     "refs": src.get("refs") or [], "scan_um": src.get("scan_um"), "zoff": src.get("zoff")})
        if src.get("mount"):
            plan["mount"] = src["mount"]
    (d / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


def write_lists(d, stem, sites):
    """The point list (NanoScope programmed move, .pgm) and its csv; returns the .pgm path."""
    lines = ["Version: 1.0\r\n"]
    for s in sites:
        lines.append(f"\\X point: {n2(s['afm'][0])}\r\n\\Y point: {n2(s['afm'][1])}\r\n"
                     + (f"\\Z point: {n2(s['afm'][2])}\r\n" if len(s["afm"]) > 2 else "") + "\\Coordinate System: 0\r\n")
    pgm = Path(d) / f"{stem}_afm_sites.pgm"
    pgm.write_bytes("".join(lines).encode("ascii"))
    with open(Path(d) / f"{stem}_afm_sites.csv", "w", newline="") as fh:
        fh.write("n,flake,terrace,afm_x,afm_y,afm_z,chip_x_um,chip_y_um,stage100_x_um,stage100_y_um,t_nm_optical\r\n")
        for s in sites:
            st = s.get("st") or ["", ""]
            fh.write(f"{s['n']},{s['flake']},{s.get('terrace', '')},{s['afm'][0]},{s['afm'][1]},{s['afm'][2] if len(s['afm']) > 2 else ''},"
                     f"{round(s['chip'][0], 2)},{round(s['chip'][1], 2)},{st[0]},{st[1]},{s.get('t_nm', '')}\r\n")
    return pgm


def pgm_parts(text):
    """A NanoScope point list (.pgm) -> (header lines, point lines, number of points): the header is what comes before the
    first backslash line ("Version: 1.0"), the points are the rest (\\X point, \\Y point, \\Z point, \\Coordinate System)."""
    lines = [q for q in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    i = next((k for k, q in enumerate(lines) if q.lstrip().startswith("\\")), len(lines))
    pts = lines[i:]
    return lines[:i], pts, sum(1 for q in pts if q.lstrip().lower().startswith("\\x point"))


def merge_pgm(first, second):
    """Two point lists (.pgm text) as one: the header of the first, all the points of the first, then all of the
    second, in their order (CRLF lines, as the designer writes them).  Returns (text, points of the first, of the second)."""
    h1, p1, n1 = pgm_parts(first)
    h2, p2, n2 = pgm_parts(second)
    if not n1 or not n2:
        raise ValueError(f"no points in the {'first' if not n1 else 'second'} file (\\X point lines)")
    return "".join(q + "\r\n" for q in (h1 or h2 or ["Version: 1.0"]) + p1 + p2), n1, n2


def _offset_sites(sites, tip):
    """afm = afm0 - the tip offset (X, Y; Z as it was)."""
    o = (tip or {}).get("offset") or [0.0, 0.0]
    for s in sites:
        a0 = s.get("afm0") or s["afm"]
        s["afm"] = [round(a0[0] - float(o[0]), 2), round(a0[1] - float(o[1]), 2)] + list(a0[2:])


def apply_tip(plan, plan_file, tip):
    """Take the tip offset (tip: dict(offset=[dx, dy] AFM stage units, site, orient, score, ...), or None: no offset)
    off every point and write the list again; returns the plan."""
    for s in plan["sites"]:
        s.setdefault("afm0", list(s["afm"]))
    _offset_sites(plan["sites"], tip)
    plan["tip"] = tip
    plan["updated"] = int(datetime.datetime.now().timestamp() * 1000)
    write_lists(Path(plan_file).parent, Path(plan["pgm"]).name[:-len("_afm_sites.pgm")], plan["sites"])
    Path(plan_file).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


# the 8 ways the AFM image's axes (u: right, v: down) can lie on the stage's X, Y: d_stage = S @ (du, dv)
ORIENTS = [np.array(m, float) for m in ([[1, 0], [0, 1]], [[1, 0], [0, -1]], [[-1, 0], [0, 1]], [[-1, 0], [0, -1]],
                                         [[0, 1], [1, 0]], [[0, 1], [-1, 0]], [[0, -1], [1, 0]], [[0, -1], [-1, 0]])]
ORIENT_NAMES = ["as the stage", "Y flipped", "X flipped", "turned 180°", "X and Y swapped", "turned 90°", "turned 270°",
                "swapped and turned 180°"]


def _edges(img, sigma=1.0):
    import cv2
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (np.percentile(m, 99) + 1e-9)


def afm_to_optical(A, orient, afm_px_um, opt_px_um):
    """The linear map from AFM image pixels (u, v) to the optical picture's pixels (x, y) for orientation `orient`:
    stage = S (du, dv) * afm_px_um * scale; chip = A^-1 stage (A = scale * an orthonormal matrix)."""
    A = np.asarray(A, float)
    sc = float(np.sqrt(abs(np.linalg.det(A))))
    return np.linalg.inv(A / sc) @ ORIENTS[orient] * (afm_px_um / opt_px_um)


def tip_offset(z, scan_um, opt_rgb, opt_px_um, site_px, A, orient=None, prefer=None, overlay=None):
    """Where the AFM image (levelled height z, scan_um square; nan = bad) really is in the site's 100x picture (opt_rgb,
    opt_px_um per pixel; the site, i.e. where the scan was meant to be centred, at site_px).  orient: force one of
    ORIENTS; else all are tried (prefer: the one used before, kept unless another is clearly better).  Returns
    dict(orient, score, scores, d_px (picture pixels from the site to the scan's centre), d_um, offset (AFM stage
    units: taken off every point), ok); overlay: a path for the AFM height turned into the picture's orientation (PNG
    with transparency, one pixel per picture pixel), its size and centre in the result."""
    import cv2
    z = np.asarray(z, float)
    good = np.isfinite(z)
    if good.mean() < 0.3:
        raise ValueError("the AFM image has too little good data")
    z = np.where(good, z, np.nanmedian(z))
    opt = np.asarray(opt_rgb, np.float32)
    gray = opt.mean(axis=2) if opt.ndim == 3 else opt
    n = max(8, int(round(scan_um / opt_px_um)))             # the scan's size in picture pixels
    zr = cv2.resize(z.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)
    Eo, Ea = _edges(gray), _edges(zr)
    afm_px_um = scan_um / n
    m = int(np.ceil(n * 1.4143)) + 2
    res = []
    for k in (range(len(ORIENTS)) if orient is None else [int(orient)]):
        L = afm_to_optical(A, k, afm_px_um, opt_px_um)
        M = np.c_[L, np.array([m / 2.0, m / 2.0]) - L @ np.array([n / 2.0, n / 2.0])]
        T = cv2.warpAffine(Ea, M, (m, m), flags=cv2.INTER_LINEAR, borderValue=0)
        W = cv2.warpAffine(np.ones((n, n), np.float32), M, (m, m), flags=cv2.INTER_NEAREST, borderValue=0)
        if Eo.shape[0] < m or Eo.shape[1] < m:
            raise ValueError("the 100x picture is smaller than the AFM scan: no room to find it")
        r = cv2.matchTemplate(Eo, T, cv2.TM_CCOEFF_NORMED, mask=W)
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)
        y, x = np.unravel_index(int(np.argmax(r)), r.shape)
        res.append(dict(orient=k, score=float(r[y, x]), centre=[x + m / 2.0, y + m / 2.0], M=M))
    res.sort(key=lambda q: -q["score"])
    best = res[0]
    if prefer is not None and orient is None:
        p = next(q for q in res if q["orient"] == int(prefer))
        if p["score"] >= best["score"] - 0.05:
            best = p
    d_px = np.array(best["centre"]) - np.asarray(site_px, float)
    d_um = d_px * opt_px_um
    off = np.asarray(A, float) @ d_um                       # the scan was at commanded + off: take off off
    second = max([q["score"] for q in res if q["orient"] != best["orient"]] or [0.0])
    out = dict(orient=best["orient"], orient_name=ORIENT_NAMES[best["orient"]], score=round(best["score"], 3),
               scores={str(q["orient"]): round(q["score"], 3) for q in res}, margin=round(best["score"] - second, 3),
               d_px=[round(float(v), 2) for v in d_px], d_um=[round(float(v), 3) for v in d_um],
               offset=[round(float(v), 3) for v in off], scan_px=n,
               ok=bool(best["score"] > 0.3 and (orient is not None or best["score"] - second > 0.03)))
    if overlay:
        lo, hi = np.percentile(zr, (2, 98))
        u = np.clip((zr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        c = cv2.applyColorMap(u, cv2.COLORMAP_INFERNO)
        a = np.full((n, n), 255, np.uint8)
        M = best["M"]
        cw = cv2.warpAffine(c, M, (m, m), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        aw = cv2.warpAffine(a, M, (m, m), flags=cv2.INTER_NEAREST, borderValue=0)
        cv2.imwrite(str(overlay), np.dstack([cw, aw]))
        out.update(overlay=Path(overlay).name, overlay_px=m)
    return out


# (2026-11-05v) AFM screening reads a copy kept on this computer of each file on the share: the .spm, the height arrays
# and the picture.  The server sets LOCAL (DiskCache.file); a file changed on the share (size or time) is copied again.
LOCAL = None


def here(p):
    """The copy of file p on this computer (made now if there is none), or p itself (no copy could be made)."""
    if LOCAL is None:
        return p
    try:
        return LOCAL(Path(p)) or p
    except OSError:
        return p


def keep_local(plan_dir, result):
    """Everything screening reads for one AFM result copied to this computer."""
    plan_dir, png = Path(plan_dir), Path(result["png"])
    for f in (plan_dir / png.name, plan_dir / (png.stem + ".npz"), Path(result.get("source") or "")):
        if f.name and f.is_file():
            here(f)
    height_full(plan_dir, result)


def load_height(plan_dir, result):
    """The levelled height of an imported AFM image (its .npz; images imported before it was kept: read again)."""
    f = Path(plan_dir) / (Path(result["png"]).stem + ".npz")
    if f.is_file():
        d = np.load(here(f))
        return d["z"].astype(float), float(d["scan_um"])
    src = Path(result.get("source") or "")
    if not src.is_file():
        raise ValueError(f"the AFM file {result.get('file')} is no longer at {src}: import it again")
    from flakeid.afm import analyse_auto, read_spm
    img = read_spm(here(src))
    ch = result.get("channel") or next(iter(img.channels))
    r = analyse_auto(img[ch], img.pixel_nm)
    return np.where(r["bad"], np.nan, r["z"]), float(img.scan_um)


def shown_height(raw, r):
    """(2026-11-04h) The levelled height of every pixel, for pictures: the analysis's line offsets and background taken
    off the raw heights also where it masked the data as bad (spikes, particles, streaks and a 2-pixel margin: they
    were painted grey, as diamonds).  Pixels with no reading (the scanner's range clipped) are taken as the tallest.
    The measurements still leave the masked pixels out (r["z"], r["bad"])."""
    raw = np.asarray(raw, float)
    lo, pl = r.get("line_off"), r.get("plane")
    v = raw - (lo[:, None] if lo is not None else 0.0) - (pl if pl is not None else 0.0)
    good = np.isfinite(r["z"]) & np.isfinite(v)
    if good.any():
        v = v + float(np.median(r["z"][good] - v[good]))   # (the same zero as the measured heights)
    fin = np.isfinite(v)
    if not fin.all():
        v = np.where(fin, v, np.nanmax(v) if fin.any() else 0.0)
    return v


def height_picture(z, bad, path, width=520, shown=None):
    """The levelled height image (nm) as a colour picture (2-98 % range of the good data); returns (lo, hi) nm.
    shown: the heights painted (shown_height: every pixel on the colour scale); without it bad data is grey."""
    import cv2
    v = np.where(bad, np.nan, z)
    good = v[np.isfinite(v)]
    lo, hi = (np.percentile(good, (2, 98)) if good.size else (0.0, 1.0))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    p = v if shown is None else shown
    u = np.clip((np.nan_to_num(p, nan=lo) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    c = cv2.applyColorMap(u, cv2.COLORMAP_INFERNO)
    c[~np.isfinite(p)] = (128, 128, 128)
    h, w = c.shape[:2]
    c = cv2.resize(c, (width, int(round(width * h / w))), interpolation=cv2.INTER_NEAREST if w < width else cv2.INTER_AREA)
    cv2.imwrite(str(path), c)
    return float(lo), float(hi)


def height_full(plan_dir, result, maxpx=768):
    """(2026-11-02k, AFM screening) the levelled height of an imported AFM image at full resolution (<= maxpx a side),
    every pixel on the colour scale (shown_height, 2026-11-04h: before, NaN where bad, painted grey); cached next to the
    plan as <png stem>.full2.npz.  Returns (z float32, pixel_nm, full: bool).
    Without its .spm any more: the reduced copy kept at import (load_height), full=False."""
    f = Path(plan_dir) / (Path(result["png"]).stem + ".full2.npz")
    if f.is_file():
        d = np.load(here(f))
        return d["z"].astype(np.float32), float(d["pixel_nm"]), True
    src = Path(result.get("source") or "")
    if not src.is_file():
        z, scan_um = load_height(plan_dir, result)
        return z.astype(np.float32), scan_um * 1000.0 / max(1, z.shape[1]), False
    from flakeid.afm import analyse_auto, read_spm
    img = read_spm(here(src))
    ch = result.get("channel") if result.get("channel") in img.channels else next(iter(img.channels))
    r = analyse_auto(img[ch], img.pixel_nm)
    z = shown_height(img[ch], r).astype(np.float32)
    st = max(1, int(np.ceil(max(z.shape) / maxpx)))
    z = z[::st, ::st]
    px = float(img.pixel_nm) * st
    np.savez_compressed(f, z=z, pixel_nm=px)
    here(f)
    return z, px, True


def gwyddion_exe():
    """Gwyddion's program on this computer, or None."""
    import shutil
    for c in (shutil.which("gwyddion"), "/opt/homebrew/bin/gwyddion", "/usr/local/bin/gwyddion",
              "/Applications/Gwyddion.app/Contents/MacOS/Gwyddion", r"C:\Program Files\Gwyddion\bin\gwyddion.exe",
              r"C:\Program Files (x86)\Gwyddion\bin\gwyddion.exe"):
        if c and Path(c).is_file():
            return Path(c)
    return None


def open_in_gwyddion(exe, spm):
    """The file opened in Gwyddion (a separate program: the designer does not wait for it).  Gwyddion running already:
    it opens the file in that window (its --remote-new)."""
    import subprocess
    import sys
    kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sys.platform == "win32":
        kw["creationflags"] = 0x00000008             # (DETACHED_PROCESS)
    else:
        kw["start_new_session"] = True
    subprocess.Popen([str(exe)] + ([] if sys.platform == "win32" else ["--remote-new"]) + [str(spm)], **kw)


def analyse_file(spm, out_png):
    """One AFM file: dict(file, scan_um, pixel_nm, channel, noise_nm, bad_frac, levels=[{step_nm, err_nm, frac}], range_nm)."""
    from flakeid.afm import analyse_auto, read_spm
    img = read_spm(here(spm))
    ch = next((k for k in ("Height Sensor", "Height", "Height Trace") if k in img.channels), None) or next(iter(img.channels))
    r = analyse_auto(img[ch], img.pixel_nm)
    lo, hi = height_picture(r["z"], r["bad"], out_png, shown=shown_height(img[ch], r))
    zz = np.where(r["bad"], np.nan, r["z"]).astype(np.float32)        # (kept, reduced, for the tip offset)
    st = max(1, int(np.ceil(max(zz.shape) / 256)))
    np.savez_compressed(Path(out_png).with_suffix(".npz"), z=zz[::st, ::st].astype(np.float16), scan_um=float(img.scan_um))
    levels = []
    for i, (L, m) in enumerate(zip(r["levels"], r["level_masks"])):
        if "step_nm" not in L:
            continue
        levels.append(dict(step_nm=round(float(L.get("step_lines_nm", L["step_nm"])), 2), step_level_nm=round(float(L["step_nm"]), 2),
                           err_nm=round(float(L["step_err_nm"]), 2) if L.get("step_err_nm") == L.get("step_err_nm") and "step_err_nm" in L else None,
                           frac=round(float(m.mean()), 3)))
    levels.sort(key=lambda q: -q["frac"])
    return dict(file=Path(spm).name, scan_um=round(float(img.scan_um), 3), pixel_nm=round(float(img.pixel_nm), 3), channel=ch,
                noise_nm=round(float(r["noise_nm"]), 3), bad_frac=round(float(r["bad_frac"]), 3), levels=levels,
                range_nm=[round(lo, 2), round(hi, 2)], png=Path(out_png).name)
