"""Thickness ladders: one flake per thickness (every nm, or every 2 nm where 1 nm leaves too many gaps), in order, in a
PDF per material, to see by eye whether the 100x thickness model puts the colours in the right order (Avram,
2026-11-12: "a large pdf, one for bn and one for graphite ... hBN 0 to 100 nm, graphite 0 to 30 nm ... save all the file
locations for all these flakes, so that if in the future the thickness model changes, you can then instantly rerun the
estimates on these flakes and generate both pdfs").

    python tools/thickness_ladder.py select  [--days 14] [--share /Volumes/<share>]   # choose the flakes, save the lists
    python tools/thickness_ladder.py render  [--share ...] [--out DIR]                  # remeasure them, make the PDFs

select: every 100x picture of the runs (folders YYYYMMDD_run_N) of the last --days days under the exfoliation folders
(ROOTS, on the share) is read from its regions file (the scan's terrace outline and its stored colour), refitted with
the current model (flakeid.thick100.fit) and binned by thickness; the best few candidates per bin (a clean, large,
unambiguous terrace near the bin's centre: score()) are measured again from their pictures and the best one kept.  The
lists (ladders/<material>.json next to this file) keep, per flake, where its files are on the share (relative to the
share's root, so the same list works from a Mac's /Volumes/<share> and a PC's Z:\\) and the reading at selection.

render: every flake of the lists measured again from its picture with the current model (thick100.colours + fit, the
same as the scan program and the designer's Thickness cursor), sorted by that thickness, into
<out>/thickness_ladder_<material>.pdf: an overview (the terrace's colour, cut out, in order), then a page per flake (its
100x picture with the terrace outlined, the reading, the reading at selection, the files).  Only reads the share.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent / "ladders"
# the exfoliations on the share (relative to its root): material -> folders holding the chip series (BN_15_auto, ...);
# set once in FLAKEID_LADDER_ROOTS as JSON, e.g. {"hbn": ["<your name>/Exfoliation/BN"], "graphite": [...]}, or --roots
try:
    ROOTS = json.loads(os.environ.get("FLAKEID_LADDER_ROOTS", "") or "{}")
except ValueError:
    ROOTS = {}
RANGE = {"hbn": (0.0, 100.0), "graphite": (0.0, 30.0)}          # nm
STEP_FILL = 0.75             # every nm when that fills this share of the 1 nm bins; else every 2 nm
CANDIDATES = 3               # per bin, measured again from the picture at selection
RUN_RE = re.compile(r"^(\d{8})_run_\d+")
# a good example only (2026-11-12: the first hBN ladder had a "1 nm" picture whose flake the scan never found, and 96 -
# 100 nm ones with poor colour fits, 4 - 6 um clean areas, tape residue)
MAX_MISFIT = 0.30            # the colour fits its thickness this well
MIN_CLEAN_D_UM = 8.0         # the terrace's largest clean disc
MAX_GLUE = 0.30              # tape residue on the terrace


def usable(reg, r):
    """A picture good enough for the ladder: the scan found the flake in it and its terrace matches the 10x one, the
    colour fits well, a clean area and little residue."""
    seed = reg.get("seed") or {}
    if seed.get("found") is False or seed.get("fits") is False:
        return False
    if r is None or r.get("poor") or float(r.get("misfit") or 0.0) > MAX_MISFIT:
        return False
    if float(reg.get("glue") or 0.0) > MAX_GLUE:
        return False
    d = reg.get("clean_d_um")
    return d is None or float(d) >= MIN_CLEAN_D_UM


def default_share():
    """The share's root: FLAKEID_SHARE_ROOT (e.g. /Volumes/<share> on a Mac, Z:\\ on a PC), else "" (give --share)."""
    return os.environ.get("FLAKEID_SHARE_ROOT", "")


def rel_to(share, p):
    """p relative to the share's root ('/' separated), or None (not on the share)."""
    try:
        return Path(p).resolve().relative_to(Path(share).resolve()).as_posix()
    except ValueError:
        return None


def on_share(share, e, key):
    """The file of entry e (key 'regions' / 'frame' / 'look') on this computer: share + its relative path, else the
    absolute path it had at selection."""
    rel = (e.get("rel") or {}).get(key)
    if rel:
        return Path(share) / Path(*rel.split("/"))
    a = (e.get("abs") or {}).get(key)
    return Path(a) if a else None


# ---------------------------------------------------------------------------------------------------- the share
def _tries(fn, n=4, wait_s=3.0):
    """fn() with a few tries on a share that times out (OSError); None when it keeps failing."""
    import time
    for k in range(n):
        try:
            return fn()
        except OSError:
            if k < n - 1:
                time.sleep(wait_s * (k + 1))
    return None


def _isfile(p):
    return p is not None and bool(_tries(lambda: Path(p).is_file()))


CACHE = Path.home() / "flakeid_thickness_ladder" / "_regions_cache"     # (local copies of the regions files read)


def _json(p):
    """A JSON file of the share; regions files from the local copy while the share's has the same size and date (the
    share drops out: a run over ~2000 pictures then only asks the dates again)."""
    st = _tries(lambda: Path(p).stat())
    if st is None:
        return None
    import hashlib
    c = CACHE / (hashlib.md5(str(p).encode("utf-8")).hexdigest()[:12] + "_" + Path(p).name)
    key = f"{st.st_size}:{int(st.st_mtime)}"
    k = c.with_suffix(".key")
    if c.is_file() and k.is_file() and k.read_text() == key:
        try:
            return json.loads(c.read_text(encoding="utf-8"))
        except ValueError:
            pass
    t = _tries(lambda: Path(p).read_text(encoding="utf-8"))
    if t is not None and str(p).endswith("_regions.json"):
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            c.write_text(t, encoding="utf-8")
            k.write_text(key)
        except OSError:
            pass
    try:
        return json.loads(t) if t is not None else None
    except ValueError:
        return None


def _imread(p):
    """A picture (BGR) read whole from the share, or None."""
    import cv2
    b = _tries(lambda: Path(p).read_bytes())
    return None if not b else cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------------------------------- measuring
def oxide_of(reg, material):
    """The 100x model's oxide of a picture (as the designer's Thickness cursor): the run's calibrated one kept in
    t100.oxide_nm, else the chip's + the material's default shift."""
    from flakeid import thick100 as T
    t = reg.get("t100") or {}
    if t.get("oxide_nm") is not None:
        return float(t["oxide_nm"])
    return float(reg.get("oxide_nm") or 300.0) + T.OXIDE_SHIFT_DEFAULT.get(material, 0.0)


def refit(reg, material):
    """thick100.fit of the colour stored in the regions file (no picture read), or None."""
    from flakeid import thick100 as T
    t = reg.get("t100") or {}
    if t.get("contrast") is None:
        return None
    return T.fit(t["contrast"], material, oxide_of(reg, material), t.get("t10"))


def measure(frame_path, reg, material):
    """The terrace's thickness measured again from its picture (<frame>.jpg: linear, white-balanced) with the current
    model, as thick100.remeasure_viewer and the designer's Thickness cursor do: fit() + substrate ('frame' / 'material')
    + terrace_rgb, or None (no picture, no terrace)."""
    import cv2
    from flakeid import thick100 as T
    from flakeid.hundred import DEMO_SUBSTRATE, WHITE, WHITE_BY_MATERIAL, _M100
    if not reg.get("terrace") or frame_path is None:
        return None
    a = _imread(frame_path)
    if a is None:
        return None
    lin = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) + 0.5
    sub_run = np.asarray(DEMO_SUBSTRATE.get(material, DEMO_SUBSTRATE["hbn"]), float) * 255.0 / WHITE_BY_MATERIAL.get(material, WHITE)
    px_um = float(reg.get("px_um") or 0.0) or float(np.sqrt(abs(np.linalg.det(_M100()))))
    tc, sub, n = T.colours(lin, reg["terrace"], px_um, sub_run)
    src = "frame"
    if tc is None and n >= T.MIN_PX:
        tc, sub, src = None, sub_run, "material"
    if tc is None:
        return None
    t10 = (reg.get("t100") or {}).get("t10")
    r = T.fit(tc / np.maximum(sub, 1e-6) - 1.0, material, oxide_of(reg, material), t10)
    if r is None:
        return None
    r.update(n_px=int(n), sub_from=src, terrace_rgb=[round(float(v), 1) for v in tc])
    return r


def score(r, reg, centre, step):
    """Lower is better: the colour's misfit, other readings (hBN colours repeat), tape residue, a small or cut terrace,
    and the distance from the bin's centre.  The 100x thickness is trusted: the 10x one plays no part (Avram,
    2026-11-12: "if 10x and 100x disagree trust 100x"); it is only shown on the flake's page."""
    s = float(r.get("misfit") or 0.0)
    s += 0.15 * len(r.get("alts") or []) + (0.2 if r.get("poor") else 0.0)
    s += 0.5 * float(reg.get("glue") or 0.0)
    um2 = float(reg.get("clean_um2") or reg.get("terrace_um2") or 0.0)
    s += 0.1 if um2 < 100 else 0.05 if um2 < 400 else 0.0
    s += 0.05 if reg.get("cut_by_frame") else 0.0
    s += 0.1 * abs(float(r["t"]) - centre) / step
    return s


# ---------------------------------------------------------------------------------------------------- selecting
def runs_of(share, material, days, roots=None):
    """[(run folder, its date)] of the last `days` days under the material's exfoliation folders."""
    since = dt.date.today() - dt.timedelta(days=days)
    out = []
    if not roots and not ROOTS.get(material):
        raise SystemExit(f"no folders to search for {material}: set FLAKEID_LADDER_ROOTS or give --roots")
    if not roots and not share:
        raise SystemExit("no share: set FLAKEID_SHARE_ROOT or give --share")
    for r in roots or [Path(share) / Path(*x.split("/")) for x in ROOTS[material]]:
        r = Path(r)
        if not r.is_dir():
            print(f"  (not found: {r})")
            continue
        # (the chip series' folders hold the runs: <root>/<series>/<YYYYMMDD_run_N>; not a walk of the whole share)
        for d in sorted(set(r.glob("*_run_*")) | set(r.glob("*/*_run_*"))):
            m = RUN_RE.match(d.name)
            if not m or not d.is_dir() or "_designer_data" in d.parts:
                continue
            day = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            if day >= since:
                out.append((d, day))
    return out


def pictures_of(run):
    """The run's 100x pictures: [(regions file, frame .jpg, _look .jpg or None)]."""
    out = []
    for rj in sorted(run.glob("*/100x/*_regions.json")):
        stem = rj.name[:-len("_regions.json")]
        fr, lk = rj.with_name(stem + ".jpg"), rj.with_name(stem + "_look.jpg")
        out.append((rj, fr, lk if _isfile(lk) else None))
    return out


def entry(share, material, run, rj, fr, lk, reg, r, target):
    files = dict(regions=rj, frame=fr, look=lk)
    return dict(material=material, target_nm=target, run=run.name, run_folder=rel_to(share, run) or str(run),
                scan=rj.parent.parent.name, label=reg.get("label"), picture=fr.name,
                rel={k: rel_to(share, v) for k, v in files.items() if v is not None},
                abs={k: str(v) for k, v in files.items() if v is not None},
                at_selection=dict(date=dt.date.today().isoformat(), t=r["t"], misfit=r.get("misfit"), alts=r.get("alts"),
                                  agree=r.get("agree"), t10=r.get("t10"), contrast=r.get("contrast"),
                                  oxide_nm=r.get("oxide_nm"), na=r.get("na")))


def select(material, share, days, roots=None, log=print):
    lo, hi = RANGE[material]
    runs = runs_of(share, material, days, roots)
    log(f"{material}: {len(runs)} runs since {dt.date.today() - dt.timedelta(days=days)}: " + ", ".join(d.name for d, _ in runs))
    cands = []                                                  # (t, run, rj, fr, lk, reg, refit)
    for run, _ in runs:
        n0 = len(cands)
        for rj, fr, lk in pictures_of(run):
            reg = _json(rj)
            if reg is None:
                continue
            r = refit(reg, material)
            if r is None and reg.get("terrace") and _isfile(fr):
                r = measure(fr, reg, material)                  # (runs from before the colour was kept: the picture)
            if r is not None and lo < r["t"] <= hi + 0.5 and usable(reg, r) and _isfile(fr):
                cands.append((float(r["t"]), run, rj, fr, lk, reg, r))
        log(f"  {run.name}: {len(cands) - n0} pictures with a reading")
    centres1 = np.arange(max(1.0, lo), hi + 1e-9, 1.0)
    fill1 = np.mean([any(abs(c[0] - x) <= 0.5 for c in cands) for x in centres1]) if len(centres1) else 0.0
    step = 1.0 if fill1 >= STEP_FILL else 2.0
    log(f"  1 nm bins filled: {fill1:.0%} -> every {step:g} nm")
    centres = np.arange(max(step, lo), hi + 1e-9, step)
    if material == "graphite":                                 # (the thinnest too: 1 and 2 layers)
        centres = np.concatenate([[0.335, 0.67], centres])
    chosen, missing, used = [], [], set()
    for x in centres:
        half = 0.17 if x < 1.0 else step / 2
        inb = [c for c in cands if abs(c[0] - x) <= half and str(c[2]) not in used]
        best = sorted(inb, key=lambda c: score(c[6], c[5], x, step))[:CANDIDATES]
        got = None
        for t, run, rj, fr, lk, reg, r in best:                 # measured again from the picture
            m = measure(fr, reg, material)
            if m is not None and abs(m["t"] - x) <= half and usable(reg, m):
                s = score(m, reg, x, step)
                if got is None or s < got[0]:
                    got = (s, run, rj, fr, lk, reg, m)
        if got is None:
            missing.append(round(float(x), 2))
            continue
        _, run, rj, fr, lk, reg, m = got
        used.add(str(rj))
        chosen.append(entry(share, material, run, rj, fr, lk, reg, m, round(float(x), 3)))
    log(f"  chosen {len(chosen)} of {len(centres)}; none at {missing if missing else '-'} nm")
    return dict(material=material, made=dt.datetime.now().isoformat(timespec="seconds"), days=days, step_nm=step,
                range_nm=[lo, hi], runs=[rel_to(share, d) or str(d) for d, _ in runs], missing_nm=missing, flakes=chosen)


# ---------------------------------------------------------------------------------------------------- the PDF
def _crop(img, polys, px_um, half_um):
    """A square of 2 * half_um (at most the terrace's widest inside disc) around its widest inside point: only the
    terrace (img, (x0, y0))."""
    import cv2
    from flakeid import thick100 as T
    M = T.terrace_mask(img.shape, polys)
    d = cv2.distanceTransform(M, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    h = max(4, int(min(round(half_um / px_um), float(d.max()) * 0.7)))
    x0, y0 = max(0, x - h), max(0, y - h)
    return img[y0:y + h, x0:x + h], (x0, y0)


def render(material, share, out_dir, lst=None, lists=None, log=print):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from flakeid import thick100 as T
    L = lst or json.loads((Path(lists or HERE) / f"{material}.json").read_text(encoding="utf-8"))
    rows = []
    for e in L["flakes"]:
        rj, fr, lk = (on_share(share, e, k) for k in ("regions", "frame", "look"))
        reg = _json(rj) if rj is not None else None
        if reg is None:
            log(f"  not readable: {rj}")
            continue
        m = measure(fr, reg, material)
        if m is None:
            log(f"  no reading: {fr}")
            continue
        pic = lk if _isfile(lk) else fr
        a = _imread(pic)
        if a is None:
            log(f"  not readable: {pic}")
            continue
        img = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        rows.append((float(m["t"]), e, reg, m, img, pic))
    rows.sort(key=lambda r: r[0])
    model = T.PROFILE_100X.get(material, T.PROFILE_DEFAULT)
    out = Path(out_dir) / f"thickness_ladder_{material}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    name = {"hbn": "hBN", "graphite": "Graphite"}[material]
    fmt = (lambda t: f"{t:.2f} nm" if t < 10 else f"{t:.1f} nm") if material == "graphite" else (lambda t: f"{t:.1f} nm")
    with PdfPages(out) as pdf:
        # overview: the terraces' colours in order, 6 x 5 per page
        per = 30
        for k in range(0, max(1, len(rows)), per):
            fig, axs = plt.subplots(5, 6, figsize=(16.5, 11.7))
            fig.suptitle(f"{name} thickness ladder — {len(rows)} flakes, 100x natural colour, sorted by the current model "
                         f"({model}; {dt.date.today().isoformat()})", fontsize=13)
            for ax in axs.flat:
                ax.axis("off")
            for ax, (t, e, reg, m, img, pic) in zip(axs.flat, rows[k:k + per]):
                c, _ = _crop(img, reg["terrace"], float(reg.get("px_um") or 0.114), 12.0)
                ax.imshow(c)
                ax.set_title(f"{fmt(t)}\n{e['run']} {e.get('label') or ''}", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
        # a page per flake
        for t, e, reg, m, img, pic in rows:
            fig = plt.figure(figsize=(16.5, 11.7))
            ax = fig.add_axes([0.02, 0.06, 0.66, 0.86])
            ax.imshow(img)
            for p in reg["terrace"]:
                q = np.asarray(p, float).reshape(-1, 2)
                q = np.vstack([q, q[:1]])
                ax.plot(q[:, 0], q[:, 1], "-", color="white", lw=0.8, alpha=0.8)
            px_um = float(reg.get("px_um") or 0.114)
            bar = 20.0 / px_um
            h, w = img.shape[:2]
            ax.plot([w * 0.04, w * 0.04 + bar], [h * 0.95] * 2, "-", color="white", lw=3)
            ax.text(w * 0.04, h * 0.93, "20 µm", color="white", fontsize=10)
            ax.axis("off")
            fig.text(0.02, 0.95, f"{name}: {fmt(t)}", fontsize=26, weight="bold")
            s = e.get("at_selection") or {}
            sw = fig.add_axes([0.71, 0.70, 0.12, 0.2])
            cr, _ = _crop(img, reg["terrace"], px_um, 6.0)
            sw.imshow(cr)
            sw.axis("off")
            sw.set_title("terrace", fontsize=9)
            lines = [
                f"thickness (current model): {fmt(t)}",
                f"  misfit {m.get('misfit')}" + (f", also fits {', '.join(fmt(a) for a in m['alts'])}" if m.get("alts") else ""),
                f"  10x thickness {m['t10']} nm (not used: the 100x is trusted)" if m.get("t10") is not None else "  10x thickness: -",
                f"  colour against the substrate R G B {m.get('contrast')}",
                f"  100x model oxide {m.get('oxide_nm')} nm, NA {m.get('na')} ({model})",
                "",
                f"at selection ({s.get('date')}): {fmt(float(s['t']))} (bin {e['target_nm']} nm)" if s.get("t") is not None else "",
                "",
                f"run {e['run']}, scan {e['scan']}, {e.get('label') or ''}",
                f"glue {reg.get('glue')}, clean terrace {reg.get('clean_um2')} µm²",
                "",
                "files (on the share):",
            ] + [f"  {k}: {(e.get('rel') or {}).get(k) or (e.get('abs') or {}).get(k)}" for k in ("frame", "look", "regions")]
            y = 0.66
            for ln in lines:
                for part in [ln[i:i + 52] for i in range(0, max(1, len(ln)), 52)]:
                    fig.text(0.70, y, part, fontsize=9, family="monospace")
                    y -= 0.02
            pdf.savefig(fig)
            plt.close(fig)
    log(f"  {out}: {len(rows)} flakes")
    return out, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "render", "both"])
    ap.add_argument("--material", choices=["hbn", "graphite", "all"], default="all")
    ap.add_argument("--share", default=default_share(), help="the share's root (Mac /Volumes/<share>, PC Z:\\; default FLAKEID_SHARE_ROOT)")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--roots", nargs="*", help="folders to search instead of FLAKEID_LADDER_ROOTS")
    ap.add_argument("--lists", default=str(HERE), help="where the lists are kept (default: ladders/ next to this file)")
    ap.add_argument("--out", default=str(Path.home() / "flakeid_thickness_ladder"))
    a = ap.parse_args(argv)
    lists = Path(a.lists)
    mats = ["hbn", "graphite"] if a.material == "all" else [a.material]
    for mat in mats:
        if a.cmd in ("select", "both"):
            L = select(mat, a.share, a.days, a.roots)
            if not L["runs"] or not L["flakes"]:            # (the share not reachable: the list kept as it was)
                sys.exit(f"{mat}: no runs or no flakes found (is the share mounted at {a.share}?): the list is not changed")
            lists.mkdir(parents=True, exist_ok=True)
            (lists / f"{mat}.json").write_text(json.dumps(L, indent=1), encoding="utf-8")
            print(f"  list: {lists / (mat + '.json')}")
        if a.cmd in ("render", "both"):
            render(mat, a.share, a.out, lists=lists)


if __name__ == "__main__":
    main()
