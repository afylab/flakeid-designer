#!/usr/bin/env python3
"""A run's flakes for the vdW Stack Designer, as a bundle anyone with edit access adds from the designer page
("Add scan results"): designer/bundle.json + a few images.

    python tools/make_designer_bundle.py <run folder on the share>              -> <run folder>/designer/
    python tools/make_designer_bundle.py <piece results> ... -o designer --run pleasework_run_1

A run folder is the copy on the share (one folder per piece with its viewer on top) or the analysis's results folder
(one folder per piece with <name>_tiles_viewer/ in it); the pieces are found in it.  The flakes are those of
tools/make_catalog_100x.py (imaged at 100x, thickness from the measured calibrations, not screened out), with ids
prefixed by the run name so that runs never collide.  Each 100x view (natural, high contrast, thickness change, IR)
is a file of its own, JPEG quality 95 at full colour resolution (--sheets: packed into sheets of SHEET images as before
2026-10-16: fewer files for the claude.ai page, which stores them per upload).  Picture names end with a hash of their
content (a rebuilt run's unchanged pictures keep their names; the designer replaces bundle.json last).  The step lines
(drawn only in the 100x picture window) are in detail_<hash>.json, read when a picture is opened; the terraces' debris
and step outlines and features (used by no page) in extra_<hash>.json; outlines are rounded to whole cut-out pixels
(0.23 um, far below what the designer resolves).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
KIND, VERSION = "flakeid-designer-bundle", 2
SHEET = 12                     # 100x frames per sheet (4 x 3: ~5500 x 3100 px)
COLS = 4
# 100x views: JPEG quality 95 at full colour resolution (4:4:4: the WebP sheets' half colour resolution showed blotchy
# substrate texture near flakes, 2026-10-15)
JPEG_VIEW = [cv2.IMWRITE_JPEG_QUALITY, 95] + ([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
                                              if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") else [])


def piece_dirs(folder: Path):
    """The piece result folders in a run folder (either layout), or [folder] if it is one itself."""
    def is_piece(d):
        return (d / "index.html").is_file() and (d / "data").is_dir() or any(d.glob("*_viewer/index.html"))
    if is_piece(folder):
        return [folder]
    out = []
    for base in (folder, folder / "results"):
        if base.is_dir():
            out += [d for d in sorted(base.iterdir()) if d.is_dir() and is_piece(d)]
    return out


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.~@+-]", "_", s)


def _ping(x):
    """(worker processes start and answer: the app's --check-builder)"""
    return 2 * x


def _terraces_one(job):
    """One flake's step terraces from its 100x image (in a worker process): (index, the flake, added, log lines)."""
    from terraces100 import add_step_terraces
    i, f, img = job
    lines = []
    k = add_step_terraces(f, Path(img), log=lines.append)
    return i, f, k, lines


def publish(src: Path, dst: Path, prog=None, keep=None):
    """The finished build (src, on this computer) into dst (e.g. the run's designer folder on the share) without a
    moment where the designer finds a half-written run: each file copied under a temporary name and renamed; files
    whose name ends with their content's hash and are already there are not copied again; bundle.json and run.json
    last; the old build's other files removed after (not ".building": the designer's lock).  keep: an update (src holds
    only the new files): the files of dst still in use, the others removed."""
    import os
    dst.mkdir(parents=True, exist_ok=True)
    names = sorted((q.name for q in src.iterdir() if q.is_file()), key=lambda n: (n == "bundle.json", n == "run.json", n))
    for k, n in enumerate(names, 1):
        s, d = src / n, dst / n
        if prog:
            prog(k / len(names), f"copying the finished files: {k} of {len(names)}")
        if n not in ("bundle.json", "run.json", "flakes.csv") and d.is_file() and d.stat().st_size == s.stat().st_size \
                and re.search(r"_[0-9a-f]{10}\.[a-z]+$", n):
            continue
        part = dst / (n + ".part")
        shutil.copyfile(s, part)
        os.replace(part, d)
    keep = set(names) | set(keep or ())
    for q in dst.iterdir():
        if q.is_file() and q.name not in keep and not q.name.startswith(".building"):
            try:
                q.unlink()
            except OSError:
                pass
    old = dst / "_catalog"                              # (a working folder left by a build before 2026-10-20)
    if old.is_dir():
        shutil.rmtree(old, ignore_errors=True)


_LAST = [0.0, -1.0]


def progress(frac, text, every=0.5):
    """The build's progress: on a terminal a bar; else a line "PROGRESS <fraction> <text>" (read by the designer, which
    draws its bar from it).  At most every `every` s (and always at the end)."""
    now = time.time()
    if frac < 1.0 and now - _LAST[0] < every:
        return
    _LAST[0] = now
    frac = max(0.0, min(1.0, float(frac)))
    try:
        if sys.stdout is not None and sys.stdout.isatty():
            w = 30
            n = int(round(w * frac))
            sys.stdout.write(f"\r[{'#' * n}{'-' * (w - n)}] {frac * 100:3.0f}%  {text[:70]:<70s}" + ("\n" if frac >= 1.0 else ""))
        else:
            sys.stdout.write(f"PROGRESS {frac:.3f} {text}\n")
        sys.stdout.flush()
    except Exception:
        pass


def terraces_parallel(flakes, todo, src: Path, workers=None, prog=None):
    """add_step_terraces on flakes[i] for i in todo, spread over worker processes (each flake is independent; the
    slowest step of a build: ~10 of 15 min for 166 hBN flakes on one core, 20261014_run_1); one at a time if
    processes cannot be started.  Returns (terraces added, flakes with any, processes used)."""
    import os
    jobs = [(i, flakes[i], str(src / flakes[i]["img"])) for i in todo]
    nw = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    nw = max(1, min(nw, len(jobs)))
    done = {}
    if nw > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            with ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn")) as ex:
                for i, f, k, lines in ex.map(_terraces_one, jobs, chunksize=2):
                    done[i] = (f, k, lines)
                    if prog:
                        prog(len(done) / max(len(jobs), 1), f"terraces in the 100x pictures: {len(done)} of {len(jobs)} flakes")
        except Exception as e:                       # (no worker processes here: the rest one at a time)
            print(f"  (100x terraces: worker processes failed ({type(e).__name__}: {e}); one at a time)")
            nw = 1
    n_new = n_fl = 0
    for n_job, job in enumerate(jobs, 1):
        i = job[0]
        if i not in done and prog:
            prog(n_job / max(len(jobs), 1), f"terraces in the 100x pictures: {n_job} of {len(jobs)} flakes")
        f, k, lines = done[i] if i in done else _terraces_one(job)[1:]
        flakes[i] = f
        for ln in lines:
            print(ln)
        n_new += k
        n_fl += bool(k)
    return n_new, n_fl, nw


def snap_outlines(flakes):
    """The refined flake outlines (flakeid.outline100, made by the catalog without crystal directions) turned onto the
    crystal directions tools/terraces100.py found (crystal_deg), before slim rounds them to whole pixels."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from flakeid.outline100 import snap_flakes
        snap_flakes(flakes, log=lambda s: print("  " + s))
    except Exception as e:                           # (the outlines then stay as refined)
        print(f"  (outlines not turned onto the crystal directions: {type(e).__name__}: {e})")


def pack_views(cat, src: Path, out: Path, log=print):
    """Replace each flake's img / img_c8 / img_dt / img_ir (file names) with {s: sheet file, x, y, w, h}."""
    sheets = []
    for f in cat["flakes"]:                         # (terrace mosaics: too big for sheets; in one-file bundles only)
        f.pop("img_mos", None), f.pop("mos", None), f.pop("img_mos_look", None)
    for key in ("img", "img_look", "img_c8", "img_dt", "img_ir"):
        items = [f for f in cat["flakes"] if f.get(key)]
        tag = {"img": "nat", "img_look": "look", "img_c8": "c8", "img_dt": "dt", "img_ir": "ir"}[key]
        for k0 in range(0, len(items), SHEET):
            chunk = items[k0:k0 + SHEET]
            ims = [cv2.imread(str(src / f[key]), cv2.IMREAD_COLOR) for f in chunk]
            cw = max(i.shape[1] for i in ims)
            ch = max(i.shape[0] for i in ims)
            rows = (len(ims) + COLS - 1) // COLS
            cols = min(COLS, len(ims))
            canvas = np.zeros((rows * ch, cols * cw, 3), np.uint8)
            # JPEG at quality 95 with full colour resolution (JPEG_VIEW)
            name = f"x100_{tag}_{k0 // SHEET:02d}.jpg"
            for j, (f, im) in enumerate(zip(chunk, ims)):
                x, y = (j % COLS) * cw, (j // COLS) * ch
                canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
                f[key] = dict(s=name, x=x, y=y, w=int(im.shape[1]), h=int(im.shape[0]))
            cv2.imwrite(str(out / name), canvas, JPEG_VIEW)
            sheets.append(name)
    log(f"  100x views packed into {len(sheets)} sheets")
    return sheets


def hashed(name, data: bytes):
    """name_<10 hex of the content's hash>.ext"""
    import hashlib
    st, ext = (name.rsplit(".", 1) + [""])[:2]
    return f"{st}_{hashlib.sha1(data).hexdigest()[:10]}" + (f".{ext}" if ext else "")


def put_hashed(src: Path, out: Path, stem: str):
    """Copy src into out as <stem>_<hash>.<ext> (the bytes as they are); returns the new name."""
    b = src.read_bytes()
    n = hashed(stem + src.suffix.lower(), b)
    (out / n).write_bytes(b)
    return n


def view_bytes(path: Path):
    """A 100x view as the designer gets it (JPEG_VIEW); a source that is already smaller (a lower quality) as it is,
    without a second loss."""
    b = path.read_bytes()
    a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise IOError(f"{path} could not be decoded")
    ok, enc = cv2.imencode(".jpg", a, JPEG_VIEW)
    return enc.tobytes() if ok and (len(enc) < len(b) or path.suffix.lower() not in (".jpg", ".jpeg")) else b


def view_files(cat, src: Path, out: Path, log=print, workers=6, prog=None):
    """Each flake's img / img_c8 / img_dt / img_ir (paths in the catalog folder) -> a file of its own in out,
    x100_<view>_<flake>_<hash>.jpg (view_bytes; in threads: OpenCV works outside Python's lock)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(f, key, tag) for key, tag in (("img", "nat"), ("img_look", "look"), ("img_c8", "c8"), ("img_dt", "dt"), ("img_ir", "ir"),
                                           ("img_mos", "mos"), ("img_mos_look", "moslook"))
            for f in cat["flakes"] if isinstance(f.get(key), str)]

    def one(job):
        f, key, tag = job
        b = view_bytes(src / f[key])
        n = hashed(f"x100_{tag}_{safe(f['id'])}.jpg", b)
        (out / n).write_bytes(b)
        return n
    names = []
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for n in ex.map(one, jobs):
            names.append(n)
            if prog:
                prog(len(names) / max(len(jobs), 1), f"writing the 100x views: {len(names)} of {len(jobs)}")
    for (f, key, _), n in zip(jobs, names):
        f[key] = n
    log(f"  100x views: {len(names)} pictures ({sum((out / n).stat().st_size for n in names) / 1e6:.0f} MB)")
    return names


def rnd(v):
    """Coordinates to whole pixels (nested lists)."""
    if isinstance(v, list):
        return [rnd(x) for x in v]
    return int(round(v)) if isinstance(v, float) else v


def slim(cat):
    """Outlines rounded to whole cut-out pixels, and what the designer's lists and layouts do not use moved out of the
    flakes -> (detail, extra): detail = the step lines (drawn in the 100x picture window: loaded when one is opened),
    {flake id: {step_lines}}; extra = the terraces' debris / step outlines and features (no page uses them: kept for
    scripts), {flake id: {terrace id: {debris, steps, features}}}."""
    det, ext = {}, {}
    for f in cat["flakes"]:
        if "step_lines" in f:
            sl = f.pop("step_lines") or []
            det[f["id"]] = dict(step_lines=[dict(x, p=rnd(x["p"])) if isinstance(x, dict) and "p" in x else x for x in sl])
            f["n_step_lines"] = len(sl)
        f["outline"] = rnd(f.get("outline") or [])
        td = {}
        for t in f["terraces"]:
            x = {k: rnd(t.pop(k)) for k in ("debris", "steps", "features") if k in t}
            if any(x.values()):
                td[t["id"]] = x
            for k in ("poly", "poly_10x", "wrinkles"):
                if k in t:
                    t[k] = rnd(t[k])
            if t.get("fine"):
                t["fine"] = [dict(q, poly=rnd(q["poly"])) for q in t["fine"]]
        if td:
            ext[f["id"]] = td
    return dict(kind="flakeid-designer-detail", flakes=det), dict(kind="flakeid-designer-extra", flakes=ext)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="a run folder, or piece result folders")
    ap.add_argument("-o", "--out", help="output folder (default: <run folder>/designer)")
    ap.add_argument("--run", help="run name (default: the run folder's name)")
    ap.add_argument("--title", help="shown in the designer's list of runs (default: the run name)")
    ap.add_argument("--workers", type=int, default=None, help="processes for the hBN 100x terraces (default: cores - 1, "
                    "at most 6; 1 = one at a time)")
    ap.add_argument("--sheets", action="store_true", help="100x views packed into sheets (the bundles before 2026-10-16; "
                    "fewer files for the claude.ai page) instead of one file per picture")
    ap.add_argument("--no-terraces100", action="store_true", help="only the terrace at the 100x image centre (no "
                    "terraces bounded by the step lines found in the 100x image, tools/terraces100.py)")
    ap.add_argument("--update", action="store_true", help="add to the run's built bundle only what is missing: flakes "
                    "whose 100x pictures were attached since (tools/reattach_100x.py), and the whole-terrace outline of "
                    "flakes with a terrace mosaic; only their pictures are read, the other files stay as they are")
    a = ap.parse_args(argv)
    ins = [Path(p).expanduser().resolve() for p in a.inputs]
    pieces = [d for p in ins for d in piece_dirs(p)]
    if not pieces:
        print("no piece results (a viewer with 100x images) found in " + ", ".join(map(str, ins)))
        return 1
    run_folder = ins[0] if len(ins) == 1 and pieces[0] != ins[0] else None
    run = safe(a.run or (run_folder.name if run_folder else re.sub(r"_\d+$", "", pieces[0].name)))
    final = Path(a.out) if a.out else (run_folder / "designer" if run_folder else Path("designer"))
    if a.update:
        return update(a, pieces, final, run, run_folder)
    # all the work in a folder on this computer, the finished files copied to `final` at the end (publish): with -o on
    # the share the build read its own working pictures back over the network (2026-10-20: the hBN terraces 2-3x
    # slower on the stacker PC), and the run's old designer files were deleted at the start
    import tempfile
    local = final.resolve() if final.exists() else final.absolute()
    staged = Path(tempfile.gettempdir()).resolve() not in local.parents
    work = Path(tempfile.mkdtemp(prefix="vdw_build_")) if staged else None
    out = work / "designer" if staged else final
    tmp = (work if staged else out) / "_catalog"
    if not staged and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True)
    print(f"run {run}: {len(pieces)} piece(s): {', '.join(p.name for p in pieces)}")
    import make_catalog_100x
    from make_catalog_100x import main as catalog
    # progress: the catalog (reading the results and 100x pictures) 0-60 %, hBN terraces 60-85 %, the views 85-98 %
    make_catalog_100x.PROGRESS = lambda f, t: progress(0.60 * f, t)
    progress(0.0, "reading the run's results")
    rc = catalog([str(p) for p in pieces] + ["-o", str(tmp)])
    if rc not in (0, None) or not (tmp / "catalog.json").is_file():
        print("the catalog could not be made")
        return 1
    cat = json.loads((tmp / "catalog.json").read_text())
    if not cat["flakes"]:
        print("no flakes imaged at 100x left in the catalog: nothing to add")
        shutil.rmtree(tmp)
        return 1
    if not a.no_terraces100:
        # the flake's other terraces in its 100x image (thickness steps), with thicknesses from their colour, and terrace
        # a's own 100x region (materials: terraces100.TERRACE_MATERIALS; graphite since 2026-10-22)
        from terraces100 import TERRACE_MATERIALS
        todo = [i for i, f in enumerate(cat["flakes"])
                if f["material"] in TERRACE_MATERIALS and isinstance(f.get("img"), str) and f["terraces"]]
        t0 = time.time()
        n_new, n_fl, nw = terraces_parallel(cat["flakes"], todo, tmp, a.workers, prog=lambda f, t: progress(0.60 + 0.25 * f, t))
        mats = sorted({cat["flakes"][i]["material"] for i in todo})
        print(f"terraces from the 100x images ({', '.join(mats) or 'none'}): {n_new} more on {n_fl} flakes "
              f"({time.time() - t0:.0f} s, {nw} process{'es' if nw > 1 else ''})")
    snap_outlines(cat["flakes"])
    files = []
    if a.sheets:                                         # (fixed names, as before)
        for n in cat["atlases"]:
            shutil.copy(tmp / n, out / n)
            files.append(n)
        for t in cat["thumbs"]:
            shutil.copy(tmp / t["file"], out / t["file"])
            files.append(t["file"])
        for s in cat["scans"].values():
            shutil.copy(tmp / s["file"], out / s["file"])
            files.append(s["file"])
        files += pack_views(cat, tmp, out)
    else:
        cat["atlases"] = [put_hashed(tmp / n, out, Path(n).stem) for n in cat["atlases"]]
        files += cat["atlases"]
        for t in cat["thumbs"]:
            t["file"] = put_hashed(tmp / t["file"], out, Path(t["file"]).stem)
            files.append(t["file"])
        for s in cat["scans"].values():
            s["file"] = put_hashed(tmp / s["file"], out, Path(s["file"]).stem)
            files.append(s["file"])
        files += view_files(cat, tmp, out, prog=lambda f, t: progress(0.85 + 0.13 * f, t))
    prefix_ids(cat, run)
    detail = extra = None
    if not a.sheets:
        for k, d in zip(("detail", "extra"), slim(cat)):
            db = json.dumps(d, separators=(",", ":")).encode("utf-8")
            n = hashed(f"{k}.json", db)
            (out / n).write_bytes(db)
            files.append(n)
            detail, extra = (n, extra) if k == "detail" else (detail, n)
    big = [n for n in files if (out / n).stat().st_size > 19.5 * 2 ** 20]
    if big:
        print(f"files over the designer's 20 MB upload limit: {big}")
        return 1
    mats = sorted({f["material"] for f in cat["flakes"]})
    bundle = dict(kind=KIND, version=VERSION, run=run, title=a.title or run, created=time.strftime("%Y-%m-%d %H:%M"),
                  note=cat.get("note"), atlas_size=cat["atlas_size"], atlases=cat["atlases"], thumbs=cat["thumbs"],
                  scans=cat["scans"], flakes=cat["flakes"], materials=mats, files=files, detail=detail, extra=extra,
                  left_out=cat.get("left_out") or {}, **run_mounts(run_folder, pieces))
    copy_mark(run_folder, bundle, a.title)
    write_meta(out, bundle)
    shutil.rmtree(tmp)
    mb = sum((out / n).stat().st_size for n in files + ["bundle.json"]) / 1e6
    if staged:
        publish(out, final, prog=lambda f, t: progress(0.98 + 0.02 * f, t))
        shutil.rmtree(work, ignore_errors=True)
        out = final
    progress(1.0, "done")
    print(f"designer bundle: {out}  ({len(cat['flakes'])} flakes, {', '.join(mats)}; {len(files) + 1} files, {mb:.0f} MB)")
    print("add it in the vdW Stack Designer: Add scan results, then choose this folder")
    return 0


def run_mounts(run_folder, pieces):
    """Mounted 1 cm chips (flakeid.scope chip_type mounted_1cm): the run file's blocks, {mounts: [{mount, corners
    [[x, y, z] 10x stage um], pieces}], chip_type}, or {} (the AFM list of a block: its chips in one stage frame)."""
    cands = []
    for d in {Path(run_folder)} if run_folder else {Path(p).parent for p in pieces}:
        cands += sorted(d.glob("*_run_*.json"))
    for f in cands:
        try:
            m = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if m.get("mounts"):
            return dict(mounts=m["mounts"], chip_type=m.get("chip_type"))
    return {}


def prefix_ids(cat, run):
    """ids prefixed by the run (":" is allowed in the designer's database paths, "/" is not)."""
    for f in cat["flakes"]:
        f["id"] = f"{run}:{f['id']}"
        for t in f["terraces"]:
            t["id"] = f"{run}:{t['id']}"
        for k in ("source", "image100"):
            f.pop(k, None)


PARTIAL = re.compile(r" \(\d+ of \d+ chips so far\)$")


def copy_mark(run_folder, bundle, title=None):
    """The run copy's copy_done.json (flakeid.scope.scan.write_copy_done) into the bundle: source = the copy it was made
    from (the designer adds what is new with --update unless the copy asks for a whole build since), and while pieces
    are still to come (copied as each one's export ends) "(2 of 4 chips so far)" after the title."""
    cd = {}
    if run_folder is not None:
        try:
            cd = json.loads((Path(run_folder) / "copy_done.json").read_text())
        except (OSError, ValueError):
            cd = {}
    p = cd.get("partial")
    base = title or PARTIAL.sub("", bundle.get("title") or bundle["run"])
    bundle.update(source=cd.get("time"), title=base + (f" ({p['done']} of {p['of']} chips so far)" if p else ""))
    if p:
        bundle["partial"] = p
    else:
        bundle.pop("partial", None)


def write_meta(out: Path, bundle):
    """bundle.json, run.json (a small summary: the portable designer lists runs from it without reading the whole
    catalog over the network) and flakes.csv (one row per terrace, for Claude Code / scripts; the designer page makes
    the same table for the loaded runs) into out."""
    import csv as _csv
    (out / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")
    run_doc = dict(kind="flakeid-designer-run", run=bundle["run"], title=bundle["title"], created=bundle["created"],
                   n_flakes=len(bundle["flakes"]), materials=bundle["materials"], scans=list(bundle["scans"]),
                   files=bundle["files"], **({"updated": bundle["updated"]} if bundle.get("updated") else {}),
                   **{k: bundle[k] for k in ("source", "partial") if bundle.get(k)})
    (out / "run.json").write_text(json.dumps(run_doc, indent=1), encoding="utf-8")
    with open(out / "flakes.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["terrace_id", "material", "thickness_nm", "alt_thickness_nm", "layers", "inscribed_d_um", "area_um2", "length_um",
                    "width_um", "score", "flags", "screening", "clean_d_um", "wrinkles", "scan", "flake_area_um2", "flake_terraces"])
        for f in bundle["flakes"]:
            for t in f["terraces"]:
                wr = t.get("wrinkle") or {}
                w.writerow([t["id"], f["material"], t["t"], (t.get("alt") or "").split(";")[0],
                            t.get("layers") if f["material"] == "graphite" and t["t"] < 3 else "", t["d"], round(t["area"]),
                            t.get("L"), t.get("W"), t.get("score", f.get("score")), t.get("quality"), t.get("tier") or "",
                            t.get("clean_d_um", ""), wr.get("wrinkle_n", "") if isinstance(wr, dict) else "", f["scan"],
                            round(f.get("area") or 0), len(f["terraces"])])


def todo_flakes(pieces, bundle):
    """{piece name: {flake numbers}} an update makes (again): flakes with 100x pictures on the viewer page that the
    bundle neither has nor left out (pictures attached since: tools/reattach_100x.py), and flakes with a terrace mosaic
    whose outline is still the picture's (bundles before 2026-10-21), and flakes without an IR view whose 100x pictures
    have one now (the stacking-domain pictures are exported after the 100x ones, at the end of the run)."""
    from make_afm_set import load_viewer_data, viewer_dir
    from make_catalog_100x import h100
    have = {}
    for f in bundle["flakes"]:
        have.setdefault(f["scan"], {})[int(f["rank"])] = f
    left = {k: set(v) for k, v in (bundle.get("left_out") or {}).items()}
    todo = {}
    for p in pieces:
        stem = load_viewer_data(p)["name"].replace("_tiles", "")
        H = h100(viewer_dir(p))
        if "left_out" not in bundle:
            # (a bundle from before 2026-10-21 does not say which flakes the catalog left out: those whose pictures
            # were on the page already before tools/reattach_100x.py (its copy of the page) and are not in the bundle)
            bak = sorted(viewer_dir(p).glob("index.html.bak-*"))
            if bak:
                try:
                    m = re.search(r"const H100=(.*?);\n", bak[0].read_text(), re.S)
                    left[stem] = {int(k) for k in json.loads(m.group(1))} - set(have.get(stem, {}))
                except (OSError, ValueError, AttributeError):
                    pass
        for k, imgs in H.items():
            fid, f = int(k), have.get(stem, {}).get(int(k))
            if f is None:
                if fid not in left.get(stem, set()):
                    todo.setdefault(stem, set()).add(fid)
            elif any(e.get("mos") for e in imgs if not e.get("domain")) and not f["terraces"][0].get("terrace_from"):
                todo.setdefault(stem, set()).add(fid)
            elif not f.get("img_ir") and not f.get("ir_none") and any(e.get("ir") for e in imgs if not e.get("domain")):
                todo.setdefault(stem, set()).add(fid)    # (its IR picture exported after the flake was added)
    return todo


def update(a, pieces, final: Path, run, run_folder=None):
    """--update: the flakes todo_flakes finds made (tools/make_catalog_100x.py --only: only their pictures are read)
    and merged into the built bundle in final: their cut-outs and thumbnails in new atlas / thumbnail sheets, their
    views as new files, a flake made again replacing the old one (same id: the designer's notes on it stay); a piece new
    to the bundle (copied to the share as its export ended) with its chip overview; the hBN screening tiers of all its
    flakes scored again (flakeid.defects100.score_flakes).  Only the new files and bundle.json / run.json / flakes.csv /
    detail / extra are written to final."""
    import tempfile
    import make_catalog_100x
    from make_catalog_100x import main as catalog
    bj = final / "bundle.json"
    if not bj.is_file():
        print(f"no built bundle in {final} to update: build it without --update")
        return 1
    bundle = json.loads(bj.read_text(encoding="utf-8"))
    if bundle.get("version") != VERSION or not isinstance((bundle.get("flakes") or [{}])[0].get("img"), str):
        print("the bundle is an older kind (sheets): build it again without --update")
        return 1
    progress(0.0, "finding what is missing")
    todo = todo_flakes(pieces, bundle)
    n = sum(len(v) for v in todo.values())
    print(f"run {run}: {n} flake(s) to add or redo" + (": " + "; ".join(f"{k}: {', '.join(map(str, sorted(v)))}"
                                                                       for k, v in sorted(todo.items())) if n else ""))
    if not n:
        # (only the run copy's state, e.g. its last pieces' records: bundle.json written again, newer than copy_done.json,
        # so that the designer does not look again)
        old = (bundle.get("source"), bundle.get("title"))
        copy_mark(run_folder, bundle, a.title)
        if run_folder is not None and (bundle.get("source"), bundle.get("title")) != old:
            work = Path(tempfile.mkdtemp(prefix="vdw_update_"))
            try:
                write_meta(work, bundle)
                publish(work, final, keep=set(bundle["files"]))
            finally:
                shutil.rmtree(work, ignore_errors=True)
        progress(1.0, "nothing to update")
        return 0
    work = Path(tempfile.mkdtemp(prefix="vdw_update_"))
    try:
        out, tmp = work / "designer", work / "_catalog"
        out.mkdir(parents=True)
        tmp.mkdir()
        (work / "only.json").write_text(json.dumps({k: sorted(v) for k, v in todo.items()}))
        make_catalog_100x.PROGRESS = lambda f, t: progress(0.70 * f, t)
        new_pieces = [k for k in todo if k not in bundle["scans"]]   # (pieces copied since: their chip overview too)
        rc = catalog([str(p) for p in pieces] + ["-o", str(tmp), "--only", str(work / "only.json")]
                     + [x for k in new_pieces for x in ("--overview", k)])
        if rc not in (0, None) or not (tmp / "catalog.json").is_file():
            print("the catalog could not be made")
            return 1
        cat = json.loads((tmp / "catalog.json").read_text())
        if not a.no_terraces100:
            from terraces100 import TERRACE_MATERIALS
            ii = [i for i, f in enumerate(cat["flakes"]) if f["material"] in TERRACE_MATERIALS and isinstance(f.get("img"), str) and f["terraces"]]
            n_new, n_fl, _ = terraces_parallel(cat["flakes"], ii, tmp, a.workers, prog=lambda f, t: progress(0.70 + 0.15 * f, t))
            print(f"terraces from the 100x images: {n_new} more on {n_fl} flakes")
        snap_outlines(cat["flakes"])
        atl =[put_hashed(tmp / q, out, Path(q).stem) for q in cat["atlases"]]
        for t in cat["thumbs"]:
            t["file"] = put_hashed(tmp / t["file"], out, Path(t["file"]).stem)
        for k, s in cat["scans"].items():
            if k not in bundle["scans"]:
                s["file"] = put_hashed(tmp / s["file"], out, Path(s["file"]).stem)
                bundle["scans"][k] = s
        view_files(cat, tmp, out, prog=lambda f, t: progress(0.85 + 0.12 * f, t))
        prefix_ids(cat, run)
        det, ext = slim(cat)
        na, nt = len(bundle["atlases"]), len(bundle["thumbs"])
        for f in cat["flakes"]:
            f["atlas"] += na
            f["frame"]["thumb"][0] += nt
        new = {f["id"] for f in cat["flakes"]}
        old_ids = {f["id"] for f in bundle["flakes"]}

        def part(name):                                 # (the bundle's detail / extra file, its flakes)
            try:
                return json.loads((final / name).read_text(encoding="utf-8"))["flakes"] if name else {}
            except (OSError, ValueError, KeyError):
                return {}
        D, E = part(bundle.get("detail")), part(bundle.get("extra"))
        for i in new:
            D.pop(i, None), E.pop(i, None)
        D.update(det["flakes"]), E.update(ext["flakes"])
        names = {}
        for k, d in (("detail", dict(kind="flakeid-designer-detail", flakes=D)), ("extra", dict(kind="flakeid-designer-extra", flakes=E))):
            b = json.dumps(d, separators=(",", ":")).encode("utf-8")
            names[k] = hashed(f"{k}.json", b)
            (out / names[k]).write_bytes(b)
        flakes = [f for f in bundle["flakes"] if f["id"] not in new] + cat["flakes"]
        # hBN: the decision model's p_keep / tier of every flake of the run again (from their stored defects: no
        # pictures read), so that the tiers stay consistent when the model centres on the run's flakes
        from flakeid.defects100 import score_flakes
        score_flakes(flakes, log=print)
        lo = {k: set(v) for k, v in (bundle.get("left_out") or {}).items()}
        for stem, v in todo.items():                    # (a flake made again but left out now keeps its old entry)
            lo[stem] = (lo.get(stem, set()) - v) | {x for x in cat.get("left_out", {}).get(stem, [])
                                                     if f"{run}:{stem}-f{x:04d}" not in old_ids}
        views = [f[k] for f in flakes for k in ("img", "img_look", "img_c8", "img_dt", "img_ir", "img_mos", "img_mos_look")
                 if isinstance(f.get(k), str)]
        atlases, thumbs = bundle["atlases"] + atl, bundle["thumbs"] + cat["thumbs"]
        files = (atlases + [t["file"] for t in thumbs] + [s["file"] for s in bundle["scans"].values()] + views
                 + [names["detail"], names["extra"]])
        bundle.update(atlases=atlases, thumbs=thumbs, flakes=flakes, files=files, detail=names["detail"], extra=names["extra"],
                      materials=sorted({f["material"] for f in flakes}), left_out={k: sorted(v) for k, v in lo.items() if v},
                      updated=time.strftime("%Y-%m-%d %H:%M"))
        copy_mark(run_folder, bundle, a.title)
        write_meta(out, bundle)
        n_add = len(new - old_ids)
        n_mos = sum(1 for f in cat["flakes"] if f["terraces"][0].get("terrace_from") == "mosaic")
        publish(out, final, prog=lambda f, t: progress(0.97 + 0.03 * f, t), keep=set(files))
        progress(1.0, "done")
        print(f"designer bundle updated: {final}  ({n_add} flakes added, {len(new) - n_add} made again, {n_mos} with the "
              f"whole-terrace outline; {len(flakes)} flakes in all; "
              f"{sum((out / q).stat().st_size for q in out.iterdir() if q.is_file()) / 1e6:.0f} MB written)")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
