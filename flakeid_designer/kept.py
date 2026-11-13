"""Kept flakes' 100x pictures copied into their run's folder (2026-10-28).

When a flake is kept at the screening (keep / discard decisions, anyone's: the latest decision per terrace wins; the
same flakes as the Library's "kept" list), all its 100x pictures are copied from the piece's viewer into the run's
folder on the share, one subfolder per kind of picture (one picture per folder and shot: no folder holds two versions
of the same picture, 2026-10-28):

    <run>/kept/raw/                     the linear 100x picture (<stem>.jpg)
    <run>/kept/raw_avg/                 the frame-averaged main picture alone (<stem>_avg.jpg)
    <run>/kept/raw_tiff/                the camera's TIFF, when it is still there (<piece>_100x/<stem>.tif)
    <run>/kept/natural/                 the white-balanced copy people look at (flakeid.look: _look)
    <run>/kept/natural_avg/             the same of the frame average (_avg_look)
    <run>/kept/color_8percent/          each channel stretched to the terrace colour +-8 % (_c8)
    <run>/kept/color_8percent_avg/      the same of the frame average (_avg_c8)
    <run>/kept/bw/                      flattened green against the terrace +-4 %, thicker darker (_sg)
    <run>/kept/bw_avg/                  the same of the frame average (_avg_sg)
    <run>/kept/thickness_change/        equivalent thickness change from the terrace (_dt)
    <run>/kept/debris/                  the terrace and its debris outlined (_debris)
    <run>/kept/ir/                      the IR (stacking-domain) picture laid over the 100x frame (_ir)
    <run>/kept/mosaic/                  the whole terrace: the scan's mosaic of 100x pictures (_mosaic)
    <run>/kept/mosaic_natural/          the same, white-balanced (_mosaic_look)
    <run>/kept/ir_domains/              stacking-domain pictures, 16-bit PNG (and their IR mosaics)
    <run>/kept/ir_domains_preview/      their JPEG previews
    <run>/kept/kept_flakes.csv          one row per kept flake: who kept it, when, its pictures

Each file is named <piece>-f<flake number>_<original name>.  A flake that is discarded (or its decision cleared)
later has its copies removed again (only the files this module copied: kept/.kept_manifest.json lists them).  The
copying runs in a background thread (the share can be slow), a few seconds after a decision, for the decision's run;
and once after the designer starts, for runs whose kept folder does not match the decisions (flakes kept before).
Standard library only (like the rest of the designer).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

KEPT_DIR = "kept"
MANIFEST = ".kept_manifest.json"
KEEP = ("keep",)                        # decisions that keep a flake: as the Library's "kept" list (2026-10-28: before also
                                        # "used", picked up into a stack, which the kept list does not show)
LAYOUT = 2                              # the folders' layout (2: the frame averages, TIFFs, previews in folders of their own)
DEBOUNCE_S = 3.0
# terrace id -> (run, piece, flake number): "<run>:<piece>-f0012.a" (drawn terraces "...-f0012.u<time>")
TID = re.compile(r"^(?P<run>.+?):(?P<piece>.+)-f(?P<n>\d+)\.[^.:]+$")
# the H100 entry's keys -> the kind's folder
KINDS = (("f", "raw"), ("a", "raw_avg"), ("c8", "color_8percent"), ("a_c8", "color_8percent_avg"), ("sg", "bw"),
         ("a_sg", "bw_avg"), ("deb", "debris"), ("ir", "ir"), ("mos", "mosaic"))
LOOK_KINDS = (("f", "natural"), ("a", "natural_avg"), ("mos", "mosaic_natural"))
DOMAIN_KINDS = (("raw", "ir_domains"), ("f", "ir_domains_preview"), ("mos_raw", "ir_domains"), ("mos", "ir_domains_preview"))
FOLDERS = ("raw", "raw_avg", "raw_tiff", "natural", "natural_avg", "color_8percent", "color_8percent_avg", "bw", "bw_avg",
           "thickness_change", "debris", "ir", "mosaic", "mosaic_natural", "ir_domains", "ir_domains_preview")


def _log(msg):
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} KEPT: {msg}\n")


def parse_tid(tid):
    m = TID.match(str(tid or ""))
    return (m.group("run"), m.group("piece"), int(m.group("n"))) if m else None


def kept_flakes(latest):
    """{(run, piece, n): dict(by, at, terraces)} of the flakes kept, from the latest decision per terrace (as the page's
    flakeDecision: a flake is kept when any of its terraces is)."""
    out = {}
    for tid, e in (latest or {}).items():
        body = (e or {}).get("body") or {}
        if body.get("decision") not in KEEP:
            continue
        k = parse_tid(tid)
        if k is None:
            continue
        r = out.setdefault(k, dict(by=e.get("by") or "", at=int(e.get("updated") or 0), terraces=[]))
        r["terraces"].append(tid)
        if int(e.get("updated") or 0) > r["at"]:
            r.update(by=e.get("by") or "", at=int(e.get("updated") or 0))
    return out


def _js_const(text, name):
    """The JSON after "const <name>=" in a viewer page (up to ";\\n"), or None."""
    i = text.find(f"const {name}=")
    if i < 0:
        return None
    i += len(name) + 7
    j = text.find(";\n", i)
    try:
        return json.loads(text[i:j if j > 0 else None])
    except ValueError:
        return None


class Pieces:
    """A run folder's pieces: piece name -> (viewer folder, H100), read from the viewer pages (cached by their time)."""

    def __init__(self):
        self.cache = {}                 # index.html path -> (mtime_ns, piece name, H100)

    def viewers(self, run_dir):
        out = []
        try:
            kids = sorted(p for p in Path(run_dir).iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
                          and p.name not in (KEPT_DIR, "designer"))
        except OSError:
            return out
        for c in kids:
            if (c / "index.html").is_file():
                out.append(c)
            else:
                try:
                    out += [v for v in c.glob("*_viewer") if (v / "index.html").is_file()]
                except OSError:
                    pass
        return out

    def read(self, viewer):
        f = Path(viewer) / "index.html"
        try:
            mt = f.stat().st_mtime_ns
        except OSError:
            return None, {}
        hit = self.cache.get(str(f))
        if hit and hit[0] == mt:
            return hit[1], hit[2]
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None, {}
        D = _js_const(text, "D") or {}
        name = str(D.get("name") or "").replace("_tiles", "") or None
        H = _js_const(text, "H100") or {}
        self.cache[str(f)] = (mt, name, H if isinstance(H, dict) else {})
        return name, self.cache[str(f)][2]

    def find(self, run_dir, piece):
        """(viewer folder, H100) of a piece in the run's folder, or (None, {}).  The folder whose name has the piece's
        name in it is read first (one page, usually)."""
        vs = self.viewers(run_dir)
        vs.sort(key=lambda v: (piece not in v.name and piece not in v.parent.name, v.name))
        for v in vs:
            name, H = self.read(v)
            if name == piece:
                return v, H
        return None, {}


def _tiff(run_dir, viewer, stem):
    """The camera's TIFF of a 100x picture if it is still there (it is deleted after the export, unless --keep-tiles)."""
    for d in {Path(viewer), Path(viewer).parent, Path(run_dir)}:
        try:
            dirs = [p for p in d.glob("*_100x") if p.is_dir()]
        except OSError:
            continue
        for p in dirs:
            for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
                if (p / (stem + ext)).is_file():
                    return p / (stem + ext)
    return None


def sources(run_dir, viewer, entries):
    """[(kind folder, source file)] of one flake's 100x pictures (the viewer's H100 entries of the flake)."""
    out, seen = [], set()

    def add(kind, rel):
        if not rel or not isinstance(rel, str):
            return
        p = Path(viewer) / rel.split("?")[0]
        if str(p) not in seen and p.is_file():
            seen.add(str(p))
            out.append((kind, p))

    for e in entries or []:
        if not isinstance(e, dict):
            continue
        if e.get("domain"):
            for k, kind in DOMAIN_KINDS:
                add(kind, e.get(k))
            continue
        for k, kind in KINDS:
            add(kind, e.get(k))
        for k, kind in LOOK_KINDS:
            add(kind, (e.get("look") or {}).get(k))
        f = e.get("f")
        if isinstance(f, str) and f.endswith(".jpg"):
            add("thickness_change", f[:-4] + "_dt.jpg")
            if not e.get("ir"):
                add("ir", f[:-4] + "_ir.jpg")
            t = _tiff(run_dir, viewer, Path(f).stem)
            if t is not None and str(t) not in seen:
                seen.add(str(t))
                out.append(("raw_tiff", t))
    return out


def _same(src, dst):
    try:
        a, b = src.stat(), dst.stat()
    except OSError:
        return False
    return a.st_size == b.st_size and b.st_mtime >= a.st_mtime - 2


def _copy(src, dst):
    tmp = dst.with_name("." + dst.name + ".part")
    shutil.copy2(str(src), str(tmp))
    for i in range(5):                                  # (Windows: a virus scanner can hold the file a moment)
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == 4:
                raise
            time.sleep(0.2 * (i + 1))


def _read_manifest(kdir):
    try:
        j = json.loads((kdir / MANIFEST).read_text(encoding="utf-8"))
        return j if isinstance(j, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_text(path, text):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def flake_key(piece, n):
    return f"{piece}-f{n:04d}"


def needs_sync(run_dir, keys):
    """Whether the run's kept folder lists other flakes than keys (one small file read)."""
    man = _read_manifest(Path(run_dir) / KEPT_DIR)
    if not man and not keys:
        return False
    return set(man.get("flakes", {})) != set(keys) or man.get("layout") != LAYOUT


STRAY = re.compile(r"^.+-f\d{4,}_.+")         # a picture this module names (<piece>-f0012_<name>)


def sync_run(run_dir, kept, pieces=None, full=False):
    """Make <run_dir>/kept/ hold the 100x pictures of the kept flakes {(piece, n): info} and nothing of other flakes.
    full (Update kept folder): also pictures in the kind folders that the manifest does not list (copied by an older
    version, or left over) are removed.  -> dict(copied, removed, flakes, missing)."""
    run_dir = Path(run_dir)
    kdir = run_dir / KEPT_DIR
    pieces = pieces or Pieces()
    man = _read_manifest(kdir)
    had = man.get("flakes", {}) if isinstance(man.get("flakes"), dict) else {}
    st = dict(copied=0, removed=0, flakes=0, missing=[])
    if not kept and not had:
        return st
    kdir.mkdir(exist_ok=True)
    now, rows = {}, []
    for (piece, n), info in sorted(kept.items()):
        key = flake_key(piece, n)
        viewer, H = pieces.find(run_dir, piece)
        entries = H.get(str(n)) if viewer is not None else None
        if not entries:
            st["missing"].append(key)
            if key in had:                              # (the viewer not readable now: its copies stay)
                now[key] = had[key]
            continue
        files = []
        for kind, src in sources(run_dir, viewer, entries):
            d = kdir / kind
            d.mkdir(exist_ok=True)
            dst = d / f"{key}_{src.name}"
            if not _same(src, dst):
                try:
                    _copy(src, dst)
                    st["copied"] += 1
                except OSError as e:
                    _log(f"{src} not copied: {e}")
                    continue
            files.append(f"{kind}/{dst.name}")
        now[key] = files
        st["flakes"] += 1
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.get("at", 0) / 1000)) if info.get("at") else ""
        rows.append([key, piece, n, info.get("by", ""), ts, len(files), ";".join(files)])
    for key, files in had.items():                      # flakes no longer kept: their copies only
        keep_files = set(now.get(key, []))
        for rel in files:
            if rel in keep_files:
                continue
            p = kdir / rel
            try:
                if p.is_file():
                    p.unlink()
                    st["removed"] += 1
            except OSError as e:
                _log(f"{p} not removed: {e}")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["flake", "piece", "flake_number", "kept_by", "kept_at", "n_pictures", "files"])
    w.writerows(rows)
    _write_text(kdir / "kept_flakes.csv", buf.getvalue())
    if full:                                            # (named like ours, not in the folder's list now: left over)
        listed = {f for v in now.values() for f in v}
        for d in list(kdir.iterdir()) if kdir.is_dir() else []:
            if not d.is_dir():
                continue
            for q in list(d.iterdir()):
                rel = f"{d.name}/{q.name}"
                if q.is_file() and rel not in listed and (STRAY.match(q.name) or q.name.startswith(".")):
                    try:
                        q.unlink()
                        st["removed"] += 1
                    except OSError as e:
                        _log(f"{q} not removed: {e}")
    for d in kdir.iterdir() if kdir.is_dir() else []:   # (folders left empty: an older layout's, or no such pictures now)
        try:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    _write_text(kdir / MANIFEST, json.dumps(dict(updated=int(time.time() * 1000), layout=LAYOUT, flakes=now), indent=1))
    return st


class Kept:
    """The background copier: request(store, terrace id) after a decision; the run is synced a few seconds later."""

    def __init__(self):
        self.lock = threading.Lock()
        self.todo = {}                  # run name -> time it is due
        self.store = None
        self.pieces = Pieces()
        self.wake = threading.Event()
        self.thread = None
        self.last = {}                  # run name -> last result (for /api/kept)
        self.sync_lock = threading.Lock()

    def _start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._loop, daemon=True, name="kept pictures")
            self.thread.start()

    def request(self, store, tid=None, run=None, delay=DEBOUNCE_S):
        k = parse_tid(tid) if tid else None
        run = run or (k[0] if k else None)
        if not run:
            return
        with self.lock:
            self.store = store
            self.todo[run] = time.time() + delay
        self._start()
        self.wake.set()

    def initial(self, store, wait_s=120):
        """Once after the start: runs whose kept folder does not match the decisions (flakes kept before this)."""
        def go():
            store.first_scan.wait(wait_s)
            try:
                latest = store.review_state()[2]
            except Exception as e:                      # (never fails the designer)
                _log(f"decisions not read: {e}")
                return
            by_run = {}
            for (run, piece, n), info in kept_flakes(latest).items():
                by_run.setdefault(run, {})[(piece, n)] = info
            with store.lock:
                runs = {v["doc"].get("run"): v["dir"].parent for v in store.runs.values()}
            for run, d in runs.items():
                try:
                    if needs_sync(d, [flake_key(p, n) for p, n in by_run.get(run, {})]):
                        self.request(store, run=run, delay=0)
                except OSError:
                    pass
        threading.Thread(target=go, daemon=True, name="kept pictures (start)").start()

    def _loop(self):
        while True:
            with self.lock:
                due = [r for r, t in self.todo.items() if t <= time.time()]
                wait = min((t for t in self.todo.values()), default=None)
            if not due:
                self.wake.wait(None if wait is None else max(0.2, wait - time.time()))
                self.wake.clear()
                continue
            for run in due:
                with self.lock:
                    self.todo.pop(run, None)
                    store = self.store
                try:
                    self._sync(store, run)
                except Exception as e:                  # (never ends the thread)
                    _log(f"run {run}: {type(e).__name__}: {e}")

    def sync_now(self, store, run, full=False):
        """Bring the run's kept folder(s) up to date now (the Library's "Update kept folder") -> [result per folder]."""
        self._sync(store, run, full=full)
        return [self.last[k] for k in (run,) if k in self.last]

    def _sync(self, store, run, full=False):
        sig, _, latest = store.review_state()
        if sig == "none":                               # (the decisions cannot be read now: never empty a kept folder)
            raise OSError("the keep / discard decisions cannot be read (is the library's data folder reachable?)")
        kept = {(p, n): v for (r, p, n), v in kept_flakes(latest).items() if r == run}
        with store.lock:
            dirs = [v["dir"].parent for v in store.runs.values() if v["doc"].get("run") == run]
        for d in dirs:
            t0 = time.time()
            with self.sync_lock:                        # (the background copier and the button: one at a time)
                st = sync_run(d, kept, self.pieces, full=full)
            self.last[run] = dict(st, dir=str(d / KEPT_DIR), at=int(time.time() * 1000))
            if st["copied"] or st["removed"] or st["missing"]:
                _log(f"{d / KEPT_DIR}: {st['flakes']} kept flakes, {st['copied']} pictures copied, {st['removed']} removed"
                     + (f", no 100x pictures found for {', '.join(st['missing'][:5])}" if st["missing"] else "")
                     + f" ({time.time() - t0:.1f} s)")


KEPT = Kept()


# ---------------------------------------------------------------- AFM centers next to the 100x pictures (2026-10-29)
# A flake's AFM center (Library / 100x window: the red cross) is kept with the library's decisions (review collection,
# "<run>:<piece>-f0012.clean").  So that it travels with the 100x pictures, it is also written into the piece's viewer:
#     <piece viewer>/100x/afm_centers.json   {"version": 1, "centers": {"<100x picture file name>": {flake, rel, um,
#                                             by, at}}}
# rel: the point as a fraction of the 100x picture (x / width, y / height, the picture as the viewer shows it; may be
# outside 0-1 on the whole-terrace mosaic); um: the same as micrometres from the picture's center.  Both are relative to
# the 100x picture, not stage coordinates: a run made again from the same 100x pictures with new stage positions
# (a chip found slightly shifted) keeps the points, and its new coordinates apply.  Copy this file with the 100x
# pictures; the designer reads it for flakes of a run that have no AFM center in the library yet
# (GET /api/afmcenters?run=...), matched by the 100x picture's file name (or the flake number).
#
# After a remeasure (a run made again from the same 100x pictures with new positions) the AFM center is the picture's
# NEW position plus the um offset (2026-10-29: not the fraction through a fixed camera matrix, which would miss the
# chip's rotation).  A remeasured chip that is turned against the old pictures: "rot_deg" (degrees, clockwise as the
# pictures are shown, y down) at the file's top level or per center turns the um offset into the new run's frame.

CENTERS = "afm_centers.json"
_centers_lock = threading.Lock()


def _main_100x(entries):
    """The flake's main 100x picture's file name (the first one that is not a stacking-domain picture), or None."""
    for e in entries or []:
        if isinstance(e, dict) and not e.get("domain") and isinstance(e.get("f"), str):
            return Path(e["f"].split("?")[0]).name
    return None


def write_center(store, tid, body, pieces=None):
    """One flake's AFM center (body: clean, rel, um; clean None: removed) into its viewer's 100x/afm_centers.json."""
    k = parse_tid(tid[:-len(".clean")] + ".a") if str(tid).endswith(".clean") else None
    if k is None:
        return
    run, piece, n = k
    pieces = pieces or KEPT.pieces
    with store.lock:
        dirs = [v["dir"].parent for v in store.runs.values() if v["doc"].get("run") == run]
    for d in dirs:
        viewer, H = pieces.find(d, piece)
        if viewer is None:
            continue
        name = _main_100x(H.get(str(n)))
        if not name:
            continue
        f = Path(viewer) / "100x" / CENTERS
        with _centers_lock:
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                j = {}
            C = j.get("centers") if isinstance(j.get("centers"), dict) else {}
            if body.get("clean") and body.get("rel"):
                C[name] = dict(flake=n, rel=body.get("rel"), um=body.get("um"), by=body.get("by") or "",
                               at=int(body.get("updated") or time.time() * 1000))
            else:
                C.pop(name, None)
            try:
                _write_text(f, json.dumps(dict(version=1, centers=C), indent=1))
            except OSError as e:
                _log(f"{f} not written: {e}")


def read_centers(store, run, pieces=None):
    """{"<piece>-f0012": {rel, um}} of a run from its viewers' 100x/afm_centers.json, matched to the run's flakes by the
    100x picture's file name (else the flake number written with it)."""
    pieces = pieces or KEPT.pieces
    # (the library's centers of this run that have rel but are not next to the pictures yet, e.g. saved while the
    # designer was older: written now)
    try:
        latest = store.review_state()[2]
    except Exception:
        latest = {}
    have = None
    for tid, e in latest.items():
        b = (e or {}).get("body") or {}
        if not (tid.startswith(run + ":") and tid.endswith(".clean") and b.get("clean") and b.get("rel")):
            continue
        if have is None:
            have = {}
            with store.lock:
                ds = [v["dir"].parent for v in store.runs.values() if v["doc"].get("run") == run]
            for d in ds:
                for viewer in pieces.viewers(d):
                    try:
                        C = json.loads((Path(viewer) / "100x" / CENTERS).read_text(encoding="utf-8")).get("centers") or {}
                    except (OSError, ValueError, AttributeError):
                        C = {}
                    for c in C.values():
                        if isinstance(c, dict):
                            have[(Path(viewer).name, c.get("flake"), tuple(c.get("rel") or ()))] = 1
        k = parse_tid(tid[:-len(".clean")] + ".a")
        if k and not any(key[1] == k[2] and key[2] == tuple(b["rel"]) for key in have):
            write_center(store, tid, dict(b, by=e.get("by") or ""), pieces)
    with store.lock:
        dirs = [v["dir"].parent for v in store.runs.values() if v["doc"].get("run") == run]
    out = {}
    for d in dirs:
        for viewer in pieces.viewers(d):
            f = Path(viewer) / "100x" / CENTERS
            try:
                C = json.loads(f.read_text(encoding="utf-8")).get("centers") or {}
            except (OSError, ValueError, AttributeError):
                continue
            piece, H = pieces.read(viewer)
            if not piece:
                continue
            by_name = {}
            for num, entries in H.items():
                nm = _main_100x(entries)
                if nm:
                    by_name[nm] = num
            try:
                top = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                top = {}
            for name, c in C.items():
                if not isinstance(c, dict) or not (c.get("um") or c.get("rel")):
                    continue
                num = by_name.get(name) or (str(c.get("flake")) if str(c.get("flake")) in H else None)
                if num is None:
                    continue
                out[f"{piece}-f{int(num):04d}"] = dict(rel=c.get("rel"), um=c.get("um"), image=name,
                                                       rot_deg=float(c.get("rot_deg", top.get("rot_deg", 0)) or 0))
    return out
