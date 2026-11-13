"""vdW Stack Designer, portable version: the designer page served from this computer, with the flakes of the scan runs
found on the share (each run's designer/ folder, written by the scan program) and the lab's designs and keep/discard
decisions kept in a shared folder.  Only Python 3.8+ is needed (no other packages).

    python flakeid_designer                      start (opens the designer in the browser)
    python flakeid_designer --root "Z:\\scans"    also look for runs there
    python flakeid_designer --install-shortcut   desktop shortcut (Windows) / app icon (Mac)

Runs: every <run>/designer/bundle.json up to a few folders below each root (add roots from the page: Scan results ->
Add a folder).  Shared data (designs, decisions): --data, else <first root>/_designer_data.  Settings per person:
~/.flakeid_designer.json.  The server stops by itself 15 minutes after the last designer tab is closed.

"Export to KLayout..." runs marks_to_klayout.py (next to this file) with this Python as a separate process; that tool
needs numpy, opencv-python and gdstk (python -m pip install numpy opencv-python gdstk).  Its choices are kept in the
settings file under "klayout" (template_gds, lyp, last_lys, last_replace, last_image_dir, out_dir).
"""
from __future__ import annotations

import argparse
import datetime
import atexit
import email.utils
import getpass
import hashlib
import json
import math
import shutil
import mimetypes
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:                                                 # the stacker version's live camera (needs OpenCV)
    from . import camera as CAM
except ImportError:
    try:
        import camera as CAM                         # type: ignore  (run as: python flakeid_designer)
    except ImportError:
        CAM = None
try:                                                 # stacks kept in a folder of their own (stackdir.py)
    from . import stackdir as SD
except ImportError:
    import stackdir as SD                            # type: ignore
try:                                                 # kept flakes' 100x pictures into the run's folder (kept.py)
    from . import kept as KP
except ImportError:
    import kept as KP                                # type: ignore
try:                                                 # stacks kept on this computer (offline.py)
    from . import offline as OF
except ImportError:
    import offline as OF                             # type: ignore
try:                                                 # updates from GitHub (update.py)
    from . import update as UPD
except ImportError:
    import update as UPD                             # type: ignore

HERE = Path(__file__).resolve().parent
for _r in (HERE.parent, HERE / "builder"):         # (2026-11-05y) `python flakeid_designer` puts only this folder on the path:
    if (_r / "flakeid" / "__init__.py").is_file():   # the flakeid package beside it (Thickness cursor, AFM lists) was not found
        if str(_r) not in sys.path:                  # until a build had run (appended: this folder's modules keep priority)
            sys.path.append(str(_r))
        break
APP = "flakeid-designer"
VERSION = "2026-11-12c"
CONFIG = Path.home() / ".flakeid_designer.json"
LOG = Path.home() / ".flakeid_designer_log.txt"
DEFAULT_PORT = 8765
IDLE_S = 15 * 60
SKIP = re.compile(r"(_viewer|_tiles|_100x|_domains|^tiles|^100x|^domains|^data|^_catalog|^old_coords$|^\.)", re.I)
# (^old_coords: coordinates saved before a remeasure, not a run: it was listed as "an older run without designer files")
DEPTH = 5
SAFE_ID = re.compile(r"[^A-Za-z0-9_.~:@+-]")
# a run file named by its content (make_designer_bundle.hashed; with a letter: never an older build's number or date)
HASHED = re.compile(r"_(?=[0-9]*[a-f])[0-9a-f]{10}\.[a-z]+$")


def _lenient_json(text):
    """JSON as people edit it by hand: // comments and trailing commas allowed."""
    t = re.sub(r"(?m)^\s*//.*$", "", text)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return json.loads(t)


def load_config():
    """This computer's designer settings (~/.flakeid_designer.json: libraries, the transfer station's camera and its
    scale per objective, ...).  A file that cannot be read (2026-10-20: a hand edit with a missing comma made the
    designer start empty and then save over it, and the stacker's calibrations were lost) is kept as
    .flakeid_designer.json.unreadable-<time> and never written over; the last good copy is in .bak."""
    try:
        text = CONFIG.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        return json.loads(text)
    except ValueError:
        pass
    try:
        return _lenient_json(text)
    except ValueError as e:
        keep = CONFIG.with_name(CONFIG.name + ".unreadable-" + time.strftime("%Y%m%d_%H%M%S"))
        try:
            raw = CONFIG.read_bytes()
            if not any(q.read_bytes() == raw for q in CONFIG.parent.glob(CONFIG.name + ".unreadable-*")):
                shutil.copy2(str(CONFIG), str(keep))
                sys.stderr.write(f"{time.strftime('%H:%M:%S')} SETTINGS: {CONFIG} cannot be read ({e}); kept as {keep.name}"
                                 f"{' (the last good one: ' + CONFIG.name + '.bak)' if CONFIG.with_name(CONFIG.name + '.bak').is_file() else ''}\n")
        except OSError:
            pass
        _CONFIG_BROKEN[0] = True
        bak = CONFIG.with_name(CONFIG.name + ".bak")
        try:
            return json.loads(bak.read_text(encoding="utf-8"))      # (the last good settings, while the file is fixed)
        except (OSError, ValueError):
            return {}


_CONFIG_BROKEN = [False]


def library_info(cfg, store):
    """The current library (a person's scan-results folder, with the folders imported into it and its own designer data)
    and the ones used before on this computer, newest first."""
    libs = cfg.get("libraries") or {}
    cur = cfg.get("library") or (store.roots[0] if store.roots else None)
    recent = sorted(libs.items(), key=lambda kv: -float(kv[1].get("used", 0)))
    return dict(library=cur, libraries=[dict(path=k, name=Path(k).parent.name if Path(k).name.lower() == "scanresults" else Path(k).name,
                                             roots=v.get("roots", [k])) for k, v in recent])


def lib_owns(path, roots, data_dir):
    """True when saved folders belong to the library `path`: the folder is one of its roots, or its designer data is
    inside it.  (2026-11-03f: a folder window left open while switching libraries saved one library's folders under
    another one's name, and BN_15_auto/20261101_run_1 then opened an empty GR13' folder: its decisions and AFM lists
    looked lost.)"""
    try:
        p = Path(path)
        if any(Path(r) == p for r in roots or []):
            return True
        return bool(data_dir) and (Path(data_dir) == p or p in Path(data_dir).parents)
    except (TypeError, ValueError):
        return False


def own_library(path):
    """A library's own folders: just the folder, its designer data in its _designer_data."""
    return dict(roots=[str(Path(path))], data_dir=str(Path(path) / "_designer_data"))


def repair_libraries(cfg):
    """(2026-11-03f) Saved libraries whose folders belong to another library go back to their own folder and its
    _designer_data (nothing on disk is changed: the data was always in that folder); the current one too.  Returns
    the libraries repaired."""
    fixed = []
    libs = cfg.get("libraries") if isinstance(cfg.get("libraries"), dict) else {}
    for k, v in list(libs.items()):
        v = v if isinstance(v, dict) else {}
        if not lib_owns(k, v.get("roots"), v.get("data_dir")):
            libs[k] = dict(own_library(k), used=v.get("used", 0))
            fixed.append(k)
    cur = cfg.get("library")
    if cur and not lib_owns(cur, cfg.get("roots"), cfg.get("data_dir")):
        e = libs.get(cur) or own_library(cur)
        cfg["roots"], cfg["data_dir"] = list(e.get("roots") or [cur]), e.get("data_dir")
        if cur not in fixed:
            fixed.append(cur)
    return fixed


def start_library(cfg, extra_roots=()):
    """The folders and designer data to start with (the current library's, repaired if they were saved under the wrong
    name), and the library they belong to."""
    fixed = repair_libraries(cfg)
    if fixed:
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} SETTINGS: folders saved under the wrong library put back to the "
                         f"library's own folder: {'; '.join(fixed)}\n")
        save_config(cfg)
    roots = list(dict.fromkeys([str(Path(r)) for r in cfg.get("roots", [])] + [str(Path(r)) for r in extra_roots]))
    data = cfg.get("data_dir") or (str(Path(roots[0]) / "_designer_data") if roots else None)
    return roots, data, cfg.get("library") or (roots[0] if roots else None)


LIBRARY_LOCK = threading.Lock()         # one library change (switch, add / remove a folder) at a time
PICKING = threading.Lock()              # one folder window for the library at a time


def remember_library(cfg, store):
    """Keep the current library's folders in the config (to come back to it after switching).  The library is the one
    the store holds (store.library), not whatever the settings file says: another window or request may have changed
    that meanwhile, and the folders would be saved under the wrong name."""
    cur = getattr(store, "library", None) or cfg.get("library") or (store.roots[0] if store.roots else None)
    if not cur:
        return cfg
    store.library = cur
    data = str(store.data) if store.data else None
    if not lib_owns(cur, store.roots, data):
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} SETTINGS: {cur} not saved with folders that are not its own "
                         f"({', '.join(store.roots)}; data {data})\n")
        return cfg
    libs = cfg.setdefault("libraries", {})
    libs[cur] = dict(roots=list(store.roots), data_dir=data, used=time.time())
    cfg["library"], cfg["roots"], cfg["data_dir"] = cur, list(store.roots), data
    return cfg


def switch_library(cfg, store, path):
    """Make `path` the library: its own folders and designer data (a new one, or one saved with another library's
    folders: just that folder, data in its _designer_data)."""
    path = str(Path(path))
    remember_library(cfg, store)
    lib = (cfg.get("libraries") or {}).get(path) or {}
    if not lib_owns(path, lib.get("roots"), lib.get("data_dir")):
        lib = own_library(path)
    roots = [str(Path(r)) for r in lib.get("roots") or [path]]
    store.roots = roots
    store.data = Path(lib.get("data_dir") or Path(path) / "_designer_data")
    store.library = cfg["library"] = path
    remember_library(cfg, store)
    save_config(cfg)
    store.scan()


def save_added(store):
    """(2026-11-05j) Keep the store's added run folders in the settings (this computer, any library)."""
    with LIBRARY_LOCK:
        cfg = load_config()
        cfg["added_runs"] = list(store.added or [])
        save_config(cfg)


def save_config(c):
    """Written safely (a new file renamed over the old one); the previous good file kept as .bak."""
    try:
        if CONFIG.is_file():
            try:
                json.loads(CONFIG.read_text(encoding="utf-8"))
                shutil.copy2(str(CONFIG), str(CONFIG.with_name(CONFIG.name + ".bak")))
            except ValueError:
                pass                                     # (an unreadable one is kept by load_config, not as .bak)
        tmp = CONFIG.with_name(CONFIG.name + ".part")
        tmp.write_text(json.dumps(c, indent=1), encoding="utf-8")
        os.replace(str(tmp), str(CONFIG))
    except OSError:
        pass


def user_name():
    try:
        return SAFE_ID.sub("_", f"{getpass.getuser()}@{socket.gethostname().split('.')[0]}")
    except Exception:
        return "user"


def run_pieces(run_dir):
    """(2026-11-05m) Each chip's material and SiO2 thickness from the scan program's <run>/<run>.json (the Library
    run card's top left): [{name, material, oxide_nm}], [] when unreadable."""
    try:
        j = json.loads((Path(run_dir) / (Path(run_dir).name + ".json")).read_text(encoding="utf-8"))
        return [dict(name=p.get("name"), material=p.get("material"), oxide_nm=p.get("oxide_nm"))
                for p in (j.get("pieces") or []) if isinstance(p, dict)]
    except (OSError, ValueError, AttributeError):
        return []


class Store:
    """Runs found under the roots, and the shared designs / decisions."""
    SCAN_S = 180                        # the share is walked for runs in the background this often (and on Refresh)

    def __init__(self, roots, data_dir, library=None, added=None):
        self.roots = [str(Path(r)) for r in roots]
        # (2026-11-05j, Avram: "the folder is irrelevant, you add the runs") the run folders added by hand (＋ Add run,
        # Recent runs › Add run): only these are read.  None: not chosen yet on this computer (the page seeds it once
        # with the runs ticked then, /api/runs/seed); before that the library's folder is walked as before
        self.added = [str(Path(r)) for r in added] if isinstance(added, list) else None
        self.data = Path(data_dir) if data_dir else None
        self.library = library or (self.roots[0] if self.roots else None)     # (the library these folders belong to)
        self.related = []               # other designer-data folders with decisions about these runs (related_data)
        self.lock = threading.Lock()    # self.runs only (held briefly: never around share I/O)
        self.scan_lock = threading.Lock()                # one walk of the share at a time
        self.seen_lock, self.review_lock = threading.Lock(), threading.Lock()
        self.first_scan = threading.Event()
        self.runs = {}                  # id -> dict(dir, doc)
        self.runs_sig = "0"
        self.scanned = 0.0
        self.cache = {}                 # bundle path -> (mtime, summary)
        self._files = {}                # shared-data file path -> ((mtime_ns, size), parsed)
        self._lists = {}                # collection -> (signature, list)
        self._review = {}               # review .jsonl path -> dict(key, pos, latest)
        self.folders = SD.Folders(load_config, save_config, user_name)   # folder stacks (Open..., Open recent)

    # ---- runs
    def scan(self):
        """Walk the roots for runs (one walk at a time: a request that comes during one waits for it)."""
        with self.scan_lock:
            return self._scan()

    def _scan(self):
        found = {}
        todo, unbuilt = [], []                       # (local: request threads and the builder scan too, 2026-10-16)
        for root in self.walked_roots():
            base = Path(root)
            try:
                if not base.is_dir():
                    continue
                # (an added run's own folder, 2026-11-05b: walked as a run, as one found inside a folder would be)
                added = self.added is not None or (bool(self.library) and not in_folder(root, self.library))
                dirs = list(self._walk(base, 1 if added else 0, todo, unbuilt))
            except OSError as e:                         # (a share that does not answer: its runs are left out this time)
                sys.stderr.write(f"runs in {root} not read: {e}\n")
                continue
            for d in dirs:
                doc = self._summary(d)
                if doc is None:
                    continue
                rid = SAFE_ID.sub("_", doc["run"])
                k, n = rid, 2
                while k in found and found[k]["dir"] != d:
                    k, n = f"{rid}~{n}", n + 1
                # (?v=: a rebuilt run keeps its file names; the browser keeps images for an hour, so a new version is a
                # new address.  Not for names that end with their content's hash (builds from 2026-10-16): those stay
                # in the browser across rebuilds)
                v = int(doc.get("added") or 0) // 1000
                doc = dict(doc, bundle=f"{k}/bundle.json", files={f: f"{k}/{f}" + ("" if HASHED.search(f) else f"?v={v}")
                                                                  for f in doc.get("files_list", [])})
                doc.pop("files_list", None)
                found[k] = dict(dir=d, doc=doc)
        sig = _sig(json.dumps(sorted((k, v["doc"]) for k, v in found.items()), sort_keys=True, default=str))
        with self.lock:
            self.runs, self.runs_sig, self.scanned = found, sig, time.time()
        try:
            self.related = self.related_data(found)
        except OSError:
            pass
        self.first_scan.set()
        RUN_DATA.sync_soon(self)                     # (decisions copied into each run's own folder, 2026-11-10b)
        # (2026-11-05k) builds only for the runs added by hand: before that list exists (the page seeds it at once),
        # a walk of the library's folder offered every run of the folder for building, and those rows stayed
        BUILDER.consider(self, todo if self.added is not None else [], unbuilt if self.added is not None else [])
        return found

    def walked_roots(self):
        """(2026-11-04g) Only the folders inside the library's own folder (Directory at the top): a library that also
        had another folder added (GR13'_auto + BN_16_auto) listed the other folder's runs too."""
        if self.added is not None:
            return list(self.added)
        if not self.library:
            return list(self.roots)
        out = [r for r in self.roots if in_folder(r, self.library)]
        # (2026-11-05b) and the runs added one by one from any folder (＋ Add run…): a run folder itself
        return (out or [self.library]) + [r for r in self.roots if not in_folder(r, self.library) and is_run_dir(r)]

    def runs_list(self, wait_s=60):
        """(signature, the runs) from the last walk; the first request waits for the first walk."""
        if not self.first_scan.is_set():
            self.first_scan.wait(wait_s)
        with self.lock:
            return self.runs_sig, [dict(id=k, data=v["doc"]) for k, v in self.runs.items()]

    def start_scanner(self, period=None):
        """Walk the share for new / rebuilt runs every SCAN_S seconds in a background thread (the page's polls only read
        the last result: a walk of a slow share no longer holds up a request, 2026-10-16)."""
        def loop():
            while True:
                time.sleep(period or self.SCAN_S)
                try:
                    self.scan()
                except Exception as e:                   # (never ends the thread)
                    sys.stderr.write(f"{time.strftime('%H:%M:%S')} run search failed: {e}\n")
        threading.Thread(target=loop, daemon=True, name="run search").start()

    def _walk(self, d, depth, todo, unbuilt):
        b = d / "designer" / "bundle.json"
        mark = d / "copy_done.json"                      # (a run's copy is complete: written last by the scan program)
        if mark.is_file():
            try:
                if not b.is_file() or b.stat().st_mtime < mark.stat().st_mtime:
                    todo.append(d)                       # no designer files yet, or older than the run's last update
            except OSError:
                pass
        elif not b.is_file() and depth > 0 and looks_like_run(d):
            unbuilt.append(d)                            # (an older run, copied before the marker: built on request)
            return
        if b.is_file():
            yield d / "designer"
            return
        if depth >= DEPTH:
            return
        try:
            subs = sorted(p for p in d.iterdir() if p.is_dir() and not SKIP.search(p.name))
        except OSError:
            return
        for s in subs:
            yield from self._walk(s, depth + 1, todo, unbuilt)

    def _summary(self, ddir):
        """Run summary from designer/run.json (written with the bundle), else from bundle.json (cached)."""
        rj, bj = ddir / "run.json", ddir / "bundle.json"
        try:
            src = rj if rj.is_file() else bj
            mt = src.stat().st_mtime
            hit = self.cache.get(str(src))
            if hit and hit[0] == mt:
                return hit[1]
            j = json.loads(src.read_text(encoding="utf-8"))
            if j.get("kind") not in ("flakeid-designer-bundle", "flakeid-designer-run"):
                return None
            doc = dict(run=j.get("run") or ddir.parent.name, title=j.get("title") or j.get("run") or ddir.parent.name,
                       created=j.get("created"), added=int(mt * 1000),
                       n_flakes=j.get("n_flakes", len(j.get("flakes", []))), materials=j.get("materials", []),
                       scans=list(j.get("scans", {}) if isinstance(j.get("scans"), dict) else j.get("scans", [])),
                       files_list=j.get("files", []), path=str(ddir.parent), started=run_started(ddir.parent),
                       pieces=run_pieces(ddir.parent))
            self.cache[str(src)] = (mt, doc)
            return doc
        except (OSError, ValueError):
            return None

    def run_file(self, rid, name):
        with self.lock:
            r = self.runs.get(rid)
        if r is None or "/" in name or "\\" in name or name.startswith("."):
            return None
        return r["dir"] / name                           # (whether it is there: one stat, in Handler._file)

    # ---- shared data
    # The page asks for each collection every 20 s.  A collection's signature (its files' names, sizes and times: one
    # directory listing) is its ETag; files are read only when new or changed, and an unchanged collection is answered
    # with 304 (before: every file of every collection read over the network on every poll, 2026-10-16).
    RELATED_UP = 2                      # parent folders of the library's folders whose designer data is read too

    def related_data(self, runs):
        """(2026-11-03g) The designer-data folders of other libraries that hold decisions about this library's runs:
        a run screened while its own folder was the library (BN_15_auto/20261101_run_1/_designer_data) showed as
        "not screened" in the parent library (BN_15_auto), and its AFM lists were missing.  Read: every
        _designer_data in a run's folder or between it and the library's folder, and in the library folders' parents
        (up to RELATED_UP).  Decisions are merged with the library's own (the latest wins); new ones are written to the
        library's own data, which is read by the others in turn."""
        own = str(Path(self.data)) if self.data is not None else None
        out, seen = [], {own}
        roots = [Path(r) for r in self.roots]

        def add(folder):
            q = Path(folder) / "_designer_data"
            if str(q) not in seen:
                seen.add(str(q))
                try:
                    if q.is_dir():
                        out.append(q)
                except OSError:
                    pass
        for v in runs.values():
            d = Path(v["dir"])
            # (2026-11-05t) a run added by hand from outside the library's folder (GR13'_auto/20261102_run_1 with the
            # library BN_15_auto): its own and its folder's data (GR13'_auto/_designer_data: its decisions and AFM
            # lists) were not read, so it looked unscreened with no AFM lists
            rd = d.parent if d.name == "designer" else d          # (v["dir"]: the run's designer folder)
            add(rd)
            add(rd.parent)
            for anc in [d] + list(d.parents):
                inside = [r for r in roots if anc == r or r in anc.parents]
                if not inside:
                    break
                add(anc)
        for r in roots:
            for anc in list(r.parents)[:self.RELATED_UP]:
                add(anc)
        return out

    def _data_dirs(self, sub):
        """[the library's own data/<sub> (made if missing), then the related folders' that exist]; [] without data."""
        d = self._dir(sub)
        if d is None:
            return []
        out = [d]
        for q in list(self.related):
            try:
                if (q / sub).is_dir():
                    out.append(q / sub)
            except OSError:
                pass
        return out

    def _dir(self, sub):
        if self.data is None:
            return None
        d = self.data / sub
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return d

    @staticmethod
    def _entries(d, ext, skip_under=False):
        """[(name, mtime_ns, size)] of the files *ext in d (not .hidden / temporary ones), sorted, or None."""
        out = []
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.endswith(ext) and not e.name.startswith(".") and not (skip_under and e.name.startswith("_")) and e.is_file():
                        st = e.stat()
                        out.append((e.name, st.st_mtime_ns, st.st_size))
        except OSError:
            return None
        return sorted(out)

    def _read_json(self, p, key):
        """A shared-data file's JSON, parsed again only when its (mtime, size) changed; None if unreadable."""
        c = self._files.get(str(p))
        if c and c[0] == key:
            return c[1]
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        self._files[str(p)] = (key, j)
        return j

    MERGED = ("terraces",)              # collections read from the related data folders too (the library's own win)

    def collection(self, sub, skip_under=False):
        """(signature, [dict(id, data)]) of the JSON files of data/<sub> (and of the related folders' for MERGED)."""
        d = self._dir(sub)
        ents = self._entries(d, ".json", skip_under) if d else None
        if ents is None:
            return "none", []
        dirs = [(d, ents)]
        if sub in self.MERGED:
            for q in self._data_dirs(sub)[1:]:
                e = self._entries(q, ".json", skip_under)
                if e:
                    dirs.append((q, e))
        sig = _sig("|".join(f"{q}" + repr(e) for q, e in dirs))
        hit = self._lists.get(sub)
        if hit and hit[0] == sig:
            return hit
        out, ids = [], set()
        for q, e in dirs:
            for name, mt, sz in e:
                if name[:-5] in ids:
                    continue
                j = self._read_json(q / name, (mt, sz))
                if j is None:
                    continue
                if sub == "stacks" and isinstance(j, dict):  # (the version the page opened: sent back with its next save)
                    j = dict(j, _sig=f"{j.get('rev', 0)}-{mt}-{sz}")
                ids.add(name[:-5])
                out.append(dict(id=name[:-5], data=j))
        self._lists[sub] = (sig, out)
        return sig, out

    def list_designs(self):
        return self.collection("designs")[1]

    def put_design(self, did, body):
        d = self._dir("designs")
        if d is None:
            raise OSError("no shared data folder")
        body = dict(body, by=body.get("by") or user_name())
        tmp = d / f".{did}.tmp"
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, d / f"{did}.json")

    # stacks: one file per stack (plan, stacking steps, final state, exports: docs/stack_workflow.md)
    def list_stacks(self):
        return self.stacks_listing()[1]

    def stacks_listing(self):
        """The stacks of the shared data folder (older ones) and the folder stacks opened in this session."""
        s1, a = self.collection("stacks", skip_under=True)
        s2, b = self.folders.listing()
        return (_sig(s1 + "#" + s2) if s2 else s1), a + b

    def put_stack(self, sid, body):
        """Save a stack.  body._base_sig: the _sig of the version the page opened (its rev, file time and size): if the
        file changed since (saved by another page, another computer, or an older designer), StackConflict instead
        (body._force: save anyway).  The server counts the revisions (rev) and stamps `updated` later than the version
        it replaces: computers' clocks differ, so their times alone cannot order saves (2026-10-16).  Pages from before
        send _base (the `updated` they opened) instead.  Returns dict(sig, rev, updated) of the saved file."""
        import platform
        if SD.is_folder_id(sid):
            return self.folders.put(sid, body, platform.node())
        d = self._dir("stacks")
        if d is None:
            raise OSError("no shared data folder")
        body = dict(body, by=body.get("by") or user_name())
        base, force = body.pop("_base", None), body.pop("_force", False)
        bsig = body.pop("_base_sig", None)
        body.pop("_sig", None)
        body.pop("_offline", None)                       # (the offline listing's state of the stack: not saved)
        old = d / f"{sid}.json"
        cur, csig = None, None
        try:
            st = old.stat()
            cur = json.loads(old.read_text(encoding="utf-8"))
            csig = f"{cur.get('rev', 0)}-{st.st_mtime_ns}-{st.st_size}" if isinstance(cur, dict) else None
        except (OSError, ValueError):
            cur = None
        cur = cur if isinstance(cur, dict) else None
        other = cur and not force and cur.get("session") != body.get("session")
        if other and (csig != bsig if bsig else (base is not None and (cur.get("updated") or 0) > base)):
            raise StackConflict(dict(by=cur.get("by"), at=cur.get("updated"), host=cur.get("saved_on")))
        if not body.get("legacy") and cur and cur.get("legacy"):   # (the original design files embedded in a stack are kept)
            body["legacy"] = cur["legacy"]
        body["saved_on"] = platform.node()
        body["rev"] = int((cur or {}).get("rev") or 0) + 1
        body["updated"] = max(int(body.get("updated") or 0), int(time.time() * 1000), int((cur or {}).get("updated") or 0) + 1)
        tmp = d / f".{sid}.tmp"
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, old)
        st = old.stat()
        return dict(sig=f"{body['rev']}-{st.st_mtime_ns}-{st.st_size}", rev=body["rev"], updated=body["updated"])

    def stack_lock(self, sid, session, release=False, stale_s=150):
        """A note that the stack is open on this computer (renewed every minute by the page): (ok, the other holder or
        None).  Another page's note younger than stale_s is not taken over; release: remove ours."""
        import platform
        if SD.is_folder_id(sid):
            return self.folders.lock(sid, session, platform.node(), release, stale_s)
        d = self._dir("stacks")
        if d is None or not re.match(r"^[A-Za-z0-9_.~:@+-]{1,200}$", str(sid)):
            return True, None
        f = d / f"{sid}.lock"
        other = None
        try:
            other = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
        except (OSError, ValueError):
            other = None
        mine = other is not None and other.get("session") == session
        if release:
            if mine:
                try:
                    f.unlink()
                except OSError:
                    pass
            return True, None
        if other and not mine and time.time() - (other.get("beat") or 0) < stale_s:
            return False, dict(by=other.get("by"), host=other.get("host"), since=other.get("since"))
        f.write_text(json.dumps(dict(session=session, by=user_name(), host=platform.node(), beat=time.time(),
                                     since=(other or {}).get("since") if mine else time.time())), encoding="utf-8")
        return True, None

    def delete_stack(self, sid):
        """Not deleted: moved to stacks/_deleted/ (a stack holds a whole assembly's record).  A folder stack: only
        taken off the recent list (its folder is the record)."""
        if SD.is_folder_id(sid):
            return self.folders.forget(sid)
        d = self._dir("stacks")
        if d and (d / f"{sid}.json").is_file():
            (d / "_deleted").mkdir(exist_ok=True)
            os.replace(d / f"{sid}.json", d / "_deleted" / f"{sid}_{time.strftime('%Y%m%d_%H%M%S')}.json")

    # flakes imported by hand (Library: Import flake): one file each
    # AFM lists and results per chip (flakeid_designer/afmplan.py): <data>/afm/<plan>/plan.json, lists, pictures
    def afm_plans(self, scan=None, drafts=False):
        """Every AFM list: the library's own and the related data folders' (2026-11-03g; a list stays in the folder it
        was made in, and is changed there)."""
        out, ids = [], set()
        for d in self._data_dirs("afm"):
            for f in sorted(d.glob("*/plan.json")):
                if f.parent.name in ids:
                    continue
                try:
                    pl = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                ids.add(f.parent.name)
                if (scan is None or pl.get("scan") == scan) and (drafts or not pl.get("draft")):
                    out.append(pl)                      # (drafts: Library › AFM runs only, 2026-11-03e)
        return sorted(out, key=lambda q: q.get("updated") or 0, reverse=True)

    def afm_plan(self, pid):
        if not re.match(r"^[A-Za-z0-9_-]{1,120}$", str(pid or "")):
            return None, None
        for d in self._data_dirs("afm"):
            f = d / pid / "plan.json"
            try:
                return json.loads(f.read_text(encoding="utf-8")), f
            except (OSError, ValueError):
                continue
        return None, None

    # device photos (the scanner's "Device photos": <device>_<date>/ with <name>_photos.json and photos/) for the
    # Fabrication view: found in device_photos folders next to the library, imported into a stack's own files
    def device_photo_roots(self):
        cand = []
        if self.data is not None:
            cand += [self.data.parent / "device_photos", self.data.parent.parent / "device_photos"]
        for r in self.roots:
            cand += [Path(r) / "device_photos", Path(r).parent / "device_photos"]
        out = []
        for c in cand:
            try:
                if c.is_dir() and str(c) not in out:
                    out.append(str(c))
            except OSError:
                pass
        return out

    @staticmethod
    def device_photo_set(d):
        """(record of <d>/<name>_photos.json, its path) or (None, None)."""
        try:
            j = next(iter(sorted(Path(d).glob("*_photos.json"))), None)
            return (json.loads(j.read_text(encoding="utf-8")), j) if j else (None, None)
        except (OSError, ValueError):
            return None, None

    def list_device_photos(self):
        sets = []
        for root in self.device_photo_roots():
            try:
                subs = sorted((x for x in Path(root).iterdir() if x.is_dir() and not x.name.startswith(".")),
                              key=lambda x: x.stat().st_mtime, reverse=True)
            except OSError:
                continue
            for d in subs[:200]:
                rec, j = self.device_photo_set(d)
                if rec is None:
                    continue
                ph = rec.get("photos") or []
                sets.append(dict(name=d.name, path=str(d), device=rec.get("device") or d.name, n=len(ph),
                                 objectives=sorted({x.get("objective", "") for x in ph}, key=lambda o: -float(o[:-1] or 0) if o[:-1].replace(".", "").isdigit() else 0),
                                 taken=int(j.stat().st_mtime * 1000)))
        return dict(roots=self.device_photo_roots(), sets=sets)

    def import_device_photos(self, sid, path):
        """Copy a device-photo set's JPEGs into <data>/stack_files/<sid>/fab/<set>/; returns its record for the stack."""
        rec, j = self.device_photo_set(path)
        if rec is None:
            raise ValueError(f"{path}: no device photos here (a folder of the scanner's Device photos, with *_photos.json)")
        d = self._dir("stack_files")
        if d is None:
            raise RuntimeError("no shared data folder")
        src = Path(path)
        name = re.sub(r"[^A-Za-z0-9_.~@+-]+", "_", src.name).strip("._")[:100] or "photos"
        dst = d / sid / "fab" / name
        dst.mkdir(parents=True, exist_ok=True)

        def cp(rel):
            if not rel:
                return None
            f = src / rel
            fn = re.sub(r"[^A-Za-z0-9_.~@+-]+", "_", f.name)
            if not (dst / fn).is_file():
                shutil.copy2(str(f), str(dst / fn))
            return f"/stackfile/{sid}/fab/{name}/{fn}"
        photos = []
        for ph in rec.get("photos") or []:
            try:
                photos.append(dict(objective=ph.get("objective"), label=ph.get("label"), um_per_px=ph.get("um_per_px"), w=ph.get("w"),
                                   h=ph.get("h"), url=cp(ph.get("jpg")), thumb=cp(ph.get("thumb")), terrace_map=cp(ph.get("terrace_map")),
                                   shifts_combined=ph.get("shifts_combined"), flat_field=ph.get("flat_field"),
                                   terraces=[dict(n=t.get("n"), area_um2=t.get("area_um2"), contrast=t.get("contrast"), hc=cp(t.get("hc")))
                                             for t in ph.get("terraces") or []]))
            except OSError as e:
                raise RuntimeError(f"{ph.get('jpg')}: not copied ({e})")
        return dict(set=src.name, device=rec.get("device") or src.name, oxide_nm=rec.get("oxide_nm"), source=str(src),
                    taken=int(j.stat().st_mtime * 1000), photos=photos)

    def new_fab_photos(self, sid, title, photos):
        """A stack kept in the designer's data: pictures taken any way into <data>/stack_files/<sid>/fab/<step>_<time>/;
        returns the step's photos for the page (as import_device_photos)."""
        d = self._dir("stack_files")
        if d is None:
            raise RuntimeError("no shared data folder")
        name = (re.sub(r"[^A-Za-z0-9_.~@+-]+", "_", title or "").strip("._")[:60] or "step") + time.strftime("_%Y%m%d_%H%M%S")
        dst = d / sid / "fab" / name
        dst.mkdir(parents=True, exist_ok=True)
        out = []
        for p in photos:
            fn = SD.unique_name(dst, p.get("name") or "picture.jpg")
            (dst / fn).write_bytes(p["data"])
            e = SD.photo_entry(p, fn)
            e.update(url=f"/stackfile/{sid}/fab/{name}/{fn}", thumb=f"/stackfile/{sid}/fab/{name}/{fn}")
            e.pop("file", None)
            out.append(e)
        return dict(set=name, device=None, source="designer: new step from pictures", taken=int(time.time() * 1000), photos=out)

    # colour profiles of picture sources (imported flakes' thickness from colour): gamma from flakes of known thickness
    def colour_profiles(self):
        d = self._dir("colour_profiles")
        try:
            return json.loads((d / "profiles.json").read_text(encoding="utf-8")) if d is not None and (d / "profiles.json").is_file() else {}
        except (OSError, ValueError):
            return {}

    def add_colour_sample(self, name, sample):
        """Add a flake of known thickness to profile `name`; its gamma = the median of its flakes'.  Returns the profile."""
        d = self._dir("colour_profiles")
        if d is None:
            raise RuntimeError("no shared data folder")
        with self.lock:
            P = self.colour_profiles()
            pr = P.get(name) or dict(name=name, samples=[])
            pr["samples"] = [x for x in pr["samples"] if x.get("picture") != sample.get("picture")] + [sample]
            import statistics
            pr["gamma"] = round(float(statistics.median([x["gamma"] for x in pr["samples"]])) if pr["samples"] else 2.2, 3)
            pr["updated"] = int(time.time() * 1000)
            P[name] = pr
            tmp = d / "profiles.json.tmp"
            tmp.write_text(json.dumps(P, indent=1), encoding="utf-8")
            os.replace(str(tmp), str(d / "profiles.json"))
        return pr

    def list_manual(self):
        return self.collection("manual")[1]

    def put_manual(self, mid, body):
        d = self._dir("manual")
        if d is None:
            raise OSError("no shared data folder")
        body = dict(body, by=body.get("by") or user_name())
        tmp = d / f".{mid}.tmp"
        tmp.write_text(json.dumps(body), encoding="utf-8")
        os.replace(tmp, d / f"{mid}.json")

    def delete_manual(self, mid):
        """Not deleted: moved to manual/_deleted/."""
        d = self._dir("manual")
        if d and (d / f"{mid}.json").is_file():
            (d / "_deleted").mkdir(exist_ok=True)
            os.replace(d / f"{mid}.json", d / "_deleted" / f"{mid}_{time.strftime('%Y%m%d_%H%M%S')}.json")

    def legacy_info(self):
        """Old design files and whether they were moved to stacks (the page offers the move)."""
        d, s = (self.data / "designs") if self.data else None, (self.data / "stacks") if self.data else None
        n = len(list(d.glob("*.json"))) if d and d.is_dir() else 0
        return dict(designs=n, migrated=bool(s and (s / "_migration.json").is_file()))

    def delete_design(self, did):
        d = self._dir("designs")
        if d and (d / f"{did}.json").is_file():
            (d / f"{did}.json").unlink()

    # terraces drawn by hand in the designer (one file each: also data for the automatic terrace finding)
    def list_terraces(self):
        return self.collection("terraces")[1]

    def put_terrace(self, tid, body):
        d = self._dir("terraces")
        if d is None:
            raise OSError("no shared data folder")
        body = dict(body, by=body.get("by") or user_name())
        tmp = d / f".{tid}.tmp"
        tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
        os.replace(tmp, d / f"{tid}.json")

    def delete_terrace(self, tid):
        d = self._dir("terraces")
        if d and (d / f"{tid}.json").is_file():
            (d / f"{tid}.json").unlink()

    def review_state(self):
        """Keep / discard decisions: every person appends to their own file; the latest decision per terrace wins.
        -> (signature, [dict(id, data)], terrace id -> latest entry).  The files only grow: each is read on from where the
        last read ended (before: all of them in full on every poll, 2026-10-16)."""
        d = self._dir("review")
        ents = self._entries(d, ".jsonl") if d else None
        if ents is None:
            return "none", [], {}
        files = [(d, name, mt, sz) for name, mt, sz in ents]
        for q in self._data_dirs("review")[1:]:          # (the related libraries' decisions about these runs, 2026-11-03g)
            files += [(q, name, mt, sz) for name, mt, sz in (self._entries(q, ".jsonl") or [])]
        sig = _sig("review" + repr([(str(q), n, mt, sz) for q, n, mt, sz in files]))
        with self.review_lock:
            hit = self._lists.get("review")
            if hit and hit[0] == sig:
                return hit
            for d, name, mt, sz in files:
                c = self._review.get(str(d / name))
                if c is None or sz < c["pos"] or (sz == c["size"] and mt != c["mt"]):     # new, or written anew
                    c = dict(pos=0, size=0, mt=0, latest={})
                if sz > c["pos"]:
                    try:
                        with open(d / name, "rb") as fh:
                            fh.seek(c["pos"])
                            chunk = fh.read()
                    except OSError:
                        continue
                    end = chunk.rfind(b"\n") + 1             # (complete lines only: the rest is read next time)
                    for line in chunk[:end].decode("utf-8", "replace").splitlines():
                        try:
                            e = json.loads(line)
                        except ValueError:
                            continue
                        k = e.get("id") if isinstance(e, dict) else None
                        if k and (k not in c["latest"] or e.get("updated", 0) >= c["latest"][k].get("updated", 0)):
                            c["latest"][k] = e
                    c["pos"] += end
                c["size"], c["mt"] = sz, mt
                self._review[str(d / name)] = c
            latest = {}
            for d, name, _, _ in files:
                for k, e in self._review.get(str(d / name), {}).get("latest", {}).items():
                    if k not in latest or e.get("updated", 0) >= latest[k].get("updated", 0):
                        latest[k] = e
            out = [dict(id=k, data=e.get("body", {})) for k, e in latest.items()
                   if any((e.get("body") or {}).get(k) for k in ("decision", "stars", "clean", "afmoff", "tchoice", "afmdec", "tset", "hist"))]   # ("<flake>.stars", ".clean", ".afmoff", ".tchoice", ".afmdec", "<terrace>.tset")
            self._lists["review"] = (sig, out, latest)
            return sig, out, latest

    def list_review(self):
        return self.review_state()[1]

    def seen_state(self):
        """(signature, runs already offered for screening (in this library, on any computer): [dict(id=run, data={by, at})])."""
        d = self._dir("")
        f = d / "screened_runs.json" if d else None
        try:
            st = f.stat()
            key = (st.st_mtime_ns, st.st_size)
            j = self._read_json(f, key)
        except (OSError, AttributeError):
            key, j = None, None
        j = dict(j) if isinstance(j, dict) else {}
        sig = f"seen{f}{key}"
        for q in list(self.related) if d else []:                     # (runs screened in a related library, 2026-11-03g)
            g = q / "screened_runs.json"
            try:
                st = g.stat()
            except OSError:
                continue
            o = self._read_json(g, (st.st_mtime_ns, st.st_size))
            for k, v in (o.items() if isinstance(o, dict) else ()):
                j.setdefault(k, v)
            sig += f"|{g}{st.st_mtime_ns}"
        if key is None and sig == f"seen{f}{key}":
            return "none", []
        return _sig(sig), [dict(id=k, data=v) for k, v in j.items()]

    def list_seen(self):
        return self.seen_state()[1]

    def put_seen(self, rid, body):
        d = self._dir("")
        if d is None:
            raise OSError("no shared data folder")
        with self.seen_lock:                             # (not self.lock: that one serves the runs' files)
            f = d / "screened_runs.json"
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                j = {}
            j[rid] = dict(by=user_name(), at=body.get("at") or int(time.time() * 1000))
            tmp = d / ".screened_runs.tmp"
            tmp.write_text(json.dumps(j, indent=1), encoding="utf-8")
            os.replace(tmp, f)

    def archived_state(self):
        """(signature, runs archived in this library (Library › Archive: left out of the flake list and screening, on
        every computer): [dict(id=run, data={run, archived, by, at})])."""
        d = self._dir("")
        f = d / "archived_runs.json" if d else None
        try:
            st = f.stat()
            key = (st.st_mtime_ns, st.st_size)
        except (OSError, AttributeError):
            return "none", []
        j = self._read_json(f, key)
        j = j if isinstance(j, dict) else {}
        return _sig(f"archived{f}{key}"), [dict(id=k, data=v) for k, v in j.items()]

    def put_archived(self, rid, body):
        """Archive (archived: true) or bring back (false) a run; nothing of the run itself is touched."""
        d = self._dir("")
        if d is None:
            raise OSError("no shared data folder")
        with self.seen_lock:
            f = d / "archived_runs.json"
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                j = {}
            j[rid] = dict(run=str(body.get("run") or rid), archived=bool(body.get("archived")), by=user_name(),
                          at=body.get("at") or int(time.time() * 1000))
            tmp = d / ".archived_runs.tmp"
            tmp.write_text(json.dumps(j, indent=1), encoding="utf-8")
            for i in range(5):                           # (Windows: a virus scanner or indexer can hold the file a moment)
                try:
                    os.replace(tmp, f)
                    break
                except PermissionError:
                    if i == 4:
                        raise
                    time.sleep(0.2 * (i + 1))

    def put_review(self, tid, body):
        """One decision appended to this person's file.  Stamped later than the latest decision on that terrace this
        computer has seen (a hybrid clock: a decision made after seeing another one wins even when this computer's
        clock is behind, 2026-10-16)."""
        d = self._dir("review")
        if d is None:
            raise OSError("no shared data folder")
        prev = self.review_state()[2].get(tid) or {}
        upd = max(int(time.time() * 1000), int(prev.get("updated") or 0) + 1)
        e = dict(id=tid, updated=upd, by=user_name(), body=dict(body, updated=upd))
        with self.review_lock, open(d / f"{user_name()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(e) + "\n")
        RUN_DATA.mirror(self, [e])                       # (also in the run's own folder: seen by any computer that adds it)
        try:                                             # the kept flakes' 100x pictures into the run's kept/ folder
            if "decision" in body:                       # (not for a flake's stars)
                KP.KEPT.request(self, tid)
                if body.get("decision") in KP.KEEP:
                    KEPT_COPY.want(self, (KP.parse_tid(tid) or [None])[0])
            if tid.endswith(".clean"):                   # (the AFM center, also next to the 100x pictures: kept.py)
                threading.Thread(target=KP.write_center, args=(self, tid, dict(body, by=user_name())), daemon=True).start()
        except Exception as ex:                          # (never fails the decision)
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} kept pictures not requested: {ex}\n")


# Saved coordinates of a run (Library › Restore coordinates): the scan program's "Remeasure flake coordinates" moves
# every stage coordinate of the run in place and first copies each file it changes to <run>/old_coords/<YYYYmmdd_HHMMSS>/
# (flakeid/scope/remeasure.py).  Restoring copies such a set back; the files it replaces are first saved as a new set
# ("..._before_restore"), so a restore can itself be undone.
COORDS_DIR = "old_coords"
_COORDS_META = ("remeasure_report.json", "restore_note.json")


class RunData:
    """(2026-11-10b) A run's decisions travel with the run.  Keep / discard, AFM screening keep / discard, stars, the AFM
    center and the thickness set by hand were written only to the library's own designer data (<library>/_designer_data):
    another computer that added the run from a different library did not see them.  Each decision is now also
    appended to <run folder>/_designer_data/review/<person>.jsonl (which every computer reads for an added run,
    Store.related_data), and after each walk of the runs the latest decision on every terrace of every run is copied
    there when the run's folder does not have it yet (decisions made before, or on a computer without this version).
    New AFM lists are made in <run folder>/_designer_data/afm (run_dir).  All share I/O in one background thread, in order;
    it never fails a decision (the library's copy is written first, as before)."""

    def __init__(self):
        import queue
        self.q = queue.Queue()
        self.thread = None
        self.lock = threading.Lock()
        self.pending_sync = False
        self.done = 0                                    # entries written (tests)

    def _start(self):
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._work, daemon=True, name="run-data")
                self.thread.start()

    def _work(self):
        while True:
            job = self.q.get()
            try:
                job()
            except Exception as ex:                      # (a share that does not answer: tried again at the next walk)
                sys.stderr.write(f"{time.strftime('%H:%M:%S')} decisions not copied to the run's folder: {ex}\n")
            finally:
                self.q.task_done()

    def wait(self, timeout=30.0):
        """(tests) until every queued write is done"""
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.02)

    @staticmethod
    def run_dir(store, run, sub):
        """<the run's folder>/_designer_data/<sub> (made), or None: unknown run, a run name two added runs share, or
        the library's own data folder (already written)."""
        if not run:
            return None
        with store.lock:
            ds = [v["dir"] for k, v in store.runs.items() if v["doc"].get("run") == run or k == run]
        if len({str(d) for d in ds}) != 1:
            return None
        d = Path(ds[0])
        rd = d.parent if d.name == "designer" else d     # (as Store.related_data reads it)
        q = rd / "_designer_data"
        try:
            if store.data is not None and q.resolve() == Path(store.data).resolve():
                return None
            (q / sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        if q not in store.related:                       # (read here too from now on)
            store.related = list(store.related) + [q]
        return q / sub

    def mirror(self, store, entries):
        """Decision entries (as written to the library's file) appended to their runs' own folders, in the background."""
        ents = [e for e in entries if isinstance(e, dict) and ":" in str(e.get("id") or "")]
        if not ents:
            return
        self.q.put(lambda: self._append(store, ents))
        self._start()

    def _append(self, store, ents):
        by_run = {}
        for e in ents:
            by_run.setdefault(str(e["id"]).split(":", 1)[0], []).append(e)
        for run, es in by_run.items():
            d = self.run_dir(store, run, "review")
            if d is None:
                continue
            with open(d / f"{user_name()}.jsonl", "a", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(e) + "\n" for e in es))
            self.done += len(es)

    def sync_soon(self, store):
        """After a walk of the runs: every run's latest decisions copied into its own folder where missing or older."""
        with self.lock:
            if self.pending_sync:
                return
            self.pending_sync = True

        def job():
            with self.lock:
                self.pending_sync = False
            self.sync(store)
        self.q.put(job)
        self._start()

    def sync(self, store):
        sig, _, latest = store.review_state()
        if sig == "none":
            return 0
        with store.lock:
            runs = sorted({v["doc"].get("run") for v in store.runs.values() if v["doc"].get("run")})
        n = 0
        for run in runs:
            mine = [e for tid, e in latest.items() if tid.startswith(run + ":")]
            if not mine:
                continue
            d = self.run_dir(store, run, "review")
            if d is None:
                continue
            have = {}                                    # what the run's folder already holds: id -> latest stamp
            for name, _, _ in Store._entries(d, ".jsonl") or []:
                try:
                    with open(d / name, encoding="utf-8") as fh:
                        for line in fh:
                            try:
                                e = json.loads(line)
                            except ValueError:
                                continue
                            if isinstance(e, dict) and e.get("id"):
                                have[e["id"]] = max(have.get(e["id"], 0), int(e.get("updated") or 0))
                except OSError:
                    continue
            new = [e for e in mine if int(e.get("updated") or 0) > have.get(e.get("id"), -1)]
            if new:
                with open(d / f"{user_name()}.jsonl", "a", encoding="utf-8") as fh:
                    fh.write("".join(json.dumps(e) + "\n" for e in new))
                n += len(new)
                self.done += len(new)
        return n


RUN_DATA = RunData()


def _run_dirs(store, run):
    with store.lock:
        return [v["dir"].parent for v in store.runs.values() if v["doc"].get("run") == run]


AFM_SCANS = "afm_scans"           # (2026-11-02l) <run>/afm_scans: the scanner makes it empty in every run folder; the AFM
                                  # images (.spm) of the run's flakes are saved there, and are read from there by default


AFM_PGM = "afm_pgm"               # (2026-11-03q) <run>/afm_pgm: the scanner makes it empty in every run folder (py-2026-11-03a);
                                  # Generate AFM file saves the AFM point lists (.pgm / .csv) there, made if missing


def afm_pgm_dir(store, run):
    """The run's afm_pgm folder (made if the run is older than the scanner's py-2026-11-03a), else None."""
    try:
        rd = _run_dirs(store, run) if run else []
    except Exception:
        rd = []
    for r in rd:
        d = Path(r) / AFM_PGM
        try:
            d.mkdir(exist_ok=True)
        except OSError:
            pass
        if d.is_dir():
            return str(d)
    return None


def afm_typed(store, key, data=None, write=False):
    """(2026-11-03q) The AFM X / Y / Z typed for an AFM list's reference points, saved as they are typed so that a lost
    connection or a reload of the page loses nothing: <data>/afm_typed/<hash of key>.json {key, data, saved}.
    key = the chip (or block) and the list (its id, or "new"); write with data None removes it."""
    if not key or not isinstance(key, str) or len(key) > 400:
        return dict(error="no key")
    d = store._dir("afm_typed")
    if d is None:
        return dict(error="no shared data folder")
    f = d / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:24] + ".json")
    if write:
        if data is None:
            try:
                f.unlink()
            except OSError:
                pass
            return dict(ok=True)
        txt = json.dumps(dict(key=key, data=data, saved=int(time.time() * 1000)))
        if len(txt) > 200000:
            return dict(error="too much")
        tmp = f.with_suffix(".tmp")
        tmp.write_text(txt, encoding="utf-8")
        os.replace(str(tmp), str(f))
        return dict(ok=True)
    try:
        j = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(data=None)
    return dict(data=j.get("data") if j.get("key") == key else None)


AFM_DONE_MSG = ("this AFM list is marked AFM scan complete: its flakes and positions stay as they were scanned "
                "(remove the complete mark in Library › AFM runs to change them, or Redo… for a new list)")


def afm_folder(store, pl):
    """Where an AFM list's AFM images (.spm) are read from: (folder or None, is_default).  The folder chosen for that
    list ("Change folder…", kept in its plan as afm_folder) while it exists, else its run's afm_scans folder (made if
    the run is older than the scanner's py-2026-11-02e), else None (then a dialog asks)."""
    f = pl.get("afm_folder")
    if f and Path(f).is_dir():
        return str(f), False
    try:
        rd = _run_dirs(store, pl.get("run")) if pl.get("run") else []
    except Exception:
        rd = []
    for r in rd:
        d = Path(r) / AFM_SCANS
        try:
            d.mkdir(exist_ok=True)
        except OSError:
            pass
        if d.is_dir():
            return str(d), True
    return None, False


def afm_result(pl, n, k=0):
    """(2026-11-04l) An AFM list point's AFM result: k = 0 its first (results), k >= 1 the ones added by hand
    ("AFM file assignment": more[n], e.g. a flake scanned again), or None."""
    n, k = str(int(n)), int(k or 0)
    if k <= 0:
        return (pl.get("results") or {}).get(n)
    m = (pl.get("more") or {}).get(n) or []
    return m[k - 1] if k - 1 < len(m) else None


def afm_files(folder):
    """The AFM files (.spm, or NanoScope's numbered .001 …) of a folder in name order as the Finder / Explorer show it:
    case ignored, numbers by value (scan_9 before scan_10)."""
    nat = lambda f: [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", f.name)]
    try:
        return [str(f) for f in sorted((f for f in Path(folder).iterdir() if f.is_file() and (f.suffix.lower() == ".spm" or re.search(r"\.\d{3}$", f.name))), key=nat)]
    except OSError:
        return []


def coord_points(store, run):
    """The run's saved coordinate sets, newest first: [dict(id, label, files)]."""
    out = []
    for d in _run_dirs(store, run):
        base = d / COORDS_DIR
        if not base.is_dir():
            continue
        for p in sorted((q for q in base.iterdir() if q.is_dir()), key=lambda q: q.name, reverse=True):
            n = sum(1 for f in p.rglob("*") if f.is_file() and f.name not in _COORDS_META)
            label = p.name
            try:
                if (p / "restore_note.json").is_file():
                    nt = json.loads((p / "restore_note.json").read_text(encoding="utf-8"))
                    label = f"{nt.get('date', p.name)}: as before restoring {nt.get('restored', '?')}"
                elif (p / "remeasure_report.json").is_file():
                    rp = json.loads((p / "remeasure_report.json").read_text(encoding="utf-8"))
                    pcs = rp.get("pieces") or []
                    label = (f"{rp.get('date', p.name)}: before remeasure"
                             + "".join(f" ({q.get('old')}: {q.get('rot_deg', 0):+.2f}°, {q['shift_um'][0]:+.0f}, {q['shift_um'][1]:+.0f} µm)"
                                       for q in pcs if q.get("shift_um")))
            except (OSError, ValueError, KeyError, TypeError):
                pass
            out.append(dict(id=p.name, label=label, files=n))
    return out


def restore_coords(store, run, pid):
    """Put a saved coordinate set back (copy its files over the run's); the current ones are saved first.  -> dict."""
    dirs = _run_dirs(store, run)
    if not dirs:
        raise FileNotFoundError(f"run {run} not found in this library's folders")
    d = dirs[0]
    src = d / COORDS_DIR / pid
    if not pid or "/" in pid or "\\" in pid or not src.is_dir():
        raise FileNotFoundError(f"no saved coordinates {pid} in {d / COORDS_DIR}")
    files = [f for f in src.rglob("*") if f.is_file() and f.name not in _COORDS_META]
    keep = d / COORDS_DIR / (time.strftime("%Y%m%d_%H%M%S") + "_before_restore")
    for f in files:                                     # (all saved before any is replaced)
        cur = d / f.relative_to(src)
        if cur.is_file():
            (keep / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cur, keep / f.relative_to(src))
    rm = d / "remeasure.json"                           # (the run's "remeasured" report goes with the coordinates)
    if rm.is_file() and not (src / "remeasure.json").is_file():
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rm, keep / "remeasure.json")
    keep.mkdir(parents=True, exist_ok=True)
    (keep / "restore_note.json").write_text(json.dumps(dict(date=time.strftime("%Y-%m-%d %H:%M"), restored=pid, by=user_name()),
                                                       indent=1), encoding="utf-8")
    for f in files:
        dst = d / f.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
    if rm.is_file() and not (src / "remeasure.json").is_file():
        rm.unlink()
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} {d}: coordinates restored from {COORDS_DIR}/{pid} ({len(files)} files); "
                     f"the replaced ones in {keep}\n")
    threading.Thread(target=store.scan, daemon=True).start()
    return dict(restored=len(files), saved_as=keep.name, dir=str(d))


# The 10x picture of a flake (the 100x window and the screening, key 3; 2026-10-29): read from the chip's viewer in the run's
# folder on the share (<run>/<piece>/index.html and its tile pyramid tiles/look, else tiles/image), at the place and with
# the outline the 10x analysis found the flake (its stage position can be off from it by up to ~150 um where the 10x
# scan's tiles were misplaced, so the 100x position is not used to find it)
_TENX = {}                          # viewer page -> ((mtime_ns, size), its summary)
_TENX_LOCK = threading.Lock()
_TENX_NAME = re.compile(r"[A-Za-z0-9_.~@+-]{1,160}")


def _tenx_piece(store, run, scan):
    """The piece folder (with its viewer) of run `run`, or None."""
    scan = str(scan or "").rsplit(":", 1)[-1]           # (a block's flakes: "<run>:<piece>")
    if not _TENX_NAME.fullmatch(scan) or scan in (".", ".."):
        return None
    for d in _run_dirs(store, run):
        p = d / scan
        if (p / "index.html").is_file():
            return p
    return None


def tenx_viewer(store, run, scan):
    """(piece folder, summary) of a piece's viewer: dict(pyr, w, h, level, tile, um_px, flakes: {id: dict(x, y, bb, poly)}).
    Read once (the page is several MB on the share) and again only when it changes."""
    p = _tenx_piece(store, run, scan)
    if p is None:
        raise FileNotFoundError(f"no viewer of {scan} in the folder of run {run}")
    page = p / "index.html"
    st = page.stat()
    sig = (st.st_mtime_ns, st.st_size)
    with _TENX_LOCK:
        c = _TENX.get(str(page))
        if c and c[0] == sig:
            return p, c[1]
    html = page.read_text(encoding="utf-8")
    m = re.search(r"const D=(\{.*?\});\n", html, re.S)
    if not m:
        raise ValueError(f"{page}: not a chip viewer")
    D = json.loads(m.group(1))
    info = D.get("info") or {}
    pyr = "look" if info.get("look") and (p / "tiles" / "look").is_dir() else "image"
    I = info.get(pyr) or {}
    if not I.get("width"):
        raise ValueError(f"{page}: no 10x picture")
    fl = {}
    for f in D.get("flakes") or []:
        try:
            fl[int(f["id"])] = dict(x=f.get("x"), y=f.get("y"), bb=[f.get("bx"), f.get("by"), f.get("bw"), f.get("bh")],
                                    poly=f.get("poly") or [])
        except (KeyError, TypeError, ValueError):
            continue
    s = dict(pyr=pyr, w=int(I["width"]), h=int(I["height"]), level=int(I["maxLevel"]), tile=int(I.get("tileSize") or 256),
             um_px=float(D.get("pixel_um") or 1.0911), flakes=fl)
    with _TENX_LOCK:
        _TENX[str(page)] = (sig, s)
        while len(_TENX) > 8:                           # (the last few chips: ~1 MB each)
            _TENX.pop(next(iter(_TENX)))
    return p, s


def tenx_flake(store, run, scan, fid):
    """What the page needs to draw flake `fid`'s 10x picture: the pyramid's size and the flake's place and outline (10x
    pixels), or an error."""
    p, s = tenx_viewer(store, run, scan)
    f = s["flakes"].get(int(fid))
    if f is None:
        raise KeyError(f"flake {fid} is not in the viewer of {scan}")
    return dict({k: s[k] for k in ("pyr", "w", "h", "level", "tile", "um_px")}, flake=f)


_H100 = {}                          # viewer page -> ((mtime_ns, size), its 100x pictures per flake: const H100)


def frame_entries(store, run, scan, fid, img=None):
    """(piece folder, the flake's 100x frame entries in the chip viewer's H100: the designer's own first) or (None, []).
    Of several frames, the one whose natural picture is the same size as the designer's copy of it (img: its
    /_blob/<run>/<file> address; builds copy it byte for byte) comes first."""
    p = _tenx_piece(store, run, scan)
    if p is None:
        return None, []
    page = p / "index.html"
    st = page.stat()
    sig = (st.st_mtime_ns, st.st_size)
    with _TENX_LOCK:
        c = _H100.get(str(page))
    if c and c[0] == sig:
        H = c[1]
    else:
        html = page.read_text(encoding="utf-8")
        i = html.find("const H100=")
        H = json.loads(html[i + 11:html.index(";\n", i)]) if i >= 0 else {}
        with _TENX_LOCK:
            _H100[str(page)] = (sig, H)
            while len(_H100) > 8:
                _H100.pop(next(iter(_H100)))
    ents = [e for e in H.get(str(int(fid))) or [] if isinstance(e, dict) and e.get("f") and not e.get("domain")]
    if len(ents) > 1 and img:
        parts = urllib.parse.unquote(urllib.parse.urlparse(str(img)).path).split("/")
        mine = store.run_file(parts[2], parts[3]) if len(parts) == 4 and parts[1] == "_blob" else None
        try:
            size = mine.stat().st_size if mine is not None else None
        except OSError:
            size = None
        same = [e for e in ents if size is not None and (p / e["f"]).is_file() and (p / e["f"]).stat().st_size == size]
        if not same and mine is not None:
            same = _same_picture(mine, p, ents)
        ents = same or ents
    return p, ents


_SAMEPIC = {}


def _same_picture(mine, p, ents):
    """(2026-11-05) [the entry whose picture is the designer's copy] by content, else []: the copy is not always byte
    for byte (re-encoded, smaller), and a flake can hold a picture of another (a provisional label's, BN_16 run 4
    flake 17: flake 24's _015 first) -- the reading must use the picture shown. Thumbnails (64 x 48 grey, each
    normalised) compared by mean absolute difference; the best one when it is clearly the same picture."""
    import cv2
    import numpy as np

    def th(q):
        try:
            st = q.stat()
        except OSError:
            return None
        k = (str(q), st.st_mtime_ns, st.st_size)
        if k in _SAMEPIC:
            return _SAMEPIC[k]
        a = cv2.imread(str(q), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        v = None
        if a is not None:
            v = cv2.resize(a, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
            v = (v - v.mean()) / max(float(v.std()), 1e-3)
        _SAMEPIC[k] = v
        while len(_SAMEPIC) > 64:
            _SAMEPIC.pop(next(iter(_SAMEPIC)))
        return v
    m = th(mine)
    if m is None:
        return []
    sc = []
    for e in ents:
        q = _in_piece(p, e.get("f"))
        v = th(q) if q is not None else None
        if v is not None and v.shape == m.shape:
            sc.append((float(np.abs(v - m).mean()), e))
    sc.sort(key=lambda x: x[0])
    if sc and sc[0][0] < 0.25 and (len(sc) == 1 or sc[1][0] > 2 * sc[0][0]):
        return [sc[0][1]]
    return []


def _in_piece(p, rel):
    """p / rel when it is a file inside the chip's folder, else None."""
    rel = str(rel or "")
    q = p / rel
    return q if rel and ".." not in Path(rel).parts and not Path(rel).is_absolute() and q.is_file() else None


# (2026-11-05k) "Natural" is the frame's own <frame>.jpg: what the 100x window's Natural view shows (the designer's
# copy of it) and what the thickness is read from; "Look" the brightened _look.jpg (not shown in the designer)
FRAME_KINDS = (("f", "Natural", ("f",)), ("a", "Natural, average", ("a",)), ("sg", "B/W green", ("sg",)),
               ("a_sg", "B/W green, average", ("a_sg",)), ("c8", "High contrast", ("c8",)),
               ("a_c8", "High contrast, average", ("a_c8",)), ("dt", "Thickness change", ("dt",)),
               ("deb", "Debris", ("deb",)), ("mos", "Mosaic", ("mos",)), ("look", "Look", ("look", "f")),
               ("look_a", "Look, average", ("look", "a")), ("look_mos", "Mosaic, look", ("look", "mos")))


def frame_files(store, run, scan, fid, img=None):
    """(2026-11-05g) The files of a flake's 100x frame, for the 100x window's "Copy file path or image": [dict(k,
    label, path, folder)] of the kinds in FRAME_KINDS that are files inside the chip's folder (the frame: frame_entries,
    the designer's own picture first)."""
    p, ents = frame_entries(store, run, scan, fid, img)
    e = ents[0] if ents else None
    if p is None or e is None:
        return []
    out = []
    for k, label, keys in FRAME_KINDS:
        v = e
        for kk in keys:
            v = v.get(kk) if isinstance(v, dict) else None
        q = _in_piece(p, v) if isinstance(v, str) else None
        if q is None and len(keys) == 1 and k != "f" and isinstance(e.get("f"), str):
            q = _in_piece(p, re.sub(r"\.jpg$", f"_{k.replace('a_', 'avg_')}.jpg" if k.startswith("a_") else f"_{k}.jpg", e["f"]))
            if k == "deb" and q is None:
                q = _in_piece(p, re.sub(r"\.jpg$", "_debris.jpg", e["f"]))
        if q is not None:
            out.append(dict(k=k, label=label, path=str(q), folder=str(q.parent)))
    return out


def sg_picture(store, run, scan, fid, img=None):
    """(2026-11-03m) The scanner's black-and-white green picture of a flake's 100x frame (<frame>_sg.jpg, next to the
    colour high-contrast _c8.jpg: the flattened green channel against the terrace, +-4 %, flakeid.hundred.analyse), or
    None (the frame: frame_entries)."""
    p, ents = frame_entries(store, run, scan, fid, img)
    if p is None:
        return None
    for e in ents:
        rel = str(e.get("sg") or re.sub(r"\.jpg$", "_sg.jpg", e["f"]))
        q = p / rel
        if ".." not in Path(rel).parts and not Path(rel).is_absolute() and q.is_file():   # (inside the chip's folder)
            return q
    return None


def tenx_tile(store, run, scan, pyr, level, name):
    """A tile of a piece's 10x pyramid: its file, or None."""
    if pyr not in ("look", "image") or not re.fullmatch(r"\d{1,2}", level or "") or not re.fullmatch(r"\d{1,4}_\d{1,4}\.jpg", name or ""):
        return None
    p = _tenx_piece(store, run, scan)
    return p / "tiles" / pyr / level / name if p is not None else None


def _reset_review(store, run, afm=False):
    """Every keep / discard decision of a run cleared (Library › Reset screening), for everyone: one cleared entry per
    decided terrace, appended to this person's file, stamped later than the decision it clears.  Stars and picked-up
    flakes stay.  afm: the AFM screening's keep / discard ("<flake>.afmdec", Library › Reset AFM screening) instead.
    -> the number cleared."""
    d = store._dir("review")
    if d is None:
        raise OSError("no shared data folder")
    sig, _, latest = store.review_state()
    if sig == "none":
        raise OSError("the decisions cannot be read")
    now, lines = int(time.time() * 1000), []
    for tid, e in latest.items():
        if not tid.startswith(run + ":"):
            continue
        if afm:
            if tid.endswith(".afmdec") and ((e or {}).get("body") or {}).get("afmdec") in ("keep", "discard"):
                upd = max(now, int(e.get("updated") or 0) + 1)
                lines.append(json.dumps(dict(id=tid, updated=upd, by=user_name(), body=dict(
                    afmdec=None, flake=tid[:-len(".afmdec")], run=run, reset=True, updated=upd))))
            continue
        if tid.endswith(".stars"):
            continue
        if ((e or {}).get("body") or {}).get("decision") in ("keep", "discard"):
            upd = max(now, int(e.get("updated") or 0) + 1)
            lines.append(json.dumps(dict(id=tid, updated=upd, by=user_name(),
                                         body=dict(decision=None, terrace=tid, reset=True, updated=upd))))
    if lines:
        with store.review_lock, open(d / f"{user_name()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        RUN_DATA.mirror(store, [json.loads(q) for q in lines])
        if afm:
            return len(lines)
        try:
            KP.KEPT.request(store, run=run)
        except Exception:
            pass
    return len(lines)


def _sig(text):
    """A short signature (ETag) of a text."""
    import hashlib
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:20]


def child_python():
    """The Python for helper processes: on Windows python.exe, not pythonw.exe (its output would be lost)."""
    exe = sys.executable or "python"
    if sys.platform == "win32":
        exe = re.sub(r"pythonw\.exe$", "python.exe", exe, flags=re.I)
    return exe


def _ps_pick(kind, prompt, start, exts):
    """The Windows file / folder dialog through PowerShell (a packaged app has no python.exe for the Tk chooser)."""
    q = lambda v: "'" + str(v or "").replace("'", "''") + "'"        # noqa: E731
    if kind == "folder":
        cmd = ("Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.FolderBrowserDialog; "
               f"$d.Description = {q(prompt)}; $d.SelectedPath = {q(start)}; if ($d.ShowDialog() -eq 'OK') {{ $d.SelectedPath }}")
    else:
        flt = (" / ".join(exts).upper() + " files|" + ";".join("*." + e for e in exts) + "|") if exts else ""
        cmd = ("Add-Type -AssemblyName System.Windows.Forms; $d = New-Object System.Windows.Forms.OpenFileDialog; "
               f"$d.Title = {q(prompt)}; $d.Filter = {q(flt + 'All files|*.*')}; $d.InitialDirectory = {q(start)}; "
               "if ($d.ShowDialog() -eq 'OK') { $d.FileName }")
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", cmd], capture_output=True, text=True, timeout=600,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout.strip() or None


def _osa_str(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _start_dir(start):
    if not start:
        return None
    p = Path(str(start))
    p = p if p.is_dir() else p.parent
    return str(p) if p.is_dir() else None


NATIVE_PICK = None       # the desktop app (app.py): fn(kind "folder" | "file", prompt, start, exts) -> path or None


def pick_folder(prompt="Folder with scan results", start=None):
    """A native folder picker on this computer (a separate process: Tk / AppleScript must own their main thread)."""
    start = _start_dir(start)
    if NATIVE_PICK is not None:
        try:
            return NATIVE_PICK("folder", prompt, start, None)
        except Exception as e:                       # (the app's own dialog failed: the system's chooser instead)
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} folder dialog failed ({type(e).__name__}: {e}); system chooser\n")
    try:
        if sys.platform == "darwin":
            loc = " default location (POSIX file %s)" % _osa_str(start) if start else ""
            r = subprocess.run(["osascript", "-e", "POSIX path of (choose folder with prompt %s%s)" % (_osa_str(prompt), loc)],
                               capture_output=True, text=True, timeout=600)
            return r.stdout.strip() or None
        if sys.platform == "win32" and getattr(sys, "frozen", False):
            return _ps_pick("folder", prompt, start, None)
        code = ("import sys,tkinter as t;from tkinter import filedialog as f;r=t.Tk();r.withdraw();r.attributes('-topmost',1);"
                "print(f.askdirectory(title=sys.argv[1],initialdir=sys.argv[2] or None) or '')")
        r = subprocess.run([child_python(), "-c", code, prompt, start or ""], capture_output=True, text=True, timeout=600,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout.strip() or None
    except Exception:
        return None


def pick_file(prompt, exts, start=None):
    """A native file picker (one file) limited to the extensions `exts` (without dots) when the system allows it."""
    start = _start_dir(start)
    if NATIVE_PICK is not None:
        try:
            return NATIVE_PICK("file", prompt, start, exts)
        except Exception as e:                       # (the app's own dialog failed, e.g. a file filter it refuses)
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} file dialog failed ({type(e).__name__}: {e}); system chooser\n")
    try:
        if sys.platform == "darwin":
            loc = " default location (POSIX file %s)" % _osa_str(start) if start else ""
            typ = " of type {%s}" % ", ".join(_osa_str(e) for e in exts) if exts else ""
            for t in (typ, ""):           # an extension macOS does not know can make the filter fail: then without it
                r = subprocess.run(["osascript", "-e", "POSIX path of (choose file with prompt %s%s%s)" % (_osa_str(prompt), t, loc)],
                                   capture_output=True, text=True, timeout=600)
                if r.returncode == 0 or "-128" in r.stderr:      # -128: cancelled
                    return r.stdout.strip() or None
            return None
        if sys.platform == "win32" and getattr(sys, "frozen", False):
            return _ps_pick("file", prompt, start, exts)
        code = ("import sys,tkinter as t;from tkinter import filedialog as f;r=t.Tk();r.withdraw();r.attributes('-topmost',1);"
                "e=sys.argv[3:];ft=[(' / '.join(e).upper()+' files',' '.join('*.'+x for x in e))] if e else [];"
                "print(f.askopenfilename(title=sys.argv[1],initialdir=sys.argv[2] or None,filetypes=ft+[('All files','*.*')]) or '')")
        r = subprocess.run([child_python(), "-c", code, prompt, start or ""] + list(exts), capture_output=True, text=True,
                           timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- building runs' designer files (on this computer)
# The scan program no longer builds a run's designer files on the microscope PC (slow): this computer builds them from
# the run's results on the share once its copy is complete (copy_done.json), and again when that file is newer than the
# designer files (a re-image updated pieces).  tools/make_designer_bundle.py, in this process, one run at a time.

def in_folder(p, folder):
    """p is folder or inside it (paths as written: the share is not resolved)."""
    a, b = Path(p).parts, Path(folder).parts
    return len(a) >= len(b) and a[:len(b)] == b


_STARTED = {}


def run_started(d):
    """(2026-11-04g) When the run was started, "YYYY-MM-DD HH:MM:SS", or None: the time of the first line of the scan's
    log (<run>_log.txt), on the day of the run's state file (<run>.json "updated": the day before when the log starts
    later in the day than that, a run over midnight), else the date the run's name starts with, else the log's date.
    Cached per log (its size and time)."""
    d = Path(d)
    try:
        log = d / f"{d.name}_log.txt"
        if not log.is_file():
            log = next((q for q in sorted(d.glob("*_log.txt")) if q.name != "analysis_log.txt"), None)
        if log is None:
            return None
        st = log.stat()
        key = (str(log), st.st_size, st.st_mtime)
        if key in _STARTED:
            return _STARTED[key]
        with open(log, "rb") as f:
            m = re.match(rb"\s*(\d\d):(\d\d):(\d\d)", f.read(200))
        out = None
        if m:
            hms = tuple(int(x) for x in m.groups())
            day = None
            sj = d / f"{log.name[:-len('_log.txt')]}.json"
            try:
                u = json.loads(sj.read_text(encoding="utf-8")).get("updated") if sj.is_file() else None
                ud = datetime.datetime.strptime(u, "%Y-%m-%d %H:%M:%S") if u else None
            except (OSError, ValueError, TypeError, AttributeError):
                ud = None
            if ud is not None:
                day = ud.date() - datetime.timedelta(days=1) if hms > (ud.hour, ud.minute, ud.second) else ud.date()
            if day is None:
                mm = re.match(r"(20\d\d)(\d\d)(\d\d)", d.name)
                try:
                    day = datetime.date(*map(int, mm.groups())) if mm else None
                except ValueError:
                    day = None
            if day is None:
                day = datetime.date.fromtimestamp(st.st_mtime)
            out = f"{day:%Y-%m-%d} {hms[0]:02d}:{hms[1]:02d}:{hms[2]:02d}"
        _STARTED[key] = out
        return out
    except OSError:
        return None


_THK = {}
_THK_LOCK = threading.Lock()


def _shown_colours(pic, polys, px_um):
    """(2026-11-05p) thick100.colours' regions in another picture of the same frame (the _look one): the terrace's
    median without its outer EDGE_UM and the commonest colour off the terrace (3 um around it), no substrate check."""
    from flakeid import thick100 as T
    from flakeid.hundred import _mode_rgb
    import cv2
    import numpy as np
    M = T.terrace_mask(pic.shape, polys)
    e = int(round(T.EDGE_UM / px_um))
    Me = cv2.erode(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))) if e > 0 else M
    M2 = Me if Me.sum() >= T.MIN_PX else M
    if int(M2.sum()) < T.MIN_PX:
        return None, None
    tc = np.median(pic[M2 > 0], axis=0).astype(float)
    g = int(round(3.0 / px_um))
    off = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) == 0
    px = pic[::4, ::4][off[::4, ::4]]
    sub = None
    if len(px) >= 500:
        fm, frac = _mode_rgb(px.reshape(-1, 1, 3), step=3)
        sub = np.asarray(fm, float) if frac >= T.SUB_MIN_FRAC else None
    return tc, sub


def thick_frame(store, run, scan, fid, img=None, material=None):
    """(2026-11-04i) What the 100x window's Thickness needs to read the thickness under the cursor, as the scan program
    measures it (flakeid.thick100): the frame's linear picture (<frame>.jpg, white-balanced; and the terrace mosaic's
    _mosaic.jpg), the bare substrate's colour in it (thick100.colours: the commonest colour of the frame off the
    terrace, else the material's), and the thin-film model at 100x for the run's own calibrated 100x oxide (the
    regions file's t100.oxide_nm: the chip's oxide + the run's shift from thick100.recalibrate; without it the
    chip's oxide + the default shift): contrast = colour / substrate - 1 (R G B) against the model's.
    Returns a dict, or None (no frame)."""
    from flakeid import thick100 as T
    from flakeid.hundred import DEMO_SUBSTRATE, WHITE, WHITE_BY_MATERIAL, _M100
    import numpy as np
    p, ents = frame_entries(store, run, scan, fid, img)
    e = next((x for x in ents if _in_piece(p, x.get("f"))), None) if p is not None else None
    if e is None:
        return None
    fq = _in_piece(p, e["f"])
    rq = fq.with_name(fq.stem + "_regions.json")
    st = fq.stat()
    key = (str(fq), st.st_mtime_ns, rq.stat().st_mtime_ns if rq.is_file() else 0, material)
    with _THK_LOCK:
        if key in _THK:
            return _THK[key]
    reg = json.loads(rq.read_text(encoding="utf-8")) if rq.is_file() else {}
    t100 = reg.get("t100") or {}
    mat = (material or "hbn").lower()
    if mat not in T.T_RANGE:
        mat = "hbn"
    ox0 = float(reg.get("oxide_nm") or 300.0)
    if t100.get("oxide_nm") is not None:
        oxide, shift, cal = float(t100["oxide_nm"]), float(t100.get("oxide_shift") or 0.0), "run"
    else:
        shift = T.OXIDE_SHIFT_DEFAULT.get(mat, 0.0)
        oxide, cal = ox0 + shift, "default"
    import cv2
    a = cv2.imread(str(fq))
    if a is None:
        return None
    lin = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) + 0.5          # (as thick100.remeasure_viewer)
    sub_run = np.asarray(DEMO_SUBSTRATE.get(mat, DEMO_SUBSTRATE["hbn"]), float) * 255.0 / WHITE_BY_MATERIAL.get(mat, WHITE)
    px_um = float(reg.get("px_um") or 0.0) or float(np.sqrt(abs(np.linalg.det(_M100()))))
    tc, sub, n = T.colours(lin, reg["terrace"], px_um, sub_run) if reg.get("terrace") else (None, None, 0)
    sub_src = "frame"
    if sub is None:
        sub, sub_src = sub_run, "material"
    # (2026-11-05p) the same terrace and substrate in the picture the Natural view shows (<frame>_look.jpg, when there
    # is one), for the colour squares next to the thickness: compared by eye with the flake below them
    shown = (None, None)
    lq = _in_piece(p, (e.get("look") or {}).get("f")) if isinstance(e.get("look"), dict) else None
    if lq is None:
        cand = fq.with_name(fq.stem + "_look.jpg")
        lq = cand if cand.is_file() else None
    if tc is not None and lq is not None:
        b = cv2.imread(str(lq))
        if b is not None and b.shape[:2] == a.shape[:2]:
            look = cv2.cvtColor(b, cv2.COLOR_BGR2RGB).astype(np.float32)
            shown = _shown_colours(look, reg["terrace"], px_um)
            if sub_src != "frame":
                shown = (shown[0], None)
    ts, C = T._model(mat, oxide)
    # (2026-11-05u) few-layer graphite: the page reads whole-layer thicknesses (in nm) as thick100.layers_of does
    # ((2026-11-12) from all three channels: thin when the best fit is at most tmax, then the nearest layer colour)
    lm = T.layer_model(mat, oxide)
    layers = None if lm is None else dict(n=[int(v) for v in lm[0]], C=[[round(float(v), 4) for v in c] for c in lm[1]],
                                          nm=T.LAYER_NM, tmax=round(T.layers_tmax(), 4))
    mq = _in_piece(p, e.get("mos"))
    q = urllib.parse.urlencode
    out = dict(material=mat, oxide_nm=round(oxide, 1), chip_oxide_nm=round(ox0, 1), shift=round(shift, 1), calibrated=cal,
               sub=[round(float(v), 3) for v in sub], sub_from=sub_src, add=0.5,
               t=[round(float(v), 3) for v in ts], C=[[round(float(v), 4) for v in c] for c in C],
               alt_ratio=T.ALT_RATIO, alt_abs=T.ALT_ABS, fit_max=T.FIT_MAX, px_um=px_um, layers=layers,
               w=int(a.shape[1]), h=int(a.shape[0]), file=e["f"],
               terrace=None if not t100 else {k: t100.get(k) for k in ("t", "alts", "misfit", "contrast")},
               terrace_contrast=None if tc is None else [round(float(v), 3) for v in tc / np.maximum(sub, 1e-6) - 1.0],
               terrace_rgb=None if tc is None else [round(float(v), 1) for v in tc],
               shown_rgb=None if shown[0] is None else [round(float(v), 1) for v in shown[0]],
               shown_sub=None if shown[1] is None else [round(float(v), 1) for v in shown[1]],
               shown_file=None if shown[0] is None else "look",
               img="/api/thick100/img?" + q(dict(run=run, scan=scan, flake=int(fid), img=img or "", which="f")),
               mos=("/api/thick100/img?" + q(dict(run=run, scan=scan, flake=int(fid), img=img or "", which="mos"))) if mq else None)
    with _THK_LOCK:
        _THK[key] = out
        while len(_THK) > 32:
            _THK.pop(next(iter(_THK)))
    return out


def thick_picture(store, run, scan, fid, img=None, which="f"):
    """The linear picture thick_frame reads (the frame's <frame>.jpg, or its _mosaic.jpg), or None."""
    p, ents = frame_entries(store, run, scan, fid, img)
    e = next((x for x in ents if _in_piece(p, x.get("f"))), None) if p is not None else None
    return None if e is None else _in_piece(p, e.get("mos" if which == "mos" else "f"))


def is_run_dir(p):
    """(2026-11-05b) A scan run's folder: its designer files, the scan program's copy mark, or its piece folders."""
    d = Path(p)
    try:
        return (d / "designer" / "bundle.json").is_file() or (d / "copy_done.json").is_file() or looks_like_run(d)
    except OSError:
        return False


def looks_like_run(d):
    """A run's results folder (the share layout): piece folders with a viewer page and data."""
    try:
        return any((c / "index.html").is_file() and (c / "data").is_dir() for c in d.iterdir() if c.is_dir())
    except OSError:
        return False


def builder_root():
    """Where the builder is: the repository around this folder, a builder/ folder beside this file (the share's copy),
    or the packaged app."""
    env = os.environ.get("FLAKEID_BUILDER")              # (another copy of the builder: tests, or a newer one)
    for r in ([Path(env)] if env else []) + [HERE.parent, HERE / "builder"]:
        if (r / "flakeid" / "__init__.py").is_file() and (r / "tools" / "make_designer_bundle.py").is_file():
            return r
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", str(HERE)))
    return None


def builder_main():
    """tools/make_designer_bundle.main, or RuntimeError(why it cannot build here)."""
    import importlib.util
    root = builder_root()
    if root is None:
        raise RuntimeError("the builder (the flakeid package and tools/) is not next to the designer: copy the share's "
                           "flakeid_designer folder again (it has a builder folder), or use the app")
    if not getattr(sys, "frozen", False):
        for q in (str(root), str(root / "tools")):
            if q not in sys.path:
                sys.path.append(q)                       # (appended: this folder's own modules keep priority)
    miss = [pkg for m, pkg in (("numpy", "numpy"), ("scipy", "scipy"), ("cv2", "opencv-python"), ("skimage", "scikit-image"))
            if importlib.util.find_spec(m) is None]
    if miss:
        raise RuntimeError(f"building runs needs {', '.join(miss)} ({child_python()} -m pip install {' '.join(miss)})")
    from make_designer_bundle import main             # type: ignore
    return main


class StackConflict(Exception):
    """A stack file saved by another page after this page opened it: .info = who, when, on which computer."""
    def __init__(self, info):
        super().__init__("saved elsewhere since")
        self.info = info


def build_here():
    """Does this computer build new runs' designer files by itself?  Setting "build_here" (the builds list); default
    off in the Stacker version (the stacking PC with the camera: a build there froze its window on Windows, 2026-10-20),
    on elsewhere.  Builds asked for by hand go ahead either way."""
    v = load_config().get("build_here")
    if v is None:
        return "stacker" not in Path(sys.executable).name.lower()
    return bool(v)


def free_memory_gb():
    """Memory free for new processes (GB), or None if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / 2**30
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            n = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative"))
            return n * page / 2**30
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        return None


class Builder:
    """The runs to build, one at a time in a background thread; their state for the page (GET /api/builds)."""
    LOCK_H = 3.0                                     # a lock older than this is a computer that gave up

    def __init__(self):
        self.lock = threading.Lock()
        self.procs = set()                              # (build processes running: ended when the designer closes)
        self.queue, self.state, self.thread, self.store = [], {}, None, None
        self.started = {}                               # run folder -> run_started (read once)

    def consider(self, store, todo, unbuilt):
        self.store = store
        with self.lock:
            # (2026-11-05k) runs not in this walk any more (removed from the list, or seen in an older walk of a folder)
            # leave the list unless a build of theirs is queued, running or just ended
            keep = {str(d) for d in list(todo) + list(unbuilt)}
            listed = {str(Path(r)) for r in store.added} if store is not None and store.added is not None else None
            for k in [k for k, v in self.state.items()
                      if (v.get("state") in ("unbuilt", "off") and k not in keep)
                      or (listed is not None and k not in listed and v.get("state") not in ("queued", "building"))]:
                del self.state[k]
            for d in unbuilt:
                k = str(d)
                if k not in self.state:
                    self.state[k] = dict(run=d.name, dir=k, state="unbuilt", msg="an older run without designer files", t=time.time())
        if not build_here():
            with self.lock:
                for d in todo:
                    k = str(d)
                    if self.state.get(k, {}).get("state") not in ("queued", "building", "done"):
                        self.state[k] = dict(run=d.name, dir=k, state="off", t=time.time(),
                                             msg="not built here: building new runs is off on this computer")
            return
        for d in todo:
            self.want(d, auto=True)

    def want(self, d, auto=False, full=False):
        """Queue a build of run folder d.  full: build it all again (Library › Rebuild: the chip picture and every flake),
        not only what is new (make_designer_bundle --update), whatever the run's copy asks for."""
        k = str(d)
        with self.lock:
            st = self.state.get(k, {})
            if st.get("state") in ("queued", "building") or (auto and st.get("state") in ("failed", "slow")
                                                              and time.time() - st.get("t", 0) < 3600):
                return
            self.state[k] = dict(run=Path(d).name, dir=k, state="queued", msg="waiting", t=time.time())
            self.queue.append((Path(d), auto, full))
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, daemon=True, name="designer builds")
                self.thread.start()

    def status(self):
        with self.lock:
            out = sorted((dict(v) for v in self.state.values()), key=lambda x: (x["state"] != "building", x["state"] != "queued", -x["t"]))
        for b in out:                                    # (when each run was started: the Library shows it, 2026-11-04g)
            if b.get("dir") not in self.started:
                self.started[b["dir"]] = run_started(b["dir"])
            b["started"] = self.started[b["dir"]]
        return out

    def _set(self, d, **kw):
        with self.lock:
            self.state.setdefault(str(d), dict(run=Path(d).name, dir=str(d))).update(kw, t=time.time())

    def _loop(self):
        while True:
            with self.lock:
                if not self.queue:
                    return
                d, auto, full = self.queue.pop(0)
            try:
                if auto:                                  # (started by itself: not over a slow connection unasked)
                    slow = self.too_slow(d)
                    if slow:
                        self._set(d, state="slow", msg=slow)
                        continue
                self._build(d, full=full)
            except Exception as e:
                self._set(d, state="failed", msg=f"{type(e).__name__}: {e}")
                sys.stderr.write(f"{time.strftime('%H:%M:%S')} designer build of {d} failed: {e}\n")

    SLOW_MIN = 10.0                                  # an automatic build expected to take longer than this is asked first

    def too_slow(self, d):
        """Would building this run here take long because the share is slow from this computer (Wi-Fi, VPN: 2026-10-20,
        ~0.3 MB/s, an hour+ for an hBN run)?  A few of its 100x pictures are read and timed (with the time per file:
        the build reads hundreds); the build reads all of them and writes about half as much back.  -> a message
        (too slow) or None."""
        try:
            pics = []
            for piece in sorted(q for q in Path(d).iterdir() if q.is_dir()):
                h = piece / "100x"
                if h.is_dir():
                    pics += [q for q in h.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if not pics:
                return None
            sample, nbytes, t0 = pics[:: max(1, len(pics) // 3)][:3], 0, time.time()
            for q in sample:
                with open(q, "rb") as fh:
                    while True:
                        b = fh.read(1 << 20)
                        if not b:
                            break
                        nbytes += len(b)
            dt = max(time.time() - t0, 1e-3)
            per_file, rate = dt / len(sample), nbytes / dt
            est = len(pics) * per_file * 1.5 + 30.0           # (reading every picture, writing about half back, the rest)
        except OSError:
            return None
        if est <= self.SLOW_MIN * 60:
            return None
        return (f"not built here: the share is slow from this computer ({rate / 2**20:.1f} MB/s), building would take about "
                f"{est / 60:.0f} min. A computer on the lab network builds it in minutes; or build it here anyway.")

    @staticmethod
    def can_update(d, ddir):
        """Add only what is new to the built run (make_designer_bundle --update) instead of building it all again?  Yes
        when the run's copy_done.json asks for no whole build since the copy the bundle was made from ("rebuild" not newer
        than run.json's "source": pieces copied as their export ended, the IR pictures at the end).  A re-image, a copy
        by hand or a copy from before 2026-10-21 (no "rebuild") -> the whole build."""
        try:
            cd = json.loads((d / "copy_done.json").read_text())
            src = json.loads((ddir / "run.json").read_text()).get("source")
        except (OSError, ValueError):
            return False
        return bool(src) and isinstance(cd.get("rebuild"), str) and cd["rebuild"] <= src and (ddir / "bundle.json").is_file()

    LOCK_STALE_S = 600                               # a lock not renewed for this long: its computer stopped building
    LOCK_BEAT_S = 60

    def take_lock(self, lock):
        """The run's build lock, taken atomically (created only if absent: two computers that both found the run unbuilt
        cannot both build it, 2026-10-16).  -> None (taken) or who holds it.  A lock not renewed for LOCK_STALE_S (a
        computer that stopped mid-build) is taken over; one from an older designer (no renewal) after LOCK_H hours."""
        import platform
        import uuid
        me = json.dumps(dict(by=user_name(), host=platform.node(), since=time.strftime("%Y-%m-%d %H:%M:%S"), beat=self.LOCK_BEAT_S))
        for attempt in (0, 1):
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            except FileExistsError:
                try:
                    info = json.loads(lock.read_text() or "{}")
                except (OSError, ValueError):
                    info = {}
                try:
                    age = time.time() - lock.stat().st_mtime
                except OSError:
                    continue                             # (just removed: try again)
                stale = age > (self.LOCK_STALE_S if info.get("beat") else self.LOCK_H * 3600)
                if not stale or attempt:
                    return info.get("by") or "another computer"
                # moved aside, not deleted: of two computers that both find it stale only one can move it
                aside = lock.with_name(f".building.stale.{uuid.uuid4().hex[:8]}")
                try:
                    os.replace(lock, aside)
                    aside.unlink()
                except OSError:
                    pass
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(me)
            return None
        return "another computer"

    def _beat(self, lock, stop):
        """Renew the lock's time while building (others see a live build)."""
        while not stop.wait(self.LOCK_BEAT_S):
            try:
                os.utime(lock, None)
            except OSError:
                pass

    BUILD_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))

    def _run_build(self, args, buf, d):
        """make_designer_bundle in a separate process at low priority (2026-10-20: run inside the app, with a worker per
        core but one at normal priority, it left the stacker PC's window unusable): the window keeps its share of the
        computer and of Python.  Its output goes to buf; the latest line is the build's progress in the page.
        -> exit code."""
        import subprocess
        nw, gb = self.BUILD_WORKERS, free_memory_gb()
        if gb is not None:
            nw = max(1, min(nw, int((gb - 1.0) // 1.5)))          # (leave this computer 1 GB; ~1.5 GB per worker)
        args = list(args) + ["--workers", str(nw)]
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--helper", "bundle"] + args
        else:
            root = builder_root()
            cmd = [child_python(), "-u", str(root / "tools" / "make_designer_bundle.py")] + args
        kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                  encoding="utf-8", errors="replace", bufsize=1)
        if sys.platform == "win32":
            kw["creationflags"] = 0x00004000 | 0x08000000    # below normal priority (inherited by its workers), no window
        else:
            kw["preexec_fn"] = lambda: os.nice(10)             # (inherited by its workers)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.Popen(cmd, env=env, **kw)
        self.procs.add(p)
        logf = None
        try:                                              # (every build's whole output, on this computer)
            ld = Path.home() / ".flakeid_designer_builds"
            ld.mkdir(exist_ok=True)
            lp = ld / f"{Path(str(d)).name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            logf = open(lp, "w", encoding="utf-8", buffering=1)
            self._set(d, log=str(lp))
            logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} designer {VERSION}: {' '.join(cmd)}\n")
        except OSError:
            logf = None
        t0 = time.time()
        try:
            for line in p.stdout:
                ln = line.strip()
                if ln.startswith("PROGRESS "):            # (make_designer_bundle.progress: the bar in the page)
                    try:
                        _, fr, text = (ln.split(" ", 2) + [""])[:3]
                        fr = float(fr)
                        el = time.time() - t0
                        eta = el / fr * (1 - fr) if fr > 0.03 else None
                        self._set(d, state="building", msg=text[:160], frac=round(fr, 3), eta_s=None if eta is None else round(eta))
                    except ValueError:
                        pass
                    continue
                buf.write(line)
                if logf:
                    logf.write(line)
            rc = p.wait()
            if logf:
                logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exit {rc}\n")
            return rc
        finally:
            self.procs.discard(p)
            if logf:
                logf.close()

    def stop_all(self):
        """The designer is closing: end the builds it started (their computer stops renewing the lock)."""
        for p in list(self.procs):
            try:
                p.terminate()
            except Exception:
                pass

    def _build(self, d, full=False):
        import contextlib
        import io
        import tempfile
        ddir = d / "designer"
        lock = ddir / ".building"
        # (no builder_main() here: importing the builder's libraries into the designer's own process, next to its window,
        # is the likely cause of the Windows app hanging as a build started, 2026-10-20; the build process imports them)
        if not getattr(sys, "frozen", False) and builder_root() is None:
            raise RuntimeError("the builder (the flakeid package and tools/) is not next to the designer: copy the share's "
                               "flakeid_designer folder again (it has a builder folder), or use the app")
        ddir.mkdir(parents=True, exist_ok=True)
        who = self.take_lock(lock)
        if who is not None:
            self._set(d, state="elsewhere", msg=f"being built by {who}")
            return
        stop = threading.Event()
        threading.Thread(target=self._beat, args=(lock, stop), daemon=True).start()
        buf = io.StringIO()
        try:
            self._set(d, state="building", msg="reading the run's results", frac=0.0, eta_s=None, detail=None, log=None)
            t0 = time.time()
            if not full and self.can_update(d, ddir):
                # only what is new (pieces copied as their export ended, IR pictures): their pictures read, the new files
                # and bundle.json written straight into the run's designer folder (publish: bundle.json last)
                self._set(d, state="building", msg="adding what is new")
                rc = self._run_build([str(d), "--run", d.name, "--update", "-o", str(ddir)], buf, d)
                if rc in (0, None):
                    out = buf.getvalue().strip().splitlines()
                    n = next((ln for ln in reversed(out) if ln.startswith(("designer bundle updated:", "run "))), "")
                    self._set(d, state="done", msg=f"updated in {time.time() - t0:.0f} s" + (f" ({n.split('(', 1)[1].split(';')[0]})" if "(" in n else ""))
                    sys.stderr.write(f"{time.strftime('%H:%M:%S')} designer files of {d} updated ({time.time() - t0:.0f} s)\n")
                    return
                sys.stderr.write(f"{time.strftime('%H:%M:%S')} update of {d} failed (exit {rc}): building it again\n")
                buf = io.StringIO()
            tmp = Path(tempfile.mkdtemp(prefix="vdw_build_")) / "designer"
            try:
                rc = self._run_build([str(d), "--run", d.name, "-o", str(tmp)], buf, d)
                out = buf.getvalue().strip().splitlines()
                if rc not in (0, None) or not (tmp / "bundle.json").is_file():
                    raise RuntimeError(build_error(out) or f"exit {rc}")
            except Exception as e:
                # why it failed, on the share for everyone (the page's message stays on this computer): who, when, the
                # error and the builder's last lines; removed by the next build that works
                import platform
                import traceback
                try:
                    (ddir / "build_failed.txt").write_text(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {user_name()} on {platform.node()} ({platform.system()}), "
                        f"designer {VERSION}{' app' if getattr(sys, 'frozen', False) else ''}\n{type(e).__name__}: {e}\n\n"
                        "builder output (last lines):\n" + "\n".join(buf.getvalue().strip().splitlines()[-30:]) + "\n\n"
                        + traceback.format_exc(), encoding="utf-8")
                except OSError:
                    pass
                # the whole error for the page ("Show full error") and, run from the source folder, a copy next to the
                # designer (last_build_error.txt) to send or look at (2026-10-28: only the last line was shown)
                rep = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {d}\ndesigner {VERSION}, {sys.executable} {sys.version.split()[0]} "
                       f"({platform.system()} {platform.machine()})\n{type(e).__name__}: {e}\n\nbuilder output (last lines):\n"
                       + "\n".join(ln for ln in buf.getvalue().strip().splitlines()[-200:] if not ln.startswith("PROGRESS ")))
                self._set(d, detail=rep[-60000:])
                if not getattr(sys, "frozen", False):
                    try:
                        (HERE.parent / "last_build_error.txt").write_text(rep, encoding="utf-8")
                    except OSError:
                        pass
                raise
            self._set(d, state="building", msg="copying to the share")
            install_build(tmp, ddir)
            shutil.rmtree(tmp.parent, ignore_errors=True)
            n = next((ln for ln in reversed(out) if ln.startswith("designer bundle:")), "")
            self._set(d, state="done", msg=f"built in {time.time() - t0:.0f} s" + (f" ({n.split('(', 1)[1].split(';')[0]})" if "(" in n else ""))
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} designer files of {d} built ({time.time() - t0:.0f} s)\n")
        finally:
            stop.set()
            try:
                lock.unlink()
            except OSError:
                pass
            if self.store is not None:                   # (also after an update, which returns early)
                try:
                    self.store.scan()
                except Exception as e:
                    sys.stderr.write(f"{time.strftime('%H:%M:%S')} run search failed: {e}\n")


def build_error(lines):
    """The builder's error from its output: the last exception line ("ImportError: ...") with the lines that continue
    it (a message over several lines), else the last line.  (Before: only the last line, e.g. "your python interpreter
    from there." of NumPy's three-line ImportError, 2026-10-28.)"""
    lines = [ln.rstrip() for ln in lines if ln.strip() and not ln.startswith("PROGRESS ")]
    exc = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)\b:?")
    for i in range(len(lines) - 1, -1, -1):
        if exc.match(lines[i]):
            tail = [lines[i]]
            for ln in lines[i + 1:i + 6]:
                if exc.match(ln) or ln.startswith(("Traceback", "  File ")):
                    break
                tail.append(ln.strip())
            return " ".join(tail)[:600]
    return lines[-1][:600] if lines else ""


def install_build(tmp, ddir):
    """A build's files (tmp) into the run's designer folder (ddir, on the share) without a moment where a page finds a
    half-written run: the new files first (each copied under a temporary name and renamed: never half a file), run.json,
    then bundle.json replaced in one step; the files of the older build that this one does not use removed last.  New
    builds' picture names carry their content's hash, so the older bundle.json's pictures stay valid until it is
    replaced, and pictures that did not change are not copied again (before: bundle.json deleted first, then ~75 MB
    copied, 2026-10-16)."""
    names = sorted((q.name for q in tmp.iterdir() if q.is_file()), key=lambda n: (n == "bundle.json", n == "run.json", n))
    for n in names:
        src, dst = tmp / n, ddir / n
        try:
            if n not in ("bundle.json", "run.json", "flakes.csv") and dst.is_file() and dst.stat().st_size == src.stat().st_size \
                    and HASHED.search(n):
                continue                                 # (the same content: the name is its hash)
        except OSError:
            pass
        part = ddir / f".{n}.part"
        shutil.copy2(src, part)
        os.replace(part, dst)
    keep = set(names) | {".building"}
    for q in ddir.iterdir():
        if q.is_file() and q.name not in keep and not q.name.startswith(".building"):
            try:
                q.unlink()
            except OSError:
                pass


BUILDER = Builder()
atexit.register(BUILDER.stop_all)                      # (a closing designer ends its builds)


# ---------------------------------------------------------------- the live camera's scale from a chip's alignment marks
# ("Calibrate from marks" in the live bar): the marks of the KLayout export's template layout found in a few averaged live
# frames (marks_to_klayout.calibrate_frame, in this process: it needs numpy, opencv and gdstk)
CAL = dict(tpl=None, gds=None, mtime=None, check=None)


def marks_module():
    """(the marks_to_klayout module, None), or (None, the package that is missing)."""
    import importlib.util
    for m, pkg in (("numpy", "numpy"), ("cv2", "opencv-python"), ("gdstk", "gdstk")):
        if importlib.util.find_spec(m) is None:
            return None, pkg
    try:
        from . import marks_to_klayout as MK
    except ImportError:
        import marks_to_klayout as MK                # type: ignore
    return MK, None


def cal_template(MK, gds):
    """The template's alignment marks (kept until the file changes); the simulated camera shows them too."""
    mt = Path(gds).stat().st_mtime
    if CAL["gds"] != gds or CAL["mtime"] != mt:
        tpl, _ = MK.template_fiducials(gds)
        CAL.update(tpl=tpl, gds=gds, mtime=mt)
        if CAM is not None:
            CAM.SIM_MARKS = [p for f in tpl for p in f["polys"]]
    return CAL["tpl"]


def sim_marks():
    """Before the simulated camera starts: the template's marks for it to show, when the layout is known."""
    try:
        gds = (load_config().get("klayout") or {}).get("template_gds")
        MK, _ = marks_module()
        if MK is not None and gds and Path(gds).is_file():
            cal_template(MK, gds)
    except Exception as e:
        sys.stderr.write(f"(simulated camera without the template's marks: {e})\n")


def calibrate_live(body):
    MK, missing = marks_module()
    if MK is None:
        return dict(ok=False, why=f"the calibration needs {missing} ({child_python()} -m pip install {missing})")
    cfg = load_config()
    gds = body.get("gds") or (cfg.get("klayout") or {}).get("template_gds")
    if not gds or not Path(gds).is_file():
        return dict(ok=False, need_gds=True, why="the chip layout (GDS) with the alignment marks is needed")
    if body.get("gds"):                                  # (the same file as the KLayout export's template)
        cfg.setdefault("klayout", {})["template_gds"] = gds
        save_config(cfg)
    try:
        tpl = cal_template(MK, gds)
    except Exception as e:
        return dict(ok=False, why=f"the layout {gds} could not be read: {e}")
    if len(tpl) < 2:
        return dict(ok=False, why=f"fewer than 2 alignment marks found in {gds}")
    if CAM is None or CAM.LIVE.raw is None:
        return dict(ok=False, why="no picture from the camera (is the live view on?)")
    import cv2
    t0 = time.time()
    im = CAM.LIVE.average(int(body.get("frames") or 6))
    t_avg = time.time() - t0
    if im is None:
        return dict(ok=False, why="no picture from the camera")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    try:
        rg = body.get("rot_guess")
        t1 = time.time()
        r = MK.calibrate_frame(im, tpl, bool(body.get("mirror")), float(body.get("um_guess") or 0) or None,
                               None if rg is None else float(rg), bool(body.get("narrow")))
        t_fit = time.time() - t1
    except Exception as e:
        return dict(ok=False, why=f"the fit failed: {type(e).__name__}: {e}")
    chk = r.pop("check", None)
    if chk is not None:
        ok, buf = cv2.imencode(".jpg", chk, [cv2.IMWRITE_JPEG_QUALITY, 85])
        CAL["check"] = buf.tobytes() if ok else None
    r.update(gds=gds, frame=[int(im.shape[1]), int(im.shape[0])], has_check=chk is not None)
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} calibration {body.get('objective')}: " + (
        f"{r['um_per_px']:.5f} um/px, {r['rotation_deg']:+.2f} deg, {r['n_marks']} marks, rms {r['rms_um']:.2f} um, ok {r['ok']}"
        if "um_per_px" in r else r.get("why", "")) + f" (frames {t_avg:.1f} s, mark search {t_fit:.1f} s"
        + (", narrow: another objective's rotation and scale" if body.get("narrow") else "") + ")\n")
    return r


# ---------------------------------------------------------------- export to KLayout (marks_to_klayout.py, a subprocess)
# The pictures of a finished device are placed on the chip layout by its alignment marks; the tool needs numpy, opencv
# and gdstk, so it runs with this Python as a separate process (the designer itself stays standard library only).
KL_SCRIPT = HERE / "marks_to_klayout.py"
KL_PICS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
KL_SHOWN = (".jpg", ".jpeg", ".png", ".bmp")            # pictures a browser can show
KL_PICKS = dict(gds=("Template layout (GDS) with the alignment marks", ["gds", "gds2", "gdsii"]),
                lys=("KLayout session to add the pictures to", ["lys"]),
                lyp=("KLayout layer properties", ["lyp"]))
KL_MODULES = dict(numpy="numpy", cv2="opencv-python", gdstk="gdstk")
KL_OUTPUTS = ("lys", "gds", "check_jpg", "fit_json", "added_to", "replaced_in", "backup")


def kl_path(s):
    """A path typed or pasted by the user (quotes from 'Copy as path' removed, ~ expanded), or None."""
    s = str(s or "").strip().strip('"').strip("'").strip()
    return str(Path(os.path.expanduser(s))) if s else None


def kl_pip_command(mods=None):
    py = child_python()
    py = '"%s"' % py if " " in py else py
    return "%s -m pip install %s" % (py, " ".join(mods or KL_MODULES.values()))


_kl_ok = []


def helper_cmd(name, args):
    """A helper process: in the packaged app the app itself in its helper mode (app.py --helper: no window; its "Python"
    is the app, so "python script.py" opened another designer window, 2026-10-14), else this Python with the script."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--helper", name] + list(args)
    return [child_python(), "-u", str(KL_SCRIPT)] + list(args)


def kl_missing():
    """The packages the export needs that this Python does not have (checked once they are all there)."""
    if _kl_ok:
        return []
    if getattr(sys, "frozen", False):                # (the app: its own modules, no helper process)
        import importlib.util
        miss = [pkg for m, pkg in KL_MODULES.items() if importlib.util.find_spec(m) is None]
        if not miss:
            _kl_ok.append(True)
        return miss
    try:
        r = subprocess.run([child_python(), "-c", "import importlib,sys\nfor m in sys.argv[1:]:\n try: importlib.import_module(m)\n"
                            " except Exception: print(m)"] + list(KL_MODULES), capture_output=True, text=True, timeout=120,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        miss = [KL_MODULES[m] for m in r.stdout.split() if m in KL_MODULES]
        if r.returncode != 0 and not miss:
            miss = list(KL_MODULES.values())
    except Exception:
        miss = list(KL_MODULES.values())
    if not miss:
        _kl_ok.append(True)
    return miss


def kl_pictures(folder):
    """The pictures directly in `folder` (not its subfolders), without the tool's own check pictures."""
    out = []
    for p in sorted(Path(folder).iterdir(), key=lambda q: q.name.lower()):
        if p.is_file() and p.suffix.lower() in KL_PICS and not p.name.startswith(".") and not p.stem.endswith("_check"):
            try:
                st = p.stat()
            except OSError:
                continue
            out.append(dict(name=p.name, size=st.st_size, mtime=int(st.st_mtime), shown=p.suffix.lower() in KL_SHOWN))
    return out


def kl_reveal(p):
    """Show a file in Finder / Explorer (selected)."""
    p = str(p)
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", p])
    elif sys.platform == "win32":
        subprocess.Popen('explorer /select,"%s"' % p)
    else:
        subprocess.Popen(["xdg-open", str(Path(p).parent)])


def kl_name(body, folder):
    """The export's file names: the device's name (the open design's, sent by the page), else the pictures' folder's
    (before 2026-10-21 always the folder's: every stack's finished pictures are in "final", so each device's export was
    final.lys); characters Windows does not allow in a file name replaced by _."""
    n = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(body.get("name") or "")).strip(" ._")
    return n or Path(folder).name or "chip"


def klayout_running():
    """Is KLayout open on this computer?  (It does not notice a session file changing on disk.)"""
    try:
        if sys.platform == "win32":
            r = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=10,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return "klayout" in r.stdout.lower()
        r = subprocess.run(["ps", "-axo", "comm"], capture_output=True, text=True, timeout=10)
        return any(Path(l.strip()).name.lower() == "klayout" for l in r.stdout.splitlines())
    except Exception:
        return False


def klayout_exe():
    """The KLayout program: the one running, else the usual install places (newest first on the Mac)."""
    cands = []
    try:
        if sys.platform == "darwin":
            r = subprocess.run(["ps", "-axo", "comm"], capture_output=True, text=True, timeout=10)
            cands += [l.strip() for l in r.stdout.splitlines() if Path(l.strip()).name.lower() == "klayout"]
            for app in ("/Applications/klayout.app", "/Applications/KLayout/klayout.app",
                        str(Path.home() / "Applications/klayout.app"), "/Applications/KLayout.app"):
                cands.append(app + "/Contents/MacOS/klayout")
        elif sys.platform == "win32":
            for base in (os.environ.get("APPDATA"), os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
                if base:
                    cands.append(str(Path(base) / "KLayout" / "klayout_app.exe"))
        import shutil
        for n in ("klayout_app", "klayout"):
            w = shutil.which(n)
            if w:
                cands.append(w)
    except Exception:
        pass
    for c in cands:
        if c and Path(c).is_file():
            return c
    return None


def kl_open_klayout(lys):
    """A new KLayout window with the session restored (-u)."""
    exe = klayout_exe()
    if not exe:
        raise OSError("KLayout was not found; open the session in KLayout with File > Restore Session")
    if sys.platform == "darwin" and ".app/Contents/MacOS/" in exe:
        app = exe.split("/Contents/MacOS/")[0]
        subprocess.Popen(["open", "-n", "-a", app, "--args", "-u", str(lys)])
    else:
        subprocess.Popen([exe, "-u", str(lys)], creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))


class KlJob:
    """One run of marks_to_klayout.py: its output lines as it works, then its RESULT."""

    def __init__(self, cmd, info):
        import uuid
        self.id = uuid.uuid4().hex[:12]
        self.cmd, self.info = cmd, info
        self.lines, self.result, self.state, self.rc = [], None, "running", None
        self.started, self.ended, self.proc = time.time(), None, None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         cwd=str(HERE), env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for raw in self.proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("RESULT "):
                    try:
                        self.result = json.loads(line[7:])
                        continue
                    except ValueError:
                        pass
                if len(self.lines) < 5000:
                    self.lines.append(line)
            self.rc = self.proc.wait()
        except Exception as e:                      # (could not start it at all)
            self.lines.append("could not start the export: %s" % e)
            self.rc = -1
        r = self.result if isinstance(self.result, dict) else None
        if r is None:
            text = "\n".join(self.lines[-30:])
            if re.search(r"No module named|needs gdstk|pip3? install", text):
                msg = "This Python is missing packages the export needs. Install them with:  " + kl_pip_command()
            elif self.state == "cancelled":
                msg = "Stopped."
            else:
                tail = [x for x in self.lines if x.strip()][-1:] or ["no output"]
                msg = "The export did not finish (exit code %s): %s" % (self.rc, tail[0])
            r = dict(ok=False, message=msg)
        elif self.rc not in (0, None) and r.get("ok"):
            r = dict(r, ok=False, message=(r.get("message") or "") + " (exit code %s)" % self.rc)
        self.result = r
        self.ended = time.time()
        if self.state != "cancelled":
            self.state = "done" if r.get("ok") else "failed"

    def cancel(self):
        if self.state == "running" and self.proc is not None:
            self.state = "cancelled"
            try:
                self.proc.terminate()
            except OSError:
                pass

    def output(self, what):
        """A file the job wrote (one of KL_OUTPUTS), or None."""
        r = self.result or {}
        if what not in KL_OUTPUTS or not r.get(what):
            return None
        p = Path(r[what])
        return p if p.exists() else None

    def status(self, since=0):
        if self.state != "running" and not hasattr(self, "kl"):
            self.kl = dict(running=klayout_running(), found=bool(klayout_exe()))
        return dict(id=self.id, state=self.state, lines=self.lines[since:], n=len(self.lines), result=self.result,
                    elapsed=round((self.ended or time.time()) - self.started, 1), info=self.info,
                    klayout=getattr(self, "kl", None))


def json_finite(o):
    """o with every NaN / infinite number replaced by None (JSON has no NaN; Python writes it, browsers refuse it)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: json_finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_finite(v) for v in o]
    return o


class AlignJob(KlJob):
    """Aligning a stack's step photos (marks_to_klayout.py --fits-json) in the background: minutes for a dozen pictures
    (DQD11: ~4 min for 13), too long for one request (the Mac app's page got an empty answer, 2026-10-21).  When it
    ends, each photo's placement is put into its step.json (folder stacks) and kept for the page (fits)."""

    def __init__(self, cmd, info, td, todo, sid, gds_name, folders):
        self.td, self.todo, self.sid, self.gds_name, self.folders, self.fits = td, todo, sid, gds_name, folders, None
        super().__init__(cmd, info)

    def _run(self):
        super()._run()
        st, self.state = self.state, "running"           # (not done until the placements are written)
        try:
            out = Path(self.td) / "fits.json"
            fits = json.loads(out.read_text()) if out.is_file() else []
            by_path = {os.path.normcase(os.path.abspath(f["path"])): f for f in fits}
            res, now = {}, int(time.time() * 1000)
            for u, p in self.todo:
                f = by_path.get(os.path.normcase(os.path.abspath(str(p))))
                if f is None:
                    continue
                a = dict(how="marks" if f.get("how") == "marks" else ("matched" if f.get("how") else None), at=now,
                         gds=self.gds_name, w=f.get("w"), h=f.get("h"))
                if f.get("px_to_um"):
                    a.update(m=f["px_to_um"], um_per_px=f["um_per_px"], angle_deg=f["angle_deg"], marks=f["marks"],
                             rms_um=f["rms_um"], warning=f.get("warning"))
                a = json_finite(a)
                res[u] = a
                parts = urllib.parse.unquote(urllib.parse.urlparse(u).path)[len("/stackfile/"):].split("/")
                if SD.is_folder_id(self.sid) and len(parts) == 3:
                    self.folders.set_align(self.sid, parts[1], parts[2], a)
            self.fits = res
            if fits and st == "failed":                  # (it wrote the placements and then failed: they count)
                st = "done"
        except Exception as e:
            self.lines.append("the placements could not be written: %s: %s" % (type(e).__name__, e))
            st = "failed"
        finally:
            shutil.rmtree(self.td, ignore_errors=True)
            self.state = st

    def status(self, since=0):
        d = dict(id=self.id, state=self.state, lines=self.lines[since:], n=len(self.lines), result=self.result,
                 elapsed=round((self.ended or time.time()) - self.started, 1), of=len(self.todo))
        if self.fits is not None and self.state != "running":
            d.update(fits=self.fits, placed=sum(1 for a in self.fits.values() if a.get("m")))
        else:                                               # (progress: the pictures it has reported on so far)
            names = {Path(str(p)).name for _, p in self.todo}
            d["seen"] = len({ln.split()[0] for ln in self.lines if ln.split() and ln.split()[0] in names})
        return d


def kl_start(S, body):
    """Check the request, remember the choices and start the export.  -> (job, None) or (None, error message)."""
    gds, lyp, add_to = kl_path(body.get("gds")), kl_path(body.get("lyp")), kl_path(body.get("add_to"))
    replace = kl_path(body.get("replace"))      # update an existing session's pictures only (no GDS, no new session)
    folder = kl_path(body.get("folder"))
    if not KL_SCRIPT.is_file() and not getattr(sys, "frozen", False):     # (the app: run by its --helper mode)
        return None, "marks_to_klayout.py is not next to the designer (%s)" % HERE
    if not gds or not Path(gds).is_file():
        return None, "Choose the template layout (GDS)%s." % (": %s is not there" % gds if gds else "")
    if lyp and not Path(lyp).is_file():
        return None, "The layer properties file %s is not there." % lyp
    if add_to and (not Path(add_to).is_file() or Path(add_to).suffix.lower() != ".lys"):
        return None, "The session to add to (%s) is not a .lys file that exists." % add_to
    if body.get("replace") is not None and (not replace or not Path(replace).is_file() or Path(replace).suffix.lower() != ".lys"):
        return None, "Choose the session to update (an existing .lys)%s." % (": %s is not there" % replace if replace else "")
    if not folder or not Path(folder).is_dir():
        return None, "Choose the folder with the pictures%s." % (": %s is not reachable" % folder if folder else "")
    names = body.get("pictures")
    if names is None:
        names = [p["name"] for p in kl_pictures(folder)]
    names = [n for n in names if isinstance(n, str) and n == Path(n).name and Path(n).suffix.lower() in KL_PICS]
    pics = [str(Path(folder) / n) for n in names if (Path(folder) / n).is_file()]
    if not pics:
        return None, "No pictures to align (%s)." % ", ".join(KL_PICS)
    miss = kl_missing()
    if miss:
        return None, "This Python is missing %s, which the export needs. Install with:  %s" % (", ".join(miss), kl_pip_command(miss))
    name = kl_name(body, folder)
    if replace:
        cmd = helper_cmd("marks", pics + ["--gds", gds, "--replace-pictures", replace])
        cfg = load_config()
        k = cfg.get("klayout") if isinstance(cfg.get("klayout"), dict) else {}
        k.update(template_gds=gds, last_image_dir=folder, last_replace=replace)
        cfg["klayout"] = k
        save_config(cfg)
        job = KlJob(cmd, dict(folder=folder, pictures=names, gds=gds, replace=replace, out=str(Path(replace).with_suffix(""))))
        S.kl_jobs[job.id] = job
        return job, None
    out_dir = kl_path(body.get("out_dir")) or str(Path(folder) / (name + "_klayout"))
    cmd = helper_cmd("marks", pics + ["--gds", gds, "-o", str(Path(out_dir) / name)])
    if lyp:
        cmd += ["--lyp", lyp]
    if add_to:
        cmd += ["--add-to", add_to]
    design = body.get("design")
    if isinstance(design, dict) and isinstance(design.get("simg"), dict):
        # the stack picture: the original when it is one of the chosen pictures, else the designer's stored copy
        g = design["simg"]
        pic = None
        if g.get("name") and g["name"] in names:
            pic = str(Path(folder) / g["name"])
        else:
            m = re.match(r"^/stackimage/([A-Za-z0-9_-]{1,80}\.(jpg|png))$", str(g.get("url") or ""))
            d = S.store._dir("stack_images")
            if m and d is not None and (d / m.group(1)).is_file():
                pic = str(d / m.group(1))
        if not pic:
            return None, "The design's stack picture was not found (%s); add it to the picture folder or untick the design." % (
                g.get("name") or g.get("url"))
        design["picture"] = pic
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        dj = Path(out_dir) / (name + "_design.json")
        dj.write_text(json.dumps(design), encoding="utf-8")
        cmd += ["--design", str(dj)]
    cfg = load_config()                              # remember the choices (the other settings stay as they are)
    k = cfg.get("klayout") if isinstance(cfg.get("klayout"), dict) else {}
    k.update(template_gds=gds, last_image_dir=folder)
    if lyp:
        k["lyp"] = lyp
    if add_to:
        k["last_lys"] = add_to
    if body.get("remember_out") and body.get("out_dir"):   # (an output folder chosen by hand: the next exports' default,
        k["out_dir"] = out_dir                            # e.g. a local Lithography folder; "Back to the default" forgets it)
    elif body.get("forget_out"):
        k.pop("out_dir", None)
    cfg["klayout"] = k
    save_config(cfg)
    job = KlJob(cmd, dict(folder=folder, pictures=names, gds=gds, lyp=lyp, add_to=add_to, out=str(Path(out_dir) / name)))
    S.kl_jobs[job.id] = job
    return job, None


# ---------------------------------------------------------------- copies of the share's run files on this computer
# A run's bundle.json and pictures are read over the network by every page load (2 MB + ~3 MB per 100x view sheet):
# kept in this computer's cache folder, keyed by the share file's path, size and time (a rebuilt file is a new copy);
# the oldest are removed beyond CACHE_MB (2026-10-16).  FLAKEID_DESIGNER_CACHE=0 turns it off.
def cache_dir():
    env = os.environ.get("FLAKEID_DESIGNER_CACHE")
    if env == "0":
        return None
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / APP
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / APP


class DiskCache:
    def __init__(self, root, cap_mb=None):
        self.root = Path(root) if root else None
        self.cap = int((cap_mb or float(os.environ.get("FLAKEID_DESIGNER_CACHE_MB") or 2048)) * 2 ** 20)
        self.lock = threading.Lock()
        self.used = None                                 # (bytes in the cache: counted at the first write)

    def _name(self, p, st):
        return f"{_sig(str(p))}_{st.st_size}_{st.st_mtime_ns}"

    def get(self, p, st):
        if self.root is None:
            return None
        f = self.root / self._name(p, st)
        try:
            b = f.read_bytes()
        except OSError:
            return None
        if len(b) != st.st_size:
            return None
        try:
            os.utime(f, None)                            # (recently used: removed last)
        except OSError:
            pass
        return b

    def file(self, p):
        """(2026-11-05v) The path of the copy of file p kept here, made now if there is none or p changed; None: no copy."""
        if self.root is None:
            return None
        st = p.stat()
        f = self.root / self._name(p, st)
        try:
            if f.stat().st_size == st.st_size:
                os.utime(f, None)
                return f
        except OSError:
            pass
        self.put(p, st, p.read_bytes())
        return f if f.is_file() else None

    def drop(self, p):
        """(2026-11-05v) Every copy kept here of file p removed (Refresh AFM files): the next read fetches it again."""
        if self.root is None or not self.root.is_dir():
            return
        with self.lock:
            for q in self.root.glob(_sig(str(p)) + "_*"):
                try:
                    sz = q.stat().st_size
                    q.unlink()
                    if self.used is not None:
                        self.used -= sz
                except OSError:
                    pass

    def put(self, p, st, data):
        if self.root is None or len(data) != st.st_size or len(data) > self.cap // 8:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            name = self._name(p, st)
            for old in self.root.glob(name.split("_", 1)[0] + "_*"):     # (older versions of the same file)
                if old.name != name and not old.name.endswith(".part"):
                    try:
                        sz = old.stat().st_size
                        old.unlink()
                        if self.used is not None:
                            self.used -= sz
                    except OSError:
                        pass
            tmp = self.root / f"{name}.{threading.get_ident()}.part"
            tmp.write_bytes(data)
            os.replace(tmp, self.root / name)
        except OSError:
            return
        with self.lock:
            if self.used is None:
                self.used = sum(q.stat().st_size for q in self.root.iterdir() if q.is_file())
            else:
                self.used += len(data)
            if self.used > self.cap:
                files = sorted((q for q in self.root.iterdir() if q.is_file()), key=lambda q: q.stat().st_mtime)
                for q in files:
                    if self.used <= 0.8 * self.cap:
                        break
                    try:
                        sz = q.stat().st_size
                        q.unlink()
                        self.used -= sz
                    except OSError:
                        pass


DISK = DiskCache(None)
# (2026-11-05v) AFM screening's copies, in their own folder and with their own limit: <cache>/afm, oldest removed beyond
# FLAKEID_DESIGNER_AFM_CACHE_MB (20 GB)
AFM_DISK = DiskCache(None, float(os.environ.get("FLAKEID_DESIGNER_AFM_CACHE_MB") or 20480))
AFM_COPYING, AFM_COPY_LOCK = {}, threading.Lock()
# (2026-11-05w) Kept flakes' 100x pictures on this computer: <cache>/kept, oldest removed beyond FLAKEID_DESIGNER_KEPT_CACHE_MB
# (20 GB), in a folder of their own so that browsing other flakes (DISK) does not push them out
KEPT_DISK = DiskCache(None, float(os.environ.get("FLAKEID_DESIGNER_KEPT_CACHE_MB") or 20480))
KEPT_MIN_FREE = 10 * 2 ** 30                     # (bytes left free on this computer: no copies below this)


def kept_files(store, run):
    """The files to keep on this computer for a run's kept flakes (kept.kept_flakes: as the Library's "kept" list): their
    views in the run's designer folder (the bundle's img*), then their JPEG pictures in the chip's viewer (kept.sources)."""
    sig, _, latest = store.review_state()
    if sig == "none":
        raise OSError("the keep / discard decisions cannot be read")
    kept = {(p, n) for (r, p, n) in KP.kept_flakes(latest) if r == run}
    with store.lock:
        ddirs = [v["dir"] for v in store.runs.values() if v["doc"].get("run") == run]
    for ddir in ddirs if kept else []:
        ddir = Path(ddir)
        doc = json.loads((ddir / "bundle.json").read_text(encoding="utf-8"))
        for f in doc.get("flakes") or []:
            m = re.match(r"^.+?:(.+)-f(\d+)$", str(f.get("id") or ""))
            if m and (m.group(1), int(m.group(2))) in kept:
                for k, v in f.items():
                    if k.startswith("img") and isinstance(v, str) and "/" not in v:
                        yield ddir / v
        for piece, n in sorted(kept):
            viewer, H = KP.KEPT.pieces.find(ddir.parent, piece)
            if viewer is not None:
                for _, q in KP.sources(ddir.parent, viewer, H.get(str(n))):
                    if q.suffix.lower() == ".jpg":
                        yield q


class KeptCopy:
    """(2026-11-05w) A run's kept flakes' 100x pictures copied to this computer in the background, one file at a time:
    when the run is opened (flakes others kept too) and when a flake is kept here.  Asked again while running: goes
    through the run once more afterwards."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs, self.again, self.running = [], set(), False

    def want(self, store, run):
        if KEPT_DISK.root is None or not run:
            return
        with self.lock:
            if run in self.jobs:
                self.again.add(run)
                return
            self.jobs.append(run)
            if not self.running:
                self.running = True
                threading.Thread(target=self._work, args=(store,), daemon=True).start()

    def _work(self, store):
        while True:
            with self.lock:
                if not self.jobs:
                    self.running = False
                    return
                run = self.jobs[0]
                self.again.discard(run)
            try:
                for f in kept_files(store, run):
                    if shutil.disk_usage(KEPT_DISK.root.parent).free < KEPT_MIN_FREE:
                        raise OSError("this computer's disk is nearly full")
                    try:
                        KEPT_DISK.file(f)
                    except OSError as e:
                        print(f"Kept local copy of {f.name}: {e}", file=sys.stderr)
            except (OSError, ValueError, KeyError) as e:   # (the share gone, the disk full: asked again at the next opening)
                print(f"Kept local copies of {run} stopped: {e}", file=sys.stderr)
            with self.lock:
                self.jobs.remove(run)
                if run in self.again:
                    self.jobs.append(run)


KEPT_COPY = KeptCopy()


def afm_drop_list(pl, pf, folder_files=()):
    """(2026-11-05v, Refresh AFM files) The copies kept here of a list's files dropped, so they come again from the
    share: its results' .spm, pictures and height arrays, and the .spm files now in its AFM folder."""
    d = Path(pf).parent
    for r in list((pl.get("results") or {}).values()) + [r for m in (pl.get("more") or {}).values() for r in m]:
        if r.get("source"):
            AFM_DISK.drop(Path(r["source"]))
        if r.get("png"):
            stem = Path(r["png"]).stem
            for name in (r["png"], stem + ".npz", stem + ".full2.npz"):
                AFM_DISK.drop(d / name)
    for f in folder_files:
        AFM_DISK.drop(Path(f))


def afm_copy_list(store, AP, pid):
    """(2026-11-05v) Every AFM result of a list copied to this computer, in the background, one at a time, so that
    screening it does not wait for the share.  Asked again while running: goes through the list once more afterwards."""
    with AFM_COPY_LOCK:
        if pid in AFM_COPYING:
            AFM_COPYING[pid] = True
            return
        AFM_COPYING[pid] = False

    def run():
        while True:
            pl, pf = store.afm_plan(pid)
            results = list((pl or {}).get("results", {}).values()) + [r for m in (pl or {}).get("more", {}).values() for r in m]
            for r in results:
                try:
                    AP.keep_local(pf.parent, r)
                except Exception as e:                   # (a file no longer there, a broken file: the others still)
                    print(f"AFM local copy of {r.get('file')}: {e}", file=sys.stderr)
            with AFM_COPY_LOCK:
                if not AFM_COPYING[pid]:
                    del AFM_COPYING[pid]
                    return
                AFM_COPYING[pid] = False

    threading.Thread(target=run, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = f"{APP}/{VERSION}"

    def log_message(self, fmt, *a):       # only errors (4xx/5xx) go to the log
        if a and str(a[1] if len(a) > 1 else "").startswith(("4", "5")) and "favicon" not in str(a[0]):
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {fmt % a}\n")

    # ---- helpers
    def _send(self, code, body=b"", ctype="application/json", extra=None):
        extra = dict(extra or {})
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(extra.pop("_length", len(body))))
        cc = extra.pop("_cache", None) or ("no-store" if ctype == "application/json" or ctype.startswith("text/html") or extra.pop("_nocache", None)
                                           else "max-age=3600")
        extra.pop("_nocache", None)
        self.send_header("Cache-Control", cc)
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and code != 304:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        try:
            data = json.dumps(obj, allow_nan=False)
        except ValueError:                               # (NaN / Infinity, e.g. in a file written by Python: not JSON for
            data = json.dumps(json_finite(obj))          # a browser, DQD11's step.json 2026-10-21: the stack did not open)
        self._send(code, data.encode("utf-8"))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}

    def _local(self):
        return self.client_address[0] in ("127.0.0.1", "::1")

    def _etag_ok(self, etag):
        """The page already has this version (If-None-Match): 304."""
        inm = self.headers.get("If-None-Match")
        return bool(inm) and etag in [t.strip() for t in inm.split(",")]

    def _list(self, sig, items):
        """A collection with its signature as ETag: 304 when the page has it already (the page's 20 s polls)."""
        etag = f'"{sig}"'
        if self._etag_ok(etag):
            return self._send(304, b"", "application/json", {"ETag": etag, "_cache": "no-cache"})
        return self._send(200, json.dumps(items).encode("utf-8"), "application/json", {"ETag": etag, "_cache": "no-cache"})

    def _file(self, p, extra=None, cached=False, alt=None):
        """A file with ETag / Last-Modified: 304 when the browser has it (only a stat of the file), HEAD without reading
        it; cached: a copy kept on this computer (DISK) serves it when the share's file is unchanged (2026-10-16).
        alt: where else it may be when not there (a stack's file added offline and not uploaded yet: offline.py)."""
        try:
            if p is None:
                raise OSError("no file")
            st = p.stat()
            if not stat.S_ISREG(st.st_mode):
                raise OSError("not a file")
        except OSError:
            q = None
            try:
                q = alt() if alt is not None else None
            except Exception:
                q = None
            if q is None:
                return self._send(404, b"not found", "text/plain")
            p, cached = q, False
            try:
                st = p.stat()
            except OSError:
                return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix.lower() == ".webp":
            ctype = "image/webp"
        etag = f'"{st.st_size:x}-{st.st_mtime_ns:x}"'
        hd = dict(extra or {}, ETag=etag, **{"Last-Modified": email.utils.formatdate(st.st_mtime, usegmt=True)})
        if HASHED.search(p.name):
            hd["_cache"] = "max-age=604800, immutable"   # (named by its content: never changes)
        elif ctype == "application/json" or ctype.startswith("text/"):
            hd["_cache"] = "no-cache"                   # (bundle.json: asked again each load, answered 304 if unchanged)
        # (Chromium sends only If-Modified-Since to a HTTP/1.0 server; equal, not later: a file put back from a backup
        # is older)
        ims = None if self.headers.get("If-None-Match") else self.headers.get("If-Modified-Since")
        try:
            ims = ims and int(email.utils.parsedate_to_datetime(ims).timestamp()) == int(st.st_mtime)
        except (TypeError, ValueError, IndexError, OverflowError):
            ims = None
        if self._etag_ok(etag) or ims:
            return self._send(304, b"", ctype, hd)
        if self.command == "HEAD":
            return self._send(200, b"", ctype, dict(hd, _length=st.st_size))
        store = cached if isinstance(cached, DiskCache) else DISK     # (cached: True, or which copies: AFM_DISK)
        data = (KEPT_DISK.get(p, st) if cached is True else None) or (store.get(p, st) if cached else None)
        if data is None:
            data = p.read_bytes()
            if cached:
                store.put(p, st, data)
        self._send(200, data, ctype, hd)

    # ---- export to KLayout
    def _stackfile_path(self, u):
        """The local file of a /stackfile/<stack>/... URL (a folder stack's file, or the designer's stack files), or None."""
        parts = urllib.parse.unquote(urllib.parse.urlparse(str(u)).path)
        if not parts.startswith("/stackfile/"):
            return None
        parts = parts[len("/stackfile/"):].split("/")
        if not parts or any(q in ("", ".", "..") or "\\" in q for q in parts):
            return None
        if SD.is_folder_id(parts[0]):
            return self.server.store.folders.file(parts[0], parts[1:])
        d = self.server.store._dir("stack_files")
        p = d.joinpath(*parts) if d is not None and all(re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", q) for q in parts) else None
        return p if p is not None and p.is_file() else None

    def _klayout_get(self, path, q):
        S = self.server
        if not self._local():
            return self._json(dict(error="only from this computer"), 403)
        arg = lambda k: (q.get(k) or [""])[0]
        if path == "/api/klayout/config":
            cfg = load_config()
            run = [j.id for j in S.kl_jobs.values() if j.state == "running"]
            miss = kl_missing() if arg("check") else None
            return self._json(dict(klayout=cfg.get("klayout") if isinstance(cfg.get("klayout"), dict) else {},
                                   platform=sys.platform, python=child_python(), script=KL_SCRIPT.is_file() or bool(getattr(sys, "frozen", False)),
                                   missing=miss, pip=kl_pip_command(miss) if miss else None, running=run[0] if run else None))
        if path == "/api/klayout/job":
            job = S.kl_jobs.get(arg("id"))
            if job is None:
                return self._json(dict(error="no such export"), 404)
            try:
                since = max(0, int(arg("since") or 0))
            except ValueError:
                since = 0
            return self._json(job.status(since))
        if path == "/api/klayout/thumb":
            # only pictures directly in a folder listed for the export in this session
            folder, name = kl_path(arg("folder")), arg("name")
            p = Path(folder) / name if folder in S.kl_folders and name == Path(name).name else None
            ok = p is not None and p.suffix.lower() in KL_SHOWN and p.is_file()
            return self._file(p) if ok else self._send(404, b"not found", "text/plain")
        if path == "/api/klayout/out":
            job = S.kl_jobs.get(arg("job"))
            p = job.output(arg("what")) if job is not None and arg("what") == "check_jpg" else None
            return self._file(p) if p is not None and p.is_file() else self._send(404, b"not found", "text/plain")
        return self._json(dict(error="unknown"), 404)

    def _klayout_post(self, path, body):
        S = self.server
        if path == "/api/klayout/pick":
            kind = body.get("kind")
            start = kl_path(body.get("start"))
            if kind in ("folder", "outdir"):
                prompt = "Folder with the pictures of the chip" if kind == "folder" else "Folder for the KLayout files"
                return self._json(dict(picked=pick_folder(prompt, start)))
            if kind not in KL_PICKS:
                return self._json(dict(error="unknown kind"), 400)
            prompt, exts = KL_PICKS[kind]
            return self._json(dict(picked=pick_file(prompt, exts, start)))
        if path == "/api/klayout/pictures":
            folder = kl_path(body.get("folder"))
            if not folder or not Path(folder).is_dir():
                return self._json(dict(error="%s is not a folder that can be reached" % folder if folder else "no folder"), 400)
            try:
                pics = kl_pictures(folder)
            except OSError as e:
                return self._json(dict(error=str(e)), 400)
            S.kl_folders.add(folder)
            name = kl_name(body, folder)
            # sessions this device's export would have written (for "Update pictures"): the remembered output folder's,
            # the default one's
            k = load_config().get("klayout") or {}
            cands = [Path(d) / (name + ".lys") for d in (k.get("out_dir"), str(Path(folder) / (name + "_klayout"))) if d]
            sessions = [str(c) for c in cands if c.is_file()]
            return self._json(dict(folder=folder, pictures=pics, default_out=str(Path(folder) / (name + "_klayout")), name=name,
                                   sessions=sessions))
        if path == "/api/klayout/export":
            if any(j.state == "running" for j in S.kl_jobs.values()):
                return self._json(dict(error="An export is still running; wait for it or stop it."), 409)
            job, err = kl_start(S, body)
            if err:
                return self._json(dict(error=err), 400)
            return self._json(dict(job=job.id, cmd=job.cmd))
        job = S.kl_jobs.get(str(body.get("job") or ""))
        if job is None:
            return self._json(dict(error="no such export"), 404)
        if path == "/api/klayout/cancel":
            job.cancel()
            return self._json(dict(ok=True))
        if path == "/api/klayout/openkl":
            p = job.output(body.get("what") or "lys")
            if p is None or not str(p).lower().endswith(".lys"):
                return self._json(dict(error="that session is not there"), 404)
            try:
                kl_open_klayout(p)
            except OSError as e:
                return self._json(dict(error=str(e)), 500)
            return self._json(dict(ok=True))
        if path == "/api/klayout/reveal":
            p = job.output(body.get("what") or "lys")
            if p is None:
                return self._json(dict(error="that file is not there"), 404)
            try:
                kl_reveal(p)
            except OSError as e:
                return self._json(dict(error=str(e)), 500)
            return self._json(dict(ok=True))
        return self._json(dict(error="unknown"), 404)

    # ---- offline (offline.py): the share not reachable, the stacks kept on this computer
    def _offline_get(self, path):
        """What the page asks for, answered from the packs on this computer.  -> True if answered."""
        O = self.server.offline
        lists = {"/api/db/runs": O.runs_listing, "/api/db/stacks": O.stacks_listing, "/api/db/review": O.review_listing,
                 "/api/db/seen": O.seen_listing, "/api/db/terraces": lambda: O.snap_listing("terraces"),
                 "/api/db/manual": lambda: O.snap_listing("manual")}
        if path in lists:
            self._list(*lists[path]())
            return True
        if path.startswith("/api/db/"):
            self._list("offline-none", [])
            return True
        if path.startswith("/_blob/"):
            parts = path[len("/_blob/"):].split("/")
            p = O.run_file(parts[0], parts[1]) if len(parts) == 2 else None
            if p is None:
                self._send(404, b"not on this computer (offline)", "text/plain")
            else:
                self._file(p)
            return True
        if path.startswith(("/stackimage/", "/stackfile/")):
            p = O.file_for(path)
            if p is None:
                self._send(404, b"not on this computer (offline)", "text/plain")
                return True
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            disp = "inline" if ctype.startswith(("image/", "text/")) or ctype == "application/pdf" else "attachment"
            self._file(p, {"Content-Disposition": f'{disp}; filename="{urllib.parse.quote(Path(path).name)}"'})
            return True
        simple = {"/api/stackdir/recent": [], "/api/stacks/legacy": dict(designs=0, migrated=True, offline=True),
                  "/api/devicephotos": dict(roots=[], sets=[], offline=True), "/api/afm/plans": [], "/api/afm/lists": [],
                  "/api/stackfiles/dir": dict(error="offline: the stack's folders are on the drive")}
        if path in simple:
            self._json(simple[path])
            return True
        if path.startswith("/afmfile/"):
            self._send(404, b"not on this computer (offline)", "text/plain")
            return True
        return False

    def _offline_post(self, path):
        """Offline: pictures and files added to a stack kept here (under the URL they will have on the share), the
        lock note (nothing to note), and a clear answer for what needs the share.  -> True if answered."""
        S = self.server
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        import uuid
        if path == "/api/stackimage":
            n = int(self.headers.get("Content-Length") or 0)
            ctype = (self.headers.get("Content-Type") or "").lower()
            if not 0 < n <= 40 * 1024 * 1024 or ctype not in ("image/jpeg", "image/png"):
                self._json(dict(error="a JPEG or PNG picture up to 40 MB"), 400)
                return True
            ext = "png" if ctype == "image/png" else "jpg"
            sid = (q.get("stack") or [""])[0]
            if SD.is_folder_id(sid):
                url = SD.url(sid, f"01_stacking/{stamp}_{uuid.uuid4().hex[:6]}_picture.{ext}")
            else:
                url = f"/stackimage/{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:12]}.{ext}"
            try:
                S.offline.store_new(url, self.rfile.read(n))
            except OSError as e:
                self._json(dict(error=f"not kept on this computer: {e}"), 500)
                return True
            self._json(dict(url=url, offline=True))
            return True
        if path == "/api/stackfile":
            if self.headers.get("X-Designer") != "1":
                self._json(dict(error="designer page only"), 403)
                return True
            sid, name = (q.get("stack") or [""])[0], (q.get("name") or ["file"])[0]
            n = int(self.headers.get("Content-Length") or 0)
            if not re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", sid) or not 0 < n <= 500 * 1024 * 1024:
                self._json(dict(error="a stack id and a file up to 500 MB"), 400)
                return True
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name).strip("._")[-120:] or "file"
            fin = (q.get("sub") or [""])[0] == "final"
            fn = safe if fin else f"{stamp}_{safe}"
            if SD.is_folder_id(sid):
                url = SD.url(sid, ("final/" if fin else "01_stacking/") + fn)
            else:
                url = f"/stackfile/{sid}/" + ("final/" if fin else "") + fn
            try:
                p = S.offline.store_new(url, self.rfile.read(n))
            except OSError as e:
                self._json(dict(error=f"not kept on this computer: {e}"), 500)
                return True
            self._json(dict(url=url, name=fn if fin else Path(name).name, bytes=n, path=str(p), offline=True))
            return True
        if path == "/api/stacklock":
            self._json(dict(ok=True, other=None, offline=True))
            return True
        if path.startswith(("/api/camera/", "/api/klayout/", "/api/station", "/api/buildhere")):
            return False                                 # (this computer only)
        self._json(dict(error="offline: this needs the drive (it is not reachable); the stacks kept on this computer "
                              "can be edited, and their changes are uploaded when it is back"), 503)
        return True

    def _offline_api(self, path):
        """/api/offline/...: the packs' state and settings (on or offline)."""
        O = self.server.offline
        try:
            b = self._body()
        except ValueError:
            b = {}
        if path == "/api/offline/settings":
            if "auto" in b:
                O.set_auto(bool(b.get("auto")))
        elif path == "/api/offline/keep":
            sid = str(b.get("id") or "")
            if not re.match(r"^[A-Za-z0-9_.~:@+-]{1,200}$", sid):
                return self._json(dict(error="no such stack"), 400)
            k = b.get("keep")
            O.set_keep(sid, None if k is None else bool(k))
        elif path == "/api/offline/sync":
            O.kick(sync=True, pack=True)
        elif path == "/api/offline/refresh":
            O.kick(pack=True)
        elif path == "/api/offline/dismiss":
            O.dismiss(str(b.get("id") or ""))
        else:
            return self._json(dict(error="unknown"), 404)
        return self._json(O.status())

    # ---- routes
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        S = self.server
        S.last = time.time()
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path in ("/", "/index.html"):
            html = (HERE / "index.html").read_text(encoding="utf-8")
            cfg = dict(user=user_name(), roots=S.store.roots, data_dir=str(S.store.data) if S.store.data else None, version=VERSION,
                       added_runs=S.store.added,
                       offline=S.offline.brief(), **library_info(load_config(), S.store))
            inj = f'<link rel="icon" href="/static/designer_64.png"><script>window.FLAKEID_LOCAL={json.dumps(cfg)};</script>'
            mb = re.search(r"<body[^>]*>", html)            # (the body tag may carry attributes: data-view)
            html = html[:mb.end()] + inj + html[mb.end():] if mb else inj + html
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/camera/status":
            st = CAM.LIVE.status() if CAM is not None else dict(available=False, sources=[], running=False)
            return self._json(dict(st, station=load_config().get("station") or {}))
        if path == "/api/camera/calibration.jpg":
            if not CAL.get("check"):
                return self._send(204, b"", "text/plain", {"_nocache": 1})
            return self._send(200, CAL["check"], "image/jpeg", {"_nocache": 1})
        if path == "/api/camera/frame.jpg":
            if CAM is None:
                return self._send(404, b"no camera", "text/plain")
            jpg, seq = CAM.LIVE.frame()
            if jpg is None:
                return self._send(204, b"", "text/plain", {"_nocache": 1})
            return self._send(200, jpg, "image/jpeg", {"_nocache": 1, "X-Frame-Seq": str(seq)})
        if path in ("/api/update/check", "/api/update/status"):
            # the header's Update…: what is new on GitHub; an install's progress (update.py)
            if not self._local():
                return self._json(dict(error="only from this computer"), 403)
            if path == "/api/update/status":
                with UPDATE["lock"]:
                    return self._json({k: UPDATE[k] for k in ("state", "log", "result", "error")})
            try:
                return self._json(dict(UPD.check(), version=VERSION, app=NATIVE_PICK is not None))
            except UPD.UpdateError as e:
                return self._json(dict(error=str(e), need_token=not UPD.token()[0] and "refused" in str(e), repo=UPD.REPO, version=VERSION,
                                       installed=UPD.installed()))
        if path == "/api/ping":
            return self._json(dict(app=APP, version=VERSION))
        if path == "/api/config":
            return self._json(dict(user=user_name(), roots=S.store.roots, data_dir=str(S.store.data) if S.store.data else None,
                                   scanned=S.store.scanned, **library_info(load_config(), S.store)))
        if path == "/api/offline":
            return self._json(S.offline.status())
        if S.offline.offline and self._offline_get(path):
            return
        if path == "/api/db/runs":                      # (the last walk of the share: Store.start_scanner, /api/rescan)
            return self._list(*S.store.runs_list())
        if path == "/api/builds":
            return self._json(BUILDER.status())
        if path == "/api/buildhere":
            return self._json(dict(on=build_here()))
        if path == "/api/coords":
            # a run's saved coordinate sets (Library › Restore coordinates)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                return self._json(dict(points=coord_points(S.store, (q.get("run") or [""])[0])))
            except OSError as e:
                return self._json(dict(error=str(e), points=[]), 503)
        if path == "/api/tenx":
            # a flake's 10x picture (100x window, key 3): the chip viewer's pyramid and the flake's place in it
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            g = lambda k: (q.get(k) or [""])[0]         # noqa: E731
            try:
                return self._json(tenx_flake(S.store, g("run"), g("scan"), g("flake")))
            except (KeyError, ValueError, FileNotFoundError) as e:
                return self._json(dict(error=str(e).strip("'\"")), 404)
            except OSError as e:
                return self._json(dict(error=f"the run's folder could not be read ({e})"), 503)
        if path == "/api/sg":
            # (2026-11-03m) a flake's black-and-white green 100x picture (100x window, key 2): the scanner's _sg.jpg
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            g = lambda k: (q.get(k) or [""])[0]         # noqa: E731
            try:
                sg = sg_picture(S.store, g("run"), g("scan"), int(g("flake") or -1), g("img"))
            except (OSError, ValueError, KeyError) as e:
                return self._send(404, f"no black-and-white green picture ({e})".encode(), "text/plain")
            return self._file(sg, cached=True) if sg is not None else self._send(404, b"no black-and-white green picture", "text/plain")
        if path in ("/api/framefiles", "/api/framefiles/img"):
            # (2026-11-05g) the 100x window's "Copy file path or image": the frame's files (frame_files), and one of them
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            g = lambda k: (q.get(k) or [""])[0]         # noqa: E731
            try:
                L = frame_files(S.store, g("run"), g("scan"), int(g("flake") or -1), g("img"))
            except (OSError, ValueError, KeyError) as e:
                return self._json(dict(error=f"the 100x pictures could not be read ({e})"), 404)
            if path.endswith("/img"):
                x = next((d for d in L if d["k"] == g("k")), None)
                return self._file(Path(x["path"]), cached=True) if x else self._send(404, b"no such picture", "text/plain")
            return self._json(dict(files=L))
        if path in ("/api/thick100", "/api/thick100/img"):
            # (2026-11-04i) the 100x window's Thickness: the frame's substrate colour and the run's calibrated model
            # (thick_frame), and its linear picture
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            g = lambda k: (q.get(k) or [""])[0]         # noqa: E731
            try:
                if path.endswith("/img"):
                    f = thick_picture(S.store, g("run"), g("scan"), int(g("flake") or -1), g("img"), g("which"))
                    return self._file(f, cached=True) if f is not None else self._send(404, b"no 100x picture", "text/plain")
                d = thick_frame(S.store, g("run"), g("scan"), int(g("flake") or -1), g("img"), g("material"))
            except Exception as e:                      # (2026-11-05y: any failure is said, the connection is not just dropped)
                return self._json(dict(error=f"the 100x picture could not be read ({e})"), 404)
            return self._json(d if d is not None else dict(error="this flake's 100x picture was not found in its run's folder"), 200 if d else 404)
        if path.startswith("/viewer/"):
            # (2026-11-03c) /viewer/<run>/<piece>/[file]: the chip's own viewer page (index.html: the 10x scan and the
            # table of all its flakes) and the files it loads (tiles, 100x pictures, ...), from the run's folder; only
            # files inside that piece folder
            parts = path[len("/viewer/"):].split("/", 2)
            p = _tenx_piece(S.store, parts[0], parts[1]) if len(parts) >= 2 and parts[0] else None
            if p is None:
                return self._send(404, b"no viewer page of this chip in the run's folder", "text/plain")
            if len(parts) == 2:                         # (relative links of the page need the trailing /)
                return self._send(301, b"", "text/plain", {"Location": urllib.parse.quote(f"/viewer/{parts[0]}/{parts[1]}/")})
            base = p.resolve()
            f = (p / (parts[2] or "index.html")).resolve()
            if f != base and base not in f.parents:
                return self._send(403, b"outside the chip's folder", "text/plain")
            return self._file(f)
        if path.startswith("/tenx/"):
            # /tenx/<run>/<piece>/<look|image>/<level>/<x>_<y>.jpg: a tile of that pyramid
            parts = path[len("/tenx/"):].split("/")
            p = tenx_tile(S.store, *parts) if len(parts) == 5 else None
            return self._file(p) if p is not None else self._send(404, b"not found", "text/plain")
        if path == "/api/afmcenters":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                return self._json(dict(centers=KP.read_centers(S.store, (q.get("run") or [""])[0])))
            except Exception as e:
                return self._json(dict(error=str(e), centers={}), 500)
        if path == "/api/db/designs":
            return self._list(*S.store.collection("designs"))
        if path == "/api/db/stacks":
            return self._list(*S.store.stacks_listing())
        if path == "/api/db/manual":
            return self._list(*S.store.collection("manual"))
        if path == "/api/afm/lasttip":
            return self._json(load_config().get("afm_tip") or {})
        if path == "/api/afm/plans":
            q_ = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return self._json(S.store.afm_plans((q_.get("scan") or [None])[0], drafts=(q_.get("drafts") or [""])[0] == "1"))
        if path == "/api/afm/typed":
            q_ = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return self._json(afm_typed(S.store, (q_.get("key") or [None])[0]))
        if path == "/api/afm/lists":
            # flake lists to measure (e.g. a contrast calibration set): <data>/afm/lists/*.json {name, scan, flakes: [ids], note}
            sc = (urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("scan") or [None])[0]
            d, out = S.store._dir("afm"), []
            for f in sorted((d / "lists").glob("*.json")) if d is not None and (d / "lists").is_dir() else []:
                try:
                    j = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(j, dict) and (sc is None or j.get("scan") in (sc, sc.split(":")[-1])):
                    out.append(dict(j, file=f.name))
            return self._json(out)
        if path.startswith("/afmfile/"):
            parts = path[len("/afmfile/"):].split("/")
            ok = len(parts) == 2 and all(re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", q) and q not in (".", "..") for q in parts)
            hit = next((d / parts[0] / parts[1] for d in (S.store._data_dirs("afm") if ok else [])
                        if (d / parts[0] / parts[1]).is_file()), None)      # (a related library's list too, 2026-11-03g)
            return self._file(hit, cached=AFM_DISK) if hit is not None else self._send(404, b"not found", "text/plain")
        if path == "/api/devicephotos":
            return self._json(S.store.list_device_photos())
        if path == "/api/stacks/legacy":
            return self._json(S.store.legacy_info())
        if path == "/api/stackdir/recent":
            return self._json(S.store.folders.recents())
        if path == "/api/stackfiles/dir":
            # the folder of a stack's final pictures (originals, for the KLayout export)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            sid = (q.get("stack") or [""])[0]
            sub = (q.get("sub") or ["final"])[0]          # (or a fabrication step's folder, e.g. 03_slide)
            if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_. ~@+-]{0,120}$", sub) or ".." in sub:
                return self._json(dict(error="bad folder"), 400)
            if SD.is_folder_id(sid):
                fd = S.store.folders.dir_of(sid, sub)
                return self._json(dict(path=str(fd), klayout=str(fd.parent / "klayout")) if fd else dict(error="no such stack"))
            d = S.store._dir("stack_files")
            if d is None or not re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", sid):
                return self._json(dict(error="no such stack"), 400)
            return self._json(dict(path=str(d / sid / "final")))
        if path == "/api/db/review":
            return self._list(*S.store.review_state()[:2])
        if path == "/api/db/seen":
            return self._list(*S.store.seen_state())
        if path == "/api/db/archived":
            return self._list(*S.store.archived_state())
        if path == "/api/db/terraces":
            return self._list(*S.store.collection("terraces"))
        if path.startswith("/api/klayout/"):
            return self._klayout_get(path, urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query))
        if path == "/api/fabalign/job":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            job = (S.__dict__.get("align_jobs") or {}).get((q.get("id") or [""])[0])
            if job is None:
                return self._json(dict(error="no such alignment"), 404)
            try:
                since = max(0, int((q.get("since") or ["0"])[0]))
            except ValueError:
                since = 0
            return self._json(job.status(since))
        if path.startswith("/api/db/"):
            return self._json([])
        if path.startswith("/_blob/"):
            parts = path[len("/_blob/"):].split("/")
            p = S.store.run_file(parts[0], parts[1]) if len(parts) == 2 else None
            return self._file(p, cached=True) if p else self._send(404, b"not found", "text/plain")
        if path.startswith("/stackfile/"):
            # a stack step's attachments (AFM data, pictures from other microscopes, ...)
            parts = path[len("/stackfile/"):].split("/")
            if SD.is_folder_id(parts[0]):               # (a folder stack: any file in its folder)
                p = S.store.folders.file(parts[0], parts[1:]) or S.offline.file_for(path)
                if p is None:
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                disp = "inline" if ctype.startswith(("image/", "text/")) or ctype == "application/pdf" else "attachment"
                return self._file(p, {"Content-Disposition": f'{disp}; filename="{urllib.parse.quote(p.name)}"'})
            d = S.store._dir("stack_files")
            ok = (len(parts) == 2 or (len(parts) == 3 and parts[1] == "final") or (len(parts) == 4 and parts[1] == "fab")) and \
                all(re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", q) and q not in (".", "..") for q in parts)
            p = d.joinpath(*parts) if d is not None and ok else None
            if p is None:
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            disp = "inline" if ctype.startswith(("image/", "text/")) or ctype == "application/pdf" else "attachment"
            return self._file(p, {"Content-Disposition": f'{disp}; filename="{parts[-1]}"'}, alt=lambda: S.offline.file_for(path))
        if path.startswith("/stackimage/"):
            # pictures of assembled stacks (as-built designs), in the shared data folder
            name = Path(path).name
            d = S.store._dir("stack_images")
            p = d / name if d is not None and re.match(r"^[A-Za-z0-9_-]{1,80}\.(jpg|png)$", name) else None
            # (a picture added offline and not uploaded yet: from this computer)
            return self._file(p, cached=True, alt=lambda: S.offline.file_for(path))
        if path.startswith("/static/"):
            p = (HERE / "static" / Path(path).name)
            return self._file(p) if p.is_file() else self._send(404, b"not found", "text/plain")
        self._send(404, b"not found", "text/plain")

    def do_PUT(self):
        S = self.server
        S.last = time.time()
        if not self._local():
            return self._json(dict(error="only from this computer"), 403)
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        m = re.match(r"^/api/db/(designs|review|seen|archived|terraces|stacks|manual)/([A-Za-z0-9_.~:@+-]{1,200})$", path)
        if not m:
            return self._json(dict(error="unknown"), 404)
        if S.offline.offline:                            # (the share not reachable: offline.py)
            try:
                body = self._body()
                if m.group(1) == "stacks":
                    return self._json(dict(S.offline.put_local(m.group(2), body), ok=True, offline=True))
                if m.group(1) == "review":
                    S.offline.queue_review(m.group(2), body)
                    return self._json(dict(ok=True, offline=True, queued=True))
                return self._json(dict(error="offline: not saved (this is saved only on the drive, which is not reachable)"), 503)
            except StackConflict as e:
                return self._json(dict(conflict=True, **e.info), 409)
            except (OSError, ValueError) as e:
                return self._json(dict(error=str(e)), 503)
        try:
            body = self._body()
            r = dict(designs=S.store.put_design, review=S.store.put_review, seen=S.store.put_seen,
                     archived=S.store.put_archived,
                     terraces=S.store.put_terrace, stacks=S.store.put_stack, manual=S.store.put_manual)[m.group(1)](m.group(2), body)
            if m.group(1) == "stacks":
                try:
                    S.offline.after_online_save(m.group(2), body)
                except Exception as e:                   # (never fails the save)
                    sys.stderr.write(f"{time.strftime('%H:%M:%S')} offline copy of {m.group(2)} not updated: {e}\n")
            return self._json(dict(r or {}, ok=True))
        except (StackConflict, SD.Conflict) as e:
            return self._json(dict(conflict=True, **e.info), 409)
        except (OSError, ValueError) as e:
            return self._json(dict(error=str(e)), 500)

    def do_DELETE(self):
        S = self.server
        S.last = time.time()
        if not self._local():
            return self._json(dict(error="only from this computer"), 403)
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        m = re.match(r"^/api/db/(designs|terraces|stacks|manual)/([A-Za-z0-9_.~:@+-]{1,200})$", path)
        if not m:
            return self._json(dict(error="unknown"), 404)
        if S.offline.offline:
            return self._json(dict(error="offline: nothing is removed until the drive is back"), 503)
        dict(designs=S.store.delete_design, terraces=S.store.delete_terrace, stacks=S.store.delete_stack,
             manual=S.store.delete_manual)[m.group(1)](m.group(2))
        return self._json(dict(ok=True))

    def do_POST(self):
        S = self.server
        S.last = time.time()
        if not self._local():
            return self._json(dict(error="only from this computer"), 403)
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/offline/"):
            return self._offline_api(path)
        if S.offline.offline and self._offline_post(path):
            return
        if path == "/api/stacks/migrate":
            # the old design files -> stack files, after a backup of the whole data folder checked file by file
            if S.store.data is None:
                return self._json(dict(error="no shared data folder"), 400)
            try:
                try:
                    from . import stacks as SK
                except ImportError:
                    import stacks as SK                  # type: ignore   (run as a folder: python flakeid_designer)
                lines = []
                bk, stacks, written = SK.migrate(S.store.data, log=lines.append)
                sys.stderr.write("".join(f"{time.strftime('%H:%M:%S')} {ln}\n" for ln in lines))
                return self._json(dict(ok=True, backup=str(bk), stacks=len(stacks), written=len(written), log=lines))
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e} (the old design files were not changed)"), 500)
        if path == "/api/stackfile":
            # an attachment for a stack step: any file up to 500 MB, kept in <data>/stack_files/<stack>/ (the header:
            # only the designer's page sends it, so another web page cannot store files here)
            if self.headers.get("X-Designer") != "1":
                return self._json(dict(error="designer page only"), 403)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            sid, name = (q.get("stack") or [""])[0], (q.get("name") or ["file"])[0]
            n = int(self.headers.get("Content-Length") or 0)
            if not re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", sid) or not 0 < n <= 500 * 1024 * 1024:
                return self._json(dict(error="a stack id and a file up to 500 MB"), 400)
            if SD.is_folder_id(sid):                    # (a folder stack: 01_stacking/, the originals in final/)
                fin = (q.get("sub") or [""])[0] == "final"
                try:
                    u, fn, fp = S.store.folders.store(sid, self.rfile.read(n), name, "final" if fin else "01_stacking", stamp=not fin)
                except OSError as e:
                    return self._json(dict(error=str(e)), 500)
                return self._json(dict(url=u, name=fn if fin else Path(name).name, bytes=n, path=str(fp)))
            d = S.store._dir("stack_files")
            if d is None:
                return self._json(dict(error="no shared data folder"), 500)
            safe = re.sub(r"[^A-Za-z0-9_.~@+-]+", "_", Path(name).name).strip("._") or "file"
            if (q.get("sub") or [""])[0] == "final":
                # a final picture's original: its own name (the KLayout export lists the folder), never overwritten
                fd = d / sid / "final"
                fd.mkdir(parents=True, exist_ok=True)
                stem, ext = os.path.splitext(safe[-120:])
                fn, k = stem + ext, 2
                while (fd / fn).exists():
                    fn, k = f"{stem}_{k}{ext}", k + 1
                (fd / fn).write_bytes(self.rfile.read(n))
                return self._json(dict(url=f"/stackfile/{sid}/final/{fn}", name=fn, bytes=n, path=str(fd / fn)))
            fn = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe[-120:]}"
            (d / sid).mkdir(parents=True, exist_ok=True)
            (d / sid / fn).write_bytes(self.rfile.read(n))
            return self._json(dict(url=f"/stackfile/{sid}/{fn}", name=Path(name).name, bytes=n))
        if path.startswith("/api/afm/"):
            if self.headers.get("X-Designer") != "1":
                return self._json(dict(error="designer page only"), 403)
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
                if path == "/api/afm/typed":             # (2026-11-03q) reference X / Y / Z as typed
                    r = afm_typed(S.store, b.get("key"), b.get("data"), True)
                    return self._json(r, 400 if r.get("error") else 200)
                builder_main()                           # (numpy, flakeid on the path)
                try:
                    from . import afmplan as AP
                except ImportError:
                    import afmplan as AP                 # type: ignore
                AP.LOCAL = AFM_DISK.file
                d = S.store._dir("afm")
                if d is None:
                    return self._json(dict(error="no shared data folder"), 500)
                if path == "/api/afm/plan":
                    old, had = S.store.afm_plan(b.get("id")) if b.get("id") else (None, None)
                    if old is not None and old.get("scanned"):
                        # (2026-11-03h) a list marked "AFM scan complete" keeps its flakes and positions as scanned
                        return self._json(dict(error=AFM_DONE_MSG), 409)
                    # (made again: where it is; a new one in the run's own folder, 2026-11-10b: seen on every computer)
                    return self._json(AP.make_plan(b, had.parent.parent if had is not None
                                                   else RUN_DATA.run_dir(S.store, b.get("run"), "afm") or d))
                if path == "/api/afm/newlist":
                    # (2026-11-03e) Library › AFM runs: ＋ a named list (a draft until it is made), or Redo of a list
                    src = None
                    if b.get("from"):
                        src, _ = S.store.afm_plan(b.get("from"))
                        if src is None:
                            return self._json(dict(error="no such AFM list"), 404)
                    rd_ = RUN_DATA.run_dir(S.store, b.get("run") or (src or {}).get("run"), "afm") or d
                    with S.store.lock:
                        return self._json(AP.new_draft(b, rd_, src))
                if path == "/api/afm/export":
                    # (2026-10-30) an AFM list's flakes as a flake list (JSON): in the run's folder (the microscope's
                    # "Remeasure flake coordinates" › Custom list) and in <data>/afm/lists/ (a "list:" choice here)
                    fl = [str(x) for x in b.get("flakes") or [] if x]
                    if not fl:
                        return self._json(dict(error="no flakes"), 400)
                    nm = re.sub(r"[^A-Za-z0-9_-]+", "_", str(b.get("name") or "flakes")).strip("_")[:60] or "flakes"
                    doc = dict(kind="flakeid-flakes", source="export", name=str(b.get("name") or nm), run=b.get("run"), scan=b.get("scan"), flakes=fl,
                               note=str(b.get("note") or ""), by=b.get("by"), created=time.strftime("%Y-%m-%d %H:%M"))
                    fn = f"flakes_{nm}_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    txt = json.dumps(doc, indent=1)
                    out = []
                    try:
                        rd = _run_dirs(S.store, b.get("run")) if b.get("run") else []
                    except Exception:
                        rd = []
                    for q in rd[:1]:
                        try:
                            (q / fn).write_text(txt, encoding="utf-8")
                            out.append(str(q / fn))
                        except OSError:
                            pass
                    (d / "lists").mkdir(parents=True, exist_ok=True)
                    (d / "lists" / fn).write_text(txt, encoding="utf-8")
                    return self._json(dict(saved=out[0] if out else str(d / "lists" / fn), files=out + [str(d / "lists" / fn)],
                                           in_run=bool(out)))
                if path == "/api/afm/merge":
                    # (2026-11-01) two point lists as one: every point of the first, then every point of the second;
                    # written next to the first as "<first>+<second>.pgm" (never over a file)
                    try:
                        rd = _run_dirs(S.store, b.get("run")) if b.get("run") else []
                    except Exception:
                        rd = []
                    f1 = b.get("first") or pick_file("AFM point list 1 (.pgm): its points come first", ["pgm"], str(rd[0]) if rd else None)
                    if not f1:
                        return self._json(dict(cancelled=True))
                    f2 = b.get("second") or pick_file("AFM point list 2 (.pgm): its points come after those of list 1", ["pgm"], str(Path(f1).parent))
                    if not f2:
                        return self._json(dict(cancelled=True))
                    p1, p2 = Path(f1), Path(f2)
                    txt, n1, n2 = AP.merge_pgm(p1.read_text(encoding="latin-1"), p2.read_text(encoding="latin-1"))
                    stem = f"{p1.stem}+{p2.stem}"[:150]
                    out, k = p1.parent / f"{stem}.pgm", 2
                    while out.exists():
                        out, k = p1.parent / f"{stem}_{k}.pgm", k + 1
                    out.write_bytes(txt.encode("latin-1"))
                    return self._json(dict(saved=str(out), first=str(p1), second=str(p2), n1=n1, n2=n2, n=n1 + n2))
                pl, pf = S.store.afm_plan(b.get("id"))
                if pl is None:
                    return self._json(dict(error="no such AFM list"), 404)
                if path in ("/api/afm/rename", "/api/afm/scanned"):
                    # (2026-11-03e) Library › AFM runs: a list's name; its "AFM scan complete" mark (when, the folder)
                    with S.store.lock:
                        pl, pf = S.store.afm_plan(b.get("id"))
                        if path == "/api/afm/rename":
                            nm = str(b.get("name") or "").strip()[:80]
                            if not nm:
                                return self._json(dict(error="a name is needed"), 400)
                            pl["name"] = nm
                        elif b.get("scanned") is False:
                            # (2026-11-03h) the complete mark removed: the list can be changed (made again) again
                            if pl.pop("scanned", None):
                                pl["unmarked"] = int(time.time() * 1000)
                        else:
                            if pl.get("draft"):
                                return self._json(dict(error="this list is not made yet: no points were scanned"), 400)
                            pl["scanned"] = int(time.time() * 1000)
                            if b.get("folder"):
                                pl["scanned_folder"] = str(b["folder"])
                        pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
                    return self._json(pl)
                if path == "/api/afm/save":
                    # a copy of the point list (and its csv) in a folder chosen on this computer (for the AFM PC)
                    # (2026-10-30) the dialog opens in the run's own folder (where its designer build is), else as before
                    try:
                        rd = _run_dirs(S.store, pl.get("run")) if pl.get("run") else []
                    except Exception:
                        rd = []
                    # (2026-11-03k) {auto: true} (Generate AFM file): no dialog; by hand ("Save the list to a folder…")
                    # a dialog.  (2026-11-03q) both default to the run's afm_pgm folder (made if missing); only when the
                    # run's folder is not known, the folder last chosen by hand (config afm_save_dir), else the run's
                    cfg = load_config()
                    last = cfg.get("afm_save_dir")
                    last = last if last and Path(last).is_dir() else None
                    pgm = afm_pgm_dir(S.store, pl.get("run"))
                    start = pgm or last or (str(rd[0]) if rd else None)
                    if b.get("auto") and start:
                        to = start
                    else:
                        to = pick_folder("Folder for the AFM point list (.pgm)", start)
                        if not to:
                            return self._json(dict(cancelled=True))
                        if to != cfg.get("afm_save_dir"):
                            cfg["afm_save_dir"] = to
                            save_config(cfg)
                    # (2026-11-03j) the copies are named with the time they are saved, to the second
                    # (<chip>_afm_sites_20261103_142507.pgm and .csv): every save a new file, never one written over
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    names, k = {}, 1
                    while True:
                        sfx = f"_{stamp}" + (f"_{k}" if k > 1 else "")
                        names = {n: f"{Path(n).stem}{sfx}{Path(n).suffix}" for n in (pl["pgm"], pl["csv"])}
                        if not any((Path(to) / m).exists() for m in names.values()):
                            break
                        k += 1
                    for n, m in names.items():
                        shutil.copy2(str(pf.parent / n), str(Path(to) / m))
                    saved = str(Path(to) / names[pl["pgm"]])
                    with self.server.store.lock:                 # (the list shows where it was saved last)
                        pl, pf = S.store.afm_plan(b.get("id"))
                        pl["saved_to"] = saved
                        pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
                    return self._json(dict(saved=saved, plan=pl))
                if path in ("/api/afm/pickdir", "/api/afm/folder"):
                    # (2026-11-02l) the AFM images are read from the list's folder: by default its run's afm_scans
                    # (afm_folder); /api/afm/folder: that folder and its files, no dialog ({reset: true}: back to the
                    # default); /api/afm/pickdir {change: true}: a dialog (opening there) chooses another, kept for the list
                    fo, dflt = afm_folder(S.store, pl)
                    if path == "/api/afm/folder" and b.get("reset"):
                        with self.server.store.lock:
                            pl, pf = S.store.afm_plan(b.get("id"))
                            pl.pop("afm_folder", None)
                            pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
                        fo, dflt = afm_folder(S.store, pl)
                    if path == "/api/afm/pickdir" and (b.get("change") or fo is None):
                        try:
                            rd = _run_dirs(S.store, pl.get("run")) if pl.get("run") else []
                        except Exception:
                            rd = []
                        to = pick_folder("Folder with the AFM images of this list (.spm)", fo or (str(rd[0]) if rd else None))
                        if not to:
                            return self._json(dict(cancelled=True, folder=fo, default=dflt))
                        to = str(Path(to))
                        with self.server.store.lock:
                            pl, pf = S.store.afm_plan(b.get("id"))
                            if dflt and fo and Path(to) == Path(fo):
                                pl.pop("afm_folder", None)      # (the run's afm_scans chosen: stays the default)
                            else:
                                pl["afm_folder"] = to
                            pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
                        fo, dflt = afm_folder(S.store, pl)
                    return self._json(dict(folder=fo, default=dflt, files=afm_files(fo) if fo else []))
                if path == "/api/afm/localdrop":
                    fo, _ = afm_folder(S.store, pl)
                    afm_drop_list(pl, pf, afm_files(fo) if fo else ())
                    return self._json(dict(ok=True))
                if path == "/api/afm/localcopy":
                    afm_copy_list(S.store, AP, b.get("id"))
                    return self._json(dict(ok=True))
                if path == "/api/afm/analyse":
                    f = Path(str(b.get("file") or ""))
                    if not f.is_file():
                        return self._json(dict(error=f"{f}: not found"), 400)
                    n = int(b.get("site") or 0)
                    png = pf.parent / f"{n:03d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', f.name)}.png"
                    for c in png.parent.glob(png.stem + ".full*.npz"):
                        try:                                  # (Refresh AFM files: a file scanned again under its name)
                            c.unlink()
                        except OSError:
                            pass
                    r = AP.analyse_file(f, png)
                    r.update(site=n, source=str(f), at=int(time.time() * 1000))
                    with self.server.store.lock:              # (the plan file: one writer at a time)
                        pl, pf = S.store.afm_plan(b.get("id"))
                        R = pl.setdefault("results", {})
                        if b.get("add") and R.get(str(n)) is not None and R[str(n)].get("source") != str(f):
                            # (2026-11-04l, AFM file assignment: ＋ Add) another AFM file of the same flake, after its first
                            M = pl.setdefault("more", {}).setdefault(str(n), [])
                            i = next((j for j, q in enumerate(M) if q.get("source") == str(f)), None)
                            if i is None:
                                M.append(r)
                            else:
                                M[i] = r
                        else:
                            R[str(n)] = r
                        pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
                    return self._json(r)
                if path == "/api/afm/unlink":
                    # (2026-11-04l, AFM file assignment: ✕) one AFM file taken off a point (k: 0 the first, then the added);
                    # the first taken off: the next added one becomes the first
                    n, k = str(int(b.get("site") or 0)), int(b.get("k") or 0)
                    with self.server.store.lock:
                        pl, pf = S.store.afm_plan(b.get("id"))
                        R, M = pl.setdefault("results", {}), pl.setdefault("more", {})
                        L = [x for x in [R.get(n)] + list(M.get(n) or []) if x is not None]
                        if 0 <= k < len(L):
                            L.pop(k)
                        if L:
                            R[n] = L[0]
                        else:
                            R.pop(n, None)
                        if len(L) > 1:
                            M[n] = L[1:]
                        else:
                            M.pop(n, None)
                        if not M:
                            pl.pop("more", None)
                        pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
                    return self._json(pl)
                if path == "/api/afm/pickfile":
                    # (2026-11-04l, AFM file assignment: ＋ Add) one AFM file, the dialog opening in the list's AFM folder
                    fo, _ = afm_folder(S.store, pl)
                    to = pick_file("AFM file of this flake (.spm)", ["spm"], fo)
                    return self._json(dict(file=str(Path(to))) if to else dict(cancelled=True))
                if path == "/api/afm/tipoffset":
                    # (optional) where the tip really was at a scanned point: its AFM image matched to the site's 100x
                    # picture (afmplan.tip_offset); nothing changes until /api/afm/tip
                    n = int(b.get("site") or 0)
                    r0 = (pl.get("results") or {}).get(str(n))
                    site = next((q for q in pl["sites"] if int(q["n"]) == n), None)
                    if r0 is None or site is None:
                        return self._json(dict(error=f"no AFM image imported for point {n}"), 400)
                    u = urllib.parse.urlparse(str(b.get("img") or "")).path
                    parts = u[len("/_blob/"):].split("/") if u.startswith("/_blob/") else []
                    ip = S.store.run_file(parts[0], parts[1]) if len(parts) == 2 else None
                    if ip is None or not ip.is_file():
                        return self._json(dict(error="the flake's 100x picture was not found"), 400)
                    import cv2
                    rgb = cv2.imread(str(ip), cv2.IMREAD_COLOR)[:, :, ::-1]
                    z, scan_um = AP.load_height(pf.parent, r0)
                    pref = (load_config().get("afm_tip") or {}).get("orient")
                    A = pl["transform"]["A"]
                    if site.get("c2s"):                 # (a block's list: AFM <- stage <- this chip's micrometres)
                        C = site["c2s"]
                        A = [[A[i][0] * C[0][0] + A[i][1] * C[1][0], A[i][0] * C[0][1] + A[i][1] * C[1][1]] for i in range(2)]
                    out = AP.tip_offset(z, scan_um, rgb, float(b["px_um"]), b.get("px") or site["px"], A,
                                        orient=b.get("orient"), prefer=pref, overlay=pf.parent / f"{n:03d}_tip_overlay.png")
                    out.update(site=n, scan_um=scan_um, picture_px_um=float(b["px_um"]))
                    return self._json(out)
                if path == "/api/afm/tip" and pl.get("scanned"):
                    return self._json(dict(error=AFM_DONE_MSG), 409)
                if path == "/api/afm/tip":
                    # the tip offset taken off every point of the list (the list is written again), or removed (offset null)
                    with self.server.store.lock:
                        pl, pf = S.store.afm_plan(b.get("id"))
                        tip = None
                        if b.get("offset") is not None:
                            tip = dict(offset=[float(v) for v in b["offset"]], site=int(b.get("site") or 0), orient=b.get("orient"),
                                       score=b.get("score"), d_um=b.get("d_um"), how=b.get("how") or "matched", at=int(time.time() * 1000))
                            cfg = load_config()
                            cfg["afm_tip"] = dict(tip, plan=pl["id"])         # (the orientation and last offset, for next time)
                            save_config(cfg)
                        pl = AP.apply_tip(pl, pf, tip)
                    return self._json(pl)
                if path == "/api/afm/height":
                    # (2026-11-02k, AFM screening) one point's levelled AFM height image, full resolution (<= 768 px),
                    # for the picture next to the 100x one: float32 little-endian, base64, NaN where the data is bad
                    r0 = afm_result(pl, int(b.get("site") or 0), b.get("k"))
                    if r0 is None:
                        return self._json(dict(error="no AFM image for this point"), 404)
                    z, px_nm, full = AP.height_full(pf.parent, r0)
                    import base64
                    return self._json(dict(w=int(z.shape[1]), h=int(z.shape[0]), pixel_nm=px_nm, full=full, file=r0.get("file"),
                                           source=r0.get("source"), z=base64.b64encode(z.astype("<f4").tobytes()).decode("ascii")))
                if path == "/api/afm/gwyddion":
                    # (2026-11-02k) the point's .spm opened in Gwyddion on this computer, to work on it there
                    r0 = afm_result(pl, int(b.get("site") or 0), b.get("k"))
                    src = Path(str((r0 or {}).get("source") or ""))
                    if r0 is None or not src.is_file():
                        return self._json(dict(error="the AFM file is not at " + str(src) if r0 else "no AFM image for this point"), 404)
                    exe = AP.gwyddion_exe()
                    if not exe:
                        return self._json(dict(error="Gwyddion was not found on this computer (gwyddion.net; on a Mac: brew install gwyddion)"), 404)
                    AP.open_in_gwyddion(exe, src)
                    return self._json(dict(ok=True, exe=str(exe), file=str(src)))
                if path == "/api/afm/assign":
                    # a result moved to another site (a file matched to the wrong point), or dropped (site 0)
                    with self.server.store.lock:
                        pl, pf = S.store.afm_plan(b.get("id"))
                        R = pl.setdefault("results", {})
                        r = R.pop(str(int(b.get("from") or 0)), None)
                        if r is not None and int(b.get("to") or 0) > 0:
                            r["site"] = int(b["to"])
                            R[str(int(b["to"]))] = r
                        pf.write_text(json.dumps(pl, indent=1), encoding="utf-8")
                    return self._json(pl)
                return self._json(dict(error="unknown"), 404)
            except (ValueError, RuntimeError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
        if path == "/api/devicephotos/pick":
            return self._json(dict(picked=pick_folder("Folder of device photos (the scanner's Device photos)",
                                                      (S.store.device_photo_roots() or [None])[0])))
        if path in ("/api/fabalign/substrate", "/api/fabalign"):
            # the aligned photos: the substrate's layout (GDS) kept with the stack, and every step photo placed on its
            # alignment marks (marks_to_klayout --fits-json; pictures without enough marks matched to placed ones)
            if self.headers.get("X-Designer") != "1":
                return self._json(dict(error="designer page only"), 403)
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
                sid = str(b.get("stack") or "")
                if not re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", sid):
                    return self._json(dict(error="save the stack first"), 400)
                if path == "/api/fabalign/substrate":
                    src = Path(str(b.get("path") or ""))
                    if not src.is_file() or src.suffix.lower() not in (".gds", ".gds2", ".gdsii"):
                        return self._json(dict(error=f"{src}: not a GDS file that can be read here"), 400)
                    if SD.is_folder_id(sid):
                        u, fn, fp = S.store.folders.put_file(sid, src)
                    else:
                        d = S.store._dir("stack_files")
                        if d is None:
                            return self._json(dict(error="no shared data folder"), 500)
                        (d / sid).mkdir(parents=True, exist_ok=True)
                        fn = SD.unique_name(d / sid, src.name)
                        shutil.copy2(str(src), str(d / sid / fn))
                        u, fp = f"/stackfile/{sid}/{fn}", d / sid / fn
                    return self._json(dict(name=fn, url=u, path=str(fp), source=str(src)))
                gds = self._stackfile_path(str(b.get("substrate") or ""))
                if gds is None:
                    return self._json(dict(error="choose the substrate's layout (GDS) first"), 400)
                urls = [str(e.get("url") if isinstance(e, dict) else e) for e in b.get("photos") or []]
                paths = {u: self._stackfile_path(u) for u in urls}
                todo = [(u, p) for u, p in paths.items() if p is not None]
                if not todo:
                    return self._json(dict(error="no step pictures to align"), 400)
                miss = kl_missing()
                if miss:
                    return self._json(dict(error="This Python is missing %s, which the alignment needs. Install with:  %s" % (
                        ", ".join(miss), kl_pip_command(miss))), 400)
                jobs = S.__dict__.setdefault("align_jobs", {})
                if any(j.state == "running" for j in jobs.values()):
                    return self._json(dict(error="an alignment is still running: wait for it"), 409)
                import tempfile
                td = tempfile.mkdtemp(prefix="vdw_align_")
                exp = {Path(str(p)).name: float(e["um_per_px"]) for e in b.get("photos") or [] if isinstance(e, dict)
                       and e.get("um_per_px") for u, p in todo if u == e.get("url")}
                (Path(td) / "expect.json").write_text(json.dumps(exp))
                cmd = helper_cmd("marks", [str(p) for _, p in todo] + ["--gds", str(gds), "--fits-json", str(Path(td) / "fits.json"),
                                                                     "--outline-max-um-px", "0", "--expect-um", str(Path(td) / "expect.json")])
                job = AlignJob(cmd, dict(stack=sid, gds=gds.name), td, todo, sid, gds.name, S.store.folders)
                jobs[job.id] = job
                return self._json(dict(job=job.id, of=len(todo)))
            except (ValueError, OSError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
        if path == "/api/fabstep/new":
            # a fabrication step from pictures taken any way (not the scanner's Device photos): the pictures in the
            # request (base64), written to a new step folder NN_<step> (folder stacks) or the stack's files
            if self.headers.get("X-Designer") != "1":
                return self._json(dict(error="designer page only"), 403)
            try:
                import base64
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= 800 * 1024 * 1024:
                    return self._json(dict(error="up to 800 MB of pictures at once"), 400)
                b = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                sid = str(b.get("stack") or "")
                if not re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", sid):
                    return self._json(dict(error="save the stack first"), 400)
                photos = [dict(p, data=base64.b64decode(p.get("data") or "")) for p in b.get("photos") or []]
                if not photos or any(not p["data"] for p in photos):
                    return self._json(dict(error="no pictures"), 400)
                title, desc = str(b.get("title") or "").strip(), str(b.get("description") or "")
                if SD.is_folder_id(sid):
                    return self._json(dict(step=S.store.folders.new_step(sid, title, desc, user_name(), photos)))
                return self._json(S.store.new_fab_photos(sid, title, photos))
            except (ValueError, OSError, RuntimeError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
        if path == "/api/devicephotos/import":
            if self.headers.get("X-Designer") != "1":
                return self._json(dict(error="designer page only"), 403)
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
                sid = str(b.get("stack") or "")
                if not re.match(r"^[A-Za-z0-9_.~@+-]{1,200}$", sid):
                    return self._json(dict(error="save the stack first"), 400)
                if SD.is_folder_id(sid):                 # (a new step folder NN_<step> in the stack's folder)
                    src = str(b.get("path") or "")
                    rec = S.store.device_photo_set(src)[0] if not (Path(src) / "step.json").is_file() else None
                    return self._json(dict(step=S.store.folders.import_step(sid, src, str(b.get("title") or ""),
                                                                            str(b.get("description") or ""), user_name(), rec)))
                return self._json(S.store.import_device_photos(sid, str(b.get("path") or "")))
            except (ValueError, RuntimeError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
        if path == "/api/manual/profiles":
            return self._json(S.store.colour_profiles())
        if path == "/api/manual/outline":
            # an imported flake's outline from a point inside it (flakeid_designer/manual.py auto_outline)
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
                name = Path(str(b.get("url") or "")).name
                d = S.store._dir("stack_images")
                f = d / name if d is not None and re.match(r"^[A-Za-z0-9_-]{1,80}\.(jpg|png)$", name) else None
                if f is None or not f.is_file():
                    return self._json(dict(error="the picture is not in the data folder"), 400)
                builder_main()
                try:
                    from . import manual as MN
                except ImportError:
                    import manual as MN                  # type: ignore
                import cv2
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                thr = b.get("thr")
                return self._json(MN.auto_outline(img, b.get("seed") or [0, 0], float(thr) if thr not in (None, "") else None,
                                                  mode="terrace" if b.get("mode") == "terrace" else "flake"))
            except (ValueError, RuntimeError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
        if path == "/api/manual/thickness":
            # a rough thickness from an imported flake's colour against the substrate (flakeid_designer/manual.py)
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
                name = Path(str(b.get("url") or "")).name
                d = S.store._dir("stack_images")
                f = d / name if d is not None and re.match(r"^[A-Za-z0-9_-]{1,80}\.(jpg|png)$", name) else None
                if f is None or not f.is_file():
                    return self._json(dict(error="the picture is not in the data folder"), 400)
                builder_main()                           # (the flakeid package on the path, numpy / scipy / opencv there)
                try:
                    from . import manual as MN
                except ImportError:
                    import manual as MN                  # type: ignore
                import cv2
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                rng = b.get("t_range")
                rng = (float(rng[0]), float(rng[1])) if isinstance(rng, list) and len(rng) == 2 and float(rng[1]) > float(rng[0]) else None
                if b.get("calibrate_t") not in (None, ""):
                    # a flake of known thickness (AFM): the picture source's gamma, kept as a colour profile for everyone
                    c = MN.calibrate(img, b.get("outline") or [], str(b.get("material") or "hbn"), float(b.get("oxide_nm") or 285.0),
                                     float(b["calibrate_t"]))
                    prof = S.store.add_colour_sample(str(b.get("profile") or "microscope software")[:60], dict(
                        c, t_nm=float(b["calibrate_t"]), picture=name, material=str(b.get("material") or "hbn"), at=int(time.time() * 1000)))
                    return self._json(dict(calibration=c, profile=prof))
                return self._json(MN.estimate(img, b.get("outline") or [], str(b.get("material") or "hbn"),
                                              float(b.get("oxide_nm") or 285.0), linear=bool(b.get("linear")),
                                              gamma=float(b["gamma"]) if b.get("gamma") else None, t_range=rng))
            except (ValueError, RuntimeError) as e:
                return self._json(dict(error=str(e)), 400)
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
        if path == "/api/stackdir/open":
            # a folder stack: {path} (a folder or its .stack file; none: a folder chooser), new: a new stack there
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
            except ValueError:
                return self._json(dict(error="bad request"), 400)
            p = b.get("path") or pick_folder("The stack's folder (the device's folder)" if not b.get("new") else
                                              "A folder for the new stack (make one for the device)", b.get("start"))
            if not p:
                return self._json(dict(cancelled=True))
            try:
                sid, folder, fname, exists = S.store.folders.register(p)
            except (OSError, ValueError) as e:
                return self._json(dict(error=str(e)), 400)
            data = S.store.folders.read(sid) if exists else None
            if exists and data is None:
                return self._json(dict(error=f"{folder / fname}: not a stack file (not readable)"), 400)
            S.store.folders.remember(sid, (data or {}).get("name"))
            S.store._lists.pop("stacks", None)
            return self._json(dict(id=sid, folder=str(folder), file=fname, name=Path(fname).stem, exists=exists, data=data))
        if path == "/api/stacklock":
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}")
            except ValueError:
                return self._json(dict(error="bad request"), 400)
            ok, other = S.store.stack_lock(str(b.get("id") or ""), str(b.get("session") or ""), bool(b.get("release")))
            if not b.get("release"):
                try:
                    S.offline.opened(str(b.get("id") or ""))     # (stacks opened here are kept offline: offline.py)
                except Exception:
                    pass
            return self._json(dict(ok=ok, other=other))
        if path == "/api/stackimage":
            n = int(self.headers.get("Content-Length") or 0)
            ctype = (self.headers.get("Content-Type") or "").lower()
            if not 0 < n <= 40 * 1024 * 1024 or ctype not in ("image/jpeg", "image/png"):
                return self._json(dict(error="a JPEG or PNG picture up to 40 MB"), 400)
            sid = (urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("stack") or [""])[0]
            if SD.is_folder_id(sid):                    # (a folder stack's pick-up pictures: 01_stacking/)
                try:
                    u = S.store.folders.store(sid, self.rfile.read(n), "picture." + ("png" if ctype == "image/png" else "jpg"))[0]
                except OSError as e:
                    return self._json(dict(error=str(e)), 500)
                return self._json(dict(url=u))
            d = S.store._dir("stack_images")
            if d is None:
                return self._json(dict(error="no shared data folder"), 500)
            import uuid
            name = f"{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:12]}." + ("png" if ctype == "image/png" else "jpg")
            (d / name).write_bytes(self.rfile.read(n))
            return self._json(dict(url="/stackimage/" + name))
        if path.startswith("/api/klayout/"):
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                return self._json(dict(error="JSON only"), 415)          # (a form on another site cannot start an export)
            try:
                body = self._body()
            except ValueError:
                return self._json(dict(error="bad request"), 400)
            return self._klayout_post(path, body)
        given = None
        if path == "/api/runs/add":
            # (2026-11-05i) {path}: add that run folder without the folder window (Library › Recent runs › Add run);
            # JSON only (a form on another site cannot add folders)
            try:
                b = self._body()
            except ValueError:
                b = {}
            if isinstance(b, dict) and b.get("path"):
                if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                    return self._json(dict(error="JSON only"), 415)
                given = str(b["path"])
                if not Path(given).is_dir():
                    return self._json(dict(error=f"{given} is not reachable from this computer (is the drive connected?)"), 404)
        if given is not None:
            p = given
        elif path in ("/api/roots/pick", "/api/library/pick", "/api/runs/add"):
            # (2026-11-03f) one folder window at a time, and the settings read after it closes: a window left open
            # (often behind the browser) while the library was changed saved the old settings back, with one
            # library's folders under another's name
            if not PICKING.acquire(blocking=False):
                return self._json(dict(error="A folder window is already open (it can be behind the browser): choose "
                                             "the folder there, or cancel it first."), 409)
            try:
                p = pick_folder("Run folder to add" if path == "/api/runs/add" else "Folder with scan results",
                                str(Path(S.store.library).parent) if path == "/api/runs/add" and S.store.library else None)
            finally:
                PICKING.release()
        if path == "/api/runs/add":
            # (2026-11-05b) ＋ Add run…: one run from any folder, listed with this library's runs (kept with the library)
            if not p:
                return self._json(dict(picked=None))
            p = str(Path(p))
            if not is_run_dir(p):
                return self._json(dict(error=f"{p} is not a scan run's folder (no designer files, copy_done.json or chip folders in it)"), 400)
            if not S.store.library:
                # (2026-11-05c, no Directory button any more) the first run on a computer: its folder's parent becomes
                # the library (its stacks and decisions in <parent>/_designer_data)
                with LIBRARY_LOCK:
                    switch_library(load_config(), S.store, str(Path(p).parent))
                S.store.added = [r for r in (S.store.added or []) if r != p] + [p]     # (only this run: not its folder's)
                save_added(S.store)
                S.store.scan()
                return self._json(dict(picked=p, inside=True, reload=True, roots=S.store.roots, added=S.store.added))
            if S.store.added is None and S.store.library:
                S.store.added = []                      # (no list yet: it starts with this run)
            if S.store.added is not None:
                if p not in S.store.added:
                    S.store.added.append(p)
                    save_added(S.store)
                S.store.scan()
                return self._json(dict(picked=p, inside=True, added=S.store.added, roots=S.store.roots))
            inside = in_folder(p, S.store.library)
            if not inside:
                with LIBRARY_LOCK:
                    cfg = load_config()
                    if p not in S.store.roots:
                        S.store.roots.append(p)
                        remember_library(cfg, S.store)
                        save_config(cfg)
            S.store.scan()
            return self._json(dict(picked=p, inside=inside, roots=S.store.roots))
        if path == "/api/roots/pick":
            with LIBRARY_LOCK:
                cfg = load_config()
                if p and str(Path(p)) not in S.store.roots:
                    S.store.roots.append(str(Path(p)))
                    if S.store.data is None:
                        S.store.data = Path(p) / "_designer_data"
                    if not S.store.library:
                        S.store.library = S.store.roots[0]
                    remember_library(cfg, S.store)
                    save_config(cfg)
            if p:
                S.store.scan()
            return self._json(dict(roots=S.store.roots, picked=p, data_dir=str(S.store.data) if S.store.data else None))
        if path in ("/api/library/pick", "/api/library/use"):
            if path.endswith("use"):
                p = self._body().get("path")
            if not p:
                return self._json(dict(picked=None))
            if not Path(p).is_dir():
                return self._json(dict(error=f"{p} is not reachable (is the share connected?)"), 400)
            with LIBRARY_LOCK:
                cfg = load_config()
                switch_library(cfg, S.store, p)
            return self._json(dict(picked=p, roots=S.store.roots, data_dir=str(S.store.data), **library_info(cfg, S.store)))
        if path in ("/api/runs/remove", "/api/runs/seed"):
            # (2026-11-05j) remove: a run off the list (nothing on disk changes); seed: the first list on this computer
            # (the runs ticked then), only when none was chosen yet
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                return self._json(dict(error="JSON only"), 415)
            try:
                b = self._body()
            except ValueError:
                return self._json(dict(error="bad request"), 400)
            if path == "/api/runs/seed":
                if S.store.added is None:
                    S.store.added = list(dict.fromkeys(str(Path(x)) for x in (b.get("paths") or []) if isinstance(x, str) and x))
                    save_added(S.store)
                    S.store.scan()
                return self._json(dict(added=S.store.added))
            p = str(Path(str(b.get("path") or "")))
            if S.store.added is None:
                return self._json(dict(error="no list of runs yet"), 409)
            S.store.added = [r for r in S.store.added if r != p]
            save_added(S.store)
            S.store.scan()
            return self._json(dict(added=S.store.added))
        if path == "/api/roots/remove":
            p = self._body().get("path")
            if S.store.data is not None and p and Path(p) == Path(S.store.data).parent:
                return self._json(dict(error="This folder holds the library's own data (stacks, decisions): switch libraries with Change directory instead."), 400)
            with LIBRARY_LOCK:
                cfg = load_config()
                S.store.roots = [r for r in S.store.roots if r != p]
                remember_library(cfg, S.store)
                save_config(cfg)
            S.store.scan()
            return self._json(dict(roots=S.store.roots))
        cfg = load_config()
        if path.startswith("/api/camera/") and CAM is not None:
            b = self._body()
            if path == "/api/camera/calibrate":
                return self._json(calibrate_live(b))
            if path == "/api/camera/start":
                if (b.get("source") or "sim") == "sim":
                    sim_marks()
                ok = CAM.LIVE.start(b.get("source") or "sim", int(b.get("index") or 0), b.get("serial"))
                return self._json(dict(CAM.LIVE.status(), ok=ok))
            if path == "/api/camera/stop":
                CAM.LIVE.stop()
                return self._json(CAM.LIVE.status())
            if path == "/api/camera/settings":
                err = CAM.LIVE.open_settings()
                return self._json(dict(ok=err is None, error=err))
            if path == "/api/camera/jog":
                CAM.LIVE.jog(float(b.get("dx") or 0), float(b.get("dy") or 0), float(b.get("drot") or 0), b.get("focus"))
                return self._json(dict(ok=True))
        if path == "/api/station":
            # this transfer station's settings (camera source, its placement per objective): kept on this computer
            st = self._body()
            cfg["station"] = st if isinstance(st, dict) else {}
            save_config(cfg)
            return self._json(dict(ok=True))
        if path == "/api/buildhere":
            cfg = load_config()
            cfg["build_here"] = bool(self._body().get("on"))
            save_config(cfg)
            if cfg["build_here"]:
                threading.Thread(target=S.store.scan, daemon=True).start()     # (the runs waiting: built now)
            return self._json(dict(on=cfg["build_here"]))
        if path == "/api/review/reset":
            b = self._body()
            run = str(b.get("run") or "")
            if not run:
                return self._json(dict(error="no run"), 400)
            try:
                return self._json(dict(ok=True, cleared=_reset_review(S.store, run, afm=bool(b.get("afm")))))
            except OSError as e:
                return self._json(dict(error=str(e)), 503)
        if path == "/api/coords/restore":
            b = self._body()
            run, pid = str(b.get("run") or ""), str(b.get("id") or "")
            if not run or not pid:
                return self._json(dict(error="no run / saved set"), 400)
            try:
                return self._json(dict(ok=True, **restore_coords(S.store, run, pid)))
            except FileNotFoundError as e:
                return self._json(dict(error=str(e)), 404)
            except OSError as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 503)
        if path == "/api/kept/sync":
            # "Update kept folder" (a run in the Library): its kept/ folder brought up to date with the decisions now
            run = str(self._body().get("run") or "")
            if not run:
                return self._json(dict(error="no run"), 400)
            try:
                res = KP.KEPT.sync_now(S.store, run, full=True)
            except Exception as e:
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 500)
            if not res:
                return self._json(dict(error=f"run {run} not found in this library's folders"), 404)
            return self._json(dict(ok=True, results=res))
        if path in ("/api/update/install", "/api/update/token", "/api/update/restart"):
            b = self._body()
            if path == "/api/update/token":
                try:
                    return self._json(dict(saved=UPD.save_token(b.get("token"))))
                except UPD.UpdateError as e:
                    return self._json(dict(error=str(e)), 400)
            if path == "/api/update/install":
                with UPDATE["lock"]:
                    if UPDATE["state"] == "running":
                        return self._json(dict(error="an update is already running"), 409)
                    UPDATE.update(state="running", log=[], result=None, error=None)
                threading.Thread(target=_run_update, args=(str(b.get("sha") or ""),), daemon=True, name="update").start()
                return self._json(dict(ok=True))
            # restart: the browser version starts itself again in place (same command, no new browser tab); the app
            # (or a packaged program) is opened again by hand
            if NATIVE_PICK is not None or getattr(sys, "frozen", False):
                return self._json(dict(restart=False))
            self._json(dict(restart=True))
            threading.Thread(target=_restart, args=(S,), daemon=True, name="restart").start()
            return
        if path == "/api/build":
            b = self._body()
            d = b.get("dir")
            if not d or not Path(d).is_dir():
                return self._json(dict(error="no such run folder"), 400)
            BUILDER.want(Path(d), full=bool(b.get("full")))        # (full: Library › Rebuild)
            return self._json(dict(ok=True))
        if path == "/api/keptcopy":
            KEPT_COPY.want(S.store, json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8") or "{}").get("run"))
            return self._json(dict(ok=True))
        if path == "/api/rescan":
            S.store.scan()
            return self._json(dict(n=len(S.store.runs)))
        self._json(dict(error="unknown"), 404)


def running(port):
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1.5) as r:
            return json.loads(r.read().decode()).get("app") == APP
    except Exception:
        return False


def install_shortcut():
    """Windows: desktop shortcut (pythonw, no console).  Mac: 'FlakeIdentifier Designer.app' in Applications and on the Desktop."""
    name = "FlakeIdentifier Designer"
    if sys.platform == "win32":
        import base64
        pyw = Path(sys.executable).with_name("pythonw.exe")
        pyw = pyw if pyw.is_file() else Path(sys.executable)
        if "WindowsApps" in str(pyw):
            print("Note: this is the Microsoft Store Python; the python.org one is recommended (tick \"Add python.exe to PATH\").")
        icon = HERE / "static" / "designer_win.ico"
        icon = icon if icon.is_file() else HERE / "static" / "designer.ico"
        # a copy of the icon on this computer: Windows often shows no icon (or a blank one) for a shortcut whose icon file
        # is on a network drive, e.g. when the designer is started from the shared drive.  (In the home folder, not
        # AppData: the Microsoft Store Python redirects what it writes there.)
        try:
            local = Path.home() / ".flakeid_designer_files" / "designer.ico"
            local.parent.mkdir(exist_ok=True)
            local.write_bytes(icon.read_bytes())
            icon = local
        except OSError as e:
            print(f"(icon not copied to this computer: {e})")

        def q(v):                                      # a PowerShell single-quoted string
            return "'" + str(v).replace("'", "''") + "'"
        # on the Desktop (wherever it is: OneDrive moves it) and in the Start menu; PowerShell, not VBScript (being removed
        # from Windows 11) and no script file (the Store Python redirects what it writes to %TEMP%)
        ps = f"""$ErrorActionPreference = 'Stop'
$sh = New-Object -ComObject WScript.Shell
foreach ($d in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {{
  try {{
    $p = Join-Path $d {q(name + '.lnk')}
    $l = $sh.CreateShortcut($p)
    $l.TargetPath = {q(pyw)}
    $l.Arguments = {q('"' + str(HERE) + '"')}
    $l.WorkingDirectory = {q(HERE.parent)}
    $l.IconLocation = {q(str(icon) + ',0')}
    $l.Description = 'vdW Stack Designer (FlakeIdentifier, Young Lab)'
    $l.Save()
    if (Test-Path $p) {{ Write-Output ('shortcut: ' + $p) }} else {{ Write-Output ('NOT created: ' + $p) }}
  }} catch {{ Write-Output ('NOT created in ' + $d + ': ' + $_) }}
}}
"""
        r = None
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                                base64.b64encode(ps.encode("utf-16-le")).decode()], capture_output=True, text=True, timeout=60)
            print((r.stdout + r.stderr).strip())
        except (OSError, subprocess.SubprocessError) as e:
            print(f"(PowerShell not usable: {e})")
        if r is not None and r.returncode == 0 and "shortcut: " in r.stdout:
            print(f"\nInstalled. Start it from the \"{name}\" icon on the desktop or from the Start menu.")
            return 0
        vbs = f'''Set sh = CreateObject("WScript.Shell")
Set l = sh.CreateShortcut(sh.SpecialFolders("Desktop") & "\\{name}.lnk")
l.TargetPath = "{pyw}"
l.Arguments = """{HERE}"""
l.WorkingDirectory = "{HERE.parent}"
l.IconLocation = "{icon},0"
l.Description = "vdW Stack Designer (FlakeIdentifier, Young Lab)"
l.Save
WScript.Echo "shortcut: " & sh.SpecialFolders("Desktop") & "\\{name}.lnk"
'''
        f = Path(os.environ.get("TEMP", ".")) / "flakeid_designer_shortcut.vbs"
        try:
            f.write_text(vbs, encoding="mbcs")
            r = subprocess.run(["cscript", "//nologo", str(f)], capture_output=True, text=True, timeout=60)
            print((r.stdout + r.stderr).strip())
            if r.returncode == 0:
                return 0
        except (OSError, subprocess.SubprocessError) as e:
            print(f"(VBScript not usable: {e})")
        print(f"\nThe shortcut could not be made.  Start the designer with:  \"{pyw}\" \"{HERE}\"")
        return 1
    if sys.platform == "darwin":
        # the main Applications folder (what Finder shows as "Applications") when it can be written, else ~/Applications
        apps = Path("/Applications") if os.access("/Applications", os.W_OK) else Path.home() / "Applications"
        old = Path.home() / "Applications" / f"{name}.app"
        app = apps / f"{name}.app"
        if old != app and old.is_dir():
            import shutil
            shutil.rmtree(old, ignore_errors=True)
        mac = app / "Contents" / "MacOS"
        res = app / "Contents" / "Resources"
        mac.mkdir(parents=True, exist_ok=True)
        res.mkdir(parents=True, exist_ok=True)
        (mac / "launcher").write_text(f'''#!/bin/sh
# starts the designer server in the background (it stops by itself when unused) and opens the designer
"{sys.executable}" "{HERE}" >/dev/null 2>&1 &
''')
        os.chmod(mac / "launcher", 0o755)
        icns = HERE / "static" / "designer.icns"
        if icns.is_file():
            (res / "designer.icns").write_bytes(icns.read_bytes())
        (app / "Contents" / "Info.plist").write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>{name}</string><key>CFBundleDisplayName</key><string>{name}</string>
<key>CFBundleIdentifier</key><string>edu.ucsb.younglab.flakeid-designer</string>
<key>CFBundleExecutable</key><string>launcher</string><key>CFBundleIconFile</key><string>designer</string>
<key>CFBundlePackageType</key><string>APPL</string><key>CFBundleVersion</key><string>{VERSION}</string>
<key>LSUIElement</key><true/>
</dict></plist>
''')
        desk = Path.home() / "Desktop" / f"{name}.app"
        if desk.is_symlink() and Path(os.readlink(desk)) != app:
            desk.unlink()
        if not desk.exists():
            try:
                desk.symlink_to(app)
            except OSError:
                pass
        subprocess.run(["touch", str(app)])
        print(f"\nInstalled: {app}\n(and a link on the Desktop).  Drag it to the Dock to keep it there.\n")
        return 0
    print("shortcuts: Windows and macOS only")
    return 1


UPDATE = dict(lock=threading.Lock(), state="idle", log=[], result=None, error=None)   # the header's Update… (update.py)


def _run_update(sha):
    def log(m):
        with UPDATE["lock"]:
            UPDATE["log"].append(m)
    try:
        if not sha:
            sha = UPD.latest()["sha"]
        r = UPD.install(sha, log=log)
        with UPDATE["lock"]:
            UPDATE.update(state="done", result=r)
    except Exception as e:                           # (UpdateError: its message; anything else: shown as it is)
        with UPDATE["lock"]:
            UPDATE.update(state="failed", error=str(e) if isinstance(e, UPD.UpdateError) else f"{type(e).__name__}: {e}")


def _restart(srv):
    """The browser version again, in this process (os.execv: same command and folder, --no-browser: the open page
    reloads itself), once the answer to the page is out."""
    time.sleep(0.6)
    try:
        srv.socket.close()                           # (the port free for the new one)
    except OSError:
        pass
    argv = [a for a in sys.argv if a != "--no-browser"] + ["--no-browser"]
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} restarting after the update: {sys.executable} {' '.join(argv)}\n")
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable] + argv)


def make_server(port, store):
    """The designer's server with its state (the browser version and the app both start it here: the app once lacked
    the KLayout export's, 2026-10-14)."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.store, srv.last = store, time.time()
    srv.kl_jobs, srv.kl_folders = {}, set()
    DISK.root = cache_dir()
    AFM_DISK.root = DISK.root / "afm" if DISK.root else None
    KEPT_DISK.root = DISK.root / "kept" if DISK.root else None
    store.start_scanner()
    KP.KEPT.initial(store)                               # (flakes kept before: their pictures into the runs' kept/)
    # stacks kept on this computer (offline.py): the share probed, packs made / refreshed, changes uploaded
    srv.offline = OF.Offline(store, sys.modules[__name__])
    srv.offline.start()
    return srv


def native_mac():
    """On an Apple-silicon Mac started under Rosetta (the Desktop app's launcher is a shell script, which macOS can
    open as Intel): the same command again as arm64, so builds can import this Python's arm64-only numpy / OpenCV.
    Only when that Python imports numpy natively; FLAKEID_DESIGNER_NATIVE stops a second try."""
    if sys.platform != "darwin" or getattr(sys, "frozen", False) or os.environ.get("FLAKEID_DESIGNER_NATIVE"):
        return
    try:
        r = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=10)
        if r.stdout.strip() != "1":
            return
        cmd = ["/usr/bin/arch", "-arm64", sys.executable]
        if subprocess.run(cmd + ["-c", "import numpy"], capture_output=True, timeout=120).returncode != 0:
            return
    except (OSError, subprocess.SubprocessError):
        return
    os.environ["FLAKEID_DESIGNER_NATIVE"] = "1"
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} running under Rosetta: starting again as arm64\n")
    sys.stderr.flush()
    os.execv(cmd[0], cmd + sys.argv)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flakeid_designer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=[], help="a folder to look for scan runs in (repeat; remembered)")
    ap.add_argument("--data", help="shared folder for designs and keep/discard decisions (remembered)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--install-shortcut", action="store_true")
    ap.add_argument("--offline", action="store_true", help="act as if the share were not reachable: only the stacks kept "
                    "on this computer (tests; also FLAKEID_DESIGNER_OFFLINE=1)")
    a = ap.parse_args(argv)
    if argv is None:
        native_mac()
    if a.install_shortcut:
        return install_shortcut()
    if a.offline:
        os.environ["FLAKEID_DESIGNER_OFFLINE"] = "1"
    if sys.stderr is None or not sys.stderr.isatty():
        # started from the shortcut (no console): errors go to a log file to send when something does not work
        try:
            f = open(LOG, "a", encoding="utf-8", buffering=1)
            f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {APP} {VERSION}, Python {sys.version.split()[0]}, {sys.platform}\n")
            sys.stdout = sys.stderr = f
        except OSError:
            pass
    cfg = load_config()
    roots, data, library = start_library(cfg, a.root)
    data = a.data or data
    port = a.port or cfg.get("port") or DEFAULT_PORT
    if a.root or a.data:
        cfg.update(roots=roots, data_dir=data)
        save_config(cfg)
    url = f"http://127.0.0.1:{port}/"
    if running(port):                    # already running (another shortcut click): just open it
        if not a.no_browser:
            webbrowser.open(url)
        return 0
    store = Store(roots, data, library, added=cfg.get("added_runs"))
    threading.Thread(target=store.scan, daemon=True).start()
    srv = make_server(port, store)
    print(f"vdW Stack Designer: {url}  (roots: {', '.join(roots) or 'none yet'}; designs and decisions: {data or 'none yet'})")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    def idle():
        while True:
            time.sleep(30)
            if time.time() - srv.last > IDLE_S and not any(j.state == "running" for j in srv.kl_jobs.values()):
                srv.shutdown()
                return
    threading.Thread(target=idle, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
