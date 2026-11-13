"""Offline stacks (2026-10-22): stacks kept on this computer, to work on them away from the share (a laptop at home)
and have the changes saved to the share when it is back.

Per library, in this computer's app data folder (offline_root(): Windows %LOCALAPPDATA%\\flakeid-designer\\offline,
Mac ~/Library/Application Support/flakeid-designer/offline; FLAKEID_DESIGNER_OFFLINE_DIR for tests), never on the share:

    <library>/stacks/<stack>/      meta.json (state, the drive copy it is based on: size, time, sha1, rev), stack.json
                                   (the stack as the page uses it: the local version), drive_copy.json (the drive's file
                                   as it was), files/ (the stack's pictures and attachments: step photos, final pictures)
    <library>/runs/<run>/          a small bundle per run with only the kept stacks' flakes (same ids, all their data):
                                   their 100x pictures (cut out of the shared sheets of older bundles), their atlas
                                   cut-out and thumbnail cut out of the shared sheets (refs rewritten)
    <library>/newfiles/            pictures and files added while offline (uploaded first when the share is back)
    <library>/review_queue.jsonl   keep / discard decisions made offline (flakes on this computer only)
    <library>/snap.json            decisions, drawn terraces and imported flakes of the kept stacks' flakes
    <library>/state.json           when stacks were opened here (the automatic choice), last upload

Which stacks: "Keep offline" per stack (settings file ~/.flakeid_designer.json: offline_keep {stack id: true/false}), and
by default (offline_auto, on unless set false) every active stack (not finalized / done / archived) and every stack
opened on this computer in the last 30 days.  Packs are made and refreshed in the background while the share is
reachable (one look at the stack files and the runs' bundle.json: a pack is made again only when they changed).

The share is probed in a thread with a timeout (a network drive can hang on a stat); unreachable -> offline: the
designer serves the kept stacks and their flakes from here, saves go to the pack (marked not uploaded).  When the share
is back the changed stacks are uploaded (new files first, then the stack with the usual safe write) if the drive's copy
is still the one the pack was taken from; if it changed meanwhile, both are kept: the laptop's version is saved as a
separate stack ("<name> (offline edits <date>)") and the pack follows the drive's version.  FLAKEID_DESIGNER_OFFLINE=1 (or
--offline) forces offline mode (tests).

Pictures are cut out by a helper process (OpenCV or Pillow: the designer itself stays standard library only); without
one the shared sheets are copied whole (a few MB more).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = "flakeid-designer"
DONE = {"finalized", "final", "done", "archived", "finished"}     # stack status: not kept automatically
RECENT_DAYS = 30
MAX_FILE = 25 * 2 ** 20                          # attachments bigger than this stay on the share
VIEW_KEYS = (("img", "nat"), ("img_look", "look"), ("img_c8", "c8"), ("img_dt", "dt"), ("img_ir", "ir"), ("img_mos", "mos"),
             ("img_mos_look", "moslook"))
HASHED = re.compile(r"_(?=[0-9]*[a-f])[0-9a-f]{10}\.[a-z]+$")
STACKFILE_PART = re.compile(r"^[A-Za-z0-9_.~@+-]{1,200}$")


def _env_f(name, default):
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


PROBE_S = _env_f("FLAKEID_DESIGNER_PROBE_S", 20.0)     # how often the share is looked at
PACK_S = _env_f("FLAKEID_DESIGNER_PACK_S", 300.0)      # how often the kept stacks are checked for changes
PROBE_TIMEOUT = _env_f("FLAKEID_DESIGNER_PROBE_TIMEOUT", 5.0)


def forced():
    return str(os.environ.get("FLAKEID_DESIGNER_OFFLINE") or "").strip().lower() in ("1", "true", "yes", "on")


def offline_root():
    env = os.environ.get("FLAKEID_DESIGNER_OFFLINE_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP
    elif sys.platform == "darwin":                  # (not Caches: the system may empty that, and edits live here)
        base = Path.home() / "Library" / "Application Support" / APP
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / APP
    return base / "offline"


def sha1(b):
    return hashlib.sha1(b).hexdigest()


def safe_name(s, n=80):
    return re.sub(r"[^A-Za-z0-9_.@+-]+", "_", str(s)).strip("._")[:n] or "x"


def key_of(sid):
    """A folder name for a stack id (ids may hold ':' or '~')."""
    return safe_name(sid, 60) + "_" + sha1(str(sid).encode("utf-8"))[:6]


def _log(msg):
    try:
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} offline: {msg}\n")
    except Exception:
        pass


def _replace(src, dst, tries=8):
    """os.replace with retries (Windows: a virus scanner or the indexer can hold a file a moment)."""
    for i in range(tries):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.15 * (i + 1))


def _part_name(name):
    """A short temporary name next to the file (Windows: paths over 260 characters fail)."""
    return f".{sha1(name.encode('utf-8'))[:8]}{os.getpid() % 100000}_{threading.get_ident() % 100000}.part"


def write_atomic(p, data):
    """Bytes into p: a new file renamed over the old one (never half a file)."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(_part_name(p.name))
    try:
        with open(str(tmp), "wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        _replace(tmp, p)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_json(p, obj, indent=1):
    write_atomic(p, json.dumps(obj, indent=indent).encode("utf-8"))


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def copy_atomic(src, dst):
    """A file copied under a temporary name and renamed (to the share: never half a file there)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(_part_name(dst.name))
    try:
        shutil.copyfile(str(src), str(part))
        _replace(part, dst)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def dir_size(d):
    n = 0
    try:
        for root, _, files in os.walk(str(d)):
            for f in files:
                try:
                    n += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return n


def collect_urls(obj, out=None):
    """The stack's own files referenced in it (/stackimage/..., /stackfile/...), without ?query."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            collect_urls(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_urls(v, out)
    elif isinstance(obj, str) and obj.startswith(("/stackimage/", "/stackfile/")):
        out.add(urllib.parse.urlparse(obj).path)
    return out


def active(d):
    st = str((d or {}).get("status") or "planning").lower()
    return st not in DONE and not (d or {}).get("archived")


def flake_of(tid):
    """'run:flake.a' -> 'run:flake' (a terrace id's flake)."""
    return tid.rsplit(".", 1)[0] if isinstance(tid, str) and "." in tid else None


class Probe:
    """Is a folder reachable?  Asked in a thread with a timeout: a network drive that dropped can hang a stat for
    minutes.  While the last probe still hangs, the answer is no (no second thread)."""

    def __init__(self):
        self.t = None

    def check(self, paths, timeout=PROBE_TIMEOUT):
        if self.t is not None and self.t.is_alive():
            return False
        box = {}

        def run():
            try:
                box["ok"] = all(os.path.isdir(p) for p in paths)
            except Exception:
                box["ok"] = False
        self.t = threading.Thread(target=run, daemon=True, name="drive probe")
        self.t.start()
        self.t.join(timeout)
        return False if self.t.is_alive() else bool(box.get("ok"))


class Offline:
    """The offline packs of the current library, the share's reachability and the uploads.  store: the designer's
    Store; D: the designer module (settings, user name, the run-file cache, ...)."""

    def __init__(self, store, D):
        self.S, self.D = store, D
        self.lock = threading.RLock()      # the packs' files (meta, stack.json, new files, queue): short holds only
        self.work = threading.Lock()       # one packing / upload job at a time
        self.forced = forced()
        self.offline = self.forced
        self.probe = Probe()
        self.wake = threading.Event()
        self.want_sync = self.want_pack = False
        self.busy, self.msg = None, ""
        self.last_pack_t = 0.0
        self.last_probe = None
        self.started = False
        self._fcache = {}                  # folder stack id -> (sig, page form)

    # ------------------------------------------------------------------ where
    def lib_dir(self):
        key = str(self.S.data or (self.S.roots[0] if self.S.roots else "none"))
        p = Path(key)
        name = p.parent.name if p.name.lower() == "_designer_data" else p.name
        if name.lower() == "scanresults":
            name = (p.parent.parent.name if p.name.lower() == "_designer_data" else p.parent.name) or name
        k = key.lower() if sys.platform == "win32" else key
        return offline_root() / f"{safe_name(name, 40)}_{sha1(k.encode('utf-8'))[:8]}"

    def sdir(self, sid):
        return self.lib_dir() / "stacks" / key_of(sid)

    def all_meta(self):
        out = []
        d = self.lib_dir() / "stacks"
        try:
            subs = sorted(x for x in d.iterdir() if x.is_dir())
        except OSError:
            return out
        for x in subs:
            m = read_json(x / "meta.json")
            if isinstance(m, dict) and m.get("sid"):
                out.append((x, m))
        return out

    def meta(self, sid):
        m = read_json(self.sdir(sid) / "meta.json")
        return m if isinstance(m, dict) else None

    def update_meta(self, sid, fn):
        """Read, change (fn(m) -> m) and write a pack's meta under the lock."""
        with self.lock:
            m = self.meta(sid)
            m = fn(m)
            if m is not None:
                write_json(self.sdir(sid) / "meta.json", m)
            return m

    def state(self):
        s = read_json(self.lib_dir() / "state.json", {})
        return s if isinstance(s, dict) else {}

    def put_state(self, fn):
        with self.lock:
            s = fn(self.state())
            try:
                write_json(self.lib_dir() / "state.json", s)
            except OSError as e:
                _log(f"state not written: {e}")

    # ------------------------------------------------------------------ settings
    def auto_on(self):
        v = self.D.load_config().get("offline_auto")
        return True if v is None else bool(v)

    def keep_map(self):
        k = self.D.load_config().get("offline_keep")
        return k if isinstance(k, dict) else {}

    def set_auto(self, on):
        cfg = self.D.load_config()
        cfg["offline_auto"] = bool(on)
        self.D.save_config(cfg)
        self.kick(pack=True)

    def set_keep(self, sid, v):
        cfg = self.D.load_config()
        k = cfg.get("offline_keep") if isinstance(cfg.get("offline_keep"), dict) else {}
        if v is None:
            k.pop(sid, None)
        else:
            k[sid] = bool(v)
        cfg["offline_keep"] = k
        self.D.save_config(cfg)
        self.kick(pack=True)

    def opened(self, sid):
        """A stack opened in the page (its lock note): remembered for the automatic choice."""
        if not sid:
            return
        now = time.time()
        st = self.state()
        if now - float((st.get("opened") or {}).get(sid) or 0) < 3600:
            return
        new = sid not in (st.get("opened") or {})

        def f(s):
            s.setdefault("opened", {})[sid] = now
            return s
        self.put_state(f)
        if new and self.auto_on():
            self.kick(pack=True)

    # ------------------------------------------------------------------ the background job
    def start(self):
        if self.started:
            return
        self.started = True
        if not self.forced:
            self.offline = not self.reachable()
            if self.offline:
                _log("the share is not reachable: offline (the stacks kept on this computer)")
        threading.Thread(target=self._loop, daemon=True, name="offline stacks").start()

    def kick(self, pack=False, sync=False):
        self.want_pack = self.want_pack or pack
        self.want_sync = self.want_sync or sync
        self.wake.set()

    def reachable(self):
        if self.S.data is not None:
            paths = [str(Path(self.S.data).parent)]
        elif self.S.roots:
            paths = [str(self.S.roots[0])]
        else:
            return True
        ok = self.probe.check(paths)
        self.last_probe = time.time()
        return ok

    def _loop(self):
        first = True
        while True:
            try:
                self._tick(first)
            except Exception as e:                       # (never ends the thread)
                _log(f"{type(e).__name__}: {e}")
                self.busy = None
            first = False
            self.wake.wait(PROBE_S)
            self.wake.clear()

    def _tick(self, first):
        if self.forced:
            self.offline = True
            return
        up = self.reachable()
        was = self.offline
        if not up:
            if not was:
                _log("the share is not reachable: offline (the stacks kept on this computer)")
            self.offline = True
            return
        if was or first:
            # back (or starting): the runs found again, then the changes made here uploaded before the page is served
            # from the share again
            if was:
                _log("the share is back: uploading the changes made here")
                try:
                    self.S.scan()
                except Exception as e:
                    _log(f"run search failed: {e}")
            else:
                self.S.first_scan.wait(120)
        self.sync()
        self.offline = False
        if was or first or self.want_pack or time.time() - self.last_pack_t >= PACK_S:
            self.want_pack = False
            self.pack_all()

    # ------------------------------------------------------------------ which stacks
    def _folder_page(self, sid):
        """A folder stack as the page uses it (read again only when its files changed), or None."""
        try:
            sig = self.S.folders.sig(sid)
        except Exception:
            return None
        hit = self._fcache.get(sid)
        if hit and hit[0] == sig:
            return hit[1]
        d = self.S.folders.read(sid)
        if d is not None:
            self._fcache[sid] = (sig, d)
        return d

    def candidates(self):
        """{stack id: dict(kind, data, why)} of the stacks to keep on this computer."""
        auto, keep = self.auto_on(), self.keep_map()
        opened = self.state().get("opened") or {}
        now = time.time()
        out = {}

        def why(sid, d, at=0.0):
            k = keep.get(sid)
            if k is True:
                return "kept"
            if k is False or not auto:
                return None
            if active(d):
                return "active"
            if now - max(float(opened.get(sid) or 0), at) < RECENT_DAYS * 86400:
                return "opened recently"
            return None
        if self.S.data is not None and (Path(self.S.data) / "stacks").is_dir():
            for it in self.S.collection("stacks", skip_under=True)[1]:
                d = it.get("data")
                if isinstance(d, dict):
                    w = why(it["id"], d)
                    if w:
                        out[it["id"]] = dict(kind="data", data=d, why=w)
        for r in self.S.folders.recents():
            sid = r["id"]
            if keep.get(sid) is False or (keep.get(sid) is None and not auto):
                continue
            try:
                if not Path(r["path"]).is_dir():
                    continue
            except OSError:
                continue
            d = self._folder_page(sid)
            if d is None:
                continue
            w = why(sid, d, float(r.get("at") or 0) / 1000.0)
            if w:
                out[sid] = dict(kind="folder", data=d, why=w)
        return out

    # ------------------------------------------------------------------ packing
    def pack_all(self):
        with self.work:
            self.busy = "packing"
            try:
                want = self.candidates()
                for sid, c in want.items():
                    try:
                        self.pack_one(sid, c)
                    except Exception as e:
                        _log(f"stack {sid} not packed: {type(e).__name__}: {e}")

                        def f(m, e=e):
                            if m is not None:
                                m["error"] = f"{type(e).__name__}: {e}"
                            return m
                        self.update_meta(sid, f)
                for sd, m in self.all_meta():
                    if m["sid"] not in want and m.get("state") not in ("pending",):
                        with self.lock:
                            m2 = read_json(sd / "meta.json") or {}
                            if m2.get("state") != "pending":
                                shutil.rmtree(str(sd), ignore_errors=True)
                self.update_runs()
                self.update_snapshots()
                self.msg = ""
            finally:
                self.busy = None
                self.last_pack_t = time.time()

    def stack_path(self, sid, m):
        if m.get("kind") == "folder":
            return Path(m["folder"]) / m["file"]
        return Path(self.S.data) / "stacks" / f"{sid}.json"

    def _drive_page(self, sid, kind, raw, st):
        """The page's form of the drive's stack file (raw bytes, its stat)."""
        if kind == "folder":
            d = self.S.folders.read(sid)
            if d is None:
                raise ValueError("the stack file could not be read")
            return d
        j = json.loads(raw.decode("utf-8"))
        return dict(j, _sig=f"{j.get('rev', 0)}-{st.st_mtime_ns}-{st.st_size}")

    def pack_one(self, sid, c):
        kind = c["kind"]
        sd = self.sdir(sid)
        m = self.meta(sid) or dict(sid=sid, kind=kind, state="ok", files={}, created=time.time())
        if kind == "folder":
            w = self.S.folders.where(sid)
            if not w:
                return
            m["folder"], m["file"] = str(w[0]), w[1]
        path = self.stack_path(sid, m)
        st = path.stat()
        fsig = f"{st.st_size}-{st.st_mtime_ns}"
        base = None
        if m.get("state") != "pending" and ((m.get("base") or {}).get("fsig") != fsig or not (sd / "stack.json").is_file()):
            raw = path.read_bytes()
            page = self._drive_page(sid, kind, raw, st)
            base = dict(fsig=fsig, size=st.st_size, mtime_ns=st.st_mtime_ns, sha1=sha1(raw), rev=page.get("rev", 0),
                        sig=page.get("_sig"), at=time.time())
            with self.lock:
                cur = self.meta(sid) or {}
                if cur.get("state") != "pending":          # (not edited here in the meantime)
                    write_atomic(sd / "drive_copy.json", raw)
                    write_json(sd / "stack.json", page)
                else:
                    base = None
        page = read_json(sd / "stack.json") or {}
        tids = [L.get("tid") for L in ((page.get("plan") or {}).get("layers") or []) if isinstance(L, dict) and L.get("tid")]
        flakes = sorted({f for f in (flake_of(t) for t in tids) if f})
        urls = collect_urls(page)
        man_ids = {f.split(":", 1)[1] for f in flakes if f.startswith("manual:")}
        if man_ids:
            for it in self.S.collection("manual")[1]:
                if it["id"] in man_ids:
                    collect_urls(it.get("data"), urls)
        files = self.mirror(sd, dict(m.get("files") or {}), urls)

        def f(cur):
            cur = cur or m
            if base is not None:
                cur["base"] = base
                cur["packed_at"] = time.time()
                if cur.get("state") not in ("conflict", "uploaded"):
                    cur["state"] = "ok"
            cur.update(sid=sid, kind=kind, name=page.get("name"), status=page.get("status"), tids=tids, flakes=flakes,
                       files=files, why=c.get("why"), error=None, checked_at=time.time())
            if kind == "folder":
                cur["folder"], cur["file"] = m["folder"], m["file"]
            cur.setdefault("created", time.time())
            return cur
        self.update_meta(sid, f)

    def new_index(self):
        idx = read_json(self.lib_dir() / "newfiles" / "index.json", {})
        return idx if isinstance(idx, dict) else {}

    def mirror(self, sd, files, urls):
        """The stack's files (urls) copied into its pack (only those not there yet); -> {url: dict(f, size)}."""
        fd = sd / "files"
        out = {}
        newi = self.new_index()
        for u in sorted(urls):
            e = files.get(u)
            if e and (fd / e["f"]).is_file():
                out[u] = e
                continue
            name = sha1(u.encode("utf-8"))[:12] + "_" + safe_name(Path(u).name, 80)
            try:
                ne = newi.get(u)
                src = (self.lib_dir() / "newfiles" / ne["f"]) if ne else None
                if src is None or not src.is_file():
                    src = self.drive_path(u)
                    if src is None:
                        continue
                    sz = src.stat().st_size
                    if sz > MAX_FILE:
                        out[u] = dict(f=None, size=sz, skipped="too big")
                        continue
                fd.mkdir(parents=True, exist_ok=True)
                copy_atomic(src, fd / name)
                out[u] = dict(f=name, size=(fd / name).stat().st_size)
            except OSError as e:
                out[u] = dict(f=None, error=str(e))
        keep = {e["f"] for e in out.values() if e.get("f")}
        try:
            for q in fd.iterdir():
                if q.is_file() and q.name not in keep:
                    q.unlink()
        except OSError:
            pass
        return out

    def drive_path(self, url):
        """Where a stack's file (its URL) is on the share, or None."""
        p = urllib.parse.unquote(urllib.parse.urlparse(url).path)
        if p.startswith("/stackimage/"):
            name = p[len("/stackimage/"):]
            if self.S.data is not None and re.match(r"^[A-Za-z0-9_-]{1,80}\.(jpg|png)$", name):
                return Path(self.S.data) / "stack_images" / name
            return None
        if p.startswith("/stackfile/"):
            parts = p[len("/stackfile/"):].split("/")
            if len(parts) < 2 or any(q in ("", ".", "..") or "\\" in q or ":" in q for q in parts):
                return None
            if self.D.SD.is_folder_id(parts[0]):
                w = self.S.folders.where(parts[0])
                if not w:
                    m = self.meta(parts[0])
                    if not m or not m.get("folder"):
                        return None
                    w = [Path(m["folder"]), m.get("file")]
                return Path(w[0]).joinpath(*parts[1:])
            if self.S.data is None or not all(STACKFILE_PART.match(q) for q in parts):
                return None
            return Path(self.S.data) / "stack_files" / Path(*parts)
        return None

    # ---- runs: one small bundle per run with the kept stacks' flakes
    def _run_name_map(self):
        with self.S.lock:
            runs = dict(self.S.runs)
        return runs, {v["doc"].get("run"): k for k, v in runs.items()}

    def update_runs(self):
        runs, by_name = self._run_name_map()
        need = {}
        for _, m in self.all_meta():
            for fid in m.get("flakes") or []:
                if fid.startswith("manual:"):
                    continue
                rid = by_name.get(fid.split(":", 1)[0])
                if rid is None:
                    continue
                need.setdefault(rid, set()).add(fid)
        rroot = self.lib_dir() / "runs"
        for rid, fids in need.items():
            try:
                self.build_run(rid, runs[rid], fids)
            except Exception as e:
                _log(f"run {rid} not packed: {type(e).__name__}: {e}")
        # runs no stack needs any more (only runs found on the share now: one not found keeps its pack)
        wanted = {safe_name(r, 120) for r in need}
        try:
            for d in rroot.iterdir():
                if d.is_dir() and d.name not in wanted:
                    rm = read_json(d / "meta.json") or {}
                    if rm.get("rid") in runs or not rm:
                        shutil.rmtree(str(d), ignore_errors=True)
        except OSError:
            pass

    def _read_share(self, p):
        """A run file from the share, through the designer's copies on this computer (DISK) when unchanged."""
        st = p.stat()
        disk = getattr(self.D, "DISK", None)
        b = disk.get(p, st) if disk is not None else None
        if b is None:
            b = p.read_bytes()
            if disk is not None:
                disk.put(p, st, b)
        return b, st

    def build_run(self, rid, r, fids):
        src_dir = Path(r["dir"])
        bj = src_dir / "bundle.json"
        st = bj.stat()
        ssig = f"{st.st_size}-{st.st_mtime_ns}"
        rd = self.lib_dir() / "runs" / safe_name(rid, 120)
        rm = read_json(rd / "meta.json") or {}
        if rm.get("source") == ssig and sorted(rm.get("flakes") or []) == sorted(fids) and (rd / "bundle.json").is_file() \
                and rm.get("ver") == 1:
            return
        data, st = self._read_share(bj)
        B = json.loads(data.decode("utf-8"))
        rd.mkdir(parents=True, exist_ok=True)
        fl = [json.loads(json.dumps(f)) for f in B.get("flakes") or [] if f.get("id") in fids]
        atl_src, th_src = B.get("atlases") or [], B.get("thumbs") or []

        def hname(prefix, fid, parts, ext, src_name):
            if not HASHED.search(str(src_name)):
                parts = list(parts) + [ssig]                 # (an older name, not its content's: rebuilt -> new cut-out)
            return f"{prefix}_{safe_name(fid.split(':')[-1], 60)}_{sha1('|'.join(map(str, parts)).encode('utf-8'))[:10]}.{ext}"
        jobs, copies = [], {}
        for f in fl:
            fid = f["id"]
            for key, tag in VIEW_KEYS:
                v = f.get(key)
                if isinstance(v, str) and v:
                    copies[v] = src_dir / v
                elif isinstance(v, dict) and v.get("s"):
                    box = [v.get("x", 0), v.get("y", 0), v.get("w", 0), v.get("h", 0)]
                    jobs.append(dict(src=str(src_dir / v["s"]), src_name=v["s"], box=box, fmt="jpg", kind="view", flake=fid,
                                     key=key, dst=hname("x100_" + tag, fid, [v["s"]] + box, "jpg", v["s"])))
            a, rc = f.get("atlas"), f.get("rect")
            if isinstance(a, int) and 0 <= a < len(atl_src) and isinstance(rc, list) and len(rc) == 4:
                jobs.append(dict(src=str(src_dir / atl_src[a]), src_name=atl_src[a], box=rc, fmt="webp", kind="atlas", flake=fid,
                                 dst=hname("atlas", fid, [atl_src[a]] + list(rc), "webp", atl_src[a])))
            T = (f.get("frame") or {}).get("thumb")
            if isinstance(T, list) and len(T) == 5 and isinstance(T[0], int) and 0 <= T[0] < len(th_src):
                tf = th_src[T[0]].get("file")
                jobs.append(dict(src=str(src_dir / tf), src_name=tf, box=list(T[1:]), fmt="jpg", kind="thumb", flake=fid,
                                 dst=hname("thumb", fid, [tf] + list(T[1:]), "jpg", tf)))
        # cut-outs already made (named by their source and place) are kept; the others made by the helper process
        sizes = dict((rm.get("cut") or {}))
        todo = [j for j in jobs if not ((rd / j["dst"]).is_file() and j["dst"] in sizes)]
        res = self.run_crops(todo, rd) if todo else {}
        for j in todo:
            v = res.get(j["dst"])
            if isinstance(v, list) and len(v) == 2 and (rd / j["dst"]).is_file():
                sizes[j["dst"]] = v
        new_atl, new_th, names = [], [], set()

        def atl_index(n):
            if n not in new_atl:
                new_atl.append(n)
            return new_atl.index(n)
        th_index = {}
        by = {}
        for j in jobs:
            by.setdefault(j["flake"], []).append(j)
        for f in fl:
            for j in by.get(f["id"], []):
                ok = sizes.get(j["dst"])
                if j["kind"] == "view":
                    if ok:
                        f[j["key"]] = j["dst"]
                        names.add(j["dst"])
                    else:                                    # (no cut-out: the whole sheet)
                        copies[j["src_name"]] = Path(j["src"])
                elif j["kind"] == "atlas":
                    if ok:
                        f["atlas"] = atl_index(j["dst"])
                        f["rect"] = [0, 0, ok[0], ok[1]]
                        f["atlas_w"] = ok[0]
                        names.add(j["dst"])
                    else:
                        f["atlas"] = atl_index(j["src_name"])
                        copies[j["src_name"]] = Path(j["src"])
                elif j["kind"] == "thumb":
                    T = f["frame"]["thumb"]
                    if ok:
                        k = len(new_th)
                        new_th.append(dict(file=j["dst"], w=ok[0], h=ok[1]))
                        f["frame"] = dict(f["frame"], thumb=[k, 0, 0, ok[0], ok[1]])
                        names.add(j["dst"])
                    else:
                        tf = j["src_name"]
                        if tf not in th_index:
                            th_index[tf] = len(new_th)
                            new_th.append(dict(th_src[T[0]]))
                        f["frame"] = dict(f["frame"], thumb=[th_index[tf]] + list(T[1:]))
                        copies[tf] = Path(j["src"])
        missing = []
        for n, src in copies.items():
            dst = rd / n
            try:
                if dst.is_file() and (HASHED.search(n) or dst.stat().st_size == src.stat().st_size):
                    names.add(n)
                    continue
                copy_atomic(src, dst)
                names.add(n)
            except OSError as e:
                missing.append(f"{n} ({type(e).__name__})")
        if missing:
            _log(f"run {rid}: {len(missing)} picture(s) not copied: {', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}")
        used = {f.get("scan") for f in fl}
        scans = {k: dict(v, file=None) for k, v in (B.get("scans") or {}).items() if k in used}   # (no chip overviews)
        det_name = None
        if B.get("detail"):
            try:
                db, _ = self._read_share(src_dir / B["detail"])
                dj = json.loads(db.decode("utf-8"))
                dj = dict(dj, flakes={k: v for k, v in (dj.get("flakes") or {}).items() if k in fids})
                dbytes = json.dumps(dj, separators=(",", ":")).encode("utf-8")
                det_name = f"detail_off_{sha1(dbytes)[:10]}.json"
                if not (rd / det_name).is_file():
                    write_atomic(rd / det_name, dbytes)
                names.add(det_name)
            except (OSError, ValueError) as e:
                _log(f"{rid}: the step lines not packed ({e})")
        out = dict(B, flakes=fl, atlases=new_atl, thumbs=new_th, scans=scans, detail=det_name, extra=None,
                   files=sorted(names), n_flakes=len(fl), offline=dict(source=ssig, built=time.strftime("%Y-%m-%d %H:%M")))
        # bytes per flake (the stack's size in the page): its own files, and its share of whole sheets
        own = {}
        for f in fl:
            ns = [f.get(k) for k, _ in VIEW_KEYS if isinstance(f.get(k), str)]
            if isinstance(f.get("atlas"), int) and f["atlas"] < len(new_atl):
                ns.append(new_atl[f["atlas"]])
            T = (f.get("frame") or {}).get("thumb")
            if isinstance(T, list) and T and T[0] < len(new_th):
                ns.append(new_th[T[0]]["file"])
            own[f["id"]] = ns
        users = {}
        for fid, ns in own.items():
            for n in ns:
                users.setdefault(n, set()).add(fid)
        fsize = {}
        for fid, ns in own.items():
            tot = 0
            for n in set(ns):
                try:
                    tot += (rd / n).stat().st_size / max(1, len(users.get(n, ())))
                except OSError:
                    pass
            fsize[fid] = int(tot)
        write_json(rd / "bundle.json", out, indent=None)
        write_json(rd / "meta.json", dict(ver=1, rid=rid, source=ssig, flakes=sorted(fids), files=sorted(names),
                                          cut={k: v for k, v in sizes.items() if k in names}, sizes=fsize,
                                          doc=r.get("doc") or {}, built=time.time(), cropped=bool(res) or not todo))
        keep = names | {"bundle.json", "meta.json"}
        for q in rd.iterdir():
            if q.is_file() and q.name not in keep and not q.name.startswith("."):
                try:
                    q.unlink()
                except OSError:
                    pass
        _log(f"run {rid}: {len(fl)} flake(s) on this computer ({dir_size(rd) / 1e6:.1f} MB)")

    def run_crops(self, jobs, rd):
        """Cut-outs by a helper process (OpenCV or Pillow) -> {dst: [w, h]} of those made ({} if none could be)."""
        if not jobs or os.environ.get("FLAKEID_DESIGNER_NOCROP"):
            return {}
        jf = rd / ".crop_jobs.json"
        write_json(jf, dict(dir=str(rd), jobs=[dict(src=j["src"], box=j["box"], fmt=j["fmt"], dst=j["dst"]) for j in jobs]))
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--helper", "crop", str(jf)]
        else:
            cmd = [self.D.child_python(), "-u", str(HERE / "offline.py"), "--crop", str(jf)]
        kw = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900, stdin=subprocess.DEVNULL)
        if sys.platform == "win32":
            kw["creationflags"] = 0x00004000 | 0x08000000    # below normal priority, no window
        try:
            r = subprocess.run(cmd, **kw)
        except Exception as e:
            _log(f"pictures not cut out ({type(e).__name__}: {e}): the sheets are copied whole")
            return {}
        finally:
            try:
                jf.unlink()
            except OSError:
                pass
        line = next((ln for ln in reversed((r.stdout or "").splitlines()) if ln.startswith("RESULT ")), None)
        try:
            res = json.loads(line[7:]) if line else {}
        except ValueError:
            res = {}
        if not isinstance(res, dict) or res.get("error"):
            _log(f"pictures not cut out ({(res or {}).get('error') if isinstance(res, dict) else ''}"
                 f"{(r.stderr or '').strip()[-200:]}): the sheets are copied whole")
            return {}
        return res

    def update_snapshots(self):
        """Decisions, drawn terraces and imported flakes of the kept stacks' flakes (for the page offline)."""
        flakes, tids = set(), set()
        for _, m in self.all_meta():
            flakes.update(m.get("flakes") or [])
            tids.update(m.get("tids") or [])
        snap = dict(review={}, terraces=[], manual=[], at=time.time())
        if self.S.data is not None:
            try:
                latest = self.S.review_state()[2]
                for k, e in latest.items():
                    if flake_of(k) in flakes or k in tids:
                        snap["review"][k] = e
            except Exception as e:
                _log(f"decisions not kept: {e}")
            try:
                snap["terraces"] = [it for it in self.S.collection("terraces")[1]
                                    if isinstance(it.get("data"), dict) and it["data"].get("flake") in flakes]
            except Exception:
                pass
            man = {f.split(":", 1)[1] for f in flakes if f.startswith("manual:")}
            if man:
                try:
                    snap["manual"] = [it for it in self.S.collection("manual")[1] if it["id"] in man]
                except Exception:
                    pass
        old = read_json(self.lib_dir() / "snap.json", {}) or {}
        if {k: v for k, v in old.items() if k != "at"} != {k: v for k, v in snap.items() if k != "at"}:
            write_json(self.lib_dir() / "snap.json", snap, indent=None)

    # ------------------------------------------------------------------ serving offline
    def runs_listing(self):
        out, parts = [], []
        rroot = self.lib_dir() / "runs"
        try:
            dirs = sorted(d for d in rroot.iterdir() if d.is_dir())
        except OSError:
            dirs = []
        for d in dirs:
            rm = read_json(d / "meta.json")
            try:
                st = (d / "bundle.json").stat()
            except OSError:
                continue
            if not isinstance(rm, dict) or not rm.get("rid"):
                continue
            rid, sig = rm["rid"], f"{st.st_size:x}{st.st_mtime_ns:x}"
            doc = dict(rm.get("doc") or {}, bundle=f"{rid}/bundle.json?offline={sig}",
                       files={n: f"{rid}/{n}" for n in rm.get("files") or []}, n_flakes=len(rm.get("flakes") or []),
                       offline=True)
            out.append(dict(id=rid, data=doc))
            parts.append(rid + sig)
        return self.D._sig("offline-runs|" + "|".join(parts)), out

    def run_file(self, rid, name):
        if "/" in name or "\\" in name or name.startswith("."):
            return None
        d = self.lib_dir() / "runs" / safe_name(rid, 120)
        rm = read_json(d / "meta.json") or {}
        if rm.get("rid") != rid or (name != "bundle.json" and name not in (rm.get("files") or [])):
            return None
        return d / name

    def stacks_listing(self):
        out, parts = [], []
        for sd, m in self.all_meta():
            try:
                st = (sd / "stack.json").stat()
            except OSError:
                continue
            page = read_json(sd / "stack.json")
            if not isinstance(page, dict):
                continue
            out.append(dict(id=m["sid"], data=dict(page, _offline=dict(state=m.get("state"), kind=m.get("kind")))))
            parts.append(f"{m['sid']}:{st.st_mtime_ns}:{st.st_size}:{m.get('state')}")
        return self.D._sig("offline-stacks|" + "|".join(parts)), out

    def file_for(self, url):
        """A stack's file (its URL) on this computer: added offline, or copied into a pack; None."""
        u = urllib.parse.urlparse(url).path
        ne = self.new_index().get(u)
        if ne:
            p = self.lib_dir() / "newfiles" / ne["f"]
            if p.is_file():
                return p
        for sd, m in self.all_meta():
            e = (m.get("files") or {}).get(u)
            if e and e.get("f") and (sd / "files" / e["f"]).is_file():
                return sd / "files" / e["f"]
        return None

    def snap(self):
        s = read_json(self.lib_dir() / "snap.json", {})
        return s if isinstance(s, dict) else {}

    def snap_listing(self, what):
        p = self.lib_dir() / "snap.json"
        try:
            st = p.stat()
            k = f"{st.st_mtime_ns}{st.st_size}"
        except OSError:
            k = "none"
        return self.D._sig(f"offline-{what}|{k}"), list(self.snap().get(what) or [])

    def seen_listing(self):
        sig, runs = self.runs_listing()
        return self.D._sig("seen" + sig), [dict(id=r["id"], data=dict(run=r["data"].get("run") or r["id"])) for r in runs]

    def pack_tids_flakes(self):
        fl = set()
        for _, m in self.all_meta():
            fl.update(m.get("flakes") or [])
        return fl

    def review_listing(self):
        latest = dict(self.snap().get("review") or {})
        q = self.lib_dir() / "review_queue.jsonl"
        for e in self._queue_lines(q):
            k = e.get("id")
            if k:
                latest[k] = dict(id=k, updated=e.get("updated"), body=e.get("body") or {})
        try:
            st = q.stat()
            qk = f"{st.st_size}-{st.st_mtime_ns}"
        except OSError:
            qk = "0"
        sp = self.lib_dir() / "snap.json"
        try:
            sk = sp.stat().st_mtime_ns
        except OSError:
            sk = 0
        out = [dict(id=k, data=e.get("body") or {}) for k, e in latest.items()
               if any((e.get("body") or {}).get(k) for k in ("decision", "stars", "clean"))]
        return self.D._sig(f"offline-review|{qk}|{sk}"), out

    @staticmethod
    def _queue_lines(q):
        out = []
        try:
            text = Path(q).read_text(encoding="utf-8")
        except OSError:
            return out
        for ln in text.splitlines():
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if isinstance(e, dict):
                out.append(e)
        return out

    # ------------------------------------------------------------------ edits offline
    def put_local(self, sid, body):
        """Save a stack into its pack (offline): marked not uploaded.  The page's version check as on the share
        (_base_sig: the version it opened) -> D.StackConflict.  Returns dict(sig, rev, updated)."""
        with self.lock:
            m = self.meta(sid)
            if m is None:
                raise OSError("this stack is not on this computer (it was not kept offline): it is saved when the drive is back")
            sd = self.sdir(sid)
            cur = read_json(sd / "stack.json") or {}
            body = dict(body)
            bsig, force = body.pop("_base_sig", None), body.pop("_force", False)
            body.pop("_base", None)
            for k in ("_sig", "_offline"):
                body.pop(k, None)
            if not force and bsig and cur.get("_sig") and bsig != cur.get("_sig") and cur.get("session") != body.get("session"):
                raise self.D.StackConflict(dict(by=cur.get("by"), at=cur.get("updated"), host=cur.get("saved_on")))
            n = int(m.get("n_local") or 0) + 1
            body["by"] = body.get("by") or self.D.user_name()
            body["rev"] = int(cur.get("rev") or (m.get("base") or {}).get("rev") or 0)
            body["updated"] = max(int(body.get("updated") or 0), int(time.time() * 1000), int(cur.get("updated") or 0) + 1)
            if m.get("kind") == "folder":
                body["folder"], body["file"] = m.get("folder"), m.get("file")
            raw = json.dumps(body).encode("utf-8")
            sig = f"{body['rev']}-off{n}-{len(raw)}"
            body["_sig"] = sig
            # the pack marked "not uploaded" first, then the stack written: a failure in between never lets a refresh
            # from the drive replace edits made here
            m.update(state="pending", edited_at=time.time(), n_local=n, session=body.get("session"), error=None,
                     name=body.get("name"), status=body.get("status"))
            tids = [L.get("tid") for L in ((body.get("plan") or {}).get("layers") or []) if isinstance(L, dict) and L.get("tid")]
            m["tids"] = tids
            m["flakes"] = sorted({f for f in (flake_of(t) for t in tids) if f})
            write_json(sd / "meta.json", m)
            write_json(sd / "stack.json", body, indent=None)
        return dict(sig=sig, rev=body["rev"], updated=body["updated"])

    def store_new(self, url, data):
        """A picture / file added while offline (under the URL it will have on the share)."""
        d = self.lib_dir() / "newfiles"
        name = sha1(url.encode("utf-8"))[:12] + "_" + safe_name(Path(url).name, 80)
        write_atomic(d / name, data)
        with self.lock:
            idx = self.new_index()
            idx[url] = dict(f=name, size=len(data), at=time.time(), uploaded=None)
            write_json(d / "index.json", idx)
        return d / name

    def queue_review(self, tid, body):
        fl = self.pack_tids_flakes()
        if flake_of(tid) not in fl:
            raise OSError("offline: decisions on flakes that are not on this computer wait for the drive")
        e = dict(id=tid, updated=int(time.time() * 1000), body=dict(body, updated=int(time.time() * 1000)))
        with self.lock:
            q = self.lib_dir() / "review_queue.jsonl"
            q.parent.mkdir(parents=True, exist_ok=True)
            with open(str(q), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(e) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass

    # ------------------------------------------------------------------ uploads
    def has_pending(self):
        if any(m.get("state") == "pending" for _, m in self.all_meta()):
            return True
        if any(not e.get("uploaded") for e in self.new_index().values()):
            return True
        return (self.lib_dir() / "review_queue.jsonl").is_file()

    def sync(self):
        if self.forced or not self.has_pending():
            return
        with self.work:
            self.busy = "uploading"
            try:
                self.upload_new()
                for sd, m in self.all_meta():
                    if m.get("state") != "pending":
                        continue
                    try:
                        self.sync_one(sd, m["sid"])
                    except Exception as e:
                        _log(f"stack {m['sid']} not uploaded: {type(e).__name__}: {e}")

                        def f(cur, e=e):
                            if cur is not None:
                                cur["error"] = f"not uploaded: {type(e).__name__}: {e}"
                            return cur
                        self.update_meta(m["sid"], f)
                self.sync_review()
                self.put_state(lambda s: dict(s, last_sync=time.time()))
            finally:
                self.busy = None

    def upload_new(self):
        """The files added offline, to their places on the share (before the stacks that use them)."""
        idx = self.new_index()
        done = {}
        for url, e in sorted(idx.items(), key=lambda kv: kv[1].get("at") or 0):
            if e.get("uploaded"):
                continue
            src = self.lib_dir() / "newfiles" / e["f"]
            dst = self.drive_path(url)
            if dst is None or not src.is_file():
                done[url] = dict(e, error="no place for it on the share")
                continue
            try:
                if dst.is_file():
                    if dst.stat().st_size == src.stat().st_size and dst.read_bytes() == src.read_bytes():
                        done[url] = dict(e, uploaded=time.time(), error=None)
                        continue
                    # (another file of that name: this one under a new name, the stacks here changed to it)
                    stem, ext = os.path.splitext(dst.name)
                    k = 2
                    while dst.with_name(f"{stem}_{k}{ext}").exists():
                        k += 1
                    nu = url.rsplit("/", 1)[0] + f"/{stem}_{k}{ext}"
                    copy_atomic(src, dst.with_name(f"{stem}_{k}{ext}"))
                    self._rename_url(url, nu)
                    done[url] = dict(e, uploaded=time.time(), moved_to=nu, error=None)
                    done[nu] = dict(e, uploaded=time.time(), error=None)
                    continue
                copy_atomic(src, dst)
                done[url] = dict(e, uploaded=time.time(), error=None)
            except OSError as ex:
                done[url] = dict(e, error=str(ex))
        if done:
            with self.lock:
                idx = self.new_index()
                idx.update(done)
                write_json(self.lib_dir() / "newfiles" / "index.json", idx)

    def _rename_url(self, old, new):
        for sd, m in self.all_meta():
            if m.get("state") != "pending":
                continue
            with self.lock:
                p = sd / "stack.json"
                try:
                    t = p.read_text(encoding="utf-8")
                except OSError:
                    continue
                if json.dumps(old)[1:-1] in t:
                    write_atomic(p, t.replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1]).encode("utf-8"))

    def lock_holder(self, sid, m, session):
        """Another live page's note that the stack is open (stacks/<id>.lock, the folder's lock): who, or None."""
        try:
            if m.get("kind") == "folder":
                f = Path(m["folder"]) / self.D.SD.LOCK
            else:
                f = Path(self.S.data) / "stacks" / f"{sid}.lock"
            o = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
        except (OSError, ValueError):
            return None
        if o and o.get("session") != session and time.time() - float(o.get("beat") or 0) < 150:
            return o
        return None

    def sync_one(self, sd, sid):
        m = self.meta(sid)
        if not m or m.get("state") != "pending":
            return
        page = read_json(sd / "stack.json")
        if not isinstance(page, dict):
            raise ValueError("the stack kept here cannot be read")
        if m.get("kind") == "folder":
            folder = Path(m["folder"])
            if not self.probe.check([str(folder)]):
                self.update_meta(sid, lambda c: dict(c, note="its folder is not reachable: uploaded when it is") if c else c)
                return
            if not self.S.folders.where(sid):
                self.S.folders.reg[sid] = [folder, m["file"]]
        other = self.lock_holder(sid, m, page.get("session"))
        if other:
            note = f"open on {other.get('host') or 'another computer'} ({other.get('by') or '?'}): uploaded when it is closed there"
            self.update_meta(sid, lambda c: dict(c, note=note) if c else c)
            return
        path = self.stack_path(sid, m)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raw = None
        base = m.get("base") or {}
        edited = m.get("edited_at")
        body = {k: v for k, v in page.items() if k not in ("_sig", "_offline")}
        if raw is not None and sha1(raw) == base.get("sha1"):
            self.S.put_stack(sid, dict(body, _force=True))     # (the drive's copy is the one this pack was taken from)
            st = path.stat()
            raw2 = path.read_bytes()
            page2 = self._drive_page(sid, m.get("kind"), raw2, st)
            nb = dict(fsig=f"{st.st_size}-{st.st_mtime_ns}", size=st.st_size, mtime_ns=st.st_mtime_ns, sha1=sha1(raw2),
                      rev=page2.get("rev", 0), sig=page2.get("_sig"), at=time.time())
            with self.lock:
                cur = self.meta(sid) or m
                if cur.get("edited_at") == edited:           # (no new edit here while it was uploaded)
                    write_atomic(sd / "drive_copy.json", raw2)
                    write_json(sd / "stack.json", page2)
                    cur.update(state="uploaded", base=nb, uploaded_at=time.time(), note=None, error=None)
                else:
                    cur.update(base=nb, note=None, error=None)
                write_json(sd / "meta.json", cur)
            _log(f"stack {sid} uploaded")
            return
        # changed on the share since this pack was taken (or removed there): both versions are kept
        copy_id, copy_name, where = self.save_copy(sid, m, body)
        info = dict(at=time.time(), copy_id=copy_id, copy_name=copy_name, where=where)
        if raw is not None:
            try:
                d0 = json.loads(raw.decode("utf-8"))
                info.update(drive_by=d0.get("by"), drive_at=d0.get("updated"), drive_host=d0.get("saved_on"))
            except ValueError:
                pass
        with self.lock:
            cur = self.meta(sid) or m
            write_atomic(sd / f"offline_version_{time.strftime('%Y%m%d_%H%M%S')}.json", json.dumps(body).encode("utf-8"))
            if raw is not None:
                st = path.stat()
                page2 = self._drive_page(sid, m.get("kind"), raw, st)
                write_atomic(sd / "drive_copy.json", raw)
                write_json(sd / "stack.json", page2)
                cur["base"] = dict(fsig=f"{st.st_size}-{st.st_mtime_ns}", size=st.st_size, mtime_ns=st.st_mtime_ns, sha1=sha1(raw),
                                   rev=page2.get("rev", 0), sig=page2.get("_sig"), at=time.time())
            cur.update(state="conflict", conflict=info, note=None, error=None, deleted_on_drive=raw is None)
            write_json(sd / "meta.json", cur)
        _log(f"stack {sid} changed on the share meanwhile: the version from here saved as {copy_name} ({where})")

    def save_copy(self, sid, m, body):
        """The laptop's version as a stack of its own (never over the drive's): -> (id, name, where)."""
        import platform
        date, stamp = time.strftime("%Y-%m-%d %H:%M"), time.strftime("%Y%m%d_%H%M%S")
        b = {k: v for k, v in body.items() if k not in ("folder", "file", "session", "rev", "_force", "_base_sig")}
        b["name"] = f"{body.get('name') or 'stack'} (offline edits {date})"
        b["offline_copy_of"] = dict(id=sid, kind=m.get("kind"), folder=m.get("folder"), file=m.get("file"), at=int(time.time() * 1000),
                                    host=platform.node(), by=self.D.user_name())
        if self.S.data is not None:
            base = sid if m.get("kind") != "folder" else "folder_" + sid[2:]
            nid = re.sub(r"[^A-Za-z0-9_.~@+-]", "_", f"{base}_offline_{stamp}")[:180]
            self.S.put_stack(nid, b)
            return nid, b["name"], str(Path(self.S.data) / "stacks" / f"{nid}.json")
        folder = Path(m["folder"])
        p = folder / f"{Path(m['file']).stem} (offline edits {stamp}).json"
        write_atomic(p, json.dumps(b, indent=1).encode("utf-8"))
        return None, b["name"], str(p)

    def sync_review(self):
        q = self.lib_dir() / "review_queue.jsonl"
        lines = self._queue_lines(q)
        if not lines:
            if q.is_file():
                with self.lock:
                    if not self._queue_lines(q):
                        try:
                            q.unlink()
                        except OSError:
                            pass
            return
        sent = 0
        try:
            for e in lines:
                self.S.put_review(e["id"], e.get("body") or {})
                sent += 1
        finally:
            with self.lock:
                rest = self._queue_lines(q)[sent:]
                if rest:
                    write_atomic(q, "".join(json.dumps(x) + "\n" for x in rest).encode("utf-8"))
                else:
                    try:
                        q.unlink()
                    except OSError:
                        pass
        _log(f"{sent} decision(s) made offline uploaded")

    def after_online_save(self, sid, body):
        """A save to the share from the page: the pack follows it (a pending pack whose edits this page carried is
        then no longer pending: its new files go up first)."""
        m = self.meta(sid)
        if m is None:
            return
        if m.get("state") == "pending" and m.get("session") and m.get("session") == body.get("session"):
            try:
                self.upload_new()
            except Exception as e:
                _log(f"files added offline not uploaded: {e}")
            self.update_meta(sid, lambda c: dict(c, state="uploaded", uploaded_at=time.time(), base=None) if c else c)
        self.kick(pack=True)

    def dismiss(self, sid):
        self.update_meta(sid, lambda c: dict(c, state="ok", conflict=None) if c and c.get("state") == "conflict" else c)

    # ------------------------------------------------------------------ for the page
    def brief(self):
        n = len(self.all_meta())
        return dict(on=bool(self.offline), forced=self.forced, n=n)

    def status(self):
        runs = {}
        try:
            for d in (self.lib_dir() / "runs").iterdir():
                rm = read_json(d / "meta.json") or {}
                for fid, b in (rm.get("sizes") or {}).items():
                    runs[fid] = b
        except OSError:
            pass
        keep = self.keep_map()
        stacks = []
        for sd, m in self.all_meta():
            fl = m.get("flakes") or []
            have = [f for f in fl if f in runs or f.startswith("manual:")]
            size = dir_size(sd) + sum(runs.get(f, 0) for f in fl)
            stacks.append(dict(id=m["sid"], name=m.get("name"), kind=m.get("kind"), state=m.get("state"), why=m.get("why"),
                               status=m.get("status"), size=size, packed_at=m.get("packed_at"), edited_at=m.get("edited_at"),
                               uploaded_at=m.get("uploaded_at"), error=m.get("error"), note=m.get("note"),
                               conflict=m.get("conflict"), n_flakes=len(fl), n_flakes_here=len(have),
                               keep=keep.get(m["sid"]), folder=m.get("folder")))
        stacks.sort(key=lambda s: (s["state"] != "pending", s["state"] != "conflict", str(s["name"] or "").lower()))
        idx = self.new_index()
        st = self.state()
        return dict(offline=bool(self.offline), forced=self.forced, auto=self.auto_on(), busy=self.busy, msg=self.msg,
                    dir=str(self.lib_dir()), stacks=stacks, keep=keep, last_sync=st.get("last_sync"),
                    last_pack=self.last_pack_t or None, last_probe=self.last_probe,
                    pending=sum(1 for s in stacks if s["state"] == "pending"),
                    conflicts=sum(1 for s in stacks if s["state"] == "conflict"),
                    files_pending=sum(1 for e in idx.values() if not e.get("uploaded")),
                    decisions_pending=len(self._queue_lines(self.lib_dir() / "review_queue.jsonl")),
                    total=sum(s["size"] for s in stacks))


# ---------------------------------------------------------------------- the helper process: cut-outs
def crop_main(argv):
    """offline.py --crop <jobs.json>: cut each job's box out of its source picture and write it (webp / jpg) into the
    job file's dir.  Prints RESULT {dst: [w, h] or null}.  OpenCV, else Pillow."""
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out_dir = Path(job["dir"])
    res = {}
    lib = None
    try:
        import cv2
        import numpy as np
        lib = "cv2"
    except Exception:
        try:
            from PIL import Image
            lib = "pil"
        except Exception:
            print("RESULT " + json.dumps(dict(error="neither OpenCV nor Pillow")))
            return 3
    by = {}
    for j in job["jobs"]:
        by.setdefault(j["src"], []).append(j)
    for src, js in by.items():
        try:
            if lib == "cv2":
                im = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_UNCHANGED)
                if im is None:
                    raise IOError("not decoded")
                H, W = im.shape[:2]
            else:
                im = Image.open(src)
                im.load()
                W, H = im.size
        except Exception as e:
            sys.stderr.write(f"{src}: {e}\n")
            for j in js:
                res[j["dst"]] = None
            continue
        for j in js:
            try:
                x, y, w, h = [int(round(float(v))) for v in j["box"]]
                x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if x1 <= x0 or y1 <= y0:
                    res[j["dst"]] = None
                    continue
                if lib == "cv2":
                    c = im[y0:y1, x0:x1]
                    if j["fmt"] == "jpg":
                        if c.ndim == 3 and c.shape[2] == 4:
                            c = c[:, :, :3]
                        ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    else:
                        ok, buf = cv2.imencode(".webp", c, [cv2.IMWRITE_WEBP_QUALITY, 92])
                    if not ok:
                        raise IOError("not encoded")
                    data = buf.tobytes()
                else:
                    import io
                    c = im.crop((x0, y0, x1, y1))
                    b = io.BytesIO()
                    if j["fmt"] == "jpg":
                        if c.mode not in ("RGB", "L"):
                            c = c.convert("RGB")
                        c.save(b, "JPEG", quality=92)
                    else:
                        c.save(b, "WEBP", quality=92)
                    data = b.getvalue()
                write_atomic(out_dir / j["dst"], data)
                res[j["dst"]] = [x1 - x0, y1 - y0]
            except Exception as e:
                sys.stderr.write(f"{j['dst']}: {e}\n")
                res[j["dst"]] = None
    print("RESULT " + json.dumps(res))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--crop":
        sys.exit(crop_main(sys.argv[2:]))
    print("offline.py --crop <jobs.json>  (the designer's helper)")
