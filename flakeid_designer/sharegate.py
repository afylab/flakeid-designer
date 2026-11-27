"""Working on a slow share (2026-11-27a): the designer never waits on the share without a limit.

Over the VPN the share answers in ~34 ms per file when it works, but a single read can hang 30-60 s (macOS / Windows
SMB timeouts: "[Errno 60] Operation timed out").  Before, the page's requests read the share directly: one stalled
read held up the decisions' save, the lists' polls behind the same lock, and with the browser's ~6 connections to the
designer also every picture; a decision whose save failed then vanished at the next poll (2026-11-26: 75 at once).

Three pieces, used by __main__.py:

  Gate     share calls in a worker thread with a time limit (call); how they go gives the share's state (ok / slow)
           shown in the page's header.  A call that runs over its limit keeps running in its thread (a stuck SMB read
           cannot be cancelled); after MAX_STUCK such threads new calls fail at once instead of starting more.
  Lists    the collections' lists (keep / discard decisions, stacks, drawn terraces, ...) kept in memory and in a file
           on this computer per library: a request waits for a fresh read only a moment (a fast share: as before),
           else it is answered from the last copy while the read finishes in the background.
  Outbox   saves written on this computer first (a journal, flushed to disk), then to the share in one background
           thread, in order, again and again until they are there.  The request waits a moment for that upload (a
           fast share: saved there before the page hears back, as before), else answers at once: the save is safe
           here and uploaded later.  The lists show the saves not uploaded yet (__main__ overlays them).

Where on this computer: FLAKEID_DESIGNER_LOCAL_DIR, else <FLAKEID_DESIGNER_CACHE>/local when that is set (tests: one
folder each), else the offline folder's local/ (offline.offline_root(): app data, not a cache the system may empty).

Python 3.8, standard library only.
"""
from __future__ import annotations

import collections
import errno
import json
import os
import sys
import threading
import time
from pathlib import Path

MAX_STUCK = 6                   # share calls still running past their limit before new ones are refused at once
SLOW_FOR_S = 60.0               # the share counts as slow this long after a call ran over its limit or timed out
SLOW_MEDIAN_S = 1.0             # ... or while the median of the last calls takes longer than this
RETRY_S = (2.0, 5.0, 10.0, 30.0, 60.0)   # the outbox's waits after failed uploads (the last one repeats)


class ShareSlow(OSError):
    """The share did not answer within the limit (the call may still finish in the background)."""

    def __init__(self, msg):
        super().__init__(errno.ETIMEDOUT, msg)


def _log(msg):
    try:
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def local_root():
    env = os.environ.get("FLAKEID_DESIGNER_LOCAL_DIR")
    if env:
        return Path(env)
    c = os.environ.get("FLAKEID_DESIGNER_CACHE")
    if c and c != "0":
        return Path(c) / "local"
    try:
        from .offline import offline_root
    except ImportError:
        from offline import offline_root                # type: ignore
    return offline_root() / "local"


def is_timeout(e):
    return isinstance(e, ShareSlow) or (isinstance(e, OSError) and e.errno in (errno.ETIMEDOUT, getattr(errno, "EHOSTDOWN", -1)))


def write_atomic(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("." + p.name + ".part")
    with open(str(tmp), "wb") as fh:
        fh.write(data)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    for i in range(8):                                  # (Windows: a virus scanner can hold the file a moment)
        try:
            os.replace(str(tmp), str(p))
            return
        except PermissionError:
            if i == 7:
                raise
            time.sleep(0.1 * (i + 1))


# ---------------------------------------------------------------------- the gate
class Gate:
    def __init__(self, max_stuck=MAX_STUCK):
        self.lock = threading.Lock()
        self.max_stuck = max_stuck
        self.stuck = 0
        self.events = collections.deque(maxlen=40)     # (when, seconds, ok)
        self.last_slow = 0.0
        self.last_error = ""

    def note(self, ok, seconds=0.0, what="", err=None):
        """A share call's outcome, from anywhere (the background threads too)."""
        with self.lock:
            self._note(ok, seconds, what, err)

    def _note(self, ok, seconds, what, err):
        self.events.append((time.time(), float(seconds), bool(ok)))
        if err is not None and is_timeout(err):
            self.last_slow = time.time()
            self.last_error = f"{what}: {err}" if what else str(err)

    def call(self, fn, timeout, what="share"):
        """fn() in a thread; its result, or its exception, or ShareSlow after `timeout` seconds."""
        with self.lock:
            if self.stuck >= self.max_stuck:
                self.last_slow = time.time()
                raise ShareSlow(f"the share is not answering ({self.stuck} reads still waiting for it)")
        box, done, t0 = {}, threading.Event(), time.time()

        def run():
            try:
                box["v"] = fn()
            except BaseException as e:                  # (handed to the caller)
                box["e"] = e
            with self.lock:                             # (done under the lock: the caller's timeout sees it or not)
                if box.get("late"):
                    self.stuck -= 1
                e = box.get("e")
                self._note(e is None or not isinstance(e, OSError), time.time() - t0, what, e)
                done.set()
        threading.Thread(target=run, daemon=True, name=f"share: {what}"[:60]).start()
        if not done.wait(timeout):
            with self.lock:
                if not done.is_set():
                    box["late"] = True
                    self.stuck += 1
                    self.last_slow = time.time()
                    self.last_error = f"{what}: no answer in {timeout:g} s"
            if box.get("late"):
                raise ShareSlow(f"the share did not answer in {timeout:g} s ({what})")
        if "e" in box:
            raise box["e"]
        return box.get("v")

    def state(self):
        now = time.time()
        with self.lock:
            if self.stuck > 0 or now - self.last_slow < SLOW_FOR_S:
                return "slow"
            ds = sorted(d for t, d, _ in self.events if now - t < 120)
        if len(ds) >= 3 and ds[len(ds) // 2] > SLOW_MEDIAN_S:
            return "slow"
        return "ok"

    def brief(self):
        with self.lock:
            stuck, err, last = self.stuck, self.last_error, self.last_slow
        return dict(state=self.state(), stuck=stuck, error=err or None, slow_at=last or None)


# ---------------------------------------------------------------------- lists
class Lists:
    """(signature, items) per key, refreshed by fn() in a background thread (one at a time per key)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mem = {}                   # key -> (sig, items)
        self.busy = {}                  # key -> Event of the refresh running
        self.errors = {}                # key -> last error text

    @staticmethod
    def _disk(key):
        import hashlib
        return local_root() / "lists" / (hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:24] + ".json")

    def _load(self, key):
        try:
            j = json.loads(self._disk(key).read_text(encoding="utf-8"))
            if isinstance(j, dict) and j.get("key") == key and isinstance(j.get("items"), list):
                return (str(j.get("sig")), j["items"])
        except (OSError, ValueError):
            pass
        return None

    def _refresh(self, key, fn, ev):
        try:
            sig, items = fn()[:2]
            with self.lock:
                old = self.mem.get(key)
                self.mem[key] = (sig, items)
                self.errors.pop(key, None)
            if old is None or old[0] != sig:
                try:
                    write_atomic(self._disk(key), json.dumps(dict(key=key, sig=sig, items=items)).encode("utf-8"))
                except OSError as e:
                    _log(f"local copy of {key.split('|')[0]} not written: {e}")
        except Exception as e:                          # (the last copy stays)
            with self.lock:
                self.errors[key] = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.busy.pop(key, None)
            ev.set()

    def get(self, key, fn, wait=1.5, first_wait=8.0):
        """The freshest (sig, items) within `wait` seconds (first_wait when nothing is known yet), else the last one,
        else ("loading", [])."""
        with self.lock:
            cur = self.mem.get(key)
        if cur is None:
            cur = self._load(key)
            if cur is not None:
                with self.lock:
                    self.mem.setdefault(key, cur)
        with self.lock:
            ev = self.busy.get(key)
            if ev is None:
                ev = self.busy[key] = threading.Event()
                threading.Thread(target=self._refresh, args=(key, fn, ev), daemon=True, name="list refresh").start()
        ev.wait(wait if cur is not None else first_wait)
        with self.lock:
            now = self.mem.get(key)
        return now or cur or ("loading", [])

    def peek(self, key):
        with self.lock:
            return self.mem.get(key)

    def forget(self, prefix=""):
        with self.lock:
            for k in [k for k in self.mem if k.startswith(prefix)]:
                self.mem.pop(k, None)


# ---------------------------------------------------------------------- the outbox
class Outbox:
    """Saves kept on this computer until the share has them.  Each record: dict(n, coll, id, data_dir, body, t)
    (+ entry: a keep / discard decision as stamped here).  upload(rec) does the share write (set by __main__)."""

    def __init__(self, upload=None, gate=None):
        self.upload = upload
        self.gate = gate
        self.lock = threading.Condition()
        self.items = None
        self._p = None
        self.seq = 0
        self.err = None
        self.fails = 0
        self.wake = threading.Event()
        self.thread = None
        self.uploaded = 0

    def path(self):
        return local_root() / "outbox.jsonl"

    def _ensure(self):
        p = str(self.path())
        if self.items is not None and self._p == p:
            return
        self._p = p                                     # (a test's own folder: read again when it changes)
        items = []
        try:
            for line in self.path().read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("coll") and r.get("id"):
                    items.append(r)
        except OSError:
            pass
        self.items = items
        self.seq = max([int(r.get("n") or 0) for r in items] + [int(time.time() * 1000)])
        if items:
            _log(f"outbox: {len(items)} save(s) from before not uploaded yet: uploading them")

    def _rewrite(self):
        p = self.path()
        if self.items:
            write_atomic(p, "".join(json.dumps(r) + "\n" for r in self.items).encode("utf-8"))
        else:
            try:
                p.unlink()
            except OSError:
                pass

    def add(self, coll, id_, data_dir, body, entry=None):
        """The save on this computer's disk (raises OSError if even that fails), then queued.  -> its number."""
        with self.lock:
            self._ensure()
            self.seq += 1
            r = dict(n=self.seq, coll=coll, id=id_, data_dir=str(data_dir) if data_dir is not None else None,
                     body=body, t=int(time.time() * 1000))
            if entry is not None:
                r["entry"] = entry
            p = self.path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(str(p), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            self.items.append(r)
        self.start()
        self.wake.set()
        return r["n"]

    def pending(self, coll=None, data_dir=None):
        with self.lock:
            self._ensure()
            return [r for r in self.items if (coll is None or r["coll"] == coll)
                    and (data_dir is None or r.get("data_dir") == str(data_dir))]

    def count(self):
        with self.lock:
            self._ensure()
            return len(self.items)

    def wait(self, n, timeout):
        """True when save n is uploaded within timeout seconds."""
        end = time.time() + timeout
        with self.lock:
            while True:
                self._ensure()
                if not any(r["n"] == n for r in self.items):
                    return True
                left = end - time.time()
                if left <= 0:
                    return False
                self.lock.wait(min(left, 0.5))

    def flush(self, timeout=30.0):
        """Wait (tests, closing) until everything is uploaded; False if not within timeout."""
        end = time.time() + timeout
        self.wake.set()
        with self.lock:
            while True:
                self._ensure()
                if not self.items:
                    return True
                left = end - time.time()
                if left <= 0:
                    return False
                self.lock.wait(min(left, 0.5))

    def kick(self):
        self.fails = 0
        self.wake.set()

    def start(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.thread = threading.Thread(target=self._loop, daemon=True, name="outbox")
            self.thread.start()

    def _loop(self):
        while True:
            with self.lock:
                self._ensure()
                r = self.items[0] if self.items else None
            if r is None:
                self.wake.wait(30.0)
                self.wake.clear()
                continue
            t0 = time.time()
            try:
                if self.upload is None:
                    raise OSError("no uploader")
                self.upload(r)
            except Exception as e:
                self.fails += 1
                msg = f"{type(e).__name__}: {e}"
                if self.err != msg or self.fails in (1, 5, 20) or self.fails % 100 == 0:
                    _log(f"outbox: {r['coll']} {r['id']} not uploaded yet ({msg}); {self.count()} waiting, trying again")
                self.err = msg
                if self.gate is not None:
                    self.gate.note(False, time.time() - t0, "upload", e)
                self.wake.wait(RETRY_S[min(self.fails - 1, len(RETRY_S) - 1)])
                self.wake.clear()
                continue
            if self.gate is not None:
                self.gate.note(True, time.time() - t0, "upload")
            with self.lock:
                self.items = [x for x in self.items if x["n"] != r["n"]]
                try:
                    self._rewrite()
                except OSError as e:                    # (uploaded again next start: harmless, same stamp)
                    _log(f"outbox: list not rewritten: {e}")
                if self.fails:
                    _log(f"outbox: uploading again ({len(self.items)} waiting)")
                self.fails, self.err = 0, None
                self.uploaded += 1
                self.lock.notify_all()

    def brief(self):
        with self.lock:
            self._ensure()
            n = len(self.items)
            oldest = self.items[0]["t"] if self.items else None
        return dict(pending=n, error=self.err if n else None, oldest=oldest)
