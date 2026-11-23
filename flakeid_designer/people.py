"""People (2026-11-23): who is using the designer, and each person's list of runs.

One file per person, in their home library's designer data on the share, so the list follows them to any computer:

    <home library>/_designer_data/people/<Name>.json
    {"kind": "flakeid-designer-person", "name": "Jordan",
     "aliases": ["jordan@Jordans-MacBook"],               (the computer logins this person's older decisions carry)
     "runs": {"20261004_run_5": {"rel": "20261004_run_5", "path": "/Volumes/.../20261004_run_5", "on": true, "t": ms}},
     "ticked": {"keys": ["20261004_run_5"], "t": ms}}     (the runs loaded: their flakes shown)

A run is kept by its path relative to the home library (the same share is /Volumes/... on a Mac and \\\\NAS\\... or Z:\\
on Windows), with the absolute path as it was added for a run that has no relative path (another drive).  A run taken
off the list stays in the file with on: false and its time: two computers that changed the list at once are merged
entry by entry, the later change winning, and a computer that was offline cannot bring back a run removed meanwhile.
Nothing here raises on a share that does not answer: the caller keeps its copy in the computer's settings.

Python 3.8 (Windows 7 lab PCs): no newer syntax.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

KIND = "flakeid-designer-person"
SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
MAX_NAME = 40
WRITE_LOCK = threading.Lock()


def now_ms():
    return int(time.time() * 1000)


def clean_name(name):
    """A person's name as shown: spaces collapsed, at most MAX_NAME characters; '' when none is left."""
    return " ".join(str(name or "").split())[:MAX_NAME]


def file_id(name):
    """The file name part (and the key to compare names, lower case): 'Jordan Pack' -> 'Jordan_Pack'."""
    return SAFE.sub("_", clean_name(name)).strip("._") or ""


def key_of(name):
    return file_id(name).lower()


def people_dir(data_dir):
    return Path(data_dir) / "people" if data_dir else None


def person_file(data_dir, name):
    d = people_dir(data_dir)
    return d / f"{file_id(name)}.json" if d is not None and file_id(name) else None


def rel_of(path, base, mod=os.path):
    """`path` relative to `base` with / separators, or None (no base, another drive)."""
    if not base or not path or not mod.isabs(str(path)) or not mod.isabs(str(base)):
        return None                                  # (a path from another kind of computer: kept absolute)
    try:
        r = mod.relpath(mod.normpath(str(path)), mod.normpath(str(base)))
    except ValueError:                               # (Windows: another drive, or UNC against a drive letter)
        return None
    return r.replace("\\", "/")


def resolve(entry, base, mod=os.path):
    """The run's folder on this computer: the home library joined with the relative path, else the absolute path."""
    rel = entry.get("rel")
    if rel and base:
        p = mod.normpath(mod.join(str(base), rel.replace("/", mod.sep)))
    else:
        p = entry.get("path") or ""
        p = mod.normpath(p) if p else ""
    return str(Path(p)) if p and mod is os.path else p


def entry_key(path, base, mod=os.path):
    r = rel_of(path, base, mod)
    return r if r is not None else mod.normpath(str(path)).replace("\\", "/")


def new_doc(name, aliases=()):
    return dict(kind=KIND, name=clean_name(name), aliases=sorted(set(a for a in aliases if a)), runs={},
                ticked=None, updated=now_ms())


def valid(doc):
    return isinstance(doc, dict) and isinstance(doc.get("runs", {}), dict)


def load(f):
    """The person's file, or None (missing, unreadable, or the share does not answer)."""
    if f is None:
        return None
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if valid(d) else None


def merge(a, b):
    """Two copies of one person's file (the share's and this computer's): per run the later change, aliases together,
    the later tick set."""
    if not valid(a):
        return json.loads(json.dumps(b)) if valid(b) else None
    if not valid(b):
        return json.loads(json.dumps(a))
    out = dict(kind=KIND, name=a.get("name") or b.get("name"),
               aliases=sorted(set(a.get("aliases") or []) | set(b.get("aliases") or [])), runs={})
    for src in (a, b):
        for k, e in (src.get("runs") or {}).items():
            if isinstance(e, dict) and (k not in out["runs"] or int(e.get("t") or 0) > int(out["runs"][k].get("t") or 0)):
                out["runs"][k] = dict(e)
    ta, tb = a.get("ticked"), b.get("ticked")
    cand = [t for t in (ta, tb) if isinstance(t, dict) and isinstance(t.get("keys"), list)]
    out["ticked"] = max(cand, key=lambda t: int(t.get("t") or 0)) if cand else None
    out["updated"] = max(int(a.get("updated") or 0), int(b.get("updated") or 0))
    return out


def save(f, doc, tries=5):
    """Merged with what is in the file now, then written (a new file renamed over it, tried again while Windows or the
    share holds it).  -> the merged copy, or None when it could not be written (the caller keeps its own copy)."""
    if f is None or not valid(doc):
        return None
    f = Path(f)
    with WRITE_LOCK:
        for i in range(tries):
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                cur = merge(load(f), doc)
                cur["updated"] = now_ms()
                tmp = f.with_name(f.name + f".{os.getpid()}.part")
                tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False), encoding="utf-8")
                os.replace(str(tmp), str(f))
                return cur
            except OSError:
                time.sleep(0.2 * (i + 1))
    return None


def set_run(doc, path, on, base, t=None, mod=os.path):
    """Put a run on (on=True) or take it off the person's list."""
    k = match(doc, base, mod).get(str(path)) or entry_key(path, base, mod)
    old = (doc.get("runs") or {}).get(k)
    if old and bool(old.get("on")) == bool(on):
        return doc                                   # (no change: the entry keeps its time)
    e = dict(rel=rel_of(path, base, mod), path=str(path), on=bool(on), t=_later(t or now_ms(), old or {}))
    doc.setdefault("runs", {})[k] = e
    return doc


def run_paths(doc, base, mod=os.path):
    """The runs on the person's list (this computer's paths), in the order they were added."""
    rs = [e for e in (doc.get("runs") or {}).values() if isinstance(e, dict) and e.get("on")]
    rs.sort(key=lambda e: int(e.get("t") or 0))
    out = []
    for e in rs:
        p = resolve(e, base, mod)
        if p and p not in out:
            out.append(p)
    return out


def set_list(doc, paths, base, t=None, mod=os.path):
    """Make the list `paths`: new ones on, missing ones off (one change time for all).
    Compared by this computer's path: only the entries whose on / off changes are touched (an entry written on another
    computer keeps its key, paths and time, so an old copy elsewhere cannot outvote a change made meanwhile)."""
    t = int(t or now_ms())
    runs = doc.setdefault("runs", {})
    have = match(doc, base, mod)
    want = list(dict.fromkeys(str(p) for p in paths))
    ws = set(want)
    for p, k in have.items():
        e = runs[k]
        if e.get("on") and p not in ws:
            runs[k] = dict(e, on=False, t=_later(t, e))
    for p in want:
        k = have.get(p)
        if k is None:
            set_run(doc, p, True, base, t, mod)
        elif not runs[k].get("on"):
            runs[k] = dict(runs[k], on=True, t=_later(t, runs[k]))
    return doc


def _later(t, e):
    """A change's time: after the entry's last one even on a computer whose clock is behind."""
    return max(int(t), int(e.get("t") or 0) + 1)


def match(doc, base, mod=os.path):
    """{this computer's path: the entry's key}, an entry on the list first when two give the same path."""
    out = {}
    for k, e in sorted((doc.get("runs") or {}).items(), key=lambda kv: not (isinstance(kv[1], dict) and kv[1].get("on"))):
        if isinstance(e, dict):
            p = resolve(e, base, mod)
            if p:
                out.setdefault(p, k)
    return out


def set_ticked(doc, paths, base, t=None, mod=os.path):
    m = match(doc, base, mod)
    old = doc.get("ticked") if isinstance(doc.get("ticked"), dict) else {}
    doc["ticked"] = dict(keys=sorted(set(m.get(str(p)) or entry_key(p, base, mod) for p in paths)), t=_later(t or now_ms(), old))
    return doc


def ticked_paths(doc, base, mod=os.path):
    """The loaded (ticked) runs as this computer's paths, or None when the person never ticked any here or elsewhere."""
    tk = doc.get("ticked")
    if not isinstance(tk, dict) or not isinstance(tk.get("keys"), list):
        return None
    out = []
    for k in tk["keys"]:
        e = (doc.get("runs") or {}).get(k) or ({"path": k} if mod.isabs(k.replace("/", mod.sep)) else {"rel": k})
        p = resolve(e, base, mod)
        if p:
            out.append(p)
    return out


def listed(data_dir):
    """The people with a file in this library's designer data: [{name, file}] (one folder listed on the share)."""
    d = people_dir(data_dir)
    out = []
    try:
        for f in sorted(d.glob("*.json")) if d is not None and d.is_dir() else []:
            doc = load(f)
            if doc and doc.get("kind") == KIND and doc.get("name"):
                out.append(dict(name=doc["name"], file=str(f)))
    except OSError:
        pass
    return out
