"""Stacks kept in a folder of their own (docs/stack_workflow.md, "Stack folders"): <folder>/<name>.stack (JSON,
vdw-stack/1) with the pick-up pictures and attachments in 01_stacking/, each fabrication step in NN_<step>/ (step.json
and its photos), the better-microscope originals in final/ and the KLayout files in klayout/.

The page knows a folder stack by an id made from the folder's path ("d~" + 16 hex); its files are served at
/stackfile/<id>/<path in the folder>.  Inside the .stack file those references are written "stack:<path in the
folder>", so the folder can be moved or opened from another computer (another drive letter) as it is.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import urllib.parse
from pathlib import Path

PREFIX = "d~"
REF = "stack:"
LOCK = ".designer_lock.json"
STEP_RE = re.compile(r"^(\d{1,3})_")
MAX_RECENT = 15


class Conflict(Exception):
    def __init__(self, info):
        super().__init__("saved elsewhere since it was opened")
        self.info = info


def is_folder_id(sid):
    return isinstance(sid, str) and sid.startswith(PREFIX)


def folder_id(folder):
    key = os.path.normcase(os.path.abspath(str(folder))).rstrip("\\/")
    return PREFIX + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def safe_part(name, fallback="step"):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "")).strip("_")[:60] or fallback


def stack_files(folder):
    try:
        return sorted(p for p in Path(folder).glob("*.stack") if not p.name.startswith((".", "._")))
    except OSError:
        return []


def choose_file(folder):
    """The folder's .stack file: the only one, else the newest (as the scanner's Device photos); None if none."""
    fs = stack_files(folder)
    if len(fs) <= 1:
        return fs[0].name if fs else None
    return max(fs, key=lambda f: f.stat().st_mtime).name


def _walk(x, fn):
    if isinstance(x, dict):
        return {k: _walk(v, fn) for k, v in x.items()}
    if isinstance(x, list):
        return [_walk(v, fn) for v in x]
    return fn(x) if isinstance(x, str) else x


def to_file(body, sid):
    """Page -> file: /stackfile/<sid>/<path> -> stack:<path>."""
    pre = f"/stackfile/{sid}/"
    return _walk(body, lambda s: REF + urllib.parse.unquote(s[len(pre):]) if s.startswith(pre) else s)


def to_page(body, sid):
    """File -> page: stack:<path> -> /stackfile/<sid>/<path> (quoted)."""
    return _walk(body, lambda s: url(sid, s[len(REF):]) if s.startswith(REF) else s)


def url(sid, rel):
    return f"/stackfile/{sid}/" + "/".join(urllib.parse.quote(q) for q in str(rel).replace("\\", "/").split("/") if q)


def unique_name(d, name):
    """A safe file name for `name` not yet used in folder d (name_2.jpg, ...)."""
    safe = re.sub(r"[^A-Za-z0-9_.~@+-]+", "_", Path(str(name)).name).strip("._")[-120:] or "picture.jpg"
    stem, ext = os.path.splitext(safe)
    fn, k = safe, 2
    while (Path(d) / fn).exists():
        fn, k = f"{stem}_{k}{ext}", k + 1
    return fn


def photo_entry(p, fn):
    """A step.json photo from the page's description of a picture."""
    def num(v):
        try:
            return float(v) if v not in (None, "") and float(v) > 0 else None
        except (TypeError, ValueError):
            return None
    return dict(objective=(str(p.get("objective") or "").strip() or "picture")[:40], label=(str(p.get("label") or "").strip() or "main")[:60],
                file=fn, um_per_px=num(p.get("um_per_px")), w=int(p["w"]) if num(p.get("w")) else None,
                h=int(p["h"]) if num(p.get("h")) else None)


def step_dirs(folder):
    """[(number, name)] of the NN_<step> folders holding a step.json, in order."""
    out = []
    try:
        for d in Path(folder).iterdir():
            m = STEP_RE.match(d.name)
            if m and d.is_dir() and (d / "step.json").is_file():
                out.append((int(m.group(1)), d.name))
    except OSError:
        pass
    return sorted(out)


def next_number(folder):
    n = 1                                                  # (01_stacking counts as 01)
    try:
        for d in Path(folder).iterdir():
            m = STEP_RE.match(d.name)
            if m and d.is_dir():
                n = max(n, int(m.group(1)))
    except OSError:
        pass
    return n + 1


def read_step(folder, dname, sid):
    """A step folder's step.json as the page's fabrication step."""
    try:
        j = json.loads((Path(folder) / dname / "step.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    u = (lambda f: url(sid, f"{dname}/{f}") if f else None)
    photos = []
    for p in j.get("photos") or []:
        photos.append(dict(objective=p.get("objective"), label=p.get("label") or "main", um_per_px=p.get("um_per_px"),
                           w=p.get("w"), h=p.get("h"), url=u(p.get("file")), thumb=u(p.get("thumb") or p.get("file")),
                           align=p.get("align"),
                           terrace_map=u(p.get("terrace_map")),
                           terraces=[dict(n=t.get("n"), area_um2=t.get("area_um2"), contrast=t.get("contrast"), hc=u(t.get("hc")))
                                     for t in p.get("terraces") or []]))
    page = j.get("page") or next((h.name for h in sorted((Path(folder) / dname).glob("*.html")) if not h.name.startswith(".")), None)
    return dict(id="dir:" + dname, dir=dname, title=j.get("name") or dname, description=j.get("description") or "",
                at=j.get("at"), taken=j.get("at"), by=j.get("by"), device=j.get("device"), oxide_nm=j.get("oxide_nm"),
                source=j.get("source"), set=dname, photos=photos,
                page=u(page) if page and (Path(folder) / dname / page).is_file() else None)


class Folders:
    """The folder stacks this designer has opened (the id -> folder), its recent list (in the config), and their
    files."""

    def __init__(self, load_config, save_config, user_name):
        self.load_config, self.save_config, self.user_name = load_config, save_config, user_name
        self.reg = {}                                     # id -> [folder Path, file name]

    # -- which folder
    def recents(self):
        return [dict(r, id=folder_id(r["path"])) for r in (self.load_config().get("recent_stacks") or []) if r.get("path")]

    def remember(self, sid, name=None):
        folder, fname = self.reg[sid]
        cfg = self.load_config()
        rs = [r for r in cfg.get("recent_stacks") or [] if r.get("path") and folder_id(r["path"]) != sid]
        rs.insert(0, dict(path=str(folder), file=fname, name=name or Path(fname).stem, at=int(time.time() * 1000)))
        cfg["recent_stacks"] = rs[:MAX_RECENT]
        self.save_config(cfg)

    def forget(self, sid):
        cfg = self.load_config()
        cfg["recent_stacks"] = [r for r in cfg.get("recent_stacks") or [] if r.get("path") and folder_id(r["path"]) != sid]
        self.save_config(cfg)
        self.reg.pop(sid, None)

    def where(self, sid):
        """(folder, file name) of a folder stack id: opened in this session, else from the recent list; else None."""
        if sid in self.reg:
            return self.reg[sid]
        for r in self.recents():
            if r["id"] == sid:
                self.reg[sid] = [Path(r["path"]), r.get("file") or choose_file(r["path"]) or safe_part(Path(r["path"]).name) + ".stack"]
                return self.reg[sid]
        return None

    def register(self, path):
        """A folder (or a .stack file in it) -> (id, folder, file name, exists)."""
        p = Path(path)
        if p.suffix.lower() == ".stack" and not p.is_dir():
            folder, fname = p.parent, p.name
        else:
            folder, fname = p, choose_file(p)
        if not folder.is_dir():
            raise ValueError(f"{folder}: not a folder (or not reachable: is the share connected?)")
        exists = bool(fname) and (folder / fname).is_file()
        fname = fname or safe_part(folder.name, "stack") + ".stack"
        sid = folder_id(folder)
        self.reg[sid] = [folder, fname]
        return sid, folder, fname, exists

    # -- reading
    def sig(self, sid):
        w = self.where(sid)
        if not w:
            return "none"
        folder, fname = w
        try:
            st = (folder / fname).stat()
            s = f"{st.st_mtime_ns}-{st.st_size}"
        except OSError:
            s = "missing"
        for _, d in step_dirs(folder):
            try:
                s += f"|{d}-{(folder / d / 'step.json').stat().st_mtime_ns}"
            except OSError:
                pass
        return s

    def read(self, sid):
        """The stack as the page uses it (references as URLs, fabrication steps from the step folders, _sig, folder,
        file) or None."""
        w = self.where(sid)
        if not w:
            return None
        folder, fname = w
        f = folder / fname
        try:
            st = f.stat()
            j = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(j, dict):
            return None
        j = to_page(j, sid)
        fab = j.get("fab") if isinstance(j.get("fab"), dict) else {}
        removed = set(fab.get("removed") or [])
        steps = [s for s in fab.get("steps") or [] if not s.get("dir")]
        steps += [s for s in (read_step(folder, d, sid) for _, d in step_dirs(folder) if d not in removed) if s]
        j["fab"] = dict(fab, steps=steps)
        j["_sig"] = f"{j.get('rev', 0)}-{st.st_mtime_ns}-{st.st_size}"
        j["folder"], j["file"] = str(folder), fname
        return j

    def listing(self):
        """(signature, [dict(id, data)]) of the folder stacks opened in this session."""
        sids = sorted(self.reg)
        sig = "|".join(f"{s}:{self.sig(s)}" for s in sids)
        out = []
        for s in sids:
            d = self.read(s)
            if d is not None:
                out.append(dict(id=s, data=d))
        return sig, out

    # -- writing
    def put(self, sid, body, host):
        """Save the stack (as Store.put_stack: the version the page opened must still be the file's, else Conflict;
        rev counted, `updated` later than the version replaced).  The fabrication steps of step folders are not kept in
        the file: their name and description are written back to their step.json."""
        w = self.where(sid)
        if not w:
            raise OSError("this stack's folder is not known here: open it again (Open...)")
        folder, fname = w
        if not folder.is_dir():
            raise OSError(f"{folder}: not reachable (is the share connected?)")
        body = dict(body, by=body.get("by") or self.user_name())
        base, force = body.pop("_base", None), body.pop("_force", False)
        bsig = body.pop("_base_sig", None)
        for k in ("_sig", "_offline", "folder", "file"):
            body.pop(k, None)
        f = folder / fname
        cur, csig = None, None
        try:
            st = f.stat()
            cur = json.loads(f.read_text(encoding="utf-8"))
            csig = f"{cur.get('rev', 0)}-{st.st_mtime_ns}-{st.st_size}" if isinstance(cur, dict) else None
        except (OSError, ValueError):
            cur = None
        cur = cur if isinstance(cur, dict) else None
        other = cur and not force and cur.get("session") != body.get("session")
        if other and (csig != bsig if bsig else (base is not None and (cur.get("updated") or 0) > base)):
            raise Conflict(dict(by=cur.get("by"), at=cur.get("updated"), host=cur.get("saved_on")))
        fab = body.get("fab") if isinstance(body.get("fab"), dict) else None
        if fab:
            for s in fab.get("steps") or []:
                if s.get("dir"):
                    self._step_back(folder, s)
            fab = dict(fab, steps=[s for s in fab.get("steps") or [] if not s.get("dir")])
            body["fab"] = fab
        body = to_file(body, sid)
        body["format"] = body.get("format") or "vdw-stack/1"
        body["saved_on"] = host
        body["rev"] = int((cur or {}).get("rev") or 0) + 1
        body["updated"] = max(int(body.get("updated") or 0), int(time.time() * 1000), int((cur or {}).get("updated") or 0) + 1)
        tmp = folder / f".{fname}.tmp"
        tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
        os.replace(str(tmp), str(f))
        st = f.stat()
        return dict(sig=f"{body['rev']}-{st.st_mtime_ns}-{st.st_size}", rev=body["rev"], updated=body["updated"])

    @staticmethod
    def _step_back(folder, s):
        p = Path(folder) / s["dir"] / "step.json"
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        name, desc = s.get("title") or j.get("name"), s.get("description") or ""
        if j.get("name") == name and (j.get("description") or "") == desc:
            return
        j["name"], j["description"] = name, desc
        tmp = p.with_name(".step.json.tmp")
        tmp.write_text(json.dumps(j, indent=1), encoding="utf-8")
        os.replace(str(tmp), str(p))

    def lock(self, sid, session, host, release=False, stale_s=150):
        w = self.where(sid)
        if not w or not w[0].is_dir():
            return True, None
        f = w[0] / LOCK
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
        try:
            f.write_text(json.dumps(dict(session=session, by=self.user_name(), host=host, beat=time.time(),
                                         since=(other or {}).get("since") if mine else time.time())), encoding="utf-8")
        except OSError:
            pass
        return True, None

    # -- files
    def file(self, sid, parts):
        """The file at <folder>/<parts...> (no way out of the folder) or None."""
        w = self.where(sid)
        if not w or not parts or any(q in ("", ".", "..") or "\\" in q or ":" in q for q in parts):
            return None
        p = w[0].joinpath(*parts)
        return p if p.is_file() else None

    def set_align(self, sid, dname, fname, align):
        """A step photo's placement on the substrate's layout (the aligned photos), written into its step.json."""
        w = self.where(sid)
        if not w or not STEP_RE.match(dname):
            return False
        p = w[0] / dname / "step.json"
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        hit = False
        for ph in j.get("photos") or []:
            if ph.get("file") == fname:
                ph["align"], hit = align, True
        if hit:
            tmp = p.with_name(".step.json.tmp")
            tmp.write_text(json.dumps(j, indent=1, allow_nan=False), encoding="utf-8")
            os.replace(str(tmp), str(p))
        return hit

    def put_file(self, sid, src):
        """A file (the substrate's GDS) copied into the stack's folder under its own name (unchanged when the same file
        is there already, else name_2, ...); returns (its URL, its name, its path)."""
        w = self.where(sid)
        if not w:
            raise OSError("this stack's folder is not known here: open it again (Open...)")
        src = Path(src)
        dst = w[0] / src.name
        if not (dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.read_bytes() == src.read_bytes()):
            dst = w[0] / unique_name(w[0], src.name)
            shutil.copy2(str(src), str(dst))
        return url(sid, dst.name), dst.name, dst

    def dir_of(self, sid, sub):
        w = self.where(sid)
        return (w[0] / sub) if w else None

    def store(self, sid, data, name, sub="01_stacking", stamp=True):
        """Write an upload into <folder>/<sub>/ under a new name; returns (its URL, its file name, its path)."""
        w = self.where(sid)
        if not w:
            raise OSError("this stack's folder is not known here: open it again (Open...)")
        d = w[0] / sub
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.~@+-]+", "_", Path(name).name).strip("._")[-120:] or "file"
        if stamp:
            safe = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}"
        stem, ext = os.path.splitext(safe)
        fn, k = safe, 2
        while (d / fn).exists():
            fn, k = f"{stem}_{k}{ext}", k + 1
        (d / fn).write_bytes(data)
        return url(sid, f"{sub}/{fn}"), fn, d / fn

    def new_step(self, sid, title, description, by, photos):
        """A new step folder NN_<title> from pictures taken any way (another microscope, SEM, a phone): photos =
        [{name, data (bytes), objective, label, um_per_px, w, h}], written under their own names (made safe and
        unique) with a step.json.  Returns the page's step."""
        w = self.where(sid)
        if not w:
            raise OSError("this stack's folder is not known here: open it again (Open...)")
        folder, fname = w
        name = (title or "").strip() or "step"
        dname = f"{next_number(folder):02d}_{safe_part(name)}"
        dst = folder / dname
        dst.mkdir(parents=True)
        out = []
        for p in photos:
            fn = unique_name(dst, p.get("name") or "picture.jpg")
            (dst / fn).write_bytes(p["data"])
            out.append(photo_entry(p, fn))
        now = int(time.time() * 1000)
        j = dict(format="vdw-fabstep/1", name=name, description=description or "", device=Path(fname).stem, photos=out,
                 taken=now, at=now, by=by, source="designer: new step from pictures")
        (dst / "step.json").write_text(json.dumps(j, indent=1), encoding="utf-8")
        return read_step(folder, dname, sid)

    def import_step(self, sid, src, title, description, by, photo_set=None):
        """A new step folder NN_<title> from a set of device photos: a step folder of the scanner (step.json: its files
        copied) or an older Device photos folder (*_photos.json + its record from Store.device_photo_set: converted).
        Returns the page's step."""
        w = self.where(sid)
        if not w:
            raise OSError("this stack's folder is not known here: open it again (Open...)")
        folder, fname = w
        src = Path(src)
        device = Path(fname).stem
        name = (title or "").strip() or "step"
        dname = f"{next_number(folder):02d}_{safe_part(name)}"
        dst = folder / dname
        dst.mkdir(parents=True)
        now = int(time.time() * 1000)
        if (src / "step.json").is_file():
            j = json.loads((src / "step.json").read_text(encoding="utf-8"))
            for f in src.iterdir():
                if f.is_file() and f.name != "step.json" and not f.name.startswith("."):
                    shutil.copy2(str(f), str(dst / f.name))
            j.update(name=name, description=description or j.get("description") or "")
        else:
            rec = photo_set or {}
            if not rec.get("photos"):
                raise ValueError(f"{src}: no device photos here (a step folder with step.json, or the scanner's Device photos with *_photos.json)")
            stem = f"{safe_part(device, 'device')}_{safe_part(name)}"
            used, photos = set(), []

            def cp(rel, want):
                if not rel:
                    return None
                b, ext = os.path.splitext(want)
                fn, k = want, 2
                while fn in used:
                    fn, k = f"{b}_{k}{ext}", k + 1
                used.add(fn)
                shutil.copy2(str(src / rel), str(dst / fn))
                return fn
            for p in rec.get("photos") or []:
                mag = safe_part(p.get("objective"), "x")
                fn = cp(p.get("jpg"), f"{stem}_{mag}.jpg")
                photos.append(dict(objective=p.get("objective"), label=p.get("label") or "main", file=fn, um_per_px=p.get("um_per_px"),
                                   w=p.get("w"), h=p.get("h"), terrace_map=cp(p.get("terrace_map"), f"{stem}_{mag}_terraces.jpg"),
                                   terraces=[dict(n=t.get("n"), area_um2=t.get("area_um2"), contrast=t.get("contrast"),
                                                  hc=cp(t.get("hc"), f"{stem}_{mag}_T{t.get('n')}.jpg")) for t in p.get("terraces") or []]))
            j = dict(format="vdw-fabstep/1", name=name, description=description or "", device=device,
                     oxide_nm=rec.get("oxide_nm"), photos=photos, taken=int(src.stat().st_mtime * 1000))
        j.update(at=now, by=by, source="designer import", imported_from=str(src))
        (dst / "step.json").write_text(json.dumps(j, indent=1), encoding="utf-8")
        return read_step(folder, dname, sid)
