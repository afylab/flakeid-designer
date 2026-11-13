"""Update the scan program on the microscope PC from GitHub: the start screen's "Update..." button, or

    C:\\FlakeID\\run_flakeid.bat update            (asks before installing; --check: only says what is new)

A public repository needs no token.  A private one (e.g. a private fork in update_repo) needs a token that can read
it, e.g. a fine-grained personal access token with read-only "Contents" access to that repository only, in
update_token_file (default C:/FlakeID/github_token.txt, readable only by the scan's Windows account) or the
FLAKEID_GITHUB_TOKEN environment variable; never in the settings file.

An update does for the code what setup_scope_pc.bat does, from the repository instead of a kit:
  1. the newest commit of update_branch (GitHub API), compared with the installed one (installed_version.json in the
     program folder; unknown for an install from a kit)
  2. that commit downloaded (zip) and checked: every Python file of flakeid/ and tools/win7/ compiles with this
     Python (3.8 on the microscope PC), and the packages it needs (tools/win7/requirements*.txt) are those installed:
     new packages or versions need a kit (make_kit.sh; packages are not downloaded here)
  3. a backup of the program's code (C:/FlakeID/update_backups/<date>_<version>; the last 3 kept), then the new files
     copied over the old ones in place (the program folder is the running program's working folder: it cannot be
     swapped), as setup's xcopy does: the flat fields (flats/), scan_settings.json and local files stay, and so do
     files removed from the repository.  A copy that fails puts the backup back.
  4. run_flakeid.bat copied to C:/FlakeID, installed_version.json written
  5. the program restarted (the new one waits for this one to end: one program at a time holds the microscope)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the program folder (C:/FlakeID/FlakeIdentifier on the PC)
API = os.environ.get("FLAKEID_GITHUB_API", "https://api.github.com/")
VERSION_FILE = "installed_version.json"
# what the kit installs (tools/win7/make_kit.sh): these folders and files of the repository
CODE_DIRS = ("flakeid", "tools", "docs/scan_guide_images")
CODE_FILES = ("find_flakes.py", "README.md", "docs/manual.md", "docs/scan_guide.md", "docs/scan_guide.pdf")
SKIP = ("tools/mm_mock",)                            # (the 2.5 GB Micro-Manager simulator: not on the PC)
REQUIREMENTS = ("tools/win7/requirements-py38.txt", "tools/win7/requirements-mm-py38.txt")


class UpdateError(Exception):
    """An update that cannot go on (no token, GitHub not reachable, a check failed): the message says why."""


def token(s=None):
    t = os.environ.get("FLAKEID_GITHUB_TOKEN", "").strip()
    f = getattr(s, "update_token_file", "") if s is not None else ""
    if not t and f:
        try:
            with open(f, encoding="utf-8") as fh:
                t = fh.read().strip()
        except OSError:
            t = ""
    return t


def _ssl():
    import ssl
    try:                                             # (Windows 7's certificate list is out of date)
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url, tok, accept="application/vnd.github+json", timeout=30):
    import urllib.error
    import urllib.request
    rq = urllib.request.Request(url if url.startswith("http") else API + url,
                                headers={"Accept": accept, "User-Agent": "flakeid-scope-update",
                                         **({"Authorization": "Bearer " + tok} if tok else {})})
    try:
        return urllib.request.urlopen(rq, timeout=timeout, context=_ssl())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            raise UpdateError(f"GitHub refused ({e.code}): " + ("the token in update_token_file (or FLAKEID_GITHUB_TOKEN) has expired "
                              "or cannot read the repository" if tok else "the repository is private or not found: put a token "
                              "that can read it in update_token_file (see flakeid/scope/update.py)")) from None
        raise UpdateError(f"GitHub: {e}") from None
    except (urllib.error.URLError, OSError) as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            raise UpdateError(f"GitHub's certificate could not be checked ({e}): the certifi package is missing or out of date") from None
        raise UpdateError(f"GitHub not reachable ({e}): is this PC on the network?") from None


def _json(url, tok):
    with _get(url, tok) as r:
        return json.loads(r.read().decode("utf-8"))


def installed(root=ROOT):
    """The installed version: {sha, date, message, branch, installed} or {} (installed from a kit: not known)."""
    try:
        return json.loads((Path(root) / VERSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def latest(s):
    """The newest commit of update_branch: {sha, date, message}."""
    tok = token(s)                                   # (none: fine for a public repository)
    c = _json(f"repos/{s.update_repo}/commits/{s.update_branch}", tok)
    return dict(sha=c["sha"], date=c["commit"]["committer"]["date"], message=c["commit"]["message"].split("\n")[0])


def news(s, have, new):
    """The commits from the installed version `have` (sha) to `new`: [(sha, date, first line)], newest first; None
    when the installed version is not known or not on the branch."""
    if not have:
        return None
    try:
        d = _json(f"repos/{s.update_repo}/compare/{have}...{new}", token(s))
    except UpdateError:
        return None
    return [(c["sha"], c["commit"]["committer"]["date"], c["commit"]["message"].split("\n")[0]) for c in reversed(d.get("commits") or [])]


def check(s, root=ROOT):
    """What an update would bring: dict(installed, latest, news, up_to_date)."""
    if (Path(root) / ".git").exists():
        raise UpdateError(f"{root} is a git checkout: update it with git pull")
    have, new = installed(root), latest(s)
    return dict(installed=have, latest=new, news=news(s, have.get("sha"), new["sha"]), up_to_date=have.get("sha") == new["sha"])


def download(s, sha, dest, log=print):
    """The repository at commit `sha`, unpacked into dest; returns the folder with its files."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zp = dest / "update.zip"
    with _get(f"repos/{s.update_repo}/zipball/{sha}", token(s), accept="application/vnd.github+json", timeout=120) as r, open(zp, "wb") as fh:
        n, t0 = 0, time.time()
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
            n += len(b)
            if time.time() - t0 > 2:
                log(f"  downloaded {n / 1e6:.1f} MB ...")
                t0 = time.time()
    log(f"  downloaded {n / 1e6:.1f} MB")
    out = dest / "src"
    with zipfile.ZipFile(str(zp)) as z:
        for m in z.namelist():                       # (no absolute paths or .. in the archive)
            if m.startswith("/") or ".." in Path(m).parts:
                raise UpdateError(f"unexpected path in the download: {m}")
        z.extractall(str(out))
    tops = [p for p in out.iterdir() if p.is_dir()]
    if len(tops) != 1:
        raise UpdateError("unexpected download (not one folder)")
    return tops[0]


def _requirements(path):
    """{package: version} of the name==version lines of a requirements file."""
    req = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return req
    for ln in lines:
        ln = ln.split("#", 1)[0].split(";", 1)[0].strip()
        if "==" in ln:
            k, v = ln.split("==", 1)
            req[k.strip().lower()] = v.strip()
    return req


def problems(src, root=ROOT):
    """What is wrong with the new version for this PC: [text] (empty: it can be installed)."""
    src = Path(src)
    out = []
    for f in ("flakeid/scope/__main__.py", "flakeid/scope/scan.py", "flakeid/scope/ui.py", "tools/win7/run_flakeid.bat"):
        if not (src / f).is_file():
            out.append(f"{f} is missing")
    import warnings
    for d in ("flakeid", "tools/win7"):
        for p in sorted((src / d).rglob("*.py")):
            try:
                txt = p.read_text(encoding="utf-8")
                with warnings.catch_warnings():         # (only errors count: e.g. old escape sequences only warn)
                    warnings.simplefilter("ignore")
                    ast.parse(txt, filename=str(p), feature_version=(3, 8))     # (the microscope PC's Python)
                    compile(txt, str(p), "exec", dont_inherit=True)
            except (SyntaxError, ValueError, UnicodeDecodeError) as e:
                out.append(f"{p.relative_to(src)}: does not compile for Python 3.8 ({e})")
    # packages: a requirement that differs from what the installed program's kit installed must already be met here
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                              # (Python < 3.8)
        version = None
    for rf in REQUIREMENTS:
        old, new = _requirements(Path(root) / rf), _requirements(src / rf)
        for k, v in sorted(new.items()):
            if old.get(k) == v:
                continue
            have = None
            if version is not None:
                try:
                    have = version(k)
                except PackageNotFoundError:
                    have = None
            if have != v:
                out.append(f"needs {k}=={v} (installed: {have or 'none'}): install with a kit (tools/win7/make_kit.sh)")
    return out


def _files(src):
    """The files of the new version that the kit installs: [relative path]."""
    src = Path(src)
    rel = []
    for d in CODE_DIRS:
        for p in sorted((src / d).rglob("*")):
            r = p.relative_to(src).as_posix()
            if p.is_file() and "__pycache__" not in p.parts and not any(r == x or r.startswith(x + "/") for x in SKIP):
                rel.append(r)
    rel += [f for f in CODE_FILES if (src / f).is_file()]
    return rel


def _replace(src, dst):
    """Copy src over dst through a temporary file (a file open for a moment on Windows: retried)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".update_tmp")
    for k in range(10):
        try:
            shutil.copy2(str(src), str(tmp))
            os.replace(str(tmp), str(dst))
            return
        except OSError:
            if k == 9:
                raise
            time.sleep(0.3)


def install(src, info, root=ROOT, log=print):
    """Copy the new version (folder src, commit info {sha, date, message}) over the program in root, after a backup;
    a copy that fails puts the backup back.  Returns the backup folder."""
    src, root = Path(src), Path(root)
    files = _files(src)
    old = installed(root)
    bdir = root.parent / "update_backups" / (time.strftime("%Y%m%d_%H%M%S") + "_" + (old.get("sha", "kit")[:7]))
    log(f"  backup of the current code: {bdir}")
    for d in CODE_DIRS:
        if (root / d).is_dir():
            shutil.copytree(str(root / d), str(bdir / d), ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    for f in CODE_FILES + (VERSION_FILE,):
        if (root / f).is_file():
            (bdir / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(root / f), str(bdir / f))
    log(f"  copying {len(files)} files into {root} ...")
    done = []
    try:
        for r in files:
            _replace(src / r, root / r)
            done.append(r)
    except OSError as e:
        log(f"  copy failed at {r} ({e}): putting the backup back")
        for r in done:
            b = bdir / r
            try:
                if b.is_file():
                    _replace(b, root / r)
                else:
                    (root / r).unlink()              # (a file the new version added)
            except OSError:
                pass
        raise UpdateError(f"the update could not be copied ({e}); the previous version is back") from None
    bat = root.parent / "run_flakeid.bat"           # (as setup does: the launcher beside the program folder)
    if bat.is_file() and (src / "tools/win7/run_flakeid.bat").is_file():
        _replace(src / "tools/win7/run_flakeid.bat", bat)
    (root / VERSION_FILE).write_text(json.dumps(dict(info, branch=info.get("branch", ""), installed=time.strftime("%Y-%m-%d %H:%M:%S")),
                                                indent=1), encoding="utf-8")
    backups = sorted(p for p in bdir.parent.iterdir() if p.is_dir())
    for p in backups[:-3]:                          # (the last 3 backups kept)
        shutil.rmtree(str(p), ignore_errors=True)
    log(f"  installed {info['sha'][:7]} ({info['date']}): {info['message']}")
    return bdir


def update(s, log=print, root=ROOT):
    """Download, check and install the newest version.  Returns the commit installed ({sha, date, message})."""
    c = check(s, root)
    new = dict(c["latest"], branch=s.update_branch)
    tmp = tempfile.mkdtemp(prefix="flakeid_update_", dir=str(Path(root).parent))
    try:
        log(f"downloading {s.update_repo} {s.update_branch} at {new['sha'][:7]} ...")
        src = download(s, new["sha"], tmp, log)
        bad = problems(src, root)
        if bad:
            raise UpdateError("the new version cannot be installed here:\n  " + "\n  ".join(bad[:12]))
        install(src, new, root, log)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)       # (a file held for a moment on Windows must not fail the update)
    return new


def restart_command():
    """The command line that started this program (python -m flakeid.scope ...), to start it again."""
    flags = (["-X", "utf8"] if sys.flags.utf8_mode else []) + (["-s"] if sys.flags.no_user_site else [])
    return [sys.executable] + flags + ["-m", "flakeid.scope"] + sys.argv[1:]


def describe(c):
    """The lines that say what an update would bring (check's result)."""
    have, new = c["installed"], c["latest"]
    out = [f"Installed: {have['sha'][:7]} ({have.get('date', '?')[:10]}): {have.get('message', '')}" if have.get("sha")
           else "Installed: not known (installed from a kit)",
           f"On GitHub: {new['sha'][:7]} ({new['date'][:10]}): {new['message']}"]
    if c["up_to_date"]:
        out.append("The program is up to date.")
    elif c["news"]:
        out.append(f"{len(c['news'])} change{'s' if len(c['news']) > 1 else ''} since the installed version:")
        out += [f"  {d[:10]}  {m}" for _, d, m in c["news"][:15]] + (["  ..."] if len(c["news"]) > 15 else [])
    return out


def main(argv=None):
    from .settings import load
    ap = argparse.ArgumentParser(prog="run_flakeid.bat update", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default=str(ROOT.parent / "scan_settings.json"), help="scan settings (update_repo, update_branch, update_token_file)")
    ap.add_argument("--check", action="store_true", help="only say what an update would bring")
    ap.add_argument("--yes", action="store_true", help="install without asking")
    a = ap.parse_args(argv)
    s = load(a.settings if Path(a.settings).is_file() else None)
    try:
        c = check(s)
        print("\n".join(describe(c)))
        if c["up_to_date"] or a.check:
            return 0
        if not a.yes and input("Install it? [y/N] ").strip().lower() not in ("y", "yes"):
            return 0
        update(s)
        print("Done: start the scan program again.")
    except UpdateError as e:
        print(f"UPDATE NOT DONE: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
