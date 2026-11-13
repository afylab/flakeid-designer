"""Update the vdW Stack Designer from GitHub (the header's "Update…", 2026-10-30), as the scan program's Update...
(Flake-Finder flakeid/scope/update.py) does for the microscope PC.

The designer's code is the repository afylab/flakeid-designer (the designer, and the analysis code it builds runs
with: flakeid/, tools/).  A public repository needs no token.  A private one (FLAKEID_DESIGNER_REPO, e.g. a private
fork) is read with a token: FLAKEID_DESIGNER_TOKEN, else the file TOKEN_FILE in the home folder (written by the Update
window: a fine-grained personal access token with read-only "Contents" access to that repository), else the GitHub
CLI's login on this computer (gh auth token).

An update:
  1. the newest commit of BRANCH, compared with the installed one (installed_version.json in the designer's folder;
     unknown before the first update)
  2. that commit downloaded (zip) and checked: every Python file of CODE_DIRS compiles with this Python
  3. a backup of the code (BACKUP_DIR/<date>_<version>, the last 3 kept), then the new files copied over the old ones
     in place: local files (settings, the flat fields, anything not in the repository) stay, and so do files removed
     from the repository.  A copy that fails puts the backup back.
  4. installed_version.json written; the designer restarts (the browser version) or asks to be restarted (the app).
A folder that is a git checkout is not updated (git pull).  A changed requirements.txt is reported: new packages are
not installed here.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = os.environ.get("FLAKEID_DESIGNER_REPO", "afylab/flakeid-designer")
BRANCH = "main"
API = os.environ.get("FLAKEID_GITHUB_API", "https://api.github.com/")
ROOT = Path(__file__).resolve().parent.parent      # the designer's folder (flakeid_designer/ and the code it uses)
VERSION_FILE = "installed_version.json"
CODE_DIRS = ("flakeid_designer", "flakeid", "tools", "docs", "ml", "tests")
CODE_FILES = ("README.md", "LICENSE", "requirements.txt", "find_flakes.py", ".gitignore", ".gitattributes")
TOKEN_FILE = Path.home() / ".flakeid_designer_github_token"
BACKUP_DIR = Path.home() / ".flakeid_designer_update_backups"
KEEP_BACKUPS = 3


class UpdateError(Exception):
    """An update that cannot go on (no token, GitHub not reachable, a check failed): the message says why."""


def _gh_token():
    exe = shutil.which("gh") or next((p for p in ("/opt/homebrew/bin/gh", "/usr/local/bin/gh") if Path(p).is_file()), None)
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, "auth", "token"], capture_output=True, text=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def token():
    """(token, where it came from) or ("", "")."""
    t = os.environ.get("FLAKEID_DESIGNER_TOKEN", "").strip()
    if t:
        return t, "FLAKEID_DESIGNER_TOKEN"
    try:
        t = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if t:
            return t, str(TOKEN_FILE)
    except OSError:
        pass
    t = _gh_token()
    return (t, "the GitHub CLI's login (gh)") if t else ("", "")


def _ssl():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url, tok, accept="application/vnd.github+json", timeout=30):
    import urllib.error
    import urllib.request
    rq = urllib.request.Request(url if url.startswith("http") else API + url,
                                headers={"Accept": accept, "User-Agent": "flakeid-designer-update",
                                         **({"Authorization": "Bearer " + tok} if tok else {})})
    try:
        return urllib.request.urlopen(rq, timeout=timeout, context=_ssl())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            raise UpdateError(f"GitHub refused ({e.code}): " + (f"the token has expired or cannot read {REPO}" if tok else
                              f"{REPO} is private or not found: it needs a token that can read it")) from None
        raise UpdateError(f"GitHub: {e}") from None
    except (urllib.error.URLError, OSError) as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            raise UpdateError(f"GitHub's certificate could not be checked ({e}): install the certifi package "
                              "(pip install certifi) or Python's certificates (macOS: Install Certificates.command)") from None
        raise UpdateError(f"GitHub not reachable ({e}): is this computer on the network?") from None


def _json(url, tok):
    with _get(url, tok) as r:
        return json.loads(r.read().decode("utf-8"))


def installed(root=ROOT):
    """The installed version: {sha, date, message, installed} or {} (not known: before the first update)."""
    try:
        return json.loads((Path(root) / VERSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _need_token():
    """The token if this computer has one, else "" (a public repository is read without one)."""
    return token()[0]


def latest():
    c = _json(f"repos/{REPO}/commits/{BRANCH}", _need_token())
    return dict(sha=c["sha"], date=c["commit"]["committer"]["date"], message=c["commit"]["message"].split("\n")[0])


def news(have, new):
    """The commits from `have` to `new`: [(sha, date, first line)], newest first; None when `have` is unknown."""
    if not have:
        return None
    try:
        d = _json(f"repos/{REPO}/compare/{have}...{new}", _need_token())
    except UpdateError:
        return None
    return [(c["sha"], c["commit"]["committer"]["date"], c["commit"]["message"].split("\n")[0]) for c in reversed(d.get("commits") or [])]


def check(root=ROOT):
    """What an update would bring: dict(installed, latest, news, up_to_date, token_from)."""
    if (Path(root) / ".git").exists():
        raise UpdateError(f"{root} is a git checkout: update it with git pull")
    have, new = installed(root), latest()
    return dict(installed=have, latest=new, news=news(have.get("sha"), new["sha"]), up_to_date=have.get("sha") == new["sha"],
                token_from=token()[1])


def download(sha, dest):
    """The repository at commit `sha`, unpacked into dest; returns the folder with its files."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zp = dest / "designer.zip"
    with _get(f"repos/{REPO}/zipball/{sha}", _need_token(), timeout=180) as r, open(zp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest / "x")
    tops = [p for p in (dest / "x").iterdir() if p.is_dir()]
    if len(tops) != 1 or not (tops[0] / "flakeid_designer" / "__main__.py").is_file():
        raise UpdateError("the download is not the designer's repository")
    return tops[0]


def verify(src):
    """Every Python file of CODE_DIRS compiles with this Python: [errors]."""
    import warnings
    bad = []
    for d in CODE_DIRS:
        for p in sorted((Path(src) / d).rglob("*.py")):
            try:
                with warnings.catch_warnings():          # (old escape sequences: warnings, not errors)
                    warnings.simplefilter("ignore")
                    ast.parse(p.read_text(encoding="utf-8"), str(p))
            except (SyntaxError, UnicodeDecodeError, ValueError) as e:
                bad.append(f"{p.relative_to(src)}: {e}")
    return bad


def _code_files(base):
    base = Path(base)
    out = []
    for d in CODE_DIRS:
        if (base / d).is_dir():
            out += [p.relative_to(base) for p in (base / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    out += [Path(f) for f in CODE_FILES if (base / f).is_file()]
    return out


def install(sha, root=ROOT, log=print):
    """Install commit `sha` over the designer in `root`.  Returns dict(installed, files, backup, requirements_changed)."""
    root = Path(root)
    if (root / ".git").exists():
        raise UpdateError(f"{root} is a git checkout: update it with git pull")
    have = installed(root)
    with tempfile.TemporaryDirectory(prefix="designer_update_") as tmp:
        log("downloading ...")
        src = download(sha, tmp)
        log("checking the new code ...")
        bad = verify(src)
        if bad:
            raise UpdateError("the new code does not compile with this Python: " + "; ".join(bad[:5]))
        req_changed = (src / "requirements.txt").is_file() and (
            not (root / "requirements.txt").is_file()
            or (src / "requirements.txt").read_bytes() != (root / "requirements.txt").read_bytes())
        # the backup: the code as it is (what the new files replace)
        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + (have.get("sha") or "unknown")[:7]
        bk = BACKUP_DIR / stamp
        log(f"backup of the current code in {bk} ...")
        for rel in _code_files(root):
            (bk / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, bk / rel)
        new = _code_files(src)
        log(f"copying {len(new)} files ...")
        try:
            for rel in new:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, root / rel)
                if os.name != "nt" and str(rel).endswith((".command", ".sh")):
                    try:                                     # (2026-11-10a: the zip keeps no modes: without this the
                        os.chmod(root / rel, 0o755)          # Mac's "Install shortcut (Mac).command" could not be opened)
                    except OSError:
                        pass
        except OSError as e:
            log(f"copy failed ({e}): putting the backup back")
            for rel in _code_files(bk):
                try:
                    shutil.copy2(bk / rel, root / rel)
                except OSError:
                    pass
            raise UpdateError(f"the new files could not be copied ({e}); the previous code is back") from None
    c = _json(f"repos/{REPO}/commits/{sha}", _need_token())
    ver = dict(sha=sha, date=c["commit"]["committer"]["date"], message=c["commit"]["message"].split("\n")[0],
               installed=time.strftime("%Y-%m-%d %H:%M:%S"))
    (root / VERSION_FILE).write_text(json.dumps(ver, indent=1), encoding="utf-8")
    olds = sorted((p for p in BACKUP_DIR.iterdir() if p.is_dir()), key=lambda p: p.name) if BACKUP_DIR.is_dir() else []
    for p in olds[:-KEEP_BACKUPS]:
        shutil.rmtree(p, ignore_errors=True)
    log(f"installed {sha[:7]} ({ver['message']})")
    return dict(installed=ver, files=len(new), backup=str(bk), requirements_changed=bool(req_changed))


def save_token(tok):
    """Check a token against the repository, then keep it in TOKEN_FILE (readable by this account only)."""
    tok = (tok or "").strip()
    if not tok:
        raise UpdateError("no token given")
    _json(f"repos/{REPO}", tok)                       # (raises when it cannot read the repository)
    TOKEN_FILE.write_text(tok, encoding="utf-8")
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass
    return str(TOKEN_FILE)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m flakeid_designer.update", description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only say what is new")
    ap.add_argument("--root", default=str(ROOT), help="the designer's folder")
    a = ap.parse_args(argv)
    try:
        r = check(a.root)
    except UpdateError as e:
        print(f"no update: {e}")
        return 1
    print(f"installed: {(r['installed'].get('sha') or 'unknown')[:7]}; newest: {r['latest']['sha'][:7]} {r['latest']['message']}")
    if r["up_to_date"] or a.check:
        print("up to date" if r["up_to_date"] else "(--check: nothing installed)")
        return 0
    install(r["latest"]["sha"], a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
