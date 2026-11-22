"""The packaged app's entry ("thin app", 2026-11-21): the app carries Python and the libraries; the designer's own code
(flakeid_designer/, flakeid/, tools/) runs from a folder this account can write, so the header's Update... replaces it
as it does for the browser version (update.py).

Before: the code was compiled into the app's archive, which Python reads first; Update copied new .py files into the
app where nothing imported them (and writing into a signed app breaks its signature), so the Mac app could not be
updated.

CODE_DIR: ~/Library/Application Support/vdW Stack Designer/code (macOS), %LOCALAPPDATA%\\vdW Stack Designer\\code
(Windows), or FLAKEID_DESIGNER_CODE.  The app bundles a copy of the code it was built with (SEED, with its
installed_version.json); it is copied to CODE_DIR when CODE_DIR has none, or when a new app is installed whose
commit is newer than CODE_DIR's (an update installed with Update... is kept, even an older commit).  Files that are not code (settings, anything else) stay.  If the code in CODE_DIR
does not start (an update that needs a package the app lacks), the app's own copy is put back and started.
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

NAME = "vdW Stack Designer"
LOG = Path.home() / ".flakeid_designer_log.txt"
SEED_NAME = "designer_code"
VERSION_FILE = "installed_version.json"
APP_SEEN = ".app_seed"                             # (the commit of the app last started with this folder)


def code_dir():
    env = os.environ.get("FLAKEID_DESIGNER_CODE")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / NAME / "code"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / NAME / "code"
    return Path.home() / ".local" / "share" / "flakeid-designer" / "code"


def seed_dir():
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / SEED_NAME


def log(msg):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} app launcher: {msg}\n")
    except OSError:
        pass


def _version(d):
    try:
        return json.loads((Path(d) / VERSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _seed_files(seed):
    return [p.relative_to(seed) for p in seed.rglob("*") if p.is_file() and "__pycache__" not in p.parts]


def put_seed(seed, dest, why):
    """The app's own copy of the code into dest (its files replace the same files there; nothing else is removed)."""
    log(f"code from the app ({_version(seed).get('sha', '?')[:7]}) into {dest}: {why}")
    for rel in _seed_files(seed):
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed / rel, dest / rel)


def _app_seen(dest):
    try:
        return (dest / APP_SEEN).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _mark_seen(dest, sha):
    try:
        (dest / APP_SEEN).write_text(sha, encoding="utf-8")
    except OSError:
        pass


def prepare():
    """CODE_DIR ready, on the import path first; returns it.  The app's copy goes there only when there is no code, or
    when this app is not the one that was last started (a new app installed) and its code is newer than the folder's:
    an update installed with Update... stays, even one older than the app's code (2026-11-21: updating to main, older
    than the app built from a branch, was undone at every start, and the banner never went away)."""
    seed, dest = seed_dir(), code_dir()
    have, mine = _version(dest), _version(seed)
    if not (dest / "flakeid_designer" / "__main__.py").is_file():
        put_seed(seed, dest, "no code there yet")
    elif _app_seen(dest) != mine.get("sha", "") and mine.get("date") and (have.get("date") or "") < mine["date"]:
        # (a newly installed app, with newer code than the last update: commit dates, ISO 8601)
        put_seed(seed, dest, f"a newer app; older code there ({have.get('sha', 'unknown')[:7]} {have.get('date', '')})")
    _mark_seen(dest, mine.get("sha", ""))
    for p in (dest / "tools", dest):                 # (the builder's scripts import each other from tools/)
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return dest


def _start():
    import flakeid_designer.app as A                 # (from CODE_DIR: not in the app's archive)
    if len(sys.argv) > 2 and sys.argv[1] == "--helper":
        return A.run_helper(sys.argv[2], sys.argv[3:])
    if "--check-builder" in sys.argv:
        return A.check_builder()
    return A.run()


def main():
    dest = prepare()
    import multiprocessing
    multiprocessing.freeze_support()                 # (worker processes: they import from CODE_DIR too)
    try:
        return _start()
    except Exception:                                # (ImportError / SyntaxError of an update, a missing package, ...)
        log("the code in " + str(dest) + " does not start:\n" + traceback.format_exc())
        if any(a == "--helper" for a in sys.argv[1:2]):
            raise
        for m in [m for m in sys.modules if m.split(".")[0] in ("flakeid_designer", "flakeid") or m in ("update", "offline")]:
            del sys.modules[m]
        put_seed(seed_dir(), dest, "the code there did not start")
        return _start()


if __name__ == "__main__":
    sys.exit(main())
