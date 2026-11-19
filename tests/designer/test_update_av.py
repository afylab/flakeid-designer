"""Update... on a PC whose antivirus holds each new file for a moment after it is unpacked (2026-11-18o): the copy got
"[Errno 13] Permission denied" on one file, and the temporary folder's removal then failed with WinError 32 and hid it.
The copy is now tried again for a few seconds, and the temporary folder's removal never fails the update."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flakeid_designer.update as U  # noqa: E402
from test_update_move import _install, fake_github  # noqa: E402,F401


def test_file_held_for_a_moment_is_copied(tmp_path, fake_github, monkeypatch):  # noqa: F811
    root = tmp_path / "designer"
    _install(root, {"flakeid_designer/__main__.py": "VERSION = 'old'\n", "flakeid_designer/update.py": "# old\n"})
    real, held = shutil.copy2, {"n": 0}

    def copy2(a, b, *k, **kw):
        if Path(a).name == "update.py" and "designer_update_" in str(a) and held["n"] < 3:
            held["n"] += 1
            raise PermissionError(13, "Permission denied", str(a))
        return real(a, b, *k, **kw)
    monkeypatch.setattr(U.time, "sleep", lambda s: None)
    monkeypatch.setattr(U.shutil, "copy2", copy2)
    r = U.install("abc1234", root=root, log=lambda m: None)
    assert held["n"] == 3 and r["files"] == 3
    assert (root / "flakeid_designer/update.py").read_text() == "# new\n"


def test_temporary_folder_that_cannot_be_removed(tmp_path, fake_github, monkeypatch):  # noqa: F811
    root = tmp_path / "designer"
    _install(root, {"flakeid_designer/__main__.py": "VERSION = 'old'\n"})
    real = shutil.rmtree

    def rmtree(p, ignore_errors=False, *a, **k):
        if "designer_update_" in str(p) and not ignore_errors:
            raise PermissionError(32, "being used by another process", str(p))
        return real(p, ignore_errors=ignore_errors, *a, **k)
    monkeypatch.setattr(U.shutil, "rmtree", rmtree)
    r = U.install("abc1234", root=root, log=lambda m: None)
    assert (root / "flakeid_designer/__main__.py").read_text() == "VERSION = 'new'\n" and r["files"] == 3
