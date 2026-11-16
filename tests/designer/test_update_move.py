"""Update... on a designer in a folder this account cannot write (2026-11-16c): a lab PC had it unzipped into
C:\\Program Files, and every update ended in "[Errno 13] Permission denied ... the previous code is back".  The designer
is then copied to MOVE_DIR, updated there, and restarted from there; the old folder is left as it is."""
import os
import stat
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flakeid_designer.update as U  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                reason="read-only folders made with file modes (and root writes anyway)")


def _install(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def _lock(root):
    """root read-only, as C:\\Program Files is for a user without admin rights."""
    for p in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        p.chmod(0o444 if p.is_file() else 0o555)
    root.chmod(0o555)


def _unlock(root):
    for p in [root, *root.rglob("*")]:
        p.chmod(stat.S_IRWXU)


@pytest.fixture
def fake_github(tmp_path, monkeypatch):
    zp = tmp_path / "new.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("repo-abc/flakeid_designer/__main__.py", "VERSION = 'new'\n")
        z.writestr("repo-abc/flakeid_designer/update.py", "# new\n")
        z.writestr("repo-abc/requirements.txt", "numpy\n")

    def download(sha, tmp):
        zipfile.ZipFile(zp).extractall(Path(tmp) / "x")
        return Path(tmp) / "x" / "repo-abc"
    monkeypatch.setattr(U, "download", download)
    monkeypatch.setattr(U, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(U, "_need_token", lambda: "")
    monkeypatch.setattr(U, "_json", lambda *a, **k: {"commit": {"committer": {"date": "2026-11-16T00:00:00Z"}, "message": "test"}})
    monkeypatch.setattr(U, "MOVE_DIR", tmp_path / "home" / "flakeid-designer-app")
    monkeypatch.setenv("FLAKEID_INSTALL_NO_SHORTCUT", "1")


def test_read_only_designer_moves_and_updates(tmp_path, fake_github):
    root = tmp_path / "Program Files" / "flakeid-designer-main"
    _install(root, {"flakeid_designer/__main__.py": "VERSION = 'old'\n", "flakeid_designer/update.py": "# old\n",
                    "requirements.txt": "numpy\n", "flats/local_flat.txt": "this computer's own file\n"})
    _lock(root)
    try:
        assert not U.writable(root)
        log = []
        r = U.install("abc1234", root=root, log=log.append)
    finally:
        _unlock(root)
    new = U.MOVE_DIR
    assert r["moved_from"] == str(root) and r["moved_to"] == str(new)
    assert (new / "flakeid_designer" / "__main__.py").read_text() == "VERSION = 'new'\n"
    assert (new / "flats" / "local_flat.txt").read_text() == "this computer's own file\n"     # (local files go too)
    assert U.installed(new)["sha"] == "abc1234"
    assert (root / "flakeid_designer" / "__main__.py").read_text() == "VERSION = 'old'\n"     # (the old one untouched)
    assert not (root / U.VERSION_FILE).exists()
    assert any("moves to" in m for m in log) and not any("copy failed" in m for m in log)
    assert U.writable(new)


def test_writable_designer_updates_in_place(tmp_path, fake_github):
    root = tmp_path / "Documents" / "flakeid-designer-main"
    _install(root, {"flakeid_designer/__main__.py": "VERSION = 'old'\n"})
    r = U.install("abc1234", root=root, log=lambda *a: None)
    assert r["moved_to"] is None and r["moved_from"] is None
    assert (root / "flakeid_designer" / "__main__.py").read_text() == "VERSION = 'new'\n"
    assert not U.MOVE_DIR.exists()


def test_moved_into_an_earlier_copy(tmp_path, fake_github):
    """A second PC user's update (or a second try) into a MOVE_DIR that is already there: copied over it."""
    root = tmp_path / "ro"
    _install(root, {"flakeid_designer/__main__.py": "VERSION = 'old'\n"})
    _install(U.MOVE_DIR, {"flakeid_designer/__main__.py": "VERSION = 'older'\n", "flakeid_designer/stale.txt": "x"})
    _lock(root)
    try:
        r = U.install("abc1234", root=root, log=lambda *a: None)
    finally:
        _unlock(root)
    assert r["moved_to"] == str(U.MOVE_DIR)
    assert (U.MOVE_DIR / "flakeid_designer" / "__main__.py").read_text() == "VERSION = 'new'\n"


def test_writable_root_command(tmp_path, monkeypatch, capsys):
    """update.py --writable-root (the Windows installer): the folder to use, the designer copied there when needed."""
    monkeypatch.setattr(U, "MOVE_DIR", tmp_path / "moved")
    ok = tmp_path / "ok"
    _install(ok, {"flakeid_designer/__main__.py": "VERSION = 'x'\n"})
    assert U.main(["--writable-root", "--root", str(ok)]) == 0
    assert capsys.readouterr().out.strip() == str(ok)
    ro = tmp_path / "ro"
    _install(ro, {"flakeid_designer/__main__.py": "VERSION = 'x'\n"})
    _lock(ro)
    try:
        assert U.main(["--writable-root", "--root", str(ro)]) == 0
    finally:
        _unlock(ro)
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(tmp_path / "moved")
    assert (tmp_path / "moved" / "flakeid_designer" / "__main__.py").is_file()


def test_shortcut_made_again_from_the_new_folder(tmp_path, monkeypatch):
    """After a move, the shortcut is made by the moved designer (its --install-shortcut), with this Python."""
    monkeypatch.delenv("FLAKEID_INSTALL_NO_SHORTCUT", raising=False)
    new = tmp_path / "moved"
    _install(new, {"flakeid_designer/__main__.py": "import sys\nprint('shortcut: made from', __file__, sys.argv[1:])\n"})
    log = []
    U.shortcut(new, log.append)
    assert len(log) == 1 and "shortcut: made from" in log[0] and str(new / "flakeid_designer") in log[0] and "--install-shortcut" in log[0]
