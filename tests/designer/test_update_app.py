"""The desktop app (2026-11-18): its build records the commit it was made from, so the "Update available" banner is
shown only when GitHub has a newer one; its Update... says to install the new build (it has no code files to replace)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flakeid_designer.update as U  # noqa: E402


def _git(repo, *a):
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "a.txt").write_text("1")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "First line\n\nmore")
    return r


def test_build_version_records_the_commit(repo, tmp_path):
    out = tmp_path / "build" / U.VERSION_FILE
    v = U.build_version(repo, out)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert v["sha"] == sha and v["message"] == "First line" and v["date"].endswith("Z") and "changed" not in v
    assert U.installed(out.parent)["sha"] == sha


def test_build_version_with_uncommitted_changes(repo, tmp_path):
    (repo / "a.txt").write_text("2")
    assert U.build_version(repo, tmp_path / U.VERSION_FILE)["changed"] is True


def test_build_version_outside_git(tmp_path):
    assert U.build_version(tmp_path / "nothing", tmp_path / U.VERSION_FILE) == {}
    assert U.installed(tmp_path) == {}


def test_app_banner_only_when_github_is_newer(tmp_path, monkeypatch):
    (tmp_path / U.VERSION_FILE).write_text(json.dumps(dict(sha="abc")))
    monkeypatch.setattr(U, "latest", lambda: dict(sha="abc", date="", message=""))
    assert U.available(tmp_path) is None
    monkeypatch.setattr(U, "latest", lambda: dict(sha="def", date="", message=""))
    assert U.available(tmp_path)["sha"] == "def"


def test_app_update_says_install_the_new_build(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "APP", True)
    monkeypatch.setattr(U, "latest", lambda: dict(sha="def", date="", message=""))
    monkeypatch.setattr(U, "news", lambda *a: [])
    r = U.check(tmp_path)
    assert r["app_build"] is True and "new build" in r["app_how"]
    with pytest.raises(U.UpdateError, match="new build"):
        U.install("def", root=tmp_path, log=lambda *a: None)
