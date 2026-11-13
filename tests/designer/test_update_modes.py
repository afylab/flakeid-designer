"""Update... copies the new code out of GitHub's zip, which keeps no file modes: the Mac's "Install shortcut (Mac).command"
(and the .sh tools) must come out executable, or macOS refuses to open it (2026-11-10a)."""
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flakeid_designer.update as U  # noqa: E402


@pytest.mark.skipif(os.name == "nt", reason="file modes")
def test_command_files_executable_after_update(tmp_path, monkeypatch):
    zp = tmp_path / "new.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("repo-abc/flakeid_designer/__main__.py", "VERSION = 'x'\n")
        z.writestr("repo-abc/flakeid_designer/Install shortcut (Mac).command", "#!/bin/bash\necho hi\n")
        z.writestr("repo-abc/flakeid_designer/install_windows.ps1", "Write-Host hi\n")
        z.writestr("repo-abc/tools/example.sh", "#!/bin/sh\n")

    def download(sha, tmp):
        zipfile.ZipFile(zp).extractall(Path(tmp) / "x")
        return Path(tmp) / "x" / "repo-abc"
    monkeypatch.setattr(U, "download", download)
    monkeypatch.setattr(U, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(U, "_need_token", lambda: "")                # (no GitHub: the commit's details made up)
    monkeypatch.setattr(U, "_json", lambda *a, **k: {"commit": {"committer": {"date": "2026-11-10T00:00:00Z"}, "message": "test"}})
    root = tmp_path / "install"
    (root / "flakeid_designer").mkdir(parents=True)
    (root / "flakeid_designer" / "__main__.py").write_text("VERSION = 'old'\n")
    U.install("abc1234", root=root, log=lambda *a: None)
    mode = lambda p: (root / p).stat().st_mode & 0o777          # noqa: E731
    assert mode("flakeid_designer/Install shortcut (Mac).command") == 0o755
    assert mode("tools/example.sh") == 0o755
    assert mode("flakeid_designer/install_windows.ps1") & 0o111 == 0
