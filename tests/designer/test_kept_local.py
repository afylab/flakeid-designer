"""Kept flakes' 100x pictures copied to this computer (2026-11-05w): flakeid_designer KeptCopy / kept_files."""
import json
import os
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flakeid_designer import __main__ as M  # noqa: E402


def keep(*flakes, sig="s"):
    return sig, None, {f"run1:chip1-f{n:04d}.a": dict(by="a", updated=1, body=dict(decision=d)) for n, d in flakes}


class Store:
    def __init__(self, ddir, state):
        self.lock, self.state = threading.Lock(), state
        self.runs = {"run1": dict(dir=ddir, doc=dict(run="run1"))}

    def review_state(self):
        return self.state


def make_run(root):
    d, v = root / "run1" / "designer", root / "run1" / "chip1"
    d.mkdir(parents=True)
    (v / "100x").mkdir(parents=True)
    flakes, H = [], {}
    for n in (1, 2, 3):
        flakes.append(dict(id=f"run1:chip1-f{n:04d}", rank=n, img=f"x100_nat_f{n}.jpg", img_c8=f"x100_c8_f{n}.jpg", img_ir=None))
        for f in flakes[-1]:
            if f.startswith("img") and flakes[-1][f]:
                (d / flakes[-1][f]).write_bytes(f"view{n}".encode())
        for s in ("", "_sg"):
            (v / "100x" / f"F{n}{s}.jpg").write_bytes(f"pic{n}{s}".encode())
        (v / "100x" / f"F{n}.tif").write_bytes(b"tiff")
        H[str(n)] = [dict(f=f"100x/F{n}.jpg", sg=f"100x/F{n}_sg.jpg")]
    (d / "bundle.json").write_text(json.dumps(dict(run="run1", flakes=flakes)))
    (v / "index.html").write_text('<script>\nconst D={"name":"chip1_tiles"};\nconst H100=' + json.dumps(H) + ";\n</script>")
    return d


@pytest.fixture
def disk(tmp_path, monkeypatch):
    c = M.DiskCache(tmp_path / "here", 1000)
    monkeypatch.setattr(M, "KEPT_DISK", c)
    return c


def wait(copy):
    for _ in range(500):
        with copy.lock:
            if not copy.running:
                return
        time.sleep(0.01)
    raise AssertionError("still copying")


def names(disk):
    return len(list(disk.root.iterdir())) if disk.root.is_dir() else 0


def test_only_kept_flakes_are_listed(tmp_path):
    d = make_run(tmp_path)
    got = sorted(f.name for f in M.kept_files(Store(d, keep((2, "keep"), (3, "discard"))), "run1"))
    assert got == ["F2.jpg", "F2_sg.jpg", "x100_c8_f2.jpg", "x100_nat_f2.jpg"]


def test_nothing_kept_nothing_copied(tmp_path, disk):
    c = M.KeptCopy()
    c.want(Store(make_run(tmp_path), keep((1, "discard"))), "run1")
    wait(c)
    assert names(disk) == 0


def test_kept_copies_serve_the_picture(tmp_path, disk):
    d = make_run(tmp_path)
    c = M.KeptCopy()
    c.want(Store(d, keep((1, "keep"))), "run1")
    wait(c)
    assert names(disk) == 4
    f = d / "x100_nat_f1.jpg"
    st = f.stat()
    f.write_bytes(b"x" * st.st_size)                    # (changed on the share, same size and time)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert disk.get(f, f.stat()) == b"view1"


def test_keeping_another_flake_adds_it(tmp_path, disk):
    d = make_run(tmp_path)
    s, c = Store(d, keep((1, "keep"))), M.KeptCopy()
    c.want(s, "run1")
    wait(c)
    s.state = keep((1, "keep"), (2, "keep"))
    c.want(s, "run1")
    wait(c)
    assert names(disk) == 8


def test_asked_while_running_goes_again(tmp_path, disk, monkeypatch):
    d = make_run(tmp_path)
    s, c = Store(d, keep((1, "keep"))), M.KeptCopy()
    monkeypatch.setattr(M.threading, "Thread", lambda **k: type("T", (), {"start": lambda self: None})())
    c.want(s, "run1")
    c.want(s, "run1")
    assert c.jobs == ["run1"] and c.again == {"run1"}
    s.state = keep((1, "keep"), (2, "keep"))
    c._work(s)
    assert names(disk) == 8 and not c.jobs and not c.running


def test_stops_when_the_disk_is_nearly_full(tmp_path, disk, monkeypatch):
    monkeypatch.setattr(M.shutil, "disk_usage", lambda p: namedtuple("U", "free")(1))
    c = M.KeptCopy()
    c.want(Store(make_run(tmp_path), keep((1, "keep"))), "run1")
    wait(c)
    assert names(disk) == 0 and not c.jobs


def test_unreadable_decisions_or_no_cache(tmp_path, monkeypatch):
    d = make_run(tmp_path)
    with pytest.raises(OSError):
        list(M.kept_files(Store(d, ("none", None, {})), "run1"))
    monkeypatch.setattr(M, "KEPT_DISK", M.DiskCache(None))
    c = M.KeptCopy()
    c.want(Store(d, keep((1, "keep"))), "run1")
    assert not c.running and not c.jobs
