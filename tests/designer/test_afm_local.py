"""AFM screening reads copies kept on this computer (2026-11-05v): flakeid_designer.afmplan.here / DiskCache.file."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flakeid_designer import __main__ as M  # noqa: E402
from flakeid_designer import afmplan as AP  # noqa: E402


def overwrite_same_stat(f, data):
    """The share's file changed in content only: same size and time, so a read of the share would show it."""
    st = f.stat()
    assert len(data) == st.st_size
    f.write_bytes(data)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))


def setup(tmp_path, monkeypatch):
    disk = M.DiskCache(tmp_path / "here", 100)
    monkeypatch.setattr(AP, "LOCAL", disk.file)
    plan = tmp_path / "share" / "plan"
    plan.mkdir(parents=True)
    return disk, plan


def test_height_read_from_the_copy(tmp_path, monkeypatch):
    disk, plan = setup(tmp_path, monkeypatch)
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    f = plan / "001_a.spm.full2.npz"
    np.savez(f, z=z, pixel_nm=5.0)
    r = dict(png="001_a.spm.png")
    assert np.array_equal(AP.height_full(plan, r)[0], z)
    overwrite_same_stat(f, b"x" * f.stat().st_size)
    z2, px, full = AP.height_full(plan, r)
    assert np.array_equal(z2, z) and px == 5.0 and full


def test_a_changed_file_is_copied_again(tmp_path, monkeypatch):
    disk, plan = setup(tmp_path, monkeypatch)
    f = plan / "a.spm"
    f.write_bytes(b"first")
    c1 = AP.here(f)
    assert c1 != f and c1.read_bytes() == b"first"
    f.write_bytes(b"second!")
    c2 = AP.here(f)
    assert c2.read_bytes() == b"second!" and not c1.exists()          # (the old copy goes)


def test_spm_read_through_the_copy(tmp_path, monkeypatch):
    disk, plan = setup(tmp_path, monkeypatch)
    import flakeid.afm as A
    src = plan / "a.spm"
    src.write_bytes(b"spm")
    seen = []

    def stop(p):
        seen.append(Path(p))
        raise RuntimeError("stop")
    monkeypatch.setattr(A, "read_spm", stop)
    try:
        AP.height_full(plan, dict(png="001_a.spm.png", source=str(src)))
    except RuntimeError:
        pass
    assert seen and seen[0].parent == disk.root and seen[0].read_bytes() == b"spm"


def test_no_cache_reads_the_share(tmp_path, monkeypatch):
    monkeypatch.setattr(AP, "LOCAL", M.DiskCache(None).file)
    f = tmp_path / "a.spm"
    f.write_bytes(b"x")
    assert AP.here(f) == f
    monkeypatch.setattr(AP, "LOCAL", None)
    assert AP.here(f) == f
    assert AP.here(tmp_path / "missing.spm") == tmp_path / "missing.spm"


def test_keep_local_copies_every_file(tmp_path, monkeypatch):
    disk, plan = setup(tmp_path, monkeypatch)
    png, npz, src = plan / "001_a.spm.png", plan / "001_a.spm.npz", tmp_path / "share" / "a.spm"
    for f in (png, npz, src):
        f.write_bytes(f.name.encode())
    np.savez(plan / "001_a.spm.full2.npz", z=np.zeros((2, 2), np.float32), pixel_nm=1.0)
    AP.keep_local(plan, dict(png=png.name, source=str(src)))
    assert len(list(disk.root.iterdir())) == 4


def test_list_copy_survives_a_missing_file(tmp_path, monkeypatch):
    disk, plan = setup(tmp_path, monkeypatch)
    good = plan / "002_b.spm.png"
    good.write_bytes(b"png")
    np.savez(plan / "002_b.spm.full2.npz", z=np.zeros((2, 2), np.float32), pixel_nm=1.0)

    class Store:
        def afm_plan(self, pid):
            return dict(results={"1": dict(png="001_gone.png", file="gone", source=str(tmp_path / "gone.spm")),
                                 "2": dict(png=good.name, file="b")}), plan / "plan.json"
    M.afm_copy_list(Store(), AP, "L")
    for _ in range(500):
        if "L" not in M.AFM_COPYING:
            break
        import time
        time.sleep(0.01)
    assert "L" not in M.AFM_COPYING
    assert disk.get(good, good.stat()) == b"png"


def test_refresh_fetches_again_from_the_share(tmp_path, monkeypatch):
    disk, plan = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "AFM_DISK", disk)
    src, png = tmp_path / "share" / "a.spm", plan / "001_a.spm.png"
    src.write_bytes(b"old")
    png.write_bytes(b"pic")
    AP.here(src), AP.here(png)
    overwrite_same_stat(src, b"new")                      # (only a Refresh can see this one)
    assert AP.here(src).read_bytes() == b"old"
    M.afm_drop_list(dict(results={"1": dict(source=str(src), png=png.name)}), plan / "plan.json")
    assert list(disk.root.iterdir()) == []
    assert AP.here(src).read_bytes() == b"new"


def test_refresh_also_drops_files_new_in_the_folder(tmp_path, monkeypatch):
    disk, plan = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "AFM_DISK", disk)
    other = tmp_path / "share" / "b.spm"
    other.write_bytes(b"b")
    AP.here(other)
    M.afm_drop_list(dict(results={}), plan / "plan.json", [str(other)])
    assert list(disk.root.iterdir()) == []
