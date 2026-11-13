"""Unit tests of the designer bundle's builder functions (tools/make_designer_bundle.py, tools/make_catalog_100x.py):
one file per 100x view with content-hashed names, the slimmed bundle and its detail / extra files, the staged reads of
the 100x pictures.  Needs numpy and opencv (skipped otherwise)."""
import json
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
M = pytest.importorskip("make_designer_bundle")


def jpg(path, seed=0, q=80, size=(120, 160)):
    rng = np.random.default_rng(seed)
    a = np.full(size + (3,), (180, 120, 90), np.uint8)
    a[30:80, 40:110] = (90, 150, 200)
    a = np.clip(a + rng.normal(0, 4, a.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), a, [cv2.IMWRITE_JPEG_QUALITY, q])
    return path


def test_hashed_names():
    assert M.hashed("detail.json", b"x") == M.hashed("detail.json", b"x")
    assert M.hashed("detail.json", b"x") != M.hashed("detail.json", b"y")
    n = M.hashed("x100_nat_a.jpg", b"abc")
    assert n.startswith("x100_nat_a_") and n.endswith(".jpg") and len(n.split("_")[-1]) == 14


def test_view_files(tmp_path):
    src, out = tmp_path / "cat", tmp_path / "out"
    (src / "100x").mkdir(parents=True)
    out.mkdir()
    jpg(src / "100x/a.jpg", 0, q=100)                     # (the scan's quality: re-encoded at 95, smaller)
    jpg(src / "100x/a_c8.jpg", 1, q=60)                   # (already smaller than quality 95: kept as it is)
    cat = dict(flakes=[dict(id="s-f0001", img="100x/a.jpg", img_c8="100x/a_c8.jpg", img_dt=None)])
    names = M.view_files(cat, src, out, log=lambda *a: None)
    f = cat["flakes"][0]
    assert sorted(names) == sorted([f["img"], f["img_c8"]]) and f["img_dt"] is None
    assert f["img"].startswith("x100_nat_s-f0001_") and f["img_c8"].startswith("x100_c8_s-f0001_")
    assert (out / f["img"]).stat().st_size < (src / "100x/a.jpg").stat().st_size
    assert (out / f["img_c8"]).read_bytes() == (src / "100x/a_c8.jpg").read_bytes()
    assert cv2.imread(str(out / f["img"])).shape == (120, 160, 3)
    again = dict(flakes=[dict(id="s-f0001", img="100x/a.jpg")])
    M.view_files(again, src, out, log=lambda *a: None)
    assert again["flakes"][0]["img"] == f["img"]         # (the same picture: the same name)


def test_slim():
    t = dict(id="r:s-f1.a", poly=[1.4, 2.6, 10.5, 3.2], fine=[dict(dt_nm=0.4, area_um2=3, poly=[[1.2, 1.7, 5.5, 5.4]])],
             debris=[[1.1, 2.2]], steps=[], features=dict(rough=0.2), wrinkles=[[0.4, 0.6]], t=10.0, c=[3, 4])
    f = dict(id="r:s-f1", outline=[0.5, 1.5, 9.4, 2.6], terraces=[t],
             step_lines=[dict(step_nm=0.4, angle_deg=3.0, p=[1.26, 2.5, 7.7, 8.1], length_um=4.0)])
    g = dict(id="r:s-f2", outline=[0.0, 1.0], terraces=[dict(id="r:s-f2.a", poly=[0.2, 0.2], debris=[], steps=[])])
    cat = dict(flakes=[f, g])
    det, ext = M.slim(cat)
    assert f["n_step_lines"] == 1 and "step_lines" not in f and "n_step_lines" not in g
    assert det["flakes"]["r:s-f1"]["step_lines"][0]["p"] == [1, 2, 8, 8]
    assert ext["flakes"] == {"r:s-f1": {"r:s-f1.a": dict(debris=[[1, 2]], steps=[], features=dict(rough=0.2))}}
    assert t["poly"] == [1, 3, 10, 3] and t["fine"][0]["poly"] == [[1, 2, 6, 5]] and t["wrinkles"] == [[0, 1]]
    assert all(isinstance(v, int) for v in f["outline"])
    assert t["t"] == 10.0 and "debris" not in t and "features" not in t
    json.dumps(det), json.dumps(ext)


def test_stage(tmp_path):
    C = pytest.importorskip("make_catalog_100x")
    share = tmp_path / "share"
    share.mkdir()
    ps = [jpg(share / f"{i}.jpg", i, q=95, size=(400, 520)) for i in range(4)]
    st = C.Stage(tmp_path / "stage")
    try:
        st.fetch(ps + [ps[0]])
        assert all(st.path(p).parent == tmp_path / "stage" for p in ps)
        sub = st.substrate(ps)
        full = C.chip_substrate(ps)                       # (every 4th pixel of the full picture, as before)
        assert np.abs(np.asarray(sub) - np.asarray(full)).max() < 6
        st.copy(ps[1], tmp_path / "copy.jpg")
        st.copy(tmp_path / "share" / "2.jpg", tmp_path / "copy2.jpg")
        st.wait()
        assert (tmp_path / "copy.jpg").read_bytes() == ps[1].read_bytes()
        assert C.noise(st.path(ps[0]), cv2.IMREAD_REDUCED_GRAYSCALE_2) > 0
    finally:
        st.close()
    assert not (tmp_path / "stage").exists()
