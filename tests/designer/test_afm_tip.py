"""AFM tip offset (flakeid_designer/afmplan.py): a scan that landed off the site is found in the site's 100x picture
(any of the 8 ways the AFM image's axes lie on the stage, a turned / mirrored microscope -> AFM transform), and the
offset is taken off every point of the list."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
cv2 = pytest.importorskip("cv2")
from flakeid_designer import afmplan as AP  # noqa: E402


def scene(seed):
    rng = np.random.default_rng(seed)
    opt = np.full((400, 400, 3), 120, np.float32)
    for _ in range(45):
        c, r = rng.uniform(20, 380, 2), rng.uniform(10, 35)
        pts = np.array([c + r * rng.uniform(0.5, 1.2) * np.array([np.cos(a), np.sin(a)]) for a in np.sort(rng.uniform(0, 2 * np.pi, 7))], np.int32)
        v = rng.uniform(40, 100)
        cv2.fillPoly(opt, [pts], (v, v * 0.9, v * 0.8))
    return opt, rng


def scan(opt, A, k, centre, scan_um, px_um, n=128, rng=None):
    L = AP.afm_to_optical(A, k, scan_um / n, px_um)
    uu, vv = np.meshgrid(np.arange(n) + 0.5 - n / 2, np.arange(n) + 0.5 - n / 2)
    P = np.tensordot(L, np.stack([uu, vv]), axes=1)
    z = cv2.remap((120 - opt.mean(2)).astype(np.float32), (P[0] + centre[0] - 0.5).astype(np.float32),
                  (P[1] + centre[1] - 0.5).astype(np.float32), cv2.INTER_LINEAR)
    return z * 0.4 + rng.normal(0, 0.4, z.shape)


@pytest.mark.parametrize("deg,mirror,k,d", [(0, False, 1, (25, -14)), (90.4, False, 0, (-30, 8)), (33, True, 5, (12, 18)),
                                            (271, True, 7, (40, 40))])
def test_tip_offset_found(deg, mirror, k, d):
    opt, rng = scene(deg != 0)
    px = 0.114
    e = AP._edges(opt.mean(2))
    ys, xs = np.nonzero(e[150:250, 150:250] > 0.5)
    site = np.array([150 + xs[len(xs) // 2], 150 + ys[len(ys) // 2]], float)          # (sites are on an edge)
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = 1.03 * (R @ np.diag([1, -1]) if mirror else R)
    r = AP.tip_offset(scan(opt, A, k, site + d, 10.0, px, rng=rng), 10.0, opt, px, site, A)
    assert r["orient"] == k and r["ok"]
    assert np.allclose(r["d_px"], d, atol=1.5)
    assert np.allclose(r["offset"], A @ (np.array(d) * px), atol=2.2 * px * 1.03)   # (a 1.5 px error, turned)


def test_apply_tip_rewrites_list(tmp_path):
    sites = [dict(n=i + 1, flake=f"f{i}", chip=[i * 100.0, 50.0], px=[0, 0]) for i in range(3)]
    b = dict(scan="chip_1", refs=[dict(flake="a", chip=[0, 0], afm=[1000, 2000, -5]), dict(flake="b", chip=[1000, 0], afm=[2000, 2000, -6]),
                                  dict(flake="c", chip=[0, 1000], afm=[1000, 3000, -7])], sites=sites)
    pl = AP.make_plan(b, tmp_path)
    pf = Path(pl["folder"]) / "plan.json"
    a0 = [s["afm"][:] for s in pl["sites"]]
    pl = AP.apply_tip(pl, pf, dict(offset=[2.5, -1.25], site=1))
    assert [s["afm"][:2] for s in pl["sites"]] == [[x - 2.5, y + 1.25] for x, y, *_ in a0]
    assert [s["afm"][2] for s in pl["sites"]] == [z for _, _, z in a0]                  # (Z unchanged)
    assert "X point: " + AP.n2(a0[0][0] - 2.5) in (pf.parent / pl["pgm"]).read_text()
    pl2 = AP.make_plan(dict(b, id=pl["id"]), tmp_path)                                  # (made again: the offset kept)
    assert pl2["sites"][0]["afm"][:2] == pl["sites"][0]["afm"][:2]
    pl3 = AP.apply_tip(pl2, pf, None)
    assert [s["afm"] for s in pl3["sites"]] == a0
