"""(2026-11-25d) AFM centers: every position recorded (".clean" hist / hi; < and > in the 100x window), and a list marked
AFM scan complete records each flake's AFM center then with its AFM files (plan used_centers; shown in orange)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flakeid_designer.__main__ as D  # noqa: E402
from test_run_decisions import RUN, quiet, share, store_for  # noqa: E402,F401

HTML = (ROOT / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def plan(fl1, fl2):
    return dict(id="p1", name="List 1", scan_um=15.0, sites=[dict(n=1, flake=fl1, px=[10, 10]), dict(n=2, flake=fl2, px=[5, 5]),
                                                            dict(n=3, flake="nofile", px=[1, 1])],
                results={"1": dict(source="/afm/a_00001.spm")}, more={"2": [dict(source="/afm/a_00002.spm")]})


def test_used_centers_recorded(tmp_path, monkeypatch):
    run_dir, lib_a, _ = share(tmp_path)
    a = store_for(lib_a, run_dir, "avram@mac", monkeypatch)
    f1, f2 = f"{RUN}:BN7_1-f0034", f"{RUN}:BN7_1-f0035"
    a.put_review(f"{f1}.clean", dict(clean=[120.5, 80.0], rel=[0.4, 0.3], um=[1.0, 2.0], flake=f1, hist=[[100, 70], [120.5, 80.0]], hi=1))
    pl = plan(f1, f2)
    U = D.afm_used_centers(pl, a.review_state()[2], 1000)
    assert set(U) == {f1, f2}                                        # (only points with an AFM file)
    assert U[f1]["clean"] == [120.5, 80.0] and U[f1]["files"] == ["/afm/a_00001.spm"] and U[f1]["at"] == 1000 and U[f1]["scan_um"] == 15.0
    assert U[f2]["clean"] is None and U[f2]["files"] == ["/afm/a_00002.spm"]   # (no red cross set: the picture's center)
    # marked complete again after the center moved: the earlier record kept
    a.put_review(f"{f1}.clean", dict(clean=[10.0, 10.0], flake=f1, hist=[[100, 70], [120.5, 80.0], [10.0, 10.0]], hi=2))
    U = D.afm_used_centers(pl, a.review_state()[2], 2000)
    assert U[f1]["clean"] == [10.0, 10.0] and U[f1]["before"][0]["clean"] == [120.5, 80.0] and U[f1]["before"][0]["at"] == 1000
    # the history alone (red cross back at the picture's center) is still an entry the page gets
    a.put_review(f"{f1}.clean", dict(clean=None, flake=f1, hist=[[100, 70]], hi=1))
    assert any(r["id"] == f"{f1}.clean" for r in a.list_review())
    json.dumps(pl)


def test_page():
    assert '<button id="imgCtrPrev" aria-label="Previous AFM center position">&lt;</button><button id="imgCtrNext"' in HTML
    assert "hist:h,hi};CLEANSEEN.add(f.id);" in HTML and "function setCleanPt(f,p,quiet,nav)" in HTML
    assert 'stroke="#ff9a1f"' in HTML and "USEDC=U;" in HTML and "function ctrStep(d)" in HTML
