"""(2026-11-19) Recent runs are drawn as run cards (coloured bar by material, rhombohedral staircase too); a run not in
the list gets its chips from /api/runpieces (its <run>.json, read only).  Only the rhombohedral bar puts its text on a
highlight; BN and graphite keep white text.  A remembered filter value that no run has any more (e.g. the old
"Rhombohedral graphite") is dropped instead of being kept as an extra option."""
import json
from pathlib import Path

from flakeid_designer import __main__ as M

HTML = (Path(M.__file__).parent / "index.html").read_text(encoding="utf-8")
SRC = Path(M.__file__).read_text(encoding="utf-8")


def test_run_pieces_for_recent(tmp_path):
    d = tmp_path / "2-3-4_run_1"
    d.mkdir()
    (d / "2-3-4_run_1.json").write_text(json.dumps({"pieces": [{"name": "2-3-4_1", "material": "graphite", "oxide_nm": 285.0}]}))
    assert M.run_pieces(str(d))[0]["oxide_nm"] == 285.0
    assert 'if path == "/api/runpieces":' in SRC


def test_recent_runs_are_cards():
    assert 'class="runrow runcard reccard ${runKind(R,"cls")}"' in HTML
    assert "recentPieces()" in HTML and 'fetch("/api/runpieces?dir="' in HTML


def test_highlight_only_on_rhombohedral():
    assert ".runcard.mat-rh .runbar>span{background:var(--cardon);color:var(--ink)" in HTML
    assert "\n.runbar>span{" not in HTML


def test_stale_filter_dropped():
    assert "mats.push(RF.mat)" not in HTML and "who.push(RF.who)" not in HTML
    assert 'if(RF.mat&&!ids.some(k=>runMats(k).includes(RF.mat))){RF.mat="";ch=true}' in HTML
