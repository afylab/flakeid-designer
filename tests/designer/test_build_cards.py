"""(2026-11-19) A run being built is shown as a run card: /api/builds gives each build its chips' material and oxide (from
<run>/<run>.json, as the Library's run cards), the page colours its bar by them; rhombohedral graphite (graphite on 200 nm
oxide or more) has its own bar (mat-rh: the graphite colour with white staircase lines)."""
import json
from pathlib import Path

from flakeid_designer import __main__ as M

HTML = (Path(M.__file__).parent / "index.html").read_text(encoding="utf-8")


def test_builds_carry_their_pieces(tmp_path):
    d = tmp_path / "2-3-4_run_1"
    d.mkdir()
    (d / "2-3-4_run_1.json").write_text(json.dumps({"pieces": [{"name": "2-3-4_1", "material": "graphite", "oxide_nm": 285.0}]}))
    b = M.Builder()
    b._set(d, state="building", msg="reading the run's results", frac=0.0)
    st = b.status()
    assert st[0]["pieces"] == [{"name": "2-3-4_1", "material": "graphite", "oxide_nm": 285.0}]
    e = tmp_path / "no_json_run"
    e.mkdir()
    b._set(e, state="queued")
    assert {x["run"]: x["pieces"] for x in b.status()}["no_json_run"] == []


def test_page_draws_builds_as_cards():
    assert 'class="runrow runcard buildcard ${runKind(R,"cls")}"' in HTML
    assert '.runcard.mat-rh .runbar{background:#5c3859 url("data:image/svg+xml' in HTML
    assert 'kinds[0]==="Rhombohedral graphite"?"mat-rh"' in HTML
