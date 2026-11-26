"""(2026-11-25f) A search box under + Add run: the run list filtered by run name / folder."""
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_run_search():
    i, j = HTML.index('id="runAddRow"'), HTML.index('id="runFind"')
    assert i < j < HTML.index('id="pgmMergeRow"')                     # (below + Add run)
    assert "if(RFQ)a=a.filter(runMatch);" in HTML and "arch=all.filter(k=>isArch(k)&&runMatch(k))" in HTML
    assert '$("runFind").hidden=false;' in HTML
