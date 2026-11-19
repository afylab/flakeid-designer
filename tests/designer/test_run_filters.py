"""Library run list: Material / Person / Date filters and Sort (date, kept, AFM kept), remembered per computer;
the chip page button reads 'Table'. (Clicked through in Firefox: tools of the PR.)"""
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_filter_and_sort_menus():
    for key in ('sel("mat","Material"', 'sel("who","Person"', 'sel("date","Date"', '["afm","Sort: AFM kept"]', '["kept","Sort: kept"]'):
        assert key in HTML
    assert 'localStorage.setItem("fd_runfilt"' in HTML
    assert "ids=runFilt(" in HTML


def test_table_button_name():
    assert ">Flake table " not in HTML
    assert 'Table${(R.scans||[]).length>1?" "+escapeHtml(sc):""} ↗' in HTML
