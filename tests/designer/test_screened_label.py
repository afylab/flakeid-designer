"""(2026-11-19) The run card says how far its screening is in words: Not screened (red) / Partially screened (amber) /
All screened (green), the count on hover."""
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_screened_words():
    assert '${x.state==="done"?"All screened":x.state==="part"?"Partially screened":"Not screened"}' in HTML
    assert 'title="${x.d} of ${x.n} flakes screened"' in HTML
    assert ".runcnt .none{color:var(--err)}" in HTML
