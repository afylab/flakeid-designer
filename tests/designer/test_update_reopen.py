"""(2026-11-27) After an update: "Close the window and reopen the designer" (the restart in place did not work)."""
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_no_restart():
    assert 'fetch("/api/update/restart"' not in HTML and "Restarting the designer" not in HTML
    assert "<b>Close the window and reopen the designer</b>" in HTML
    assert "Then close the window and reopen the designer." in HTML
