"""(2026-11-27) View: AFM center off hides the orange (used) AFM centers too."""
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_orange_follows_red():
    assert "const used=(redHidden()?[]:usedCtrs(f)).map(" in HTML
