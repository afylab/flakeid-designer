"""(2026-11-24) Chip map: a button on each flake card (top right of its text, above Keep / Discard) and on each run card
(one per chip), the window closed with ✕, and a Flake # search at its top as the Library's (the flakes with those
numbers on the chip shown, or the chip that has them, ringed; also when not among the flakes shown)."""
from pathlib import Path

from flakeid_designer import __main__ as M

HTML = (Path(M.__file__).parent / "index.html").read_text(encoding="utf-8")


def test_card_button():
    assert 'data-cmap title="This flake on the chip\'s 10x scan (chip map)">Chip map</span>' in HTML
    # (2026-11-25) on the scores' line, right: a long thickness (142.9 nm) on the first line pushed it out of the card
    assert ".cinfo{display:grid;grid-template-columns:auto minmax(0,1fr) auto;" in HTML and ".cinfo .cmapb{grid-column:3;grid-row:2;justify-self:end" in HTML
    assert ".cinfo .ct{grid-column:2/-1;grid-row:1;" in HTML
    assert HTML.count("cmapWire(b,f,t);") == 2 and "fmapOpen(false,{f,t})" in HTML


def test_run_card_button():
    assert 'data-cmaprun="${escapeHtml(R.run||id)}|${escapeHtml(sc)}"' in HTML
    assert 'k===sc||k===r+":"+sc);if(k)fmapOpen(false,null,k)' in HTML


def test_close_is_x_and_search():
    assert '<button id="fmapClose" class="xclose" title="Close (Esc)" aria-label="Close">✕</button>' in HTML
    assert '<input type="search" id="fmapFind" placeholder="e.g. 30"' in HTML
    assert '$("fmapFind").addEventListener("input",fmapFind)' in HTML
    assert 'e.key==="Escape"&&e.target.value){e.preventDefault();e.stopPropagation()' in HTML     # (Esc clears, the window stays)


def test_close_button_room():
    # (2026-11-25e) the floating ✕ had no room kept in the chip map's header: View ↻ ran up against it
    assert "#fmapDlg .dhead,#afmFilesDlg .dhead{padding-right:64px}" in HTML
