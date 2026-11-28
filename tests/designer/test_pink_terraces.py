"""(2026-11-27m) View: Manual AFM offset (pink while on) shows / hides the pink cross; no Hide button, no pink outline.
Plan / Stack Flake info: a "Terraces" section (every terrace, the chosen one pressed, drawn ones by their name) with a
hover preview (the 100x picture, only that terrace in pink)."""
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_pink_toggle():
    assert 'id="imgOffHide"' not in HTML and "HIDEPINK" not in HTML
    assert "const pinkHidden=()=>!OFFPT.on;" in HTML and "const T=pinkHidden()?null:" in HTML
    assert '3px solid #ff5fd2' not in HTML
    assert '$("imgOff").onclick=()=>{OFFPT.on=!OFFPT.on;offSync();cleanRender()};' in HTML
    assert '#imgOff[aria-pressed="true"]{background:#ff5fd2' in HTML
    assert '$("imgOff").click()' in HTML                     # (key P)


def test_terraces_section():
    assert "Other terraces (regions of interest)" not in HTML
    assert '<div class="tersec"><div class="sechd">Terraces</div>' in HTML
    assert "function terPrevShow(btn,f,t)" in HTML and 'fill="rgba(255,47,208,.42)" stroke="#ff2fd0"' in HTML
    assert "window.innerWidth/3" in HTML
    assert '<b>${escapeHtml(o.user.note)}</b>' in HTML        # (a drawn terrace by its name)
    assert '<label class="sub">name <input type="text" id="drawNote"' in HTML
