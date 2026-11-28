"""(2026-11-27g) Plan / Stack layer cards (flake number bold top left, run top right); measuring by press-drag-release,
a red "Clear measures" only while lines are drawn."""
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2] / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_layer_card():
    assert '<div class="lhd"><b class="lfl" title=' in HTML and '<span class="lrun" title=' in HTML
    assert '<span class="lpar">${escapeHtml(fpar)}</span>' in HTML and '<span class="lrn">${escapeHtml(frun)}</span>' in HTML
    assert ".layer .lhd{display:flex;align-items:baseline;justify-content:space-between;" in HTML
    assert '<span class="lord" title="Pick-up order">${num(i)}</span>' in HTML


def test_measure_drag():
    assert '$("imgSvg").addEventListener("pointerup",measUp)' in HTML and "Math.hypot(e.clientX-D.x,e.clientY-D.y)>=4" in HTML
    assert 'if(!MEAS.a)MEAS.a=q;else{MEAS.lines.push' not in HTML           # (no click-click any more)
    assert '<button id="imgMeasClr" class="danger" hidden' in HTML and ">Clear measures</button>" in HTML
    assert '$("imgMeasClr").hidden=!MEAS.lines.length;' in HTML
