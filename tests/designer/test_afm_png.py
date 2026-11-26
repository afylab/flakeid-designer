"""(2026-11-25h) AFM panel: Copy PNG / Save PNG icons under the picture's left; Save writes next to the .spm."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import flakeid_designer.__main__ as D  # noqa: E402

HTML = (ROOT / "flakeid_designer" / "index.html").read_text(encoding="utf-8")


def test_png_path():
    assert D.afm_png_path("/afm/F.0_00001.spm") == Path("/afm/F.0_00001.png")
    assert D.afm_png_path("/afm/scan.001") == Path("/afm/scan.001.png")
    assert D.afm_png_path("/afm/F.0_00001.SPM") == Path("/afm/F.0_00001.png")


def test_page():
    a = HTML.index('id="afmSb"')
    assert a < HTML.index('id="afmCopy" class="iconb"') < HTML.index('id="afmSave" class="iconb"') < HTML.index('id="afmBar"')
    assert ">Copy PNG</button>" not in HTML and "AFM_FIT=1;" in HTML and "Math.min(w/D.w,AFM_FIT*avH/D.h)" in HTML
    assert 'afmPost("/api/afm/savepng"' in HTML and "b.innerHTML=t" in HTML


def test_compact_panel():
    # (2026-11-26) the buttons right of the picture, the profile under the histogram
    a = HTML.index('<svg id="afmBar" class="cbar"')
    assert a < HTML.index('<div class="afmtools"><button id="afmGwy"') < HTML.index('id="afmL3"') < HTML.index('id="afmHover"')
    assert HTML.index('<div class="afmhistb">') < HTML.index('<div class="afmprofb">')
    assert "#afmSide .afmlow{display:grid;grid-template-columns:minmax(0,1fr);" in HTML
