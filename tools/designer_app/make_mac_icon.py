"""(2026-11-28) The Mac icon (designer.icns and designer_mac.png) from static/designer.svg, drawn on Apple's icon grid:
the artwork 824 of 1024 px with a soft shadow, as every other app in the Dock (before: the whole square, so the
designer looked larger than the other icons).  Renders with headless Chromium (Playwright), packs with iconutil (macOS).
    python3 tools/designer_app/make_mac_icon.py"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "flakeid_designer" / "static"
SIZES = [16, 32, 128, 256, 512]


def padded_svg():
    src = (STATIC / "designer.svg").read_text(encoding="utf-8")
    inner = re.sub(r"^<svg[^>]*>|</svg>\s*$", "", src.strip())
    inner = re.sub(r"<title>.*?</title>", "", inner)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="100%" height="100%">'
            '<defs><filter id="sh" x="-10%" y="-10%" width="120%" height="125%">'
            '<feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#000" flood-opacity=".3"/></filter></defs>'
            f'<g filter="url(#sh)"><svg x="100" y="100" width="824" height="824" viewBox="0 0 64 64">{inner}</svg></g></svg>')


def render(svg, sizes):
    from playwright.sync_api import sync_playwright
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        for px in sizes:
            pg = b.new_page(viewport=dict(width=px, height=px))
            pg.set_content('<html style="margin:0;height:100%"><body style="margin:0;height:100%;background:transparent">'
                           f'<div style="width:{px}px;height:{px}px">{svg}</div></body></html>')
            out[px] = pg.screenshot(omit_background=True, clip=dict(x=0, y=0, width=px, height=px))
            pg.close()
        b.close()
    return out


def main():
    svg = padded_svg()
    pngs = render(svg, sorted({s for s in SIZES} | {2 * s for s in SIZES}))
    with tempfile.TemporaryDirectory() as t:
        ics = Path(t) / "designer.iconset"
        ics.mkdir()
        for s in SIZES:
            (ics / f"icon_{s}x{s}.png").write_bytes(pngs[s])
            (ics / f"icon_{s}x{s}@2x.png").write_bytes(pngs[2 * s])
        subprocess.run(["iconutil", "-c", "icns", str(ics), "-o", str(STATIC / "designer.icns")], check=True)
    (STATIC / "designer_mac.png").write_bytes(pngs[512])
    print("written:", STATIC / "designer.icns", STATIC / "designer_mac.png")


if __name__ == "__main__":
    main()
