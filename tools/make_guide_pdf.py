"""PDF of the scan guide (docs/scan_guide.md with its pictures) for reading and printing.

    python tools/make_guide_pdf.py                       -> docs/scan_guide.pdf
    python tools/make_guide_pdf.py --html                (also keep the intermediate docs/scan_guide.html)

Markdown -> HTML with markdown-it (CommonMark + tables: the guide's lists are indented 3 spaces, which Python-Markdown
would not nest), then printed by headless Chromium (Playwright) on US Letter with page numbers.  Each "Part" starts on
a new page; pictures and table rows are not split across pages.  Run it on the analysis Mac (needs markdown-it-py,
mdit-py-plugins and Playwright's Chromium), not on the microscope PC.
"""
from __future__ import annotations

import argparse
import datetime
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCENT = "#6c5cb4"          # the logo's violet

CSS = """
@page { size: Letter; margin: 0.75in 0.7in 0.8in 0.7in; }
* { box-sizing: border-box; }
html { font-size: 10.5pt; }
body { margin: 0; color: #1d1b26; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; line-height: 1.45;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.title { display: flex; align-items: center; gap: 14px; padding-bottom: 10px; border-bottom: 2px solid ACCENT;
         margin-bottom: 14px; }
.title img { width: 52px; height: 52px; }
.title .sub { color: #5b5770; font-size: 9.5pt; margin-top: 2px; }
h1 { font-size: 22pt; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
h2 { font-size: 15pt; color: ACCENT; margin: 0 0 8px; padding-top: 2px; break-before: page; text-wrap: balance; }
h2.first { break-before: auto; }
h3 { font-size: 12pt; margin: 18px 0 6px; break-after: avoid; text-wrap: balance; }
h2 + p, h3 + p { margin-top: 4px; }
p, li { orphans: 3; widows: 3; }
p { margin: 6px 0; }
ul, ol { margin: 6px 0; padding-left: 22px; }
li { margin: 3px 0; }
li > p { margin: 3px 0; }
a { color: ACCENT; text-decoration: none; }
strong { color: #14121c; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f1eff8; padding: 1px 4px;
       border-radius: 3px; overflow-wrap: anywhere; }
pre { background: #f1eff8; border-left: 3px solid ACCENT; padding: 8px 10px; border-radius: 3px; overflow-wrap: anywhere;
      white-space: pre-wrap; break-inside: avoid; margin: 8px 0; }
pre code { background: none; padding: 0; font-size: 8.6pt; }
hr { display: none; }
figure { margin: 10px 0 12px; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 3.8in; border: 1px solid #d9d6e6; border-radius: 4px;
             box-shadow: 0 1px 4px rgba(30, 20, 70, 0.12); }
figure img.wide { width: 100%; max-height: none; }
figcaption { font-size: 8.8pt; color: #5b5770; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; }
th, td { border: 1px solid #d9d6e6; padding: 5px 7px; vertical-align: top; text-align: left; }
th { background: #ece9f6; }
tr { break-inside: avoid; }
td:first-child { width: 24%; }
""".replace("ACCENT", ACCENT)


def slug(text):
    """GitHub-style heading anchor (the guide's internal links use these)."""
    t = re.sub(r"<[^>]+>", "", text).strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def build_html(md_path: Path) -> str:
    from markdown_it import MarkdownIt
    md_text = md_path.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", md_text, re.M).group(1).strip()
    md_text = re.sub(r"^# .+\n", "", md_text, count=1, flags=re.M)        # (the title block replaces it)
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = md.render(md_text)

    # heading anchors; every Part on a new page (not the first)
    seen = {"n": 0}

    def head(m):
        lvl, inner = m.group(1), m.group(2)
        cls = ""
        if lvl == "2":
            seen["n"] += 1
            cls = ' class="first"' if seen["n"] == 1 else ""
        return f'<h{lvl} id="{slug(inner)}"{cls}>{inner}</h{lvl}>'
    body = re.sub(r"<h([23])>(.*?)</h\1>", head, body)

    # pictures on their own: figures with the alt text as caption; wide screenshots fill the width
    from PIL import Image

    def fig(m):
        src, alt = m.group(1), html.unescape(m.group(2))
        cls = ""
        try:
            w, h = Image.open(md_path.parent / src).size
            cls = ' class="wide"' if w > 1100 else ""
        except OSError:
            pass
        return f'<figure><img src="{src}" alt="{html.escape(alt)}"{cls}><figcaption>{html.escape(alt)}</figcaption></figure>'
    body = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)" ?/?></p>', fig, body)

    logo = (ROOT / "flakeid/scope/assets/logo.svg").resolve().as_uri()
    date = datetime.date.today().strftime("%B %-d, %Y")
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style></head>'
            f'<body><div class="title"><img src="{logo}" alt=""><div><h1>{html.escape(title)}</h1>'
            f'<div class="sub">FlakeIdentifier scan program &middot; Young Lab, UCSB &middot; {date}</div></div></div>'
            f"{body}</body></html>")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default=str(ROOT / "docs/scan_guide.md"))
    ap.add_argument("-o", "--out", help="PDF to write (default: next to the Markdown file)")
    ap.add_argument("--html", action="store_true", help="keep the intermediate HTML")
    a = ap.parse_args(argv)
    md_path = Path(a.md).resolve()
    out = Path(a.out) if a.out else md_path.with_suffix(".pdf")
    htm = md_path.with_suffix(".html")             # next to the Markdown file: the pictures' relative paths resolve
    htm.write_text(build_html(md_path), encoding="utf-8")
    from playwright.sync_api import sync_playwright
    foot = ('<div style="width:100%;font:8px Helvetica,Arial,sans-serif;color:#8a86a0;padding:0 0.7in;'
            'display:flex;justify-content:space-between"><span>FlakeIdentifier: scanning chips with the autoscanner</span>'
            '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(htm.as_uri(), wait_until="networkidle")
        pg.pdf(path=str(out), format="Letter", print_background=True, display_header_footer=True,
               header_template="<div></div>", footer_template=foot,
               margin={"top": "0.75in", "bottom": "0.8in", "left": "0.7in", "right": "0.7in"})
        b.close()
    if not a.html:
        htm.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
