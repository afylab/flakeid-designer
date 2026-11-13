"""Wiki package of the scan guide (docs/scan_guide.md and its pictures), for any kind of wiki.

    python tools/make_guide_wiki.py              -> docs/scan_guide_wiki/ and docs/scan_guide_wiki.zip

The package:
  images/                    the pictures, named scan_guide_NN_....png (unique names: a wiki's files are shared by all
                             its pages)
  scan_guide.md              Markdown for Markdown wikis (GitHub / GitLab wiki, Wiki.js, BookStack, Notion import,
                             Obsidian); pictures as images/<name>
  scan_guide.mediawiki       wikitext for MediaWiki: [[File:...]] pictures (upload images/ first), {| tables |}
  scan_guide_paste.html      one self-contained page (pictures embedded): open in a browser, select all, copy, paste
                             into a rich-text editor (Confluence, Google Sites, SharePoint, Notion)
  README.txt                 which file is for which wiki
Internal links (to other sections) are kept in each format.  Needs markdown-it-py (on the analysis Mac).
"""
from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "scan_guide_"


def slug(text):
    t = re.sub(r"<[^>]+>", "", text).strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def parse(md_text):
    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    return md, md.parse(md_text)


def image_name(src):
    return PREFIX + Path(src).name


# ------------------------------------------------------------------------------------------------ MediaWiki
def _esc(s):
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"(\[\[|\]\]|\{\{|\}\}|~~~|'')", lambda m: f"<nowiki>{m.group(1)}</nowiki>", s)
    return s


def mw_inline(tok, anchors, in_table=False):
    out = []
    link = None
    for c in tok.children or []:
        t = c.type
        if t == "text":
            s = _esc(c.content)
            out.append(s.replace("|", "{{!}}") if in_table else s)
        elif t in ("softbreak", "hardbreak"):
            out.append(" " if t == "softbreak" else "<br>")
        elif t == "strong_open" or t == "strong_close":
            out.append("'''")
        elif t == "em_open" or t == "em_close":
            out.append("''")
        elif t == "code_inline":
            s = c.content.replace("|", "&#124;") if in_table else c.content
            out.append(f"<code><nowiki>{s}</nowiki></code>")
        elif t == "link_open":
            link = (c.attrs.get("href", ""), len(out))
        elif t == "link_close" and link:
            href, k = link
            label = "".join(out[k:])
            del out[k:]
            if href.startswith("#"):
                target = anchors.get(href[1:], href[1:])
                out.append(f"[[#{target}|{label}]]")
            elif re.match(r"https?://", href):
                out.append(f"[{href} {label}]")
            else:
                out.append(label)
            link = None
        elif t == "image":
            alt = "".join(x.content for x in (c.children or []))
            out.append(f"[[File:{image_name(c.attrs['src'])}|frameless|none|800px|{_esc(alt)}]]")
    return "".join(out).strip()


def to_mediawiki(tokens, anchors, sizes):
    lines, stack = [], []        # stack: list markers ('*' / '#')
    item_first = []              # per open list item: has its first paragraph been written?
    i = 0
    while i < len(tokens):
        tk = tokens[i]
        t = tk.type
        if t == "heading_open":
            lvl = int(tk.tag[1])
            text = mw_inline(tokens[i + 1], anchors)
            if lvl == 1:
                lines.append(f"<!-- page title: {text} -->")
            else:
                eq = "=" * lvl
                lines += ["", f"{eq} {text} {eq}"]
            i += 3
            continue
        if t in ("bullet_list_open", "ordered_list_open"):
            stack.append("*" if t == "bullet_list_open" else "#")
            if len(stack) == 1:
                lines.append("")
        elif t in ("bullet_list_close", "ordered_list_close"):
            stack.pop()
        elif t == "list_item_open":
            item_first.append(False)
        elif t == "list_item_close":
            item_first.pop()
        elif t == "paragraph_open":
            inl = tokens[i + 1]
            ch = inl.children or []
            only_img = len(ch) == 1 and ch[0].type == "image"
            if only_img:
                src = ch[0].attrs["src"]
                alt = "".join(x.content for x in (ch[0].children or []))
                w = sizes.get(src, 800)
                spec = f"thumb|none|800px|{_esc(alt)}" if w > 800 else f"frame|none|{_esc(alt)}"
                text = f"[[File:{image_name(src)}|{spec}]]"
            else:
                text = mw_inline(inl, anchors)
            if stack:
                pre = "".join(stack)
                if item_first and item_first[-1]:
                    pre += ":"                      # a further paragraph / picture of the same item
                elif item_first:
                    item_first[-1] = True
                lines.append(f"{pre} {text}")
            else:
                lines += ["", text]
            i += 3
            continue
        elif t == "table_open":
            lines += ["", '{| class="wikitable"']
            row = []
        elif t == "tr_open":
            row = []
        elif t in ("th_open", "td_open"):
            row.append(("!" if t == "th_open" else "|", mw_inline(tokens[i + 1], anchors, in_table=True)))
            i += 3
            continue
        elif t == "tr_close":
            if lines[-1] != '{| class="wikitable"':
                lines.append("|-")
            mark = row[0][0]
            lines.append(mark + " " + f" {mark}{mark} ".join(c for _, c in row))
        elif t == "table_close":
            lines.append("|}")
        elif t == "hr":
            pass
        i += 1
    out = "\n".join(lines).strip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", out)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default=str(ROOT / "docs/scan_guide.md"))
    ap.add_argument("-o", "--out", default=str(ROOT / "docs/scan_guide_wiki"))
    a = ap.parse_args(argv)
    from PIL import Image
    md_path = Path(a.md).resolve()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    text = md_path.read_text(encoding="utf-8")
    title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
    srcs = sorted(set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)))
    sizes = {}
    for s in srcs:
        p = md_path.parent / s
        shutil.copy2(p, out / "images" / image_name(s))
        sizes[s] = Image.open(p).size[0]

    md, tokens = parse(text)
    # heading anchors: Markdown slug -> MediaWiki anchor (the heading text)
    anchors = {}
    for k, tk in enumerate(tokens):
        if tk.type == "heading_open":
            raw = tokens[k + 1].content
            anchors[slug(raw)] = re.sub(r"[*`_]", "", raw).strip()

    # Markdown: the pictures from images/
    md_out = re.sub(r"(!\[[^\]]*\]\()([^)]+)(\))", lambda m: m.group(1) + "images/" + image_name(m.group(2)) + m.group(3), text)
    (out / "scan_guide.md").write_text(md_out, encoding="utf-8")

    # MediaWiki
    (out / "scan_guide.mediawiki").write_text(to_mediawiki(tokens, anchors, sizes), encoding="utf-8")

    # rich-text paste page: pictures embedded, heading ids for the internal links
    body = md.render(text)
    body = re.sub(r"<h([1-3])>(.*?)</h\1>", lambda m: f'<h{m.group(1)} id="{slug(m.group(2))}">{m.group(2)}</h{m.group(1)}>', body)

    def embed(m):
        data = base64.b64encode((md_path.parent / html.unescape(m.group(1))).read_bytes()).decode()
        w = sizes.get(html.unescape(m.group(1)), 800)
        return f'<img src="data:image/png;base64,{data}" alt="{m.group(2)}" style="max-width:100%;width:{min(w, 900)}px">'
    body = re.sub(r'<img src="([^"]+)" alt="([^"]*)" ?/?>', embed, body)
    body = body.replace("<table>", '<table border="1" cellpadding="5" style="border-collapse:collapse">')
    page = (f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            "<style>body{font-family:Helvetica,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;"
            "line-height:1.45;color:#1d1b26}code{background:#f1eff8;padding:1px 4px}th{background:#ece9f6}"
            "td,th{vertical-align:top;text-align:left}</style></head><body>" + body + "</body></html>")
    (out / "scan_guide_paste.html").write_text(page, encoding="utf-8")

    (out / "README.txt").write_text(f"""{title}: wiki package (made by tools/make_guide_wiki.py from docs/scan_guide.md)

Pick the file for your wiki:

  MediaWiki (Wikipedia-style):
    1. Upload every file in images/ (Special:Upload, or Special:UploadWizard for several at once), keeping the names.
    2. Create the page "{title}" and paste scan_guide.mediawiki into it.

  Markdown wikis (GitHub / GitLab wiki, Wiki.js, BookStack, Obsidian):
    Add scan_guide.md as a page and the images/ folder next to it (GitHub wiki: commit both to the wiki's git
    repository). Notion and Confluence can also import scan_guide.md: zip it together with images/ and use the
    import function (Notion: Import > Markdown; Confluence: Markdown importer or the Markdown macro).

  Rich-text editors (Confluence, Google Sites, SharePoint, Notion):
    Open scan_guide_paste.html in a browser, press Ctrl+A (Cmd+A), Ctrl+C, and paste into the wiki editor. Check the
    pictures came along; if not, insert them from images/ (they are in the same order as in the text).

The pictures come from the simulated microscope (run_flakeid.bat screenshots); the numbers in them are not real.
""", encoding="utf-8")

    zp = out.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(out.name) / f.relative_to(out))
    print(f"wrote {out}/ ({len(srcs)} pictures) and {zp} ({zp.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
