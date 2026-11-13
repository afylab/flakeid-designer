#!/usr/bin/env python3
"""Attach a finished run's 100x pictures to their flakes again (flakeid.hundred.attach, as the export does), for
viewers written before 2026-10-21: a picture taken more than 40 um from its flake's 10x position (re-centred at 100x,
or moved by hand while focusing) was left off the viewer page, and so out of the designer.  Only the page's picture
list changes (index.html, a copy kept as index.html.bak-<date>); the pictures themselves were exported already.

    python tools/reattach_100x.py <run or piece folder> [...] [--dry-run]

A piece folder is the copy on the share (<run>/<piece>/index.html, tables in data/) or the analysis's own
(<piece>/<piece>_tiles_viewer/).  Rebuild the run's designer data afterwards: the newly attached flakes are only in a
new bundle.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flakeid.hundred import attach  # noqa: E402


def pieces_of(folder: Path):
    """(viewer folder, 100x index csv, domain index csv or None) of each piece under folder."""
    out = []
    # (a few levels only: a recursive search walks every viewer tile, thousands of files over the share)
    pats = ("*_100x/*_100x_index.csv", "data/*_100x/*_100x_index.csv", "*/data/*_100x/*_100x_index.csv",
            "*/*_100x/*_100x_index.csv")
    found = sorted({p for pat in pats for p in folder.glob(pat)})
    for idx in found:
        piece = re.sub(r"_100x_index$", "", idx.stem)
        vd = None
        for c in (idx.parent.parent.parent, idx.parent.parent):   # share: <piece>/data/<piece>_100x/; local: next to it
            if (c / "index.html").is_file():
                vd = c
                break
            g = next(c.glob(f"{piece}*_viewer"), None)
            if g is not None and (g / "index.html").is_file():
                vd = g
                break
        if vd is None:
            continue
        di = next(idx.parent.parent.glob(f"{piece}_domains/*_domains_index.csv"), None)
        out.append((vd, idx, di))
    return out


def attached(page: Path):
    m = re.search(r"const H100=(.*?);\n", page.read_text(), re.S)
    try:
        H = json.loads(m.group(1)) if m else {}
    except ValueError:
        H = {}
    return H or {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="only count the pictures that would be added")
    a = ap.parse_args(argv)
    total = 0
    for fo in a.folders:
        for vd, idx, di in pieces_of(Path(fo).expanduser()):
            rows = list(csv.DictReader(open(idx, newline="")))
            drows = list(csv.DictReader(open(di, newline=""))) if di else []
            page = vd / "index.html"
            before = attached(page)
            have = {e["f"] for v in before.values() for e in v}
            with tempfile.TemporaryDirectory() as td:       # (worked on a copy: the share is written once)
                t = Path(td)
                shutil.copy(page, t / "index.html")
                try:                                        # (the pictures' statistics and mosaics are read from it)
                    (t / "100x").symlink_to(vd / "100x", target_is_directory=True)
                except OSError:
                    print("  (no link to the 100x folder: pictures added without their statistics and mosaics)")
                attach(t, rows, drows, log=lambda s: None)
                after = attached(t / "index.html")
                new = sorted({int(k) for k, v in after.items() for e in v if e["f"] not in have})
                n_new = sum(1 for v in after.values() for e in v if e["f"] not in have)
                print(f"{vd}: {len(before)} -> {len(after)} flakes with 100x pictures"
                      + (f"; {n_new} pictures added (flakes {', '.join(map(str, new))})" if n_new else "; nothing to add"))
                total += n_new
                if n_new and not a.dry_run:
                    bak = page.with_name(f"index.html.bak-{time.strftime('%Y%m%d')}")
                    if not bak.exists():
                        shutil.copy(page, bak)
                    part = page.with_name("index.html.part")
                    shutil.copy(t / "index.html", part)
                    part.replace(page)
    print(f"{total} pictures {'would be ' if a.dry_run else ''}added")
    return 0


if __name__ == "__main__":
    sys.exit(main())
