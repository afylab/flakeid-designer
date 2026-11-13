#!/usr/bin/env python3
"""Merge tools/label_page.html with <label_set>/items.json -> <label_set>/index.html."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "label_set")
tpl = (Path(__file__).parent / "label_page.html").read_text()
items = (d / "items.json").read_text()
title = json.loads(items).get("title")
if title:
    tpl = tpl.replace("<title>Flake Triage</title>", f"<title>{title}</title>", 1)
(d / "index.html").write_text(tpl.replace("__ITEMS__", items))
print("wrote", d / "index.html")
