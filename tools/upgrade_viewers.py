#!/usr/bin/env python3
"""Give existing chip viewers the current viewer page (arrow keys between flakes, large 100x view, view shortcuts),
keeping their data, tiles and 100x images.  The old page is kept as index_before_upgrade.html.

    python tools/upgrade_viewers.py "<run folder or viewer folder>" ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.zoomview import upgrade_page  # noqa: E402


def main(argv=None):
    roots = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not roots:
        print(__doc__)
        return 1
    n = 0
    for r in roots:
        pages = [r] if r.name == "index.html" else [p for p in r.rglob("index.html") if "const D=" in p.read_text(errors="ignore")[:4000000]]
        for p in pages:
            if upgrade_page(p):
                n += 1
                print(f"upgraded {p}")
    print(f"{n} viewer(s) upgraded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
