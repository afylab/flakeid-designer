#!/usr/bin/env python3
"""Update the thicknesses in a run's designer files (designer/bundle.json, flakes.csv) without rebuilding it: after a
colour-model change only the numbers change, so this reads each piece's *_live_terraces.csv (the stored 10x colours)
and rewrites the graphite terraces' thickness as tools/make_catalog_100x.py would now report it (graphite_thickness).
Seconds per run instead of a full rebuild (which re-reads and rewrites every 100x picture over the share).

    python tools/patch_thickness.py <run folder> [<run folder> ...] [--dry-run]

Flakes stay as they are (none added or removed: a flake the new model would leave out as bulk graphite is listed in
the output and kept).  hBN terraces are not touched.  Each file is written beside the old one and renamed over it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_afm_set import results_file  # noqa: E402
from make_catalog_100x import graphite_thickness  # noqa: E402


def write_atomic(path: Path, text: str):
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def patch_run(run: Path, dry=False, log=print):
    ddir = run / "designer"
    bundle = json.loads((ddir / "bundle.json").read_text(encoding="utf-8"))
    by_scan = {}
    for f in bundle["flakes"]:
        if f.get("material") == "graphite":
            by_scan.setdefault(f["scan"], []).append(f)
    new_t, changed, kept_bulk = {}, [], []
    for scan, fl in sorted(by_scan.items()):
        rows = {int(r["terrace_id"]): r for r in csv.DictReader(open(results_file(run / scan, "*_live_terraces.csv")))}
        info, skip = graphite_thickness(rows, float(fl[0]["oxide_nm"]), scan, log=log)
        for f in fl:
            for t in f["terraces"]:
                tid = t.get("terrace_id")
                if tid in skip:
                    kept_bulk.append(f"{t['id']} ({skip[tid]})")
                if tid not in info or not str(t.get("thickness_source", "")).startswith(("10x colour fit", "1 layer", "2 layer",
                                                                                           "3 layer", "4 layer")):
                    continue
                ti = info[tid]
                if t["t"] != ti["t"]:
                    changed.append((t["id"], t["t"], ti["t"]))
                t.update(t=ti["t"], tlo=ti["t"], thi=ti["t"], alt=ti["alt"], layers=ti["layers"], thickness_source=ti["note"])
                new_t[t["id"]] = t
    log(f"{run.name}: {len(changed)} graphite terraces changed" + (f"; new model would leave out {', '.join(kept_bulk)} (kept)"
                                                                   if kept_bulk else ""))
    for i, o, n in changed[:8]:
        log(f"   {i.split(':')[-1]}: {o} -> {n} nm")
    if dry or not changed:
        return len(changed)
    fc = ddir / "flakes.csv"
    if fc.is_file():                                  # same columns as make_designer_bundle writes
        rows = list(csv.reader(io.StringIO(fc.read_text(encoding="utf-8"))))
        h = rows[0]
        it, ia, il = h.index("thickness_nm"), h.index("alt_thickness_nm"), h.index("layers")
        for r in rows[1:]:
            t = new_t.get(r[0])
            if t is not None:
                r[it], r[ia] = t["t"], (t.get("alt") or "").split(";")[0]
                r[il] = t.get("layers") if t["t"] < 3 else ""
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)
        write_atomic(fc, buf.getvalue())
    write_atomic(ddir / "bundle.json", json.dumps(bundle, separators=(",", ":")))
    return len(changed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="only report what would change")
    a = ap.parse_args(argv)
    for r in a.runs:
        patch_run(Path(r).expanduser(), a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
