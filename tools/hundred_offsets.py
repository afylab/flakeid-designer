#!/usr/bin/env python3
"""Where the 100x shots really landed: each 100x picture of a piece, shrunk to the 10x scale, registered against the
piece's 10x mosaic (flakeid.scope.retarget, as the scan's re-image mode does), and the positioning error compared with
what might cause it (place on the chip, the move from the previous target, the order / time, focusing).

    python tools/hundred_offsets.py <piece results> ... [-o offsets.csv]

A piece results folder holds <name>_tiles_viewer/ (tiles + 100x/) and *_live_flakes.csv (the mosaic's pixel <-> stage
mapping), either layout of flakeid.live / the share copy.  err_x / err_y (stage um): the move that would have centred
the target (the stage that centres it = the 100x stage used + err), as the re-image mode's re-centring computes it;
peak / second: the correlation and the best one elsewhere (trust a frame when peak >= 0.5 and second <= 0.85 peak).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from flakeid.mm import DEFAULT_M  # noqa: E402
from flakeid.hundred import RATIO_100X  # noqa: E402
from flakeid.scope import retarget as RT  # noqa: E402


def viewer_and_flakes(res: Path):
    vd = next(iter(sorted(res.glob("*_viewer"))), None) or (res if (res / "index.html").is_file() else None)
    fl = next(iter(sorted(res.glob("*_live_flakes.csv"))), None) or next(iter(sorted(res.glob("*flakes.csv"))), None)
    if fl is None and vd is not None:
        fl = next(iter(sorted(vd.glob("data/*flakes.csv"))), None)
    return vd, fl


def h100(vd: Path):
    h = (vd / "index.html").read_text(encoding="utf-8")
    i = h.index("const H100=") + 11
    return json.loads(h[i:h.index(";\n", i)])


def piece(res: Path, radius_um=120.0, log=print):
    vd, fl = viewer_and_flakes(res)
    if vd is None or fl is None:
        log(f"{res}: no viewer or flake table")
        return []
    with open(fl, newline="") as fh:
        rows = list(csv.DictReader(fh))
    mos = RT.Mosaic(vd, rows)
    M = np.asarray(DEFAULT_M, float)
    out = []
    H = h100(vd)
    ents = []
    for fid, imgs in H.items():
        for e in imgs:
            if e.get("domain"):
                continue
            m = re.match(r"(\d+)_", Path(e["f"]).name)
            ents.append((int(m.group(1)) if m else 0, int(fid), e))
    ents.sort()
    prev = None
    for order, fid, e in ents:
        rj = vd / re.sub(r"\.jpg$", "_regions.json", e["f"])
        if not rj.is_file():
            continue
        R = json.loads(rj.read_text())
        stage = np.asarray(R["stage_um"], float)
        g = stage - np.asarray(R.get("offset_100x_um") or (24.0, -18.3), float)   # the 10x-equivalent position
        im = cv2.imread(str(vd / e["f"]))
        if im is None:
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
        small = cv2.resize(rgb, None, fx=1.0 / RATIO_100X, fy=1.0 / RATIO_100X, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        res_ = RT.register(RT.grey(small), mos, g, M, np.array([w / 2.0, h / 2.0]), np.eye(2), np.zeros(2), radius_um,
                           sig=(1.0, 8.0), crop=0.02)
        rec = dict(piece=res.name, order=order, flake=fid, stage_x=round(float(stage[0]), 1), stage_y=round(float(stage[1]), 1),
                   focus=e.get("focus") or "", z=e.get("z") or "", peak=None, second=None, err_x=None, err_y=None,
                   move_x=None if prev is None else round(float(stage[0] - prev[0]), 1),
                   move_y=None if prev is None else round(float(stage[1] - prev[1]), 1),
                   seed_off=json.dumps((R.get("seed") or {}).get("offset_um")) if R.get("seed") else "")
        prev = stage
        if res_ is not None:
            err = np.asarray(res_["shift_um"], float)         # the move onto the target
            rec.update(peak=round(res_["peak"], 3), second=round(res_["second"], 3),
                       err_x=round(float(err[0]), 1), err_y=round(float(err[1]), 1))
        out.append(rec)
    good = [r for r in out if r["peak"] is not None and r["peak"] >= 0.5 and r["second"] <= 0.85 * r["peak"]]
    if good:
        e = np.array([[r["err_x"], r["err_y"]] for r in good])
        log(f"{res.name}: {len(good)} of {len(out)} frames registered; error median {np.median(e, 0).round(1)} um, "
            f"spread (MAD) {np.median(np.abs(e - np.median(e, 0)), 0).round(1)} um, |error| > 20 um: {int((np.hypot(*e.T) > 20).sum())}")
    else:
        log(f"{res.name}: no frame registered confidently ({len(out)} frames)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("-o", "--out", default="hundred_offsets.csv")
    ap.add_argument("--radius", type=float, default=120.0, help="search radius (um)")
    a = ap.parse_args(argv)
    rows = []
    for r in a.results:
        try:
            rows += piece(Path(r).expanduser(), a.radius)
        except Exception as ex:                          # (one unreadable piece does not stop the others)
            print(f"{r}: {type(ex).__name__}: {ex}")
    if rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
