#!/usr/bin/env python3
"""Write the AFM scan sites as a NanoScope programmed-move point list (.pgm).

    python tools/afm_pgm.py afm_set/data.json positions/ [-o bigbntest_2_tiles_afm_sites.pgm] [--zoff 0]

data.json:  the AFM calibration page's data (tools/make_afm_set.py): scan sites in microscope stage um.
positions:  the AFM positions entered on the page: a folder of its "afmpos" documents (one JSON per flake, as
            saved by the page's database, {"x", "y", "z"} or {"data": {...}}), or one JSON file
            {"<flake id>": {"x": .., "y": .., "z": ..}, ...}.

The microscope -> AFM transform is the page's: a similarity (mirror detected with >= 3 points) through the
flakes with an AFM X, Y.  Z: the focus Z entered at the flakes, one value for all sites, or a plane through 3 or
more; --zoff is added.  Points in the page's order (the stage path), one per scan site:

    Version: 1.0
    \\X point: ...      \\Y point: ...      \\Z point: ...      \\Coordinate System: 0      (CRLF line ends)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.targets import fit_transform  # noqa: E402


def load_positions(path: Path) -> dict:
    docs = {}
    if path.is_dir():
        for f in path.rglob("*.json"):
            d = json.loads(f.read_text())
            docs[f.stem] = d.get("data", d)
    else:
        docs = json.loads(path.read_text())
    out = {}
    for k, d in docs.items():
        fid = int(str(k).lstrip("f"))
        if d.get("x") is not None and d.get("y") is not None or d.get("z") is not None:
            out[fid] = d
    return out


def n2(v):
    """2 decimals, no trailing zeros (as the page writes them: -86662.5, -105646.88, -16944)"""
    r = str(round(float(v), 2))
    return r[:-2] if r.endswith(".0") else r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("positions")
    ap.add_argument("-o", "--out")
    ap.add_argument("--zoff", type=float, default=0.0, help="added to every Z (AFM units)")
    a = ap.parse_args(argv)
    D = json.loads(Path(a.data).read_text())
    pos = load_positions(Path(a.positions))
    F = {f["flake"]: f for f in D["flakes"]}
    refs = [(F[k]["stage"], (p["x"], p["y"])) for k, p in pos.items() if k in F and p.get("x") is not None and p.get("y") is not None]
    if len(refs) < 3:
        sys.exit(f"need AFM X, Y at 3 flakes (have {len(refs)})")
    A, t, rms = fit_transform(np.array([r[0] for r in refs]), np.array([r[1] for r in refs]))
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    zp = [(p["x"], p["y"], p["z"]) for p in pos.values() if p.get("z") is not None and p.get("x") is not None]
    if not zp:
        sys.exit("no focus Z entered (enter the AFM Z reading at a reference flake)")
    if len(zp) >= 3:
        Z = np.array(zp)
        c, *_ = np.linalg.lstsq(np.c_[Z[:, :2], np.ones(len(Z))], Z[:, 2], rcond=None)
        zf = lambda x, y: float(c[0] * x + c[1] * y + c[2])          # noqa: E731
    else:
        z0 = float(np.mean([p[2] for p in zp]))
        zf = lambda x, y: z0                                            # noqa: E731
    lines = ["Version: 1.0\n"]
    for f in D["flakes"]:
        for t_ in f["terraces"]:
            if not t_.get("site"):
                continue
            x, y = A @ np.array(t_["site"]["stage"]) + t
            lines.append(f"\\X point: {n2(x)}\r\n\\Y point: {n2(y)}\r\n\\Z point: {n2(zf(x, y) + a.zoff)}\r\n\\Coordinate System: 0\r\n")
    out = Path(a.out or f"{D['meta']['scan']}_afm_sites.pgm")
    out.write_bytes("".join(lines).encode())
    print(f"wrote {out}: {len(lines) - 1} points; transform from {len(refs)} flakes, scale {scale:.5f}, "
          f"residual {rms / scale if rms else 0:.1f} um; focus Z from {len(zp)} point(s)")


if __name__ == "__main__":
    main()
