#!/usr/bin/env python3
"""Graphite wrinkle labelling set (tools/wrinkle_page.html) from 100x frames.

    python tools/make_wrinkle_label_set.py RUN [RUN ...] -o OUT [--oxide 90]

RUN is a folder of 100x RAW16 frames (mm_chip_scan.bsh targets mode; *_s1/_s2 shift frames are skipped) or a scoring
set made by tools/make_score_set.py (items.json + img/*.webp, decoded back to linear counts with its gains).  Each
run's static optics pattern is divided out, and flakeid.screen100.features gives the terrace, wrinkle lines and
bubble outlines that the page overlays.  Every frame is stored as one JPEG: the natural image above the high-contrast
one.  Labels (none / slight / significant wrinkling, folded or torn edges) go to the artifact's db.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flakeid import screen100 as S  # noqa: E402
from flakeid.hundred import _mode_rgb, static_pattern  # noqa: E402
from flakeid.mm import BAYER_CODE  # noqa: E402

SCALE = 0.75                     # stored image scale (wrinkles stay visible; keeps the publish under its size limit)
NATURAL_SUB = np.array([180.0, 147.5, 163.0])   # substrate colour of the natural view (the demo 100x graphite look)


def frames(run: Path, dark=128.0):
    """(stem, linear RGB, old label tags or None) of every frame of a run."""
    meta = run / "items.json"
    if meta.is_file():                                 # scoring set: s = 255 sqrt(d / D), d = gain * counts
        M = json.loads(meta.read_text())
        D, gain = float(M["meta"]["D"]), np.asarray(M["meta"]["gain"], np.float32)
        for it in M["items"]:
            a = cv2.imread(str(run / it["file"]), cv2.IMREAD_COLOR)
            if a is None:
                continue
            s = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32)
            yield it["id"], D * (s / 255.0) ** 2 / gain
        return
    for p in sorted(run.glob("*.tif")):
        if re.search(r"_s\d+$", p.stem):
            continue
        yield p.stem, cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), BAYER_CODE).astype(np.float32) - dark


def scale_polys(polys, k):
    return [[round(v * k, 1) for v in p] for p in polys]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--oxide", type=float, default=90.0)
    ap.add_argument("--names", nargs="*", help="display name per run (default: folder name)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for ri, run in enumerate(Path(r) for r in a.runs):
        name = (a.names or [])[ri] if a.names and ri < len(a.names) else run.name.replace("_score", "")
        lins = list(frames(run))
        pat = static_pattern(l for _, l in lins)
        print(f"{name}: {len(lins)} frames, optics pattern {'divided out' if pat is not None else 'not found'}")
        for stem, lin in lins:
            if pat is not None:
                lin = lin / pat
            m = re.search(r"_(\d+)nm", stem)
            t = float(m.group(1)) if m else 5.0
            f, reg, extra = S.features(lin, "graphite", a.oxide, t)
            sub, _ = _mode_rgb(lin[::2, ::2])
            nat = np.clip(lin / np.maximum(sub, 1.0) * NATURAL_SUB, 0, 255).astype(np.uint8)
            hc = extra["c8"]
            h, w = nat.shape[:2]
            W, H = int(round(w * SCALE)), int(round(h * SCALE))
            both = np.vstack([cv2.resize(nat, (W, H), interpolation=cv2.INTER_AREA),
                              cv2.resize(hc, (W, H), interpolation=cv2.INTER_AREA)])
            fn = f"{name}__{stem}.jpg"
            cv2.imwrite(str(out / "img" / fn), cv2.cvtColor(both, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
            items.append(dict(id=f"{name}__{stem}", run=name, frame=stem, t_nm=t, img=f"img/{fn}", w=W, h=H,
                              terrace=scale_polys(reg["terrace"], SCALE), wrinkles=scale_polys(extra["wrinkles"], SCALE),
                              bubbles=scale_polys(extra.get("bubbles", []), SCALE),
                              ridge_density=round(f.get("ridge_density", float("nan")), 4),
                              bubble_frac=round(f.get("bubble_frac", float("nan")), 4)))
    px_um = float(np.sqrt(abs(np.linalg.det(S._M100())))) / SCALE
    meta = dict(title="Graphite Wrinkle Labels", px_um=px_um, items=items)
    (out / "items.json").write_text(json.dumps(meta, separators=(",", ":")))
    pub = dict(meta, items=[{k: v for k, v in it.items() if k not in ("ridge_density", "bubble_frac")} for it in items])
    tpl = (Path(__file__).parent / "wrinkle_page.html").read_text()
    (out / "index.html").write_text(tpl.replace("__DATA__", json.dumps(pub, separators=(",", ":"))))
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
