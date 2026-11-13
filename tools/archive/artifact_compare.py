#!/usr/bin/env python3
"""Before / after images of the 100x optics-artifact removal, to check what it removes and whether it adds anything.

    python tools/artifact_compare.py "testing images/multitest/chip_1b_100x" -o OUT [--n 24]

For a run's frames, the stages of flakeid.hundred.export:
  raw      dark subtracted only
  pattern  divided by the run's static optics pattern (static_pattern: what stays at the same pixel in every frame)
  median   + median of the 3-position frames aligned on the flake (shift_median), if the run has them
each rendered as colour +-8 % around the terrace colour (the high-contrast view, where artifacts show), and the
differences pattern/raw and median/pattern as grey +-3 % (white = brighter after the step).  Frames: the ones the
median changed most (where new artifacts would come from: misalignment, flake edges, moving dust), the ones the
pattern changed most on the terrace, and a random set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flakeid.hundred import _mode_rgb, shift_median, static_pattern  # noqa: E402
from flakeid.mm import BAYER_CODE  # noqa: E402

W = 900


def linear(p, dark=128.0):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), BAYER_CODE).astype(np.float32) - dark


def c8(lin, tc):
    r = cv2.GaussianBlur(lin / np.maximum(tc, 1.0), (0, 0), 1.0)
    return np.clip((r - 0.92) / 0.16 * 255 + 0.5, 0, 255).astype(np.uint8)


def diff(a, b, tc, span=0.03):
    """grey: (a - b) relative to the terrace colour, +-span; 0 = mid grey."""
    d = cv2.GaussianBlur(((a - b) / np.maximum(tc, 1.0)).mean(axis=2), (0, 0), 1.0)
    return np.clip(128 + d / span * 127, 0, 255).astype(np.uint8), float(np.sqrt(np.mean(d ** 2)))


def small(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (W, int(W * h / w)), interpolation=cv2.INTER_AREA)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--flat", help="100x flat field .npz used by the export (an extra stage: raw -> flat)")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    FF = None
    if a.flat:
        from flakeid.mm import FlatField
        FF = FlatField.load(a.flat)

    def fl(lin_raw):
        """raw (dark 128 subtracted) -> flat-field corrected, as the export does"""
        return FF.apply(lin_raw + 128.0) if FF is not None else lin_raw
    (out / "img").mkdir(parents=True, exist_ok=True)
    idx = next(run.glob("*_100x_index.csv"), None)
    rows = [r for r in csv.DictReader(open(idx)) if (run / r["file"]).is_file()] if idx else \
        [dict(file=p.name, label=p.stem) for p in sorted(run.glob("*.tif"))]
    print(f"{run.name}: {len(rows)} frames; optics pattern ...", flush=True)
    pat = static_pattern(fl(linear(run / r["file"])) for r in rows)
    if pat is None:
        sys.exit("too few frames for an optics pattern")
    cv2.imwrite(str(out / "img" / "pattern.jpg"),
                cv2.cvtColor(small(np.clip(128 + (pat - 1.0) / 0.03 * 127, 0, 255).astype(np.uint8)), cv2.COLOR_RGB2BGR))

    def stages(r):
        raw = linear(run / r["file"])
        lin0 = fl(raw)
        sh = []
        for f_, d_ in zip((r.get("shift_files") or "").split(";"), (r.get("shift_um") or "").split(";")):
            if f_ and (run / f_).is_file() and ":" in d_:
                sh.append((fl(linear(run / f_)), tuple(float(v) for v in d_.split(":"))))
        med, resid = shift_median(lin0, sh)
        return raw, lin0, lin0 / pat, (med / pat if sh else None), resid

    # pass 1: how much each step changes each frame (on the central 60 % where the flake is)
    score = []
    for k, r in enumerate(rows):
        raw, ff, pt, md, resid = stages(r)
        h, w = raw.shape[:2]
        tc, _ = _mode_rgb(ff[h // 2 - 90:h // 2 + 90, w // 2 - 90:w // 2 + 90], step=3)
        c = (slice(int(0.2 * h), int(0.8 * h)), slice(int(0.2 * w), int(0.8 * w)))
        dp = float(np.sqrt(np.mean((((pt - ff) / tc)[c]) ** 2)))
        dm = float(np.sqrt(np.mean((((md - pt) / tc)[c]) ** 2))) if md is not None else 0.0
        score.append((k, dp, dm, resid))
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(rows)}", flush=True)
    rng = np.random.default_rng(a.seed)
    by_m = [s[0] for s in sorted(score, key=lambda s: -s[2]) if s[2] > 0][:a.n // 3]
    by_p = [s[0] for s in sorted(score, key=lambda s: -s[1]) if s[0] not in by_m][:a.n // 3]
    rest = [s[0] for s in score if s[0] not in by_m + by_p]
    rnd = list(rng.choice(rest, size=min(a.n - len(by_m) - len(by_p), len(rest)), replace=False)) if rest else []
    pick = [(k, "median changed most") for k in by_m] + [(k, "pattern changed most") for k in by_p] + [(k, "random") for k in rnd]
    items = []
    for k, why in pick:
        r = rows[k]
        raw, ff, pt, md, resid = stages(r)
        h, w = raw.shape[:2]
        tc, _ = _mode_rgb(ff[h // 2 - 90:h // 2 + 90, w // 2 - 90:w // 2 + 90], step=3)
        tr, _ = _mode_rgb(raw[h // 2 - 90:h // 2 + 90, w // 2 - 90:w // 2 + 90], step=3)
        stem = Path(r["file"]).stem
        im = dict(raw=c8(raw, tr))
        rms_f = None
        if FF is not None:
            im["flat"] = c8(ff, tc)
            # flat / raw relative change (each normalised to its own terrace colour)
            dfl = cv2.GaussianBlur((ff / tc - raw / tr).mean(axis=2), (0, 0), 1.0)
            im["d_flat"] = np.clip(128 + dfl / 0.03 * 127, 0, 255).astype(np.uint8)
            rms_f = float(np.sqrt(np.mean(dfl ** 2)))
        im["pattern"] = c8(pt, tc)
        d_p, rms_p = diff(pt, ff, tc)
        im["d_pattern"] = d_p
        rms_m = None
        if md is not None:
            im["median"] = c8(md, tc)
            im["d_median"], rms_m = diff(md, pt, tc)
        for kk, v in im.items():
            v = small(v)
            cv2.imwrite(str(out / "img" / f"{stem}_{kk}.jpg"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR) if v.ndim == 3 else v,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append(dict(id=stem, label=r.get("label", stem), why=why, stages=list(im), rms_pattern=round(100 * rms_p, 2),
                          rms_flat=None if rms_f is None else round(100 * rms_f, 2),
                          rms_median=None if rms_m is None else round(100 * rms_m, 2), shift_alignment=resid))
    meta = dict(run=run.name, n_frames=len(rows), pattern_rms=round(100 * float((pat - 1).std()), 2), items=items,
                has_median=any(i["rms_median"] is not None for i in items))
    (out / "items.json").write_text(json.dumps(meta))
    tpl = (Path(__file__).parent / "artifact_page.html").read_text()
    (out / "index.html").write_text(tpl.replace("__TITLE__", "Optics-Artifact Removal").replace("__DATA__", json.dumps(meta)))
    print(f"wrote {len(items)} frames to {out}")


if __name__ == "__main__":
    main()
