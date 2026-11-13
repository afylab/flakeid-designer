#!/usr/bin/env python3
"""Contrast-boosted renderings of 100x images, to compare ways of showing debris, residue and terraces on a flake.

    python tools/contrast_options.py "testing images/bigbntest_2_100x" --ids 055 002 042 ... -o OUT

Every option starts from the linear frame (dark subtracted, optics pattern divided out) divided by the colour of
the terrace at the frame centre, r = I / I_terrace (the terrace is 1 in every channel), and from a "flattened"
version in which slow illumination changes across the terrace are removed (r divided by its smooth background,
estimated only from terrace-coloured pixels):

  natural   the normal colour image (demo white balance, as the viewer JPEGs)
  colour8   colour, each channel stretched to terrace +-8 %  (the scoring page's "terrace contrast" button)
  colour3f  colour, flattened, +-4 %
  devgray   grey: colour distance from the terrace, flattened; white = 0, black = 12 % or more (square-root scale;
            anything that is not this terrace: debris, residue, other thicknesses)
  signgray  grey: green channel relative to the terrace, flattened, +-4 % (thicker/thinner steps are dark/bright)
  dog       grey: fine detail only (difference of Gaussians, 1.5 - 10 px): specks, bubbles, wrinkles; +-1.5 %
All but "natural" are smoothed by 1 px (0.11 um) first.
  clahe     grey: brightness with adaptive local contrast (CLAHE), no reference to the terrace
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
from flakeid.hundred import DEMO_SUBSTRATE, WHITE, _mode_rgb  # noqa: E402
from flakeid.mm import BAYER_CODE  # noqa: E402

OPTIONS = ["natural", "colour8", "colour3f", "devgray", "signgray", "dog", "clahe"]


def terrace_colour(lin, win=90):
    h, w = lin.shape[:2]
    c = lin[h // 2 - win:h // 2 + win, w // 2 - win:w // 2 + win]
    m, _ = _mode_rgb(c, step=3)
    return np.maximum(m, 1.0)


def flatten(r, tol=0.06, sigma=60.0):
    """r / its smooth background, the background estimated from pixels close to the terrace colour only."""
    m = (np.abs(r - 1.0).max(axis=2) < tol).astype(np.float32)
    num = cv2.GaussianBlur(r * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    wide = cv2.GaussianBlur(r * m[..., None], (0, 0), 4 * sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), 4 * sigma), 1e-6)[..., None]
    a = np.clip(den / 0.3, 0, 1)[..., None]
    bg = a * num / np.maximum(den, 1e-6)[..., None] + (1 - a) * wide
    bg = np.where(np.isfinite(bg) & (bg > 0.2), bg, 1.0)
    return r / bg


def u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render(lin, material="hbn", gain=None):
    tc = terrace_colour(lin)
    r = lin / tc
    # 1 px Gaussian (0.11 um, below the 100x resolution): removes most of the sensor noise the stretches amplify
    r = cv2.GaussianBlur(r, (0, 0), 1.0)
    rf = flatten(r)
    out = {}
    g = gain if gain is not None else np.asarray(DEMO_SUBSTRATE[material]) / np.maximum(np.median(lin.reshape(-1, 3), 0), 1)
    out["natural"] = u8(lin * g / WHITE)
    out["colour8"] = u8((r - 0.92) / 0.16)
    out["colour3f"] = u8((rf - 0.96) / 0.08)
    dev = np.sqrt(((rf - 1.0) ** 2).mean(axis=2))
    out["devgray"] = u8(1.0 - np.sqrt(np.clip(dev / 0.12, 0, 1)))        # (square root: small deviations stand out)
    out["signgray"] = u8(0.5 + (rf[..., 1] - 1.0) / 0.08)
    L = rf.mean(axis=2)
    dog = cv2.GaussianBlur(L, (0, 0), 1.5) - cv2.GaussianBlur(L, (0, 0), 10.0)
    out["dog"] = u8(0.5 + dog / 0.03)
    lum = cv2.cvtColor(out["natural"], cv2.COLOR_RGB2GRAY)
    out["clahe"] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lum)
    return out, tc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--ids", nargs="+", required=True, help="image order numbers (the file name prefix, e.g. 055)")
    ap.add_argument("--scores", help="folder of scoring-page documents (verdicts and tags shown with each image)")
    ap.add_argument("--pattern", help="optics pattern (.npy, tools/optics_flat.py) divided out")
    ap.add_argument("--material", default="hbn")
    ap.add_argument("--width", type=int, default=1032)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    src, out = Path(a.folder), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    idx = {r["file"].split("_")[0]: r for r in csv.DictReader(open(next(src.glob("*_100x_index.csv"))))}
    pat = np.load(a.pattern) if a.pattern else None

    def linear(f):
        lin = cv2.cvtColor(cv2.imread(str(src / f), cv2.IMREAD_UNCHANGED), BAYER_CODE).astype(np.float32) - 128.0
        return lin / pat if pat is not None else lin
    # one white balance for the whole run, from the bare substrate of the frames that show enough of it (as the
    # viewer export; a frame filled by its flake would otherwise turn the flake into "substrate" colour)
    subs = [_mode_rgb(linear(r["file"])[::2, ::2], step=8) for r in idx.values()]
    fr = np.array([f for _, f in subs])
    gain = np.asarray(DEMO_SUBSTRATE[a.material]) / np.median(np.array([m for m, f in subs if f >= min(0.25, float(np.median(fr)))]), 0)
    items = []
    for i in a.ids:
        r = idx[i]
        lin = linear(r["file"])
        imgs, tc = render(lin, a.material, gain)
        stem = Path(r["file"]).stem
        sc = {}
        if a.scores and (Path(a.scores) / f"{stem}.json").is_file():
            sc = json.loads((Path(a.scores) / f"{stem}.json").read_text())
        for k, im in imgs.items():
            im = cv2.resize(im, (a.width, int(a.width * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / "img" / f"{stem}_{k}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append(dict(id=stem, label=r["label"], verdict=sc.get("verdict"), tags=sc.get("tags") or [],
                          note=sc.get("note") or "", terrace_rgb=[round(float(v), 1) for v in tc]))
        print(stem, sc.get("verdict"), sc.get("tags"))
    (out / "items.json").write_text(json.dumps(dict(options=OPTIONS, items=items), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
