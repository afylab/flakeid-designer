"""Prepare 100x flake images (mm_chip_scan.bsh targets mode) for the scoring page.

    python tools/make_score_set.py "testing images/bigbntest_100x" \
        --flakes "testing images/full_bigbn_results/full_bigbn_flakes.csv" -o "testing images/bigbntest_100x_score"

Every RAW16 frame is demosaiced, dark-subtracted and multiplied by one set of per-channel gains
that maps the bare substrate (median over all frames of each frame's most common colour) to the
substrate colour of the designer_demo 100x images of hBN on 285 nm SiO2 (Stream exports).
The page shows these "display values" d (the substrate at the demo colour, 0..255 = the demo look).
Each image is stored as 8-bit sqrt(d / D) with D = HEADROOM x 255 (510), and the page decodes
d = D (s / 255)^2 before applying its per-channel LUT.  The page's default white point is D (the whole
stored range), so bright flakes are not shown clipped; the log prints where the camera's full scale lies.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flakeid.mm import BAYER_CODE  # noqa: E402

UM_PER_PX_100X = 1.0894 / 9.54


def mode_rgb(rgb, step=4):
    """Most common colour (the bare substrate in most frames) and the fraction of pixels near it."""
    q = (np.clip(rgb, 0, 255 * step * 16 - 1) // step).astype(np.int64)
    k = (q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]).ravel()
    v, c = np.unique(k, return_counts=True)
    sel = k == v[c.argmax()]
    flat = rgb.reshape(-1, 3)
    m = flat[sel].mean(0)
    near = np.abs(flat - m).max(1) < 3 * step
    return np.median(flat[near], 0), near.mean()


def demo_substrate(material="hbn"):
    """Substrate colour of the designer_demo 100x images: hBN on 285 nm (BN*) or graphite on 90 nm (g*)."""
    subs = []
    pat = "BN*" if material == "hbn" else "g*"
    for f in sorted(glob.glob(str(ROOT / f"example images/demo_high_magnification/{pat}/*_100x.jpg"))):
        if " X" in f:
            continue
        a = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32)
        subs.append(mode_rgb(a)[0])
    return np.median(np.array(subs), 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="targets-mode output: TIFFs + *_100x_index.csv")
    ap.add_argument("--flakes", help="<scan>_flakes.csv the targets came from (thickness, size, score)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dark", type=float, default=128.0, help="black level (12-bit counts), when no --flat is given")
    ap.add_argument("--flat", help="100x flat field (.npz from mm_chip_scan.bsh MODE = \"flat\" + python -m flakeid.mm flat): "
                    "dark frame and optics pattern (dust shadows, vignetting) removed")
    ap.add_argument("--material", choices=("hbn", "graphite"), default="hbn",
                    help="white balance from the demo 100x images of this material (hBN on 285 nm, graphite on 90 nm)")
    ap.add_argument("--subtitle", help="page subtitle (default: material, oxide, 100x)")
    ap.add_argument("--headroom", type=float, default=2.0, help="display values stored up to HEADROOM x 255 (default 2)")
    ap.add_argument("--quality", type=int, default=101, help="WebP quality; 101 = lossless (default: terrace contrast "
                    "stretches the display 10x, which makes lossy blocks visible)")
    ap.add_argument("--blur", type=float, default=1.0, help="Gaussian sigma (px) before storing; 1 px = 0.11 um, "
                    "below the 100x resolution, and lets lossless storage fit (default 1.0)")
    a = ap.parse_args(argv)
    src, out = Path(a.folder), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    idx = list(csv.DictReader(open(next(src.glob("*_100x_index.csv")))))
    flakes = {}
    if a.flakes:
        flakes = {int(r["flake_id"]): r for r in csv.DictReader(open(a.flakes))}

    from flakeid.mm import FlatField
    FF = FlatField.load(a.flat) if a.flat else None

    def linear(raw):
        rgb = cv2.cvtColor(raw, BAYER_CODE).astype(np.float32)
        return FF.apply(rgb) if FF is not None else rgb - a.dark

    target = demo_substrate(a.material)
    print(f"target substrate ({a.material} demo images):", target.round(1))
    # substrate of the 100x frames: each frame's most common colour; the median over frames
    subs, frames = [], []
    for r in idx:
        raw = cv2.imread(str(src / r["file"]), cv2.IMREAD_UNCHANGED)
        rgb = linear(raw)
        m, frac = mode_rgb(rgb[::2, ::2], step=8)
        subs.append((m, frac))
    # frames whose most common colour covers a good share of the frame (the share depends on the exposure:
    # brighter frames spread the substrate over more colour bins)
    fracs = np.array([f for _, f in subs])
    good = np.array([m for m, f in subs if f >= min(0.25, float(np.median(fracs)))])
    sub = np.median(good, 0)
    gain = target / sub
    print(f"100x substrate (median of {len(good)} frames): {sub.round(1)} counts -> gains {gain.round(4)}")
    # the camera's full scale (12-bit clipping level above black) in display units, largest channel
    black = float(np.max(FF.dark)) if FF is not None else a.dark
    fs_disp = (4095.0 - black) * gain
    D = a.headroom * 255.0
    print(f"camera full scale in display units: R {fs_disp[0]:.0f}, G {fs_disp[1]:.0f}, B {fs_disp[2]:.0f}; stored range and white point D = {D:.0f}")

    items, thumbs = [], []
    for k, r in enumerate(idx):
        raw = cv2.imread(str(src / r["file"]), cv2.IMREAD_UNCHANGED)
        rgb = linear(raw) * gain
        if a.blur > 0:
            rgb = cv2.GaussianBlur(rgb, (0, 0), a.blur)
        sat = float((raw >= 4094).mean())
        s = np.sqrt(np.clip(rgb, 0, D) / D) * 255.0
        s8 = np.clip(s + 0.5, 0, 255).astype(np.uint8)
        name = f"img/{Path(r['file']).stem}.webp"
        cv2.imwrite(str(out / name), cv2.cvtColor(s8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, a.quality])
        # terrace of interest: the image is centred on it; its most common colour in a central window
        h_, w_ = rgb.shape[:2]
        cwin = rgb[h_ // 2 - 90:h_ // 2 + 90, w_ // 2 - 90:w_ // 2 + 90]
        tcol, tfrac = mode_rgb(np.clip(cwin, 0, D), step=3)
        disp = np.clip(rgb * (255.0 / D), 0, 255).astype(np.uint8)
        thumbs.append(cv2.resize(disp, (160, 120), interpolation=cv2.INTER_AREA))
        fid = int(r["label"].split("_F")[1].split("_")[0]) if "_F" in r["label"] else None
        fl = flakes.get(fid, {})
        num = lambda v, d=1: (round(float(v), d) if v not in (None, "", "nan") else None)
        items.append(dict(
            id=Path(r["file"]).stem, file=name, order=int(r["order"]), label=r["label"], flake=fid,
            thickness=num(fl.get("best_thickness_nm")), terraces=fl.get("thicknesses_nm", ""),
            d=num(fl.get("best_terrace_d_um")), area=num(fl.get("area_um2"), 0), score=num(fl.get("score"), 3),
            quality=fl.get("quality", ""), x=num(r["stage_x_um"]), y=num(r["stage_y_um"]), z=num(r["z_um"], 2),
            focus=r.get("focus", ""), af=num(r.get("af_ratio"), 3), offset=num(r.get("offset_um"), 2),
            sat=round(sat, 5), thumb=k, tc=[round(float(v), 1) for v in tcol], tcf=round(float(tfrac), 2)))
    # thumbnail sheet
    cols = 16
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * 120, cols * 160, 3), np.uint8)
    for k, t in enumerate(thumbs):
        rr, cc = divmod(k, cols)
        sheet[rr * 120:(rr + 1) * 120, cc * 160:(cc + 1) * 160] = t
    cv2.imwrite(str(out / "thumbs.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    sub_txt = a.subtitle or (("hBN on 285 nm SiO₂" if a.material == "hbn" else "graphite on 90 nm SiO₂") +
                             f" · 100x · white balance matched to the demo 100x {a.material} images")
    meta = dict(source=str(src), n=len(items), D=D, material=a.material, subtitle=sub_txt, full_scale_display=fs_disp.round(1).tolist(), gain=gain.round(5).tolist(), target=target.round(2).tolist(),
                substrate_counts=sub.round(2).tolist(), um_per_px=UM_PER_PX_100X, width=int(raw.shape[1]),
                height=int(raw.shape[0]), thumb=dict(w=160, h=120, cols=cols, rows=rows))
    (out / "items.json").write_text(json.dumps(dict(meta=meta, items=items), separators=(",", ":")))
    tot = sum((out / it["file"]).stat().st_size for it in items)
    print(f"{len(items)} images, {tot / 1e6:.1f} MB (webp {'lossless' if a.quality > 100 else 'q%d' % a.quality}); thumbs {(out / 'thumbs.jpg').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
