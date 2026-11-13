"""Stitching accuracy of the 100x / IR terrace mosaics (flakeid.fieldmosaic) on real pictures.

A ground truth larger than a field is made by tiling real pictures (100x: the demo 100x JPEGs; IR: real stacking-domain
PNGs of graphite_20261019_run_1 if --ir is given, else grey 100x pictures at ~1.5 % contrast), laid on the stage.  The
fields are planned over a terrace outline (plan_fields), "taken" at their stage positions + a random stage error (up to
+-3 um, not known to the stitcher), with a brightness error (+-3 %), vignetting divided out by a slightly wrong flat
field, pixel noise and (IR) the camera's own fixed pattern; the IR camera mirrored and turned 5.8 deg against the stage.
Then stitched from the recorded (error-free) positions and compared with the truth: field positions (median < 1 px,
worst < 3 px over the fields with a matched overlap), and the mosaic against the truth in the overlaps against the
single-field parts (seams: no larger deviation in the overlaps).

    python tools/mm_mock/run_mock_mosaic_stitch.py [--images DIR] [--ir DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.fieldmosaic import plan_fields, stitch  # noqa: E402
from flakeid.hundred import domain_preview  # noqa: E402

M10 = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])
M100 = M10 / 9.54


def ellipse(a, b, n=36, rot=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = math.cos(rot), math.sin(rot)
    P = np.c_[a * np.cos(t), b * np.sin(t)]
    return P @ np.array([[c, s], [-s, c]])


def truth(pics, cols, g):
    """Pictures tiled into one ground-truth image (cols per row), g um per truth pixel; (image, origin um)."""
    h, w = pics[0].shape[:2]
    rows = int(math.ceil(len(pics) / cols))
    T = np.zeros((rows * h, cols * w) + pics[0].shape[2:], np.float32)
    for k, p in enumerate(pics):
        r, c = divmod(k, cols)
        T[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(p, (w, h)) if p.shape[:2] != (h, w) else p
    return T, -np.array([T.shape[1], T.shape[0]], float) * g / 2.0     # stage (0, 0) at the truth's centre


def render(T, org, g, M, wh, s):
    """A camera field at stage s: pixel p shows stage s + M (p - c), c the frame centre."""
    W, H = wh
    c = np.array([W / 2.0, H / 2.0])
    A = np.asarray(M, float) / g
    b = (np.asarray(s, float) - np.asarray(M, float) @ c - org) / g
    return cv2.warpAffine(T, np.hstack([A, b[:, None]]).astype(np.float64), (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def vignette(W, H, k=0.3):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return 1 - k * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)


def case(name, T, org, g, M, wh, poly, rng, noise, pattern=None, out=None, mono=False):
    W, H = wh
    fields, n_all = plan_fields(poly, M, wh, 0.15, 30)
    nominal = np.array([[0.0, 0.0]] + [[f[2], f[3]] for f in fields])
    err = rng.uniform(-3, 3, nominal.shape)
    err[0] = 0.0
    true = nominal + err
    vig = vignette(W, H)
    flat = vig * (1 + 0.004 * np.sin(np.linspace(0, 3, W))[None, :]).astype(np.float32)   # (a flat field not quite right)
    imgs = []
    for k, s in enumerate(true):
        f = render(T, org, g, M, wh, s)
        v = vig if f.ndim == 2 else vig[..., None]
        f = f * v * (1 + rng.uniform(-0.03, 0.03))
        f = f + rng.normal(0, noise, f.shape).astype(np.float32) * float(np.median(f))
        if pattern is not None:
            f = f * pattern
        imgs.append((f / (flat if f.ndim == 2 else flat[..., None])).astype(np.float32))
    r = stitch(imgs, nominal, M, ref=0)
    Mi = np.linalg.inv(M)
    tp = (Mi @ (true - true[0]).T).T
    ep = r["pos"] - r["pos"][0]
    e = np.hypot(*(ep - tp).T)
    linked = sorted({p[0] for p in r["pairs"] if p[6]} | {p[1] for p in r["pairs"] if p[6]})
    e_l = e[[k for k in linked if k != 0]] if len(linked) > 1 else np.zeros(0)
    e_stage = np.hypot(*(r["pos0"] - r["pos0"][0] - tp).T)
    # the mosaic against the truth seen from the ref field's true position (gain: the ref field's)
    mos, wsum = r["image"], r["weight"]
    Hm, Wm = wsum.shape
    c0 = r["ref_xy"]
    A = np.asarray(M, float) / g
    b = (true[0] - np.asarray(M, float) @ c0 - org) / g
    gt = cv2.warpAffine(T, np.hstack([A, b[:, None]]), (Wm, Hm), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    gm = mos if mos.ndim == 2 else mos.mean(axis=2)
    gg = gt if gt.ndim == 2 else gt.mean(axis=2)
    cnt = np.zeros((Hm, Wm), np.int16)
    for p in r["pos"]:
        x, y = int(round(p[0])) + 8, int(round(p[1])) + 8
        cnt[y:y + H - 16, x:x + W - 16] += 1
    ok = (wsum > 0.05) & (gg > 0)
    k_ = float(np.median(gm[ok & (cnt == 1)] / gg[ok & (cnt == 1)]))
    rel = cv2.GaussianBlur(gm / np.maximum(gg * k_, 1e-6) - 1, (0, 0), 3)     # (smoothed: the noise out, a seam stays)
    one, many = ok & (cnt == 1), ok & (cnt >= 2)
    dev1, dev2 = float(np.sqrt(np.mean(rel[one] ** 2))), float(np.sqrt(np.mean(rel[many] ** 2)))
    print(f"{name}: {len(true)} fields (plan: {n_all} for the terrace), {r['n_pairs_used']} of {len(r['pairs'])} overlaps matched; "
          f"stage error up to {np.abs(err).max():.1f} um = {e_stage.max():.1f} px")
    print(f"  position error (fields with a matched overlap: {len(e_l)} of {len(true) - 1}): median {np.median(e_l) if len(e_l) else float('nan'):.2f} px, "
          f"worst {e_l.max() if len(e_l) else float('nan'):.2f} px  (all fields: worst {e[1:].max():.2f} px)")
    print(f"  mosaic vs truth (3 px smoothed): rms {dev1 * 100:.2f} % in single-field parts, {dev2 * 100:.2f} % in overlaps")
    if out is not None:
        if mono:
            pv = domain_preview(np.where(wsum > 0, mos, 0).astype(np.float32))[0]
            pv[wsum <= 0] = 0
            cv2.imwrite(str(out / f"{name}_mosaic.jpg"), pv, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            q = float(np.percentile(mos[wsum > 0], 99.5))
            cv2.imwrite(str(out / f"{name}_mosaic.jpg"), cv2.cvtColor(np.clip(mos / q * 235, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / f"{name}_deviation.jpg"), np.clip(128 + rel * 128 / 0.03, 0, 255).astype(np.uint8))
    good = len(e_l) >= len(true) - 2 and len(e_l) > 0 and float(np.median(e_l)) < 1.0 and float(e_l.max()) < 3.0 and dev2 < max(1.5 * dev1, dev1 + 0.003)
    print(f"  {name}: {'pass' if good else 'FAIL'}")
    return good


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(ROOT / "example images" / "demo_high_magnification"))
    ap.add_argument("--ir", help="folder of real IR (stacking-domain) PNGs, e.g. <viewer>/100x/domains")
    ap.add_argument("--out", default=str(ROOT / "tools" / "mm_mock" / "out" / "mosaic_stitch"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    demo = [f for f in sorted(glob.glob(str(Path(a.images) / "*" / "*_100x.jpg"))) if " X" not in f]
    if len(demo) < 6:
        print(f"no demo 100x pictures in {a.images}")
        print("RESULT FAIL")
        return 1
    pick = [demo[i] for i in rng.choice(len(demo), 6, replace=False)]
    pics = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 * 0.8 + 0.02 for p in pick]
    pics = [cv2.resize(p, (1376, 1038), interpolation=cv2.INTER_AREA) for p in pics]
    g = 0.1144
    T, org = truth(pics, 3, g)
    ok = True
    # 100x: a terrace like graphite_20261019_1 flake 1 (~373 x 240 um), tilted
    ok &= case("100x", T, org, g, M100, (1376, 1038), ellipse(186, 120, rot=0.3), rng, 0.01, out=out)
    # IR: mono, 640 x 512 at 0.319 um / px, mirrored and turned 5.8 deg against the stage
    th = math.radians(5.8)
    Mir = 0.1144 / 0.358 * np.array([[math.cos(th), math.sin(th)], [math.sin(th), -math.cos(th)]])
    pattern = None
    if a.ir:
        irs = sorted(glob.glob(str(Path(a.ir) / "*IR*.png")))
        sel = [irs[i] for i in rng.choice(len(irs), min(6, len(irs)), replace=False)] if len(irs) >= 6 else []
        if not sel:
            print(f"(no IR PNGs in {a.ir}: grey 100x pictures instead)")
        ir = [cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) for p in sel]
        ir = [np.where(x > 0, x, np.median(x)) / float(np.median(x)) for x in ir]     # (dead border pixels: the median)
        # the IR camera's own fixed pattern (the same pixels in every frame): from the pictures' median
        if ir:
            pattern = np.median(np.array(ir), axis=0)
            pattern = (pattern / cv2.GaussianBlur(pattern, (0, 0), 20)).astype(np.float32)
            ir = [x / pattern for x in ir]
    else:
        ir = []
    if not ir:
        gr = [cv2.resize(p.mean(axis=2), (640, 512), interpolation=cv2.INTER_AREA) for p in pics]
        ir = [(1 + 0.015 * (x - x.mean()) / max(float(x.std()), 1e-6)).astype(np.float32) for x in gr]
    gI = 0.1144 / 0.358
    TI, orgI = truth(ir, 3, gI)
    ok &= case("IR", TI * 30000.0, orgI, gI, Mir, (640, 512), ellipse(250, 170, rot=-0.2), rng, 0.003, pattern=pattern, out=out, mono=True)
    print(f"pictures in {out}")
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
