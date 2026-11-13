"""Estimate the fixed optical pattern (dust shadows, rings, vignetting) of a set of 100x frames.

The pattern is the same in every frame while the sample changes, so the per-pixel median of the
frames, each normalised to its own bare-substrate level with everything else masked, estimates it.
Pass 1 masks by colour (within 6 % of the substrate chromaticity); pass 2, with pass 1 divided out,
keeps only pixels within TOL of the substrate in every channel, which also excludes thin flakes
(frames are centred on flakes, so they would otherwise bias the centre).  Pixels with too few
substrate samples are filled from a smoothed version.

    python tools/optics_flat.py "testing images/bigbntest_100x" --gain-from "testing images/bigbntest_100x_score/items.json" \
        -o "testing images/bigbntest_100x_score/flat100x.npy"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.mm import BAYER_CODE  # noqa: E402
from flakeid.progress import Progress  # noqa: E402


def fill_gaps(f, ok, sigma=25.0):
    """Values for pixels without enough substrate samples: local smoothing where there is enough data
    nearby, else a smooth (quadratic) vignetting surface fitted to the whole frame."""
    H, W = ok.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (xx - W / 2) / W, (yy - H / 2) / H
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1)
    sel = ok.copy()
    sel[::1, ::1] &= (np.add.outer(np.arange(H), np.arange(W)) % 7 == 0)      # subsample for the fit
    quad = np.empty((H, W, 3), np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A[sel], f[..., c][sel], rcond=None)
        quad[..., c] = A @ coef
    wgt = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma)
    loc = cv2.GaussianBlur(np.where(ok[..., None], f, 0).astype(np.float32), (0, 0), sigma) / np.maximum(wgt, 1e-6)[..., None]
    a = np.clip((wgt - 0.05) / 0.25, 0, 1)[..., None]          # trust the local estimate only with enough data
    return a * loc + (1 - a) * quad


def estimate(files, gain, target, dark=128.0, tol=0.025, min_cov=20, passes=2, verbose=True):
    N = len(files)
    H, W = cv2.imread(files[0], cv2.IMREAD_UNCHANGED).shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    norm = np.lib.format.open_memmap(tmp.name, mode="w+", dtype=np.float16, shape=(N, H, W, 3))
    flat = np.ones((H, W, 3), np.float32)
    for ps in range(passes):
        pr = Progress(f"optics flat pass {ps + 1}", N, verbose)
        for k, f in enumerate(files):
            rgb = (cv2.cvtColor(cv2.imread(f, cv2.IMREAD_UNCHANGED), BAYER_CODE).astype(np.float32) - dark) * gain
            rel = cv2.GaussianBlur(rgb / target / flat, (0, 0), 2)
            if ps == 0:
                ratio = rel / np.median(rel, axis=2, keepdims=True)
                m = (np.abs(ratio - 1).max(2) < 0.06) & (np.abs(rel.mean(2) - np.median(rel.mean(2))) < 0.25)
            else:
                lvl0 = np.median(rel.reshape(-1, 3)[::7], 0)
                m = (np.abs(rel / lvl0 - 1).max(2) < 2 * tol)
                lvl1 = np.median(rel[m], 0) if m.sum() > 5000 else lvl0
                m = (np.abs(rel / lvl1 - 1).max(2) < tol)
            m = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            raw = (rgb / target)
            lvl = np.median((raw / flat)[m], 0) if m.sum() > 5000 else np.median((raw / flat).reshape(-1, 3), 0)
            x = (raw / lvl).astype(np.float16)
            x[~m] = np.nan
            norm[k] = x
            pr.update()
        norm.flush()
        new = np.ones((H, W, 3), np.float32)
        cov = np.zeros((H, W), np.int32)
        for r in range(0, H, 64):
            blk = np.asarray(norm[:, r:r + 64]).astype(np.float32)
            with np.errstate(all="ignore"):
                new[r:r + 64] = np.nanmedian(blk, axis=0)
            cov[r:r + 64] = np.isfinite(blk[..., 1]).sum(0)
        ok = (cov >= min_cov) & np.isfinite(new).all(2)
        flat = cv2.GaussianBlur(np.where(ok[..., None], new, fill_gaps(new, ok)).astype(np.float32), (0, 0), 0.8)
        flat /= np.median(flat.reshape(-1, 3), 0)
        if verbose:
            print(f"  pass {ps + 1}: {(~ok).mean() * 100:.1f} % of pixels filled (fewer than {min_cov} substrate samples); "
                  f"median coverage {np.median(cov):.0f} of {N} frames")
    del norm
    Path(tmp.name).unlink(missing_ok=True)
    return flat, cov


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--gain-from", required=True, help="items.json of make_score_set.py (white-balance gains)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    meta = json.load(open(a.gain_from))["meta"]
    files = sorted(glob.glob(str(Path(a.folder) / "*.tif")))
    flat, cov = estimate(files, np.array(meta["gain"], np.float32), np.array(meta["target"], np.float32))
    np.save(a.out, flat)
    print(f"wrote {a.out}; range (0.5-99.5 %):", [np.round(np.percentile(flat[..., c], [0.5, 99.5]), 3).tolist() for c in range(3)])


if __name__ == "__main__":
    main()
