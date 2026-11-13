#!/usr/bin/env python3
"""Compare two scans of the same chip (e.g. before / after changing the acquisition).

    python tools/compare_scans.py OLD.vsi NEW.vsi hbn 285 -o comparisons/acq

Reports shading (vignetting) amplitude on bare substrate, clipped fraction,
focus of detected flakes, and flake-by-flake agreement after registering the
two mosaics.  Writes <out>.json and a figure <out>.jpg.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid import find_flakes
from flakeid.io import load_image
from flakeid.background import detect_chip


def shading(img):
    """Raw substrate brightness variation: block medians (50 um) of the green channel on bare substrate."""
    g = img.rgb[..., 1].astype(np.float32)
    s = 50
    H, W = g.shape[0] // s * s, g.shape[1] // s * s
    blk = np.median(g[:H, :W].reshape(H // s, s, W // s, s).transpose(0, 2, 1, 3).reshape(H // s, W // s, -1), axis=2)
    small = cv2.resize(img.rgb, (W // s, H // s), interpolation=cv2.INTER_AREA).astype(np.float32)
    chip = detect_chip(small, 3)
    v = blk[chip]
    med = np.median(v)
    sub = chip & (np.abs(blk / med - 1) < 0.12)          # exclude flakes / debris
    v = blk[sub] / med
    # column profile through substrate -> tile period
    prof = np.array([np.median(blk[:, j][sub[:, j]]) if sub[:, j].sum() > 5 else np.nan for j in range(blk.shape[1])])
    prof = prof / np.nanmedian(prof) - 1
    return dict(p2p_pct=float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) * 100),
                std_pct=float(np.std(v) * 100)), prof, s * img.pixel_um


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("material"); ap.add_argument("oxide", type=float)
    ap.add_argument("-o", "--out", default="comparisons/acquisition")
    a = ap.parse_args()
    out = {}
    imgs, res, profs = {}, {}, {}
    for k, p in (("old", a.old), ("new", a.new)):
        im = load_image(p)
        imgs[k] = im
        r = find_flakes(im, a.material, a.oxide, refine_oxide_thickness=False)
        res[k] = r
        sh, prof, step = shading(im)
        profs[k] = (prof, step)
        sat = [t.saturated_frac for t in r.terraces]
        foc = np.array([f.focus_um for f in r.flakes if f.focus_um == f.focus_um])
        out[k] = dict(file=str(p), size_px=list(im.rgb.shape[:2]), substrate_raw=list(np.round(r._substrate_rgb, 0)),
                      substrate_frac_of_full_scale=list(np.round(np.array(r._substrate_rgb) / im.max_value, 3)),
                      noise=r.params["substrate_noise"], shading=sh,
                      flakes=len(r.flakes), terraces=len(r.terraces),
                      flakes_score_ge_05=int(sum(f.score >= 0.5 for f in r.flakes)),
                      terraces_saturated_frac_gt_0_2=float(np.mean(np.array(sat) > 0.2)),
                      focus_um_median=float(np.median(foc)), focus_um_p90=float(np.percentile(foc, 90)),
                      flakes_focus_gt_4um=float(np.mean(foc > 4)))
    # register new -> old on 1/8-scale green
    g = {k: cv2.resize(imgs[k].rgb[..., 1].astype(np.float32), None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA) for k in imgs}
    for k in g:
        g[k] = g[k] / np.median(g[k])
    orb = cv2.ORB_create(6000)
    u8 = {k: np.clip(g[k] * 128, 0, 255).astype(np.uint8) for k in g}
    k1, d1 = orb.detectAndCompute(u8["new"], None); k2, d2 = orb.detectAndCompute(u8["old"], None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.float32([k1[x.queryIdx].pt for x in m]); p2 = np.float32([k2[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(p1, p2, ransacReprojThreshold=2)
    M[:, 2] *= 8                                   # full-res: old = M @ new
    out["registration"] = dict(matrix=M.round(4).tolist(), inliers=int(inl.sum()))
    # match flakes by best-terrace target point
    def pts(r):
        return np.array([[f.target_x_px, f.target_y_px] for f in r.flakes])
    pn = pts(res["new"]) @ M[:, :2].T + M[:, 2]
    po = pts(res["old"])
    from scipy.spatial import cKDTree
    d, j = cKDTree(po).query(pn, distance_upper_bound=15)
    ok = np.isfinite(d)
    tn = np.array([f.best_thickness_nm for f in res["new"].flakes])[ok]
    to = np.array([res["old"].flakes[i].best_thickness_nm for i in j[ok]])
    fo = np.array([res["old"].flakes[i].focus_um for i in j[ok]]); fn = np.array([f.focus_um for f in res["new"].flakes])[ok]
    # overlap region only for the found-in-one-only counts
    Ho, Wo = imgs["old"].rgb.shape[:2]
    inside = (pn[:, 0] > 50) & (pn[:, 0] < Wo - 50) & (pn[:, 1] > 50) & (pn[:, 1] < Ho - 50)
    out["matching"] = dict(new_flakes_in_old_fov=int(inside.sum()), matched=int(ok.sum()),
                           thickness_agree_within_10pct=float(np.mean(np.abs(tn - to) <= 0.1 * np.maximum(to, 1))),
                           focus_old_median_matched=float(np.nanmedian(fo)), focus_new_median_matched=float(np.nanmedian(fn)),
                           focus_improved_frac=float(np.nanmean(fn < fo - 0.3)), focus_worse_frac=float(np.nanmean(fn > fo + 0.3)))
    print(json.dumps(out, indent=2, default=float))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(out, indent=2, default=float))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, c in (("old", "#8c959f"), ("new", "#0969da")):
        prof, step = profs[k]
        ax[0].plot(np.arange(len(prof)) * step / 1000, prof * 100, color=c, lw=1, label=k)
    ax[0].set(xlabel="position (mm)", ylabel="substrate brightness (%)", title="Raw shading across the chip (column medians)")
    ax[0].legend(frameon=False)
    ax[1].scatter(fo, fn, s=6, alpha=0.5, color="#57606a"); lim = [0, max(8, np.nanpercentile(np.r_[fo, fn], 98))]
    ax[1].plot(lim, lim, color="#888", lw=0.8); ax[1].set(xlim=lim, ylim=lim, xlabel="old scan: edge width (µm)", ylabel="new scan: edge width (µm)", title="Focus of the same flakes (lower = sharper)")
    ax[2].scatter(to, tn, s=6, alpha=0.5, color="#57606a"); ax[2].plot([0, 350], [0, 350], color="#888", lw=0.8)
    ax[2].set(xlim=(0, 350), ylim=(0, 350), xlabel="old scan thickness (nm)", ylabel="new scan thickness (nm)", title="Best-terrace thickness, matched flakes")
    for x in ax: x.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(a.out + ".jpg", dpi=90, pil_kwargs={"quality": 90})
    print("wrote", a.out + ".json", a.out + ".jpg")


if __name__ == "__main__":
    main()
