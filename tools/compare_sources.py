#!/usr/bin/env python3
"""Compare flake identification from a cellSens JPEG export with the
full-resolution VSI of the same scan.

    python tools/compare_sources.py --jpg-dir A --vsi-dir B --jpg-all A_all --vsi-all B_all \
        --stem fullchipbn23 --offset 1 93 -o comparison.jpg

Each directory holds <stem>_terraces.csv/json from find_flakes (the *_all ones
run with --all).  Regions are matched by target position after shifting JPEG
coordinates by the export offset (JPEG pixel + offset = VSI pixel).  Prints a
summary and writes a one-page JPEG figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

OK = {"good", "chip_edge", "saturated", "ambiguous_thickness", "low_confidence"}


def rows(p):
    return list(csv.DictReader(open(p)))


def accepted(r):
    return not (set(r["quality"].split(";")) - OK)


def match(a, b, off, tol_px=6.0):
    """Pairs (i, j) of regions whose centroids agree within tol and whose areas agree within 40 %."""
    pa = np.array([[float(r["x_px"]) + off[0], float(r["y_px"]) + off[1]] for r in a])
    pb = np.array([[float(r["x_px"]), float(r["y_px"])] for r in b])
    if not len(pa) or not len(pb):
        return []
    t = cKDTree(pb)
    d, j = t.query(pa, distance_upper_bound=tol_px)
    out = []
    for i, (dd, jj) in enumerate(zip(d, j)):
        if np.isfinite(dd):
            aa, ab = float(a[i]["area_um2"]), float(b[jj]["area_um2"])
            if abs(aa - ab) / max(aa, ab) < 0.4:
                out.append((i, jj))
    return out


def point_in_other(a, b, off):
    """For each accepted region of a, is its target point inside the bbox of any accepted region of b
    (same flake detected, possibly split into terraces differently)?"""
    boxes = np.array([[int(r["bbox_x_px"]), int(r["bbox_y_px"]), int(r["bbox_w_px"]), int(r["bbox_h_px"])] for r in b])
    out = []
    for r in a:
        x, y = float(r["target_x_px"]) + off[0], float(r["target_y_px"]) + off[1]
        hit = ((boxes[:, 0] - 3 <= x) & (x <= boxes[:, 0] + boxes[:, 2] + 3) &
               (boxes[:, 1] - 3 <= y) & (y <= boxes[:, 1] + boxes[:, 3] + 3)) if len(boxes) else np.zeros(0, bool)
        out.append(bool(hit.any()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpg-dir", required=True)
    ap.add_argument("--vsi-dir", required=True)
    ap.add_argument("--jpg-all", required=True)
    ap.add_argument("--vsi-all", required=True)
    ap.add_argument("--stem", default="fullchipbn23")
    ap.add_argument("--offset", nargs=2, type=float, default=(1, 93))
    ap.add_argument("-o", "--output", default="jpeg_vs_vsi.jpg")
    a = ap.parse_args()
    st = a.stem
    J, V = rows(Path(a.jpg_dir) / f"{st}_terraces.csv"), rows(Path(a.vsi_dir) / f"{st}_terraces.csv")
    Ja, Va = rows(Path(a.jpg_all) / f"{st}_terraces.csv"), rows(Path(a.vsi_all) / f"{st}_terraces.csv")
    jj, vj = json.load(open(Path(a.jpg_dir) / f"{st}_flakes.json")), json.load(open(Path(a.vsi_dir) / f"{st}_flakes.json"))
    off = a.offset

    # common field of view: the JPEG area inside the VSI frame
    def in_jpg_fov(r):
        x, y = float(r["x_px"]) - off[0], float(r["y_px"]) - off[1]
        return 60 < x < 11127 - 60 and 60 < y < 10237 - 60
    V_c = [r for r in V if in_jpg_fov(r)]
    Va_c = [r for r in Va if in_jpg_fov(r)]

    # region-level agreement over all segmented regions (before quality cuts)
    pairs_all = match(Ja, Va_c, off)
    # accepted-set agreement
    acc_j = [r for r in Ja if accepted(r)]
    acc_v = [r for r in Va_c if accepted(r)]
    pairs_acc = match(acc_j, acc_v, off)
    pairs_mix = match(Ja, Va_c, off)
    only_j = len(acc_j) - len(pairs_acc)
    only_v = len(acc_v) - len(pairs_acc)
    # flake-level: does the other source also report something at this spot?
    j_found = point_in_other(acc_j, acc_v, off)
    v_found = point_in_other(acc_v, acc_j, (-off[0], -off[1]))
    flake_only_j = [r for r, f in zip(acc_j, j_found) if not f]
    flake_only_v = [r for r, f in zip(acc_v, v_found) if not f]

    cj = np.array([[float(Ja[i]["contrast_" + k]) for k in "rgb"] for i, _ in pairs_all])
    cv = np.array([[float(Va_c[j]["contrast_" + k]) for k in "rgb"] for _, j in pairs_all])
    tj = np.array([float(Ja[i]["thickness_nm"]) for i, _ in pairs_all])
    tv = np.array([float(Va_c[j]["thickness_nm"]) for _, j in pairs_all])
    aj = np.array([float(Ja[i]["area_um2"]) for i, _ in pairs_all])
    av = np.array([float(Va_c[j]["area_um2"]) for _, j in pairs_all])
    ew_j = np.array([float(Ja[i]["edge_width_um"]) for i, _ in pairs_all])
    ew_v = np.array([float(Va_c[j]["edge_width_um"]) for _, j in pairs_all])
    thin = np.linalg.norm(cj, axis=1) < 0.3

    # why do accepted sets differ? flags of the counterpart
    acc_v_keys = {(round(float(r["x_px"])), round(float(r["y_px"]))) for r in acc_v}
    reasons_j_only, reasons_v_only = {}, {}
    for i, j in pairs_mix:
        rj, rv = Ja[i], Va_c[j]
        if accepted(rj) and not accepted(rv):
            for q in set(rv["quality"].split(";")) - OK:
                reasons_j_only[q] = reasons_j_only.get(q, 0) + 1
        if accepted(rv) and not accepted(rj):
            for q in set(rj["quality"].split(";")) - OK:
                reasons_v_only[q] = reasons_v_only.get(q, 0) + 1

    summ = {
        "jpeg": dict(file=jj["image"], noise=jj["params"]["substrate_noise"], runtime_s=jj["params"]["runtime_s"],
                     oxide=jj["oxide_nm_used"], regions=len(Ja), accepted=len(acc_j)),
        "vsi": dict(file=vj["image"], noise=vj["params"]["substrate_noise"], runtime_s=vj["params"]["runtime_s"],
                    oxide=vj["oxide_nm_used"], regions_in_common_fov=len(Va_c), accepted_in_common_fov=len(acc_v),
                    accepted_total=sum(accepted(r) for r in Va)),
        "matched_regions": len(pairs_all),
        "accepted_both": len(pairs_acc), "accepted_jpeg_only": only_j, "accepted_vsi_only": only_v,
        "location_found_in_other_jpeg": float(j_found.mean()), "location_found_in_other_vsi": float(v_found.mean()),
        "location_only_jpeg": len(flake_only_j), "location_only_vsi": len(flake_only_v),
        "location_only_jpeg_median_area_um2": float(np.median([float(r["area_um2"]) for r in flake_only_j])) if flake_only_j else None,
        "location_only_vsi_median_area_um2": float(np.median([float(r["area_um2"]) for r in flake_only_v])) if flake_only_v else None,
        "why_rejected_in_vsi_when_accepted_in_jpeg": reasons_j_only,
        "why_rejected_in_jpeg_when_accepted_in_vsi": reasons_v_only,
        "contrast_diff_median_rgb": np.median(cv - cj, axis=0).round(4).tolist(),
        "contrast_diff_mad_rgb": (1.4826 * np.median(np.abs(cv - cj - np.median(cv - cj, axis=0)), axis=0)).round(4).tolist(),
        "thickness_diff_median_nm": float(np.median(tv - tj)),
        "thickness_abs_diff_p90_nm": float(np.percentile(np.abs(tv - tj), 90)),
        "thickness_agree_within_1nm_frac": float(np.mean(np.abs(tv - tj) <= 1.0)),
        "thin_regions_(|C|<0.3)_thickness_mad_nm": float(np.median(np.abs(tv[thin] - tj[thin]))) if thin.any() else None,
        "area_ratio_median": float(np.median(av / aj)),
        "edge_width_median_um": [float(np.nanmedian(ew_j)), float(np.nanmedian(ew_v))],
    }
    print(json.dumps(summ, indent=2))
    for name, sel in (("only_jpeg", flake_only_j), ("only_vsi", flake_only_v)):
        with open(Path(a.output).with_name(f"{st}_{name}.csv"), "w", newline="") as fh:
            if sel:
                w = csv.DictWriter(fh, fieldnames=sel[0].keys())
                w.writeheader()
                w.writerows(sel)

    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    fig.suptitle(f"{st}: cellSens JPEG export vs full-resolution VSI (same pipeline, same profile)", fontsize=13)
    for k, (nm, col) in enumerate([("R", "#d1242f"), ("G", "#1a7f37"), ("B", "#0969da")]):
        ax[0, 0].scatter(cj[:, k], cv[:, k], s=4, alpha=0.4, color=col, label=nm)
    lim = [-0.8, 1.3]
    ax[0, 0].plot(lim, lim, color="#888", lw=0.8)
    ax[0, 0].set(xlim=lim, ylim=lim, xlabel="JPEG region contrast", ylabel="VSI region contrast",
                 title=f"Median contrast of {len(pairs_all)} matched regions")
    ax[0, 0].legend(frameon=False)
    m = (tj < 200) & (tv < 200)
    ax[0, 1].scatter(tj[m], tv[m], s=5, alpha=0.4, color="#57606a")
    ax[0, 1].plot([0, 200], [0, 200], color="#888", lw=0.8)
    ax[0, 1].set(xlabel="JPEG thickness (nm)", ylabel="VSI thickness (nm)", title="Fitted thickness, matched regions",
                 xlim=(0, 200), ylim=(0, 200))
    ax[0, 2].hist(np.clip(tv - tj, -10, 10), bins=81, color="#57606a")
    ax[0, 2].set(xlabel="VSI - JPEG thickness (nm, clipped to ±10)", ylabel="regions",
                 title=f"{100 * summ['thickness_agree_within_1nm_frac']:.0f}% agree within 1 nm")
    ch = ["R", "G", "B"]
    x = np.arange(3)
    ax[1, 0].bar(x - 0.18, np.array(jj["params"]["substrate_noise"]) * 100, 0.36, label="JPEG", color="#8c959f")
    ax[1, 0].bar(x + 0.18, np.array(vj["params"]["substrate_noise"]) * 100, 0.36, label="VSI", color="#0969da")
    ax[1, 0].set(xticks=x, xticklabels=ch, ylabel="substrate noise per pixel (%)",
                 title="Pixel noise of bare substrate (working resolution)")
    ax[1, 0].legend(frameon=False)
    ax[1, 1].scatter(aj, av / aj, s=5, alpha=0.4, color="#57606a")
    ax[1, 1].set(xscale="log", ylim=(0.6, 1.4), xlabel="JPEG area (µm²)", ylabel="VSI / JPEG area",
                 title="Segmented area, matched regions")
    ax[1, 1].axhline(1, color="#888", lw=0.8)
    lbl = ["same region", "same spot,\nsplit differently", "JPEG only", "VSI only"]
    split = len(acc_j) - len(pairs_acc) - len(flake_only_j)
    vals = [len(pairs_acc), split, len(flake_only_j), len(flake_only_v)]
    ax[1, 2].bar(lbl, vals, color=["#1a7f37", "#8fd19e", "#8c959f", "#0969da"])
    ax[1, 2].set(ylabel="accepted regions (JPEG count, common field of view)", title="Accepted flake regions")
    for i, v in enumerate(vals):
        ax[1, 2].text(i, v, str(v), ha="center", va="bottom")
    for axx in ax.ravel():
        axx.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(a.output, dpi=90, pil_kwargs={"quality": 90})
    Path(a.output).with_suffix(".json").write_text(json.dumps(summ, indent=2))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
