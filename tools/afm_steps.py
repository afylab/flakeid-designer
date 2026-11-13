#!/usr/bin/env python3
"""Step heights from AFM images (Bruker NanoScope .spm), for the thickness calibration.

    python tools/afm_steps.py testing_afm/bigbn2_steps -o testing_afm/bigbn2_steps_results \
        [--sites "testing images/bigbn2_afm_set/data.json"]

For every file: bad data masked, levelled (flakeid.afm), and the height of every flat level above the substrate
measured.  Writes <out>/steps.csv (one row per level), <out>/steps.json and a diagnostic figure per file
(<out>/<file>.png: levelled height, mask and flat levels, histogram, per-line steps).

--sites: the AFM calibration page's data.json.  Files are matched to its scan sites in the page's (stage path)
order, which is the order of the programmed-move point list, so the n-th file is the n-th site; --first N if the
run started at site N.  Each row then gets the flake, terrace and the thickness predicted from the 10x image.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.afm import analyse_auto, read_spm  # noqa: E402

LEVEL_COLOURS = ["#3b6fb6", "#e0a32a", "#3aa57a", "#c4473a", "#8a5cc2", "#5bb6c9", "#b0703c", "#7f8c8d"]


def figure(name, spm, r, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    z = r["z"].copy()
    if "row_block" in r:                                     # other blocks: offset unknown, not shown
        z[r["row_block"] != r["main_block"], :] = np.nan
    ext = [0, spm.scan_um, spm.scan_um, 0]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1.15]))
    raw = spm["Height Sensor"]
    yy, xx = np.mgrid[0:raw.shape[0], 0:raw.shape[1]]
    g = np.isfinite(raw)
    A = np.c_[xx[g], yy[g], np.ones(g.sum())]
    c, *_ = np.linalg.lstsq(A, raw[g], rcond=None)
    flat0 = raw - (c[0] * xx + c[1] * yy + c[2])
    lo, hi = np.nanpercentile(flat0, [1, 99])
    ax[0].imshow(flat0, cmap="inferno", vmin=lo, vmax=hi, extent=ext)
    ax[0].set_title("1. height, plane only (nm)", fontsize=10)
    lab = np.full(z.shape, np.nan)
    for i, m in enumerate(r["level_masks"]):
        if r["levels"][i]["main_block"]:
            lab[m] = i
    ax[1].imshow(np.where(np.isfinite(z), z, np.nan), cmap="gray", vmin=np.nanpercentile(z, 1), vmax=np.nanpercentile(z, 99), extent=ext)
    ax[1].imshow(lab, cmap=ListedColormap(LEVEL_COLOURS), vmin=-0.5, vmax=len(LEVEL_COLOURS) - 0.5, alpha=0.55, extent=ext, interpolation="nearest")
    ax[1].imshow(np.where(r["bad"], 1.0, np.nan), cmap=ListedColormap(["#ff2d6f"]), vmin=0, vmax=1, alpha=0.85, extent=ext, interpolation="nearest")
    if "row_block" in r:
        ax[1].imshow(np.where((r["row_block"] != r["main_block"])[:, None] & ~r["bad"], 1.0, np.nan), cmap=ListedColormap(["#8a8f96"]),
                     vmin=0, vmax=1, alpha=0.8, extent=ext, interpolation="nearest")
    ax[1].set_title("2. bad data (pink), other blocks (grey), flat levels (colours)", fontsize=10)
    im = ax[2].imshow(z, cmap="inferno", vmin=np.nanpercentile(z, 1), vmax=np.nanpercentile(z, 99), extent=ext)
    ax[2].set_title(f"3. levelled ({'plane' if r['order'] == 1 else 'quadratic'} + line offsets), nm", fontsize=10)
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    v = z[np.isfinite(z)]
    lo, hi = np.percentile(v, [0.2, 99.8])
    ax[3].hist(v, bins=400, range=(lo, hi), color="#9aa3ab", log=True)
    for i, L in enumerate(r["levels"]):
        if not L["main_block"]:
            continue
        m = r["level_masks"][i]
        ax[3].hist(z[m], bins=400, range=(lo, hi), color=LEVEL_COLOURS[i % len(LEVEL_COLOURS)], log=True, alpha=0.9)
        if "step_nm" in L:
            ax[3].axvline(np.median(z[m]), color="k", lw=0.8, ls="--")
            ax[3].text(np.median(z[m]), ax[3].get_ylim()[1] * 0.6, f" {L['step_nm']:.2f} nm", fontsize=9, rotation=90, va="top")
    ax[3].set_xlabel("height above the substrate (nm)")
    ax[3].set_title("4. histogram of the levelled image", fontsize=10)
    for a in ax[:3]:
        a.set_xlabel("µm")
    fig.suptitle(name, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help=".spm files or folders")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--sites", help="AFM calibration page data.json: match files to its scan sites in order")
    ap.add_argument("--first", type=int, default=1, help="with --sites: the site number of the first file (default 1)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)
    files = []
    for p in a.inputs:
        p = Path(p)
        files += sorted(p.glob("*.spm")) if p.is_dir() else [p]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sites = []
    if a.sites:
        D = json.loads(Path(a.sites).read_text())
        sites = [(F, t) for F in D["flakes"] for t in F["terraces"] if t.get("site")]
    rows, js = [], []
    for k, f in enumerate(files):
        spm = read_spm(f)
        z = spm["Height Sensor"] if "Height Sensor" in spm.channels else spm[next(iter(spm.channels))]
        r = analyse_auto(z, spm.pixel_nm)
        alt = r.get("alt")
        site = sites[a.first - 1 + k] if a.sites and a.first - 1 + k < len(sites) else None
        levels = []
        for L in r["levels"]:
            rec = {kk: v for kk, v in L.items() if kk != "per_line"}
            # systematic: the same level with the other background model (plane vs quadratic)
            if alt and "step_nm" in L:
                cand = [M for M in alt["levels"] if "step_nm" in M and M["block"] == L["block"]]
                if cand:
                    M = min(cand, key=lambda M: abs(M["step_nm"] - L["step_nm"]))
                    rec["step_other_background_nm"] = M["step_nm"]
            levels.append(rec)
            # levels in a secondary block (cut off by a band of unusable lines) are kept only when they are large
            if "step_nm" in L and (L["main_block"] or L["area_frac"] >= 0.10):
                sys_err = abs(rec.get("step_other_background_nm", L["step_nm"]) - L["step_nm"])
                meth = abs(L.get("step_lines_nm", L["step_nm"]) - L["step_nm"])
                stat = L.get("step_err_nm", float("nan"))
                rows.append(dict(file=f.name, level=L["index"], block=L["block"], main_block=L["main_block"],
                                 step_nm=round(L["step_nm"], 3), step_lines_nm=round(L.get("step_lines_nm", float("nan")), 3),
                                 stat_err_nm=round(stat, 3), background_diff_nm=round(sys_err, 3), fit_vs_lines_nm=round(meth, 3),
                                 err_nm=round(float(np.sqrt((stat if stat == stat else 0) ** 2 + sys_err ** 2 + meth ** 2)), 3),
                                 n_lines=L.get("n_lines", 0),
                                 area_frac=round(L["area_frac"], 3), terrace_rms_nm=round(L["rms_nm"], 3),
                                 background="plane" if r["order"] == 1 else "quadratic", noise_nm=round(r["noise_nm"], 3),
                                 bad_frac=round(r["bad_frac"], 3),
                                 site=(a.first + k) if site else "", flake=site[0]["flake"] if site else "",
                                 terrace=site[1]["id"] if site else "", predicted_nm=site[1]["t"] if site else ""))
        js.append(dict(file=f.name, scan_um=spm.scan_um, pixel_nm=spm.pixel_nm, background=r["order"], noise_nm=r["noise_nm"],
                       bad_frac=r["bad_frac"], n_blocks=r.get("n_blocks", 1), levels=levels,
                       per_line={str(L["index"]): L["per_line"].tolist() for L in r["levels"] if "per_line" in L},
                       site=dict(n=a.first + k, flake=site[0]["flake"], terrace=site[1]["id"], predicted_nm=site[1]["t"],
                                 kind=site[1]["site"]["kind"]) if site else None))
        if not a.no_figures:
            figure(f.name, spm, r, out / f"{f.stem}.png")
        main = [x for x in rows if x["file"] == f.name and x["main_block"]]
        best = max(main, key=lambda x: x["area_frac"]) if main else None
        print(f"{f.name}: {len(r['levels'])} levels, bad {100 * r['bad_frac']:.0f} %, noise {r['noise_nm']:.2f} nm"
              + (f"; main step {best['step_nm']:.2f} ± {best['err_nm']:.2f} nm" if best else "; no step")
              + (f"  (site {a.first + k}: F{site[0]['flake']} T{site[1]['id']}, predicted {site[1]['t']} nm)" if site else ""))
    if rows:
        with open(out / "steps.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "steps.json").write_text(json.dumps(js, indent=1))
    print(f"wrote {out / 'steps.csv'}, steps.json" + ("" if a.no_figures else " and one figure per file"))


if __name__ == "__main__":
    main()
