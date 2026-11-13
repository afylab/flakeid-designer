#!/usr/bin/env python3
"""Choose AFM calibration targets and write a printable target sheet.

    python tools/select_afm.py --results "example images" -o AFM_targets

Writes AFM_targets.csv (one row per terrace to measure) and AFM_targets.pdf
(chip locator maps + a context and close-up crop of each site).

Selection logic
---------------
* Only a few chips (fewer sample exchanges), spanning the thickness window.
* Flat, sharp-edged, simply shaped terraces with a large inscribed circle
  (easy to find and scan), adjacent to bare SiO2 (clean step to the substrate).
* Within each thickness bin, preference for terraces where the two candidate
  camera models (tone-curve profile vs stray-light profile) predict different
  thicknesses: those measurements decide between the models.
* For hBN, cyan / green flakes whose colour fits 2nd-order interference best
  but also has a 1st-order (20-60 nm) solution are deliberately included.
* Flakes with several terraces are preferred: one AFM scan calibrates several
  thicknesses at once.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.thickness import ThicknessEstimator, load_profile  # noqa: E402

OK = {"good", "saturated", "ambiguous_thickness", "low_confidence", "chip_edge"}
# oxide thickness of the stray-light profile relative to the tone-curve profile (from the two calibrations)
OX_RATIO = {"graphite": 1.054, "hbn": 1.047}

# (material, label, lo, hi, n_sites, requires)
BINS = [
    ("graphite", "1-2 layers", 0.2, 0.8, 2, None),
    ("graphite", "3-5 layers", 0.8, 1.8, 2, None),
    ("graphite", "2-5 nm", 1.8, 5.0, 3, None),
    ("graphite", "5-10 nm", 5.0, 10.0, 3, None),
    ("graphite", "10-15 nm", 10.0, 15.0, 1, None),
    ("graphite", "15-25 nm", 15.0, 25.0, 2, None),
    ("hbn", "4-10 nm", 4.0, 10.0, 3, None),
    ("hbn", "10-20 nm", 10.0, 20.0, 3, None),
    ("hbn", "cyan/green: 20-60 nm or ~150-190 nm?", 20.0, 60.0, 4, "ambiguous"),
    ("hbn", "40-60 nm", 40.0, 60.0, 1, "unambiguous"),
    ("hbn", "60-100 nm", 60.0, 100.0, 3, "unambiguous"),
]
CHIPS = {"graphite": ["g34", "g32"], "hbn": ["fullchipbn23", "bn25"]}


def in_window(r, lo, hi):
    ts = [float(r["thickness_nm"])]
    if r["alt_thickness_nm"]:
        ts += [float(a) for a in str(r["alt_thickness_nm"]).split(";")]
    for t in ts:
        if lo <= t <= hi:
            return t
    return None


def load(results, stem, mat, prof_g, prof_f):
    d = json.load(open(Path(results) / f"{stem}_flakes.json"))
    sub = np.array(d["params"]["substrate_rgb"])
    clip = 254.5 / sub - 1
    ef = ThicknessEstimator(mat, d["oxide_nm_used"] * OX_RATIO[mat], prof_f, clip_max=clip)
    rows = d["terraces"]
    by_parent = {}
    for r in rows:
        by_parent.setdefault(r["flake_id"], []).append(r)
    for r in rows:
        r["stem"], r["material"] = stem, mat
        c = np.array([r["contrast_r"], r["contrast_g"], r["contrast_b"]])
        cen = np.array([r["saturated_frac"] > 0.5] * 3) & (c > clip - 0.02)
        f = ef.fit(c, censored=cen)
        r["t_flare"] = round(f.thickness_nm, 1)
        r["t_flare_alt"] = ";".join(f"{a:g}" for a in f.alternatives_nm)
        sib = [s for s in by_parent[r["flake_id"]] if s is not r and not (set(s["quality"].split(";")) - OK)]
        r["n_terraces"] = 1 + len(sib)
    return d, rows


def eligible(r, min_d=12.0):
    q = set(r["quality"].split(";"))
    if q - OK or "chip_edge" in q:
        return False
    return (r["inscribed_diameter_um"] >= min_d and r["fill"] >= 0.95 and r["edge_width_um"] <= 4.0
            and r["solidity"] >= 0.7 and r["uniformity"] >= 0.8)


def select(all_rows):
    chosen, used_parents = [], set()
    for mat, label, lo, hi, n, req in BINS:
        cands = []
        for r in all_rows:
            if r["material"] != mat or r["stem"] not in CHIPS[mat] or not eligible(r, 8.0 if hi <= 10 else 12.0):
                continue
            if (r["stem"], r["flake_id"]) in used_parents:
                continue
            amb = "ambiguous_thickness" in r["quality"]
            if req == "ambiguous" and not amb:
                continue
            if req == "unambiguous" and amb:
                continue
            t = in_window(r, lo, hi) if req == "ambiguous" else (
                float(r["thickness_nm"]) if lo <= float(r["thickness_nm"]) < hi else None)
            if t is None:
                continue
            if mat == "hbn" and lo < 10 and "low_confidence" in r["quality"] and t < 4.5:
                continue
            dis = abs(float(r["t_flare"]) - float(r["thickness_nm"])) / max(float(r["thickness_nm"]), 1.0)
            score = min(r["inscribed_diameter_um"], 40) * (1 + min(dis, 1.0)) * (1 + 0.25 * min(r["n_terraces"] - 1, 3))
            if r["saturated_frac"] > 0.2 and mat == "hbn" and hi <= 60:
                score *= 0.7
            if mat == "hbn" and r["stem"] == "fullchipbn23":
                score *= 1.3                     # stage coordinates available
            cands.append((score, t, r))
        cands.sort(key=lambda x: -x[0])
        per_chip = {}
        for score, t, r in cands:
            if sum(1 for c in chosen if c[0] == label) >= n:
                break
            if per_chip.get(r["stem"], 0) >= max(2, (n + 1) // 2):
                continue
            if (r["stem"], r["flake_id"]) in used_parents:
                continue
            per_chip[r["stem"]] = per_chip.get(r["stem"], 0) + 1
            used_parents.add((r["stem"], r["flake_id"]))
            chosen.append((label, t, r))
    return chosen


_N = [0]


def _save(pdf, fig, png_dir):
    pdf.savefig(fig)
    _N[0] += 1
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(png_dir) / f"p{_N[0]:02d}.png", dpi=70)
    plt.close(fig)


def crop(img, cx, cy, half):
    h, w = img.shape[:2]
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
    return img[y0:y1, x0:x1].copy(), x0, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="example images")
    ap.add_argument("--images", default="example images")
    ap.add_argument("-o", "--output", default="AFM_targets")
    ap.add_argument("--png-dir", help="also save each page as PNG")
    a = ap.parse_args()
    pg, pf = load_profile(), load_profile("bx51_xc10_6.3x_flare")
    all_rows, meta = [], {}
    for mat, stems in CHIPS.items():
        for st in stems:
            d, rows = load(a.results, st, mat, pg, pf)
            meta[st] = d
            all_rows += rows
    chosen = select(all_rows)

    # sites: the chosen terrace plus its accepted sibling terraces (measured in the same scan)
    out, site = [], 0
    for label, t, r in chosen:
        site += 1
        d = meta[r["stem"]]
        sibs = [s for s in d["terraces"] if s["flake_id"] == r["flake_id"] and s is not r
                and not (set(s["quality"].split(";")) - OK) and s["area_um2"] >= 100]
        # up to two extra terraces of clearly different thickness (one AFM scan covers them)
        extra = []
        for s_ in sorted(sibs, key=lambda s: -s["inscribed_diameter_um"]):
            if s_["inscribed_diameter_um"] < 10:
                continue
            if np.hypot(s_["target_x_um"] - r["target_x_um"], s_["target_y_um"] - r["target_y_um"]) > 120:
                continue          # parent regions can link neighbouring flakes through residue
            if all(abs(s_["thickness_nm"] - e["thickness_nm"]) > 0.15 * max(e["thickness_nm"], 1.0) for e in [r] + extra):
                extra.append(s_)
            if len(extra) == 2:
                break
        for k, s in enumerate([r] + extra):
            out.append(dict(
                site=f"A{site:02d}", terrace=chr(ord("a") + k), purpose=label if k == 0 else "extra terrace, same scan",
                chip=s["stem"], material=s["material"], terrace_id=s["terrace_id"], flake_id=s["flake_id"],
                x_um=round(s["target_x_um"], 1), y_um=round(s["target_y_um"], 1),
                stage_x_um=None if s["stage_x_um"] is None else round(s["stage_x_um"], 1),
                stage_y_um=None if s["stage_y_um"] is None else round(s["stage_y_um"], 1),
                pred_nm=s["thickness_nm"], pred_range_nm=f"{s['thickness_min_nm']}-{s['thickness_max_nm']}",
                pred_alt_nm=s["alt_thickness_nm"], pred_stray_light_model_nm=s["t_flare"],
                layers=s["layers"] if s["material"] == "graphite" else "",
                inscribed_diameter_um=round(s["inscribed_diameter_um"], 1),
                length_um=round(s["length_um"], 1), width_um=round(s["width_um"], 1),
                contrast_rgb=f"{s['contrast_r']:.3f} {s['contrast_g']:.3f} {s['contrast_b']:.3f}",
                quality=s["quality"], afm_thickness_nm="", afm_notes=""))
    fields = list(out[0].keys())
    with open(f"{a.output}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"{site} sites, {len(out)} terraces -> {a.output}.csv")

    # ---------------- PDF
    INK, MUTED = "#1f2328", "#57606a"
    imgs = {st: cv2.cvtColor(cv2.imread(str(Path(a.images) / f"{st}.jpg")), cv2.COLOR_BGR2RGB) for st in meta}
    with PdfPages(f"{a.output}.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.07, 0.95, "AFM calibration targets", fontsize=16, weight="bold", color=INK, va="top")
        txt = (f"{site} sites / {len(out)} terraces on 4 chips (g34, g32: graphite on ~90 nm SiO2; fullchipbn23, bn25: hBN on ~285 nm SiO2).\n\n"
               "For each site: measure the step from bare SiO2 onto terrace a (and b, c... when listed: other terraces of the same flake, "
               "same scan). Enter the AFM step height in the afm_thickness_nm column of AFM_targets.csv.\n\n"
               "Coordinates: x_um / y_um are measured from the top-left corner of the scan image (x right, y down), at the point "
               "deepest inside the terrace (marked +). stage_x/y_um (fullchipbn23 only) are BX51 stage coordinates of the same "
               "point from the VSI metadata; please confirm the stage axis directions on the first site.\n\n"
               "pred_nm is the current model (tone-curve camera profile); pred_stray_light_model_nm is the alternative "
               "stray-light profile. The two agree for thin graphene but differ by up to 2x for 5-25 nm graphite and change which "
               "interference order cyan/green hBN belongs to; these sites were picked to decide between them.\n\n"
               "Graphite 1-5 layers: please also take a Raman spectrum (2D peak shape / I(2D)/I(G)) for the layer count; AFM "
               "step heights of few-layer graphene on SiO2 are often offset by 0.3-1 nm from adsorbates.")
        y = 0.9
        import textwrap
        for para in txt.split("\n\n"):
            lines = textwrap.wrap(para, 100)
            fig.text(0.07, y, "\n".join(lines), fontsize=9, color=INK, va="top", linespacing=1.4)
            y -= (len(lines) * 1.4 * 9 / 72 + 0.12) / 11
        # table
        y -= 0.01
        hdr = ["site", "chip", "purpose", "pred nm", "alt model", "Ø µm", "x, y (µm)"]
        xs = [0.07, 0.13, 0.25, 0.6, 0.68, 0.77, 0.83]
        for x, h in zip(xs, hdr):
            fig.text(x, y, h, fontsize=7.5, color=MUTED, weight="bold")
        y -= 0.016
        for o in out:
            if o["terrace"] != "a":
                continue
            vals = [o["site"], o["chip"], o["purpose"][:44], f"{o['pred_nm']:g}" + (f" ({o['pred_alt_nm']})" if o["pred_alt_nm"] else ""),
                    f"{o['pred_stray_light_model_nm']:g}", f"{o['inscribed_diameter_um']:.0f}", f"{o['x_um']:.0f}, {o['y_um']:.0f}"]
            for x, v in zip(xs, vals):
                fig.text(x, y, str(v), fontsize=7.3, color=INK)
            y -= 0.0145
        _save(pdf, fig, a.png_dir)

        # locator maps
        for st, img in imgs.items():
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.07, 0.96, f"Locator: {st}", fontsize=14, weight="bold", color=INK, va="top")
            ax = fig.add_axes([0.05, 0.08, 0.9, 0.85])
            small = cv2.resize(img, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
            ax.imshow(small)
            for o in out:
                if o["chip"] != st or o["terrace"] != "a":
                    continue
                px = o["x_um"] / meta[st]["pixel_um"] / 8
                py = o["y_um"] / meta[st]["pixel_um"] / 8
                ax.plot(px, py, "o", ms=14, mfc="none", mec="white", mew=1.8)
                ax.text(px + 10, py - 10, o["site"], color="white", fontsize=9, weight="bold",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5, lw=0))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title("image top-left = x, y origin; x to the right, y down", fontsize=8, color=MUTED, loc="left")
            _save(pdf, fig, a.png_dir)

        # site pages, 3 per page
        sites = {}
        for o in out:
            sites.setdefault(o["site"], []).append(o)
        keys = list(sites)
        for p0 in range(0, len(keys), 3):
            fig = plt.figure(figsize=(8.5, 11))
            for k, sk in enumerate(keys[p0:p0 + 3]):
                ts = sites[sk]
                o = ts[0]
                st = o["chip"]
                img = imgs[st]
                pu = meta[st]["pixel_um"]
                top = 0.97 - k * 0.32
                fig.text(0.05, top, f"{sk}  ·  {st}  ·  {o['purpose']}", fontsize=11, weight="bold", color=INK, va="top")
                cx, cy = o["x_um"] / pu, o["y_um"] / pu
                c1, x0, y0 = crop(img, cx, cy, 400)
                ax = fig.add_axes([0.05, top - 0.29, 0.27, 0.26])
                ax.imshow(c1)
                ax.plot(cx - x0, cy - y0, "o", ms=16, mfc="none", mec="white", mew=1.5)
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_title("context, 800 µm field", fontsize=7, color=MUTED, loc="left")
                allx = [t["x_um"] / pu for t in ts]
                ally = [t["y_um"] / pu for t in ts]
                half = int(np.clip(max(max(allx) - min(allx), max(ally) - min(ally), 2 * o["inscribed_diameter_um"] / pu) * 1.3 + 40, 60, 300))
                c2, x0, y0 = crop(img, np.mean(allx), np.mean(ally), half)
                ax = fig.add_axes([0.34, top - 0.29, 0.27, 0.26])
                ax.imshow(c2, interpolation="lanczos")
                ax.set_xlim(0, c2.shape[1]), ax.set_ylim(c2.shape[0], 0)
                import matplotlib.patheffects as pe
                halo = [pe.withStroke(linewidth=3.5, foreground="black")]
                for t in ts:
                    ax.plot(t["x_um"] / pu - x0, t["y_um"] / pu - y0, "+", ms=12, mew=1.6, color="white", path_effects=halo)
                    ax.text(t["x_um"] / pu - x0 + 5, t["y_um"] / pu - y0 - 5, t["terrace"], color="white", fontsize=10,
                            weight="bold", path_effects=halo)
                bar = 20 if half < 120 else 50
                ax.plot([c2.shape[1] * 0.95 - bar / pu, c2.shape[1] * 0.95], [c2.shape[0] * 0.93] * 2, color="white", lw=3)
                ax.text(c2.shape[1] * 0.95 - bar / pu / 2, c2.shape[0] * 0.9, f"{bar} µm", color="white", fontsize=7, ha="center")
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(f"close-up, {2 * half * pu:.0f} µm field", fontsize=7, color=MUTED, loc="left")
                lines = []
                for t in ts:
                    st_xy = f"stage {t['stage_x_um']:.0f}, {t['stage_y_um']:.0f}" if t["stage_x_um"] is not None else ""
                    alt = f" (alt {t['pred_alt_nm']})" if t["pred_alt_nm"] else ""
                    lay = f", {t['layers']} L" if t["material"] == "graphite" and t["pred_nm"] < 3 else ""
                    lines.append(f"{t['terrace']}:  {t['pred_nm']:g} nm{alt}{lay}   [alt. model {t['pred_stray_light_model_nm']:g}]\n"
                                 f"    x {t['x_um']:.0f}, y {t['y_um']:.0f} µm  {st_xy}\n"
                                 f"    Ø {t['inscribed_diameter_um']:.0f} µm, {t['length_um']:.0f}×{t['width_um']:.0f} µm\n"
                                 f"    AFM: ________ nm")
                fig.text(0.64, top - 0.03, "\n".join(lines), fontsize=7.6, color=INK, va="top", linespacing=1.35,
                         family="DejaVu Sans Mono")
            _save(pdf, fig, a.png_dir)
    print(f"wrote {a.output}.pdf")


if __name__ == "__main__":
    main()
