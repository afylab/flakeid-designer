#!/usr/bin/env python3
"""Build the PDF report: expected flake contrast vs thickness and the
challenging edge cases found in the example scans.

    python tools/contrast_report.py --images "example images" --results DIR -o report.pdf

DIR must contain *_terraces.csv produced with the current profile (the report
picks real example flakes from them).  Regenerate after re-calibrating.
"""
from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flakeid.background import compute_contrast  # noqa: E402
from flakeid.optics import _reflectance, n_si, n_sio2  # noqa: E402
from flakeid.pipeline import find_flakes  # noqa: E402
from flakeid.thickness import ThicknessEstimator, load_profile  # noqa: E402

PAGE = (8.5, 11.0)
INK, MUTED, RULE = "#1f2328", "#57606a", "#d0d7de"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
CH = [("R", "#d1242f"), ("G", "#1a7f37"), ("B", "#0969da")]


PNG_DIR = None
_PAGE_NO = [0]


def save(pdf, fig):
    pdf.savefig(fig)
    _PAGE_NO[0] += 1
    if PNG_DIR:
        fig.savefig(Path(PNG_DIR) / f"page{_PAGE_NO[0]:02d}.png", dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------- helpers
def page(pdf, title, subtitle=None):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.955, title, fontsize=15, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.07, 0.925, subtitle, fontsize=9, color=MUTED, va="top")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.912, 0.912], color=RULE, lw=0.8))
    return fig


def para(fig, x, y, text, width=100, size=8.8, color=INK, spacing=1.35):
    lines = []
    for block in text.strip("\n").split("\n"):
        lines += textwrap.wrap(block, width) if block.strip() else [""]
    fig.text(x, y, "\n".join(lines), fontsize=size, color=color, va="top", linespacing=spacing)
    return y - len(lines) * size * spacing / 72.0 / PAGE[1]


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def crop(img, x, y, half):
    h, w = img.shape[:2]
    x0, y0 = int(max(x - half, 0)), int(max(y - half, 0))
    return img[y0:min(y0 + 2 * half, h), x0:min(x0 + 2 * half, w)], x0, y0


def show(ax, im, title=None, scalebar_um=None, px_um=1.005):
    ax.imshow(im, interpolation="lanczos")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=3)
    if scalebar_um:
        L = scalebar_um / px_um
        h, w = im.shape[:2]
        ax.add_patch(Rectangle((w - L - 0.05 * w, h - 0.08 * h), L, 0.018 * h, color="white"))
        ax.text(w - L / 2 - 0.05 * w, h - 0.1 * h, f"{scalebar_um:g} µm", color="white",
                fontsize=6.5, ha="center", va="bottom")


def render(sub_rgb, c):
    """Colour of a flake with contrast c on a substrate of (encoded) colour sub_rgb."""
    return np.clip(np.asarray(sub_rgb) * (1 + np.asarray(c)), 0, 255) / 255.0


def load_rows(results, stems):
    rows = []
    for st in stems:
        p = Path(results) / f"{st}_terraces.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                r["_stem"] = st
                rows.append(r)
    return rows


def in_window_thickness(r, lo, hi):
    """Fitted thickness, or for ambiguous regions the alternative inside [lo, hi]."""
    ts = [float(r["thickness_nm"])]
    if "ambiguous_thickness" in r["quality"] and r["alt_thickness_nm"]:
        ts += [float(a) for a in r["alt_thickness_nm"].split(";")]
    inside = [t for t in ts if lo <= t <= hi]
    return inside[0] if inside else None


def best_example(rows, lo, hi):
    ok = {"good", "saturated", "ambiguous_thickness", "low_confidence"}
    cand = [r for r in rows if in_window_thickness(r, lo, hi) is not None
            and not (set(r["quality"].split(";")) - ok)]
    # prefer unambiguous fits, then the widest flake
    cand.sort(key=lambda r: ("ambiguous_thickness" in r["quality"], -float(r["inscribed_diameter_um"])))
    return cand[0] if cand else None


def region_outline(ax, r, x0, y0):
    ax.add_patch(Rectangle((int(r["bbox_x_px"]) - x0, int(r["bbox_y_px"]) - y0), int(r["bbox_w_px"]),
                           int(r["bbox_h_px"]), fill=False, ec="white", lw=0.7, ls=(0, (2, 2))))


# --------------------------------------------------------------------------- pages
def swatch_page(pdf, title, subtitle, est, sub_rgb, targets, rows, images, note):
    fig = page(pdf, title, subtitle)
    n = len(targets)
    cols = 5
    rws = int(np.ceil(n / cols))
    top, bottom = 0.86, 0.17
    ch = (top - bottom) / rws
    cw = 0.86 / cols
    cache = {}
    for i, (label, t, lo, hi) in enumerate(targets):
        r_, c_ = divmod(i, cols)
        x = 0.07 + c_ * cw
        y = top - (r_ + 1) * ch
        c = est.model.contrast([t])[0]
        # model swatch
        ax = fig.add_axes([x + 0.005, y + ch * 0.52, cw - 0.015, ch * 0.36])
        sw = np.ones((40, 80, 3)) * np.asarray(sub_rgb) / 255.0
        sw[8:32, 24:56] = render(sub_rgb, c)
        ax.imshow(sw, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(RULE)
        ax.set_title(f"{label}", fontsize=8.5, weight="bold", color=INK, loc="left", pad=2)
        fig.text(x + 0.005, y + ch * 0.50, f"C = ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})",
                 fontsize=6.3, color=MUTED, va="top")
        # real example
        ex = best_example(rows, lo, hi)
        ax2 = fig.add_axes([x + 0.005, y + 0.004, cw - 0.015, ch * 0.40])
        if ex is not None:
            st = ex["_stem"]
            if st not in cache:
                cache[st] = rgb(Path(images) / f"{st}.jpg")
            half = int(max(int(ex["bbox_w_px"]), int(ex["bbox_h_px"])) * 0.9 + 25)
            im, x0, y0 = crop(cache[st], float(ex["x_px"]), float(ex["y_px"]), half)
            show(ax2, im)
            region_outline(ax2, ex, x0, y0)
            t_in = in_window_thickness(ex, lo, hi)
            lab = f"{st} #{ex['terrace_id']}: fit {t_in:g} nm"
            if t_in != float(ex["thickness_nm"]):
                lab = f"{st} #{ex['terrace_id']}: {t_in:g} or {float(ex['thickness_nm']):g} nm*"
            ax2.set_title(lab, fontsize=6.3, color=MUTED, loc="left", pad=1.5)
        else:
            ax2.axis("off")
            ax2.text(0.5, 0.5, "no example found", ha="center", va="center", fontsize=7, color=MUTED,
                     transform=ax2.transAxes)
    para(fig, 0.07, 0.15, note, width=110, size=8)
    save(pdf, fig)


def curves_page(pdf, title, subtitle, est, dmax, marks, spec_d, oxide, n_fl, note):
    fig = page(pdf, title, subtitle)
    d = np.linspace(0, dmax, 600)
    C = est.model.contrast(d)
    ax = fig.add_axes([0.1, 0.58, 0.83, 0.3])
    for k, (nm, col) in enumerate(CH):
        ax.plot(d, C[:, k], color=col, lw=1.6, label=nm)
    ax.axhline(0, color=RULE, lw=0.8)
    for m in marks:
        ax.axvline(m, color=RULE, lw=0.6, ls=":")
    ax.set_xlabel("thickness (nm)")
    ax.set_ylabel("RGB contrast  I/I_sub − 1  (camera-encoded)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_xlim(0, dmax)
    ax.set_title("Camera-channel contrast predicted by the calibrated model", fontsize=9, loc="left",
                 color=INK)
    # spectral contrast
    wl = np.arange(400, 721, 2.0)
    cos = np.array([[1.0]])
    Rs = _reflectance([n_sio2(wl)], [oxide], n_si(wl), wl, cos)[0]
    ax2 = fig.add_axes([0.1, 0.2, 0.83, 0.3])
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(spec_d):
        Rf = _reflectance([n_fl(wl), n_sio2(wl)], [t, oxide], n_si(wl), wl, cos)[0]
        ax2.plot(wl, Rf / Rs - 1, color=cmap(i / max(len(spec_d) - 1, 1)), lw=1.4, label=f"{t:g} nm")
    ax2.axhline(0, color=RULE, lw=0.8)
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("reflectance contrast  R/R_sub − 1  (physical)")
    ax2.legend(frameon=False, ncol=5, fontsize=7, loc="upper left")
    ax2.set_title(f"Physical spectral contrast on {oxide:g} nm SiO2 / Si (normal incidence)",
                  fontsize=9, loc="left", color=INK)
    para(fig, 0.07, 0.13, note, width=110, size=8)
    save(pdf, fig)


def case_page(pdf, title, what, rule, status, panels, images, ncols=3, extra=None):
    """panels: list of (stem, x, y, half_px, caption)."""
    fig = page(pdf, title)
    y = para(fig, 0.07, 0.895, "What it is.  " + what, width=108)
    y = para(fig, 0.07, y - 0.006, "How the pipeline handles it.  " + rule, width=108)
    y = para(fig, 0.07, y - 0.006, "Status.  " + status, width=108, color=MUTED)
    n = len(panels)
    rws = int(np.ceil(n / ncols))
    top = y - 0.02
    bottom = 0.06 if extra is None else 0.33
    h = min((top - bottom) / rws, 0.26)
    w = 0.86 / ncols
    cache = {}
    for i, (st, x, yy, half, cap) in enumerate(panels):
        r_, c_ = divmod(i, ncols)
        ax = fig.add_axes([0.07 + c_ * w + 0.005, top - (r_ + 1) * h + 0.03, w - 0.012, h - 0.045])
        if st not in cache:
            cache[st] = rgb(Path(images) / f"{st}.jpg")
        im, _, _ = crop(cache[st], x, yy, half)
        show(ax, im, scalebar_um=20 if half < 90 else 50 if half < 220 else 200)
        fig.text(0.07 + c_ * w + 0.005, top - (r_ + 1) * h + 0.026, cap, fontsize=6.6, color=INK,
                 va="top", wrap=True)
    if extra is not None:
        extra(fig)
    save(pdf, fig)


def comparison_page(pdf, prev_dir, new_dir, images):
    fig = page(pdf, "What changed since the first version",
               "Accepted flake regions per scan with default settings (previous vs current)")
    stems = ["g31", "g32", "g33", "g34", "g35", "bn24", "bn25", "bn26", "fullchipbn23"]
    bins = {"graphite": [0, 1.2, 5, 10, 15, 20, 25.1], "hbn": [0, 5, 10, 20, 40, 60, 80, 100.1]}
    rows = []
    for st in stems:
        mat = "graphite" if st.startswith("g") else "hbn"
        out = []
        for d in (prev_dir, new_dir):
            p = Path(d) / f"{st}_terraces.csv"
            t = [float(r["thickness_nm"]) for r in csv.DictReader(open(p))] if p.exists() else []
            out.append((len(t), np.histogram(t, bins=bins[mat])[0]))
        rows.append((st, mat, out))
    y = 0.87
    for mat, title in (("graphite", "Graphite (bins in nm)"), ("hbn", "hBN (bins in nm)")):
        b = bins[mat]
        hdr = ["scan", "total"] + [f"{b[i]:g}-{b[i + 1]:.0f}" for i in range(len(b) - 1)]
        fig.text(0.07, y, title, fontsize=9.5, weight="bold", color=INK)
        y -= 0.025
        xs = [0.07, 0.2] + [0.3 + 0.085 * i for i in range(len(hdr) - 2)]
        for x, h in zip(xs, hdr):
            fig.text(x, y, h, fontsize=7.5, color=MUTED)
        y -= 0.02
        for st, m, ((n0, h0), (n1, h1)) in rows:
            if m != mat:
                continue
            vals = [st, f"{n0} \u2192 {n1}"] + [f"{a} \u2192 {b_}" for a, b_ in zip(h0, h1)]
            for x, v in zip(xs, vals):
                fig.text(x, y, v, fontsize=7.8, color=INK)
            y -= 0.019
        y -= 0.02
    para(fig, 0.07, y - 0.01, COMPARISON_NOTE, width=108, size=8.6)
    save(pdf, fig)


def vignetting_page(pdf, images):
    fig = page(pdf, "7. Tile vignetting and stitching seams")
    y = para(fig, 0.07, 0.895, "What it is.  The stitched scans are not flat-field corrected: every tile is "
             "brighter in the centre (about 7-8 % peak-to-peak across a tile, see the profile below), and the seams "
             "are darker lines ~100 um wide. Both are comparable to the contrast of monolayer graphene (~6 %).",
             width=108)
    y = para(fig, 0.07, y - 0.006, "How the pipeline handles it.  The chip is detected, and the bare substrate "
             "is fitted with a smooth surface while flakes are iteratively excluded. Line-shaped kernels then "
             "remove the horizontal and vertical seams. Afterwards the band-averaged substrate is flat to ~0.3 %; "
             "the remaining per-pixel scatter (~1.4 %) is sensor and JPEG noise.", width=108)
    para(fig, 0.07, y - 0.006, "Status.  Solved in software. Enabling shading correction during acquisition "
         "would make the correction more robust near large flakes and chip edges.", width=108, color=MUTED)
    im = rgb(Path(images) / "g31.jpg")
    cm = compute_contrast(im, 1.005)
    small = cv2.resize(im, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA)
    ax = fig.add_axes([0.07, 0.36, 0.42, 0.42])
    show(ax, small, "g31 as exported (1/8 scale)")
    g = cv2.resize(cm.contrast[..., 1], None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)
    ax2 = fig.add_axes([0.51, 0.36, 0.42, 0.42])
    ax2.imshow(g, cmap="gray", vmin=-0.08, vmax=0.08)
    ax2.set_xticks([]), ax2.set_yticks([])
    ax2.set_title("green-channel contrast after flattening (±8 %)", fontsize=7.5, loc="left")
    # column profiles: median over a 1 mm horizontal band, substrate pixels only
    g_raw = im[..., 1].astype(np.float32)
    H = g_raw.shape[0]
    band = slice(H // 2 - 500, H // 2 + 500)
    work = cv2.resize(g_raw, (cm.contrast.shape[1], cm.contrast.shape[0]), interpolation=cv2.INTER_AREA)
    sub = cm.substrate_mask
    bw = slice(band.start // 2, band.stop // 2)
    rawb = np.where(sub[bw], work[bw], np.nan)
    flatb = np.where(sub[bw], cm.contrast[bw][..., 1], np.nan)
    ok_cols = np.sum(sub[bw], axis=0) > 50
    raw_prof = np.nanmedian(rawb[:, ok_cols], axis=0)
    raw_prof = raw_prof / np.nanmedian(raw_prof) - 1
    flat_prof = np.nanmedian(flatb[:, ok_cols], axis=0)
    xs = np.nonzero(ok_cols)[0] * 2 * 1.005 / 1000
    ax3 = fig.add_axes([0.1, 0.08, 0.83, 0.2])
    ax3.plot(xs, raw_prof * 100, color=MUTED, lw=0.9, label="as exported (green, substrate only)")
    ax3.plot(xs, flat_prof * 100, color="#0969da", lw=0.9, label="after flattening")
    ax3.set_ylim(-6, 6), ax3.set_xlabel("position along a 1 mm-wide band through the chip centre (mm)")
    ax3.set_ylabel("deviation from mean (%)")
    ax3.legend(frameon=False, fontsize=7)
    save(pdf, fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="example images")
    ap.add_argument("--results", required=True, help="output of find_flakes (accepted flakes)")
    ap.add_argument("--all-results", required=True, help="output of find_flakes --all (with rejected regions)")
    ap.add_argument("-o", "--output", default="FlakeIdentifier_report.pdf")
    ap.add_argument("--png-dir", help="also write every page as PNG here")
    ap.add_argument("--previous", help="directory with *_terraces.csv of an earlier version, for a comparison page")
    a = ap.parse_args()
    global PNG_DIR
    PNG_DIR = a.png_dir
    prof = load_profile()
    gr_rows = load_rows(a.results, ["g31", "g32", "g33", "g34", "g35"])
    bn_rows = load_rows(a.results, ["bn24", "bn25", "bn26", "fullchipbn23"])
    est_g = ThicknessEstimator("graphite", 90.0, prof)
    est_b = ThicknessEstimator("hbn", 285.0, prof)
    sub_g, sub_b = (126, 105, 128), (144, 120, 199)

    with PdfPages(a.output) as pdf:
        # ---------------- title / summary
        fig = page(pdf, "FlakeIdentifier — contrast reference and edge cases",
                   "Optical identification of exfoliated graphite and hBN on SiO2/Si  ·  BX51, XC10, 6.3x  ·  "
                   "example scans g31–g35, bn23–bn26")
        para(fig, 0.07, 0.89, SUMMARY, width=106, size=9)
        save(pdf, fig)

        if a.previous:
            comparison_page(pdf, a.previous, a.results, a.images)

        # ---------------- graphite reference
        L = est_g.layer_nm
        g_targets = [("1L", L, 0.2, 0.52), ("2L", 2 * L, 0.52, 0.85), ("3L", 3 * L, 0.85, 1.2)] + \
                    [(f"{t} nm", t, t - 1.0, t + 1.0) for t in range(2, 25, 2)]
        swatch_page(pdf, "Graphite on 90 nm SiO2: expected appearance",
                    "Top: model colour on the substrate (camera colour space).  Bottom: closest detected flake.",
                    est_g, sub_g, g_targets, gr_rows, a.images, GR_SWATCH_NOTE)
        curves_page(pdf, "Graphite on 90 nm SiO2: contrast vs thickness", None, est_g, 40,
                    [L, 2 * L, 3 * L, 5, 10, 15, 20, 25], [0.335, 1, 3, 6, 10, 13, 16, 20, 25, 40],
                    90.0, est_g.model_index, GR_CURVE_NOTE)
        # ---------------- hBN reference
        b_targets = [(f"{t} nm", t, t - 2.5, t + 2.5) for t in range(5, 61, 5)]
        swatch_page(pdf, "hBN on 285 nm SiO2: expected appearance",
                    "Top: model colour on the substrate in this camera's colour space.  Bottom: closest detected "
                    "flake.", est_b, sub_b, b_targets, bn_rows, a.images, BN_SWATCH_NOTE)
        curves_page(pdf, "hBN on 285 nm SiO2: contrast vs thickness", None, est_b, 120,
                    list(range(10, 101, 10)), [2, 5, 10, 20, 30, 40, 50, 60, 80, 100], 285.0,
                    est_b.model_index, BN_CURVE_NOTE)

        # ---------------- edge cases
        for case in edge_cases(a.images, a.all_results):
            case_page(pdf, images=a.images, **case)
        vignetting_page(pdf, a.images)

        # ---------------- refinement strategy
        fig = page(pdf, "Refining the identification: recommended path")
        para(fig, 0.07, 0.895, STRATEGY, width=106, size=8.8)
        save(pdf, fig)
    print(f"wrote {a.output}")


# texts and edge-case definitions live below so the layout code above stays readable
from report_text import (BN_CURVE_NOTE, BN_SWATCH_NOTE, COMPARISON_NOTE, GR_CURVE_NOTE,  # noqa: E402
                         GR_SWATCH_NOTE, STRATEGY, SUMMARY, edge_cases)

if __name__ == "__main__":
    main()
