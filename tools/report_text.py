"""Text blocks and edge-case example selection for tools/contrast_report.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SUMMARY = """
This document goes with the flakeid scripts. It has two purposes: (1) show what flakes of a given thickness are expected to look like in these scans, and (2) collect the difficult cases met while developing the automatic identification, with example images, how each is handled now, and what remains open.

How the thickness is estimated. A transfer-matrix (Fresnel) model of air / flake / SiO2 / Si, averaged over the NA 0.30 illumination cone, gives the reflectance spectrum. The camera model converts it to RGB contrast C = I/I_substrate - 1. That model has the halogen spectrum, broad R/G/B responses, a colour-correction matrix and the JPEG tone curve.

Calibration. The camera parameters were fitted without known thicknesses. Flat regions of all flakes must fall on a single curve in RGB-contrast space. In addition, the graphite (90 nm) and hBN (285 nm) chips were imaged with the same 2 ms exposure, so their substrate colours must match the model's ratio. The model curve is clipped where the camera saturates. The fit returned 84-85 nm and 288-290 nm for the oxides. The model has not yet been compared with AFM or Raman: absolute thicknesses may be off by ~20-30 %, and layer numbers for graphene by +/-1 layer.

Correction to the first version. It stated that ~13 nm graphite is "almost invisible" on 90 nm SiO2. That was an artefact of the first camera calibration, which had placed unrealistically narrow (~45 nm) colour channels exactly where the contrast changes sign. The physical spectral contrast of 13 nm graphite ranges from about +35 % (blue) to -50 % (red). With realistic channel widths (~80-115 nm, now fitted) it is clearly visible as a blue flake (page 3).

Contents:
  p.2     what changed since the first version (flake counts per scan, and why flakes were missed)
  p.3-4   graphite on 90 nm SiO2: rendered appearance (1L, 2L, 3L, 2-24 nm) with real detected examples; contrast vs thickness; spectra
  p.5-6   hBN on 285 nm SiO2: 5-60 nm in 5 nm steps; contrast vs thickness; spectra
  p.7-    edge cases (residue, smudges and defocus, facets of bulk crystals, terraces, saturation, out-of-focus thick crystals, vignetting and stitching)
  last    recommended path to refine the identification of flakes worth high-resolution imaging

Main open question (hBN). With this calibration, cyan / green hBN (blue contrast ~0-0.1) fits the second interference order (~150-190 nm) better than the first (20-60 nm). The first order would need a blue contrast of ~0.2-0.3. The fit reports the better-fitting thickness and lists the in-window alternative; such flakes are kept and flagged ambiguous_thickness. AFM on three such flakes will settle which order they belong to.

Colour rendering. Swatch colours are the measured substrate colour multiplied by the predicted (1 + C). They are therefore in this camera's colour space, not in true sRGB. A screen or printer will show them slightly differently from the microscope eyepiece.
"""

COMPARISON_NOTE = """
Why flakes were missed before, and what changed:
  - Segmentation. The first version used a watershed that split each connected patch at gradient ridges and then merged neighbours with similar colour. Blurred edges and step pixels were absorbed into regions, which made real terraces look "non-uniform"; the non_uniform and blurry_edges cuts then removed many good purple (few-layer) graphite flakes. Regions are now grown only from pixels whose colour matches a flat core, so each terrace is measured on its own clean interior.
  - Edge test. The previous "sharpness" measure compared a thin inner band with the core. It failed for small or faint flakes and for flakes touching other material. It is replaced by an edge width measured on the boundary-averaged profile, using only the edges that face bare substrate.
  - Colour model. The first calibration placed narrow colour channels where the contrast changes sign, which wrongly made ~13 nm graphite "invisible" and distorted thicknesses between 5 and 20 nm. The camera model now has broad channels and a colour-correction matrix, and it is fitted jointly with the substrate-colour ratio between the 90 nm and 285 nm chips. The model curve is also clipped where the camera saturates. Without that, cyan / green hBN (30-60 nm, blue channel clipped at +0.28) had been fitted to the wrong interference branch.
  - Residue vs thin hBN. Thin hBN is now reported down to ~3.5 nm. Faint candidates are verified on the full-resolution pixels for flatness and simple geometry, and rejected when embedded in residue haze (section 1).
  - Bulk context. The "likely_bulk" rule, which rejected 15-25 nm graphite touching thick crystals, was removed; only faint regions enclosed by thick material are still rejected.
Thickness bins also shift because the thicknesses themselves changed with the new calibration.
"""

GR_SWATCH_NOTE = """Rendering uses a 90 nm oxide (as requested). The per-image fit gives 83-85 nm for these chips; the pipeline uses the refined value per image. "Closest detected flake" is the widest accepted region in the scans whose fitted thickness is within +/-1 nm (+/-0.15 nm for 1L-3L). It is therefore a check of self-consistency, not an independent measurement. Few-layer contrast is ~6-7 % per layer in G. Because of this per-layer step, 1L/2L/3L are distinguishable in principle, but layer counts should be confirmed by Raman until AFM/Raman-labelled flakes are available for calibration."""

GR_CURVE_NOTE = """Graphite is absorbing. Thin flakes (< ~8 nm) darken all channels, red most strongly, and look purple to navy. Between ~8 and ~20 nm, interference in the SiO2 makes the blue channel rise first: the flakes look blue, then light blue. Above ~30 nm the contrast approaches the bulk value and flakes look cream / white. The 13 nm curve in the lower panel shows the physical reason 13 nm graphite is easily visible: its contrast is strongly positive in the blue and strongly negative in the red."""

BN_SWATCH_NOTE = """hBN is transparent (n ~ 2.2). Its colour comes purely from thin-film interference together with the 285 nm oxide. In this camera's colour space it runs from lavender to blue (5-15 nm) and light blue (20-25 nm), then pale cyan to green-white (30-50 nm), where blue and green clip, then yellow (55-80 nm) and orange / pink (~100 nm). The colours then repeat in second order.
* marks an example whose best fit is in the second order (value after "or") and whose first-order alternative (before "or") falls in this bin; see the summary page. The per-image fit gives 285-290 nm oxide. Contrast below ~5 nm is under ~0.25, comparable to residue films, and those identifications are flagged low_confidence."""

BN_CURVE_NOTE = """Red clipping: in the hBN scans the substrate red channel is ~145/255. Measured red contrast therefore cannot exceed +0.76, and yellow / green / orange flakes (~40-110 nm) are clipped in R (flag "saturated"; the fit then treats R as a lower bound). Lowering the exposure so that the substrate sits near 90-110/255 would remove this."""

STRATEGY = """
1. Define "worth imaging" explicitly and rank rather than filter
The CSV already has the ingredients. A candidate for high-resolution imaging is one terrace (region) with:
  - thickness inside the window, with an acceptable uncertainty range (thickness_min/max_nm);
  - usable size: inscribed_diameter_um (largest circle that fits inside) above what a device needs, e.g. >= 10-15 um. This is better than area, because long slivers have a large area but no usable space;
  - flatness and simple geometry: fill >= 0.9, fullres_solidity, low edge_width_um;
  - no warning flags, and isolation from thick material (parent flake composition).
A single score, e.g. a weighted product of these terms, sorted per chip, turns the output into a work queue for the microscope. The target_x/y columns (deepest interior point) are the intended stage targets.

2. Get a small amount of ground truth: the most valuable step
  a) Thickness: measure ~20-30 flakes by AFM, spanning 1-25 nm graphite and 3-60 nm hBN, plus Raman (2D peak) for 1-3L graphene. Include at least three cyan / green hBN flakes flagged ambiguous_thickness: they decide whether these colours are first-order (20-60 nm) or second-order (150-190 nm) hBN, the biggest open question for hBN. Adding these points to the calibration (supervised term in flakeid.calibrate) should bring thickness errors down to ~1 layer for 1-3L graphene and ~1-2 nm above that. It also removes the remaining degeneracies of the unsupervised fit, such as the exact hue of monolayers and the channel responses.
  b) Quality labels: go through ~300-500 candidates from the example chips (crops from --crops, with the CSV) and mark each as good flake / residue / smudge / multi-terrace / thick or bulk / other. A simple labelling page can be generated for this. This takes about an hour, and it is what turns the thresholds in this document from educated guesses into measured operating points.

3. Use the labels in two stages
  - Short term: tune each threshold (min contrast, fill, full-resolution solidity, edge width, haze-ring similarity, B/G) from ROC curves on the labelled set, per material and oxide. Keep the rules: they are interpretable and robust to small datasets.
  - Once ~1000+ labels exist: train a light classifier (gradient-boosted trees) on the CSV features to output P(worth imaging). A small CNN on the crops can be added later if residue or defects still confuse the features. The physics model stays in charge of thickness; the classifier only decides quality.

4. Close the loop with the microscope (active learning)
When candidates are imaged at 50-100x, record the verdict and feed it back as a label. Image all top-ranked candidates, plus a few uncertain ones near the decision boundary, so that every imaging session also improves the classifier. High-magnification images also give accurate lateral dimensions and reveal cracks, bubbles and folds that are invisible at 6.3x.

5. Acquisition changes with the largest effect on identification quality
  - Shading/flat-field correction in cellSens, or a blank-substrate reference image: removes the tile vignetting and seams that the software now has to model.
  - Fixed exposure, white balance and lamp voltage for all chips, with the substrate near 90-110/255, so that nothing clips. The camera calibration then transfers between chips and sessions.
  - Autofocus per tile or a focus map: defocused regions (e.g. parts of g31, g33) are currently rejected as "blurry".
  - Save lossless TIFF (or keep the .vsi with its _name_/ folder). JPEG blocking and chroma subsampling degrade exactly the faint, small flakes that matter most.
  - Optional: a 550 nm band-pass filter for a second graphene-optimised scan, or a 10x scan for thin hBN (< 5 nm), whose contrast is marginal at 6.3x.
"""


# ------------------------------------------------------------------ edge cases
def _rows(d, stem):
    p = Path(d) / f"{stem}_terraces.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def _c(r):
    return np.array([float(r[f"contrast_{k}"]) for k in "rgb"])


def _pick(rows, pred, n, key=lambda r: -float(r["area_um2"]), min_sep=400):
    out = []
    for r in sorted((r for r in rows if pred(r)), key=key):
        x, y = float(r["x_px"]), float(r["y_px"])
        if all(abs(x - float(o["x_px"])) + abs(y - float(o["y_px"])) > min_sep for o in out):
            out.append(r)
        if len(out) == n:
            break
    return out


def _flags(r):
    return set(r["quality"].split(";"))


def _accepted(r):
    return not (_flags(r) - {"good", "chip_edge", "saturated", "ambiguous_thickness", "low_confidence"})


def _half(r, k=1.2, lo=30, hi=260):
    return int(np.clip(max(int(r["bbox_w_px"]), int(r["bbox_h_px"])) * k, lo, hi))


def _panel(stem, r, cap):
    return (stem, float(r["x_px"]), float(r["y_px"]), _half(r), cap)


def edge_cases(images, all_dir):
    cases = []
    bn = {s: _rows(all_dir, s) for s in ["bn25", "bn26", "bn24"]}
    gr = {s: _rows(all_dir, s) for s in ["g31", "g32", "g33", "g34", "g35"]}

    # 1 residue / fragment haze vs thin hBN
    panels = []
    for st in ["bn25", "bn26"]:
        for r in _pick(bn[st], lambda r: 2 <= float(r["thickness_nm"]) < 10 and float(r["area_um2"]) > 150 and
                       _flags(r) & {"in_haze", "not_flat", "irregular_shape"}
                       and not _flags(r) & {"out_of_range", "poor_colour_match"}, 2):
            why = [q for q in ("in_haze", "not_flat", "irregular_shape") if q in _flags(r)]
            panels.append(_panel(st, r, f"REJECTED {st} ({', '.join(why)})\ncolour of {float(r['thickness_nm']):g} nm; "
                                 f"fill {float(r['fill']):.2f}\nsolidity {float(r['fullres_solidity']):.2f}; "
                                 f"haze-ring {float(r['ring_similar']):.2f}"))
    pool = [(st, r) for st in ("bn25", "bn26", "bn24") for r in bn[st]]
    for lo, hi in ((4, 7), (7, 13)):
        sel = sorted((x for x in pool if _accepted(x[1]) and lo <= float(x[1]["thickness_nm"]) < hi),
                     key=lambda x: -float(x[1]["inscribed_diameter_um"]))
        for st, r in sel[:1] if lo == 4 else sel[:2]:
            panels.append(_panel(st, r, f"ACCEPTED {st}: hBN {float(r['thickness_nm']):g} nm\nfill {float(r['fill']):.2f}; "
                                 f"solidity {float(r['fullres_solidity']):.2f}\nhaze-ring {float(r['ring_similar']):.2f}"))
    cases.append(dict(
        title="1. Tape residue and fragment haze vs thin hBN",
        what="On the hBN chips large areas are covered by fern-like / dendritic clusters of sub-micron hBN fragments "
             "and residue. Averaged over a few um, they have the same colour as a continuous 2-8 nm hBN film, and "
             "segmentation at 2 um/px produces 'uniform' patches from them.",
        rule="Faint regions (|C| < 0.4) are re-examined on the full-resolution pixels. At least 90 % of the interior "
             "must match the region's colour ('fill'), and the matching pixel set must have a simple, compact "
             "outline (full-res solidity >= 0.75). Regions with |C| < 0.6 whose surroundings (2-6 px outside) have "
             "the same hue and a similar strength are rejected as 'in_haze'. A real flake stands out from bare "
             "substrate, while a patch cut out of a haze field is surrounded by more haze.",
        status="Works well for >= 5-6 nm hBN. Between ~3.5 and 5 nm some residue patches still pass; they are "
               "flagged low_confidence. hBN < 3.5 nm (|C| < 0.15) is not reported by default. Labelled examples "
               "would allow tuning fill / solidity / haze-ring on data instead of by eye.",
        panels=panels[:6]))

    # 2 smudges / defocus
    panels = []
    for st in ["g31", "g34", "g32"]:
        for r in _pick(gr[st], lambda r: "blurry_edges" in _flags(r) and float(r["area_um2"]) > 200 and
                       not _flags(r) & {"out_of_range", "poor_colour_match"}, 1):
            panels.append(_panel(st, r, f"REJECTED {st}: edge width {float(r['edge_width_um']):.1f} um\n"
                                 f"colour fit {float(r['thickness_nm']):g} nm"))
    for st in ["g31", "g34", "g32"]:
        for r in _pick(gr[st], lambda r: _accepted(r) and float(r["thickness_nm"]) < 1.2, 1,
                       key=lambda r: -float(r["inscribed_diameter_um"])):
            panels.append(_panel(st, r, f"ACCEPTED {st}: {float(r['thickness_nm']):g} nm\n"
                                 f"edge width {float(r['edge_width_um']):.1f} um"))
    cases.append(dict(
        title="2. Diffuse smudges and defocused areas",
        what="Diffuse dark smudges (residue, contamination) and flakes in out-of-focus parts of the scan can have "
             "exactly the colour of 1-3L graphene or of a thin flake. Their edges are gradual rather than a "
             "step.",
        rule="The contrast profile across the whole region boundary is averaged as a function of distance from "
             "the edge; only boundary parts facing bare substrate are used, so neighbouring terraces do not count. "
             "The 20-80 % rise distance is the edge width. Real flake edges measure ~1.5-4 um at this "
             "magnification; regions wider than 5 um are rejected ('blurry_edges').",
        status="Works on the examples. Real flakes in defocused parts of a scan are also rejected; focus mapping "
               "during acquisition fixes this at the source.",
        panels=panels[:6]))

    # 3 grey facets vs few-layer graphene
    panels = []
    for st in ["g33", "g34", "g32"]:
        for r in _pick(gr[st], lambda r: _flags(r) & {"colour_mismatch", "enclosed_in_thick"} and
                       float(r["area_um2"]) > 200 and np.linalg.norm(_c(r)) < 0.35 and
                       "poor_colour_match" not in _flags(r), 1):
            c = _c(r)
            panels.append(_panel(st, r, f"REJECTED {st}: grey, B/G = {c[2] / c[1]:.2f}\n"
                                 f"C = ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})"))
    for st in ["g34", "g33", "g31"]:
        for r in _pick(gr[st], lambda r: _accepted(r) and float(r["thickness_nm"]) < 2.5, 1,
                       key=lambda r: -float(r["inscribed_diameter_um"])):
            c = _c(r)
            panels.append(_panel(st, r, f"ACCEPTED {st}: {float(r['thickness_nm']):g} nm, B/G = {c[2] / c[1]:.2f}\n"
                                 f"C = ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})"))
    cases.append(dict(
        title="3. Grey facets of thick crystals vs few-layer graphene",
        what="Thick graphite crystals (g32-g34) contain tilted or shadowed facets with small, neutral-grey "
             "contrast (all channels about -0.05 to -0.1). By magnitude alone this resembles 1-2L graphene.",
        rule="Few-layer graphene on ~90 nm SiO2 is green-dominated (observed B/G = 0.35-0.65, R/G ~ 0.7-0.9). "
             "Thin (< 5 nm) graphite candidates with B/G > 0.8 are rejected ('colour_mismatch'). Faint regions "
             "inside flakes that are mostly thick, and bordered mostly by strongly contrasting material, are "
             "rejected as 'enclosed_in_thick'.",
        status="The B/G rule is empirical, drawn from these scans. The calibrated model predicts a slightly too "
               "neutral monolayer colour; Raman-confirmed monolayers would fix both the model and the rule.",
        panels=panels[:6]))

    # 4 terraces
    panels = []
    for st in ["g32", "g34", "g31", "g35"]:
        pars, par_all = {}, {}
        for r in gr[st]:
            par_all.setdefault(r["flake_id"], []).append(r)
            if _accepted(r):
                pars.setdefault(r["flake_id"], []).append(r)
        # skip parents dominated by thick / unidentified material (bulk crystals with facets)
        def acc_frac(pid):
            tot = sum(float(r["area_um2"]) for r in par_all[pid])
            return sum(float(r["area_um2"]) for r in pars[pid]) / max(tot, 1.0)
        multi = sorted((v for pid, v in pars.items() if acc_frac(pid) > 0.6),
                       key=lambda v: -len({round(float(r["thickness_nm"])) for r in v}))
        for v in multi[:2 if st in ("g32", "g34") else 1]:
            xs0 = min(int(r["bbox_x_px"]) for r in v); xs1 = max(int(r["bbox_x_px"]) + int(r["bbox_w_px"]) for r in v)
            ys0 = min(int(r["bbox_y_px"]) for r in v); ys1 = max(int(r["bbox_y_px"]) + int(r["bbox_h_px"]) for r in v)
            half = int(np.clip(max(xs1 - xs0, ys1 - ys0) * 0.7, 40, 260))
            th = sorted({float(r["thickness_nm"]) for r in v})
            panels.append((st, (xs0 + xs1) / 2, (ys0 + ys1) / 2, half,
                           f"{st}: {len(v)} terraces of one flake\n" + ", ".join(f"{t:g}" for t in th[:7]) + " nm"))
    cases.append(dict(
        title="4. Flakes with many terraces / internal step edges",
        what="Many flakes consist of several terraces of different thickness. The first version merged some of "
             "them into one region, so a flake with many internal steps could appear as a single 'uniform' "
             "flake (e.g. g32 flake 1).",
        rule="Regions are now grown from flat cores and include only pixels whose contrast matches the core "
             "within max(0.045, 8 %). Each terrace is a separate row in the CSV, and terraces of one flake share "
             "a parent_id. Step edges and blurred pixels remain unassigned instead of being absorbed.",
        status="Solved for steps larger than the tolerance (~1 layer for thin graphene, ~2-3 nm for hBN). A "
               "per-flake 'number of terraces' and 'largest single-thickness area' are easy to add to the ranking.",
        panels=panels[:6], ncols=3))

    # 5 saturation
    panels = []
    for r in _pick(bn["bn24"], lambda r: "saturated" in _flags(r) and _accepted(r), 3):
        panels.append(_panel("bn24", r, f"{float(r['thickness_nm']):g} nm, clipped fraction "
                             f"{float(r['saturated_frac']):.2f}\nC = ({float(r['contrast_r']):+.2f}, "
                             f"{float(r['contrast_g']):+.2f}, {float(r['contrast_b']):+.2f})"))

    def extra(fig):
        import cv2
        im = cv2.imread(str(Path(images) / "bn24.jpg"), cv2.IMREAD_REDUCED_COLOR_2)
        ax = fig.add_axes([0.1, 0.07, 0.8, 0.2])
        for k, (nm, col) in enumerate([("B", "#0969da"), ("G", "#1a7f37"), ("R", "#d1242f")]):
            h = np.bincount(im[..., k].ravel(), minlength=256)
            ax.semilogy(np.arange(256), np.maximum(h, 1), color=col, lw=1.2, label=nm)
        ax.axvline(145, color="#d1242f", ls=":", lw=0.8)
        ax.text(147, ax.get_ylim()[1] * 0.3, "substrate R = 145", color="#d1242f", fontsize=7)
        ax.set_xlabel("8-bit value (bn24)"), ax.set_ylabel("pixels")
        ax.legend(frameon=False)
        ax.set_title("Channel histograms: pile-up at 255 = clipped flakes", fontsize=8.5, loc="left")

    cases.append(dict(
        title="5. Saturated (clipped) colour channels",
        what="The hBN scans are exposed so that the substrate red channel sits at ~145/255, so any flake brighter "
             "than 1.76x in red is clipped. This covers most green / yellow / orange hBN (~40-110 nm).",
        rule="Clipped channels are treated as lower bounds in the colour fit (a censored fit), and such flakes "
             "are flagged 'saturated' but still reported.",
        status="Works, but the thickness of clipped flakes relies on two channels. Fix at the source: lower the "
               "exposure (substrate at ~90-110/255).",
        panels=panels, extra=extra))

    # 6 out-of-focus thick chunks
    panels = []
    for st in ["g32", "g31", "g35"]:
        for r in _pick(gr[st], lambda r: "poor_colour_match" in _flags(r) and _c(r).mean() < -0.5 and
                       float(r["area_um2"]) > 1500, 1):
            c = _c(r)
            panels.append(_panel(st, r, f"REJECTED {st}: C = ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})\n"
                                 f"fit residual {float(r['fit_residual']):.2f}"))
    cases.append(dict(
        title="6. Very thick / tilted crystals and out-of-focus chunks",
        what="Thick crystals that do not lie flat, or that stick out of the focal plane, appear black or dark "
             "grey with blurred halos. Their colour is neutral dark, unlike any flat graphite film on SiO2.",
        rule="The colour-fit residual is large (tolerance max(0.2, 0.25*|C|)), giving 'poor_colour_match'. "
             "Faint regions next to them (halos) are caught by the B/G, edge-width and enclosure rules.",
        status="Robust. A few small (< ~150 um2) halo patches next to such chunks still pass.",
        panels=panels))
    return cases
