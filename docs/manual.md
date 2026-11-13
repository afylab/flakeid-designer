# FlakeIdentifier analysis reference

This document describes the analysis: how flakes are found in an optical scan, how their thickness is estimated,
how they are scored, what the outputs are, how the models are calibrated, the supporting tools and the Python API.
Scanning chips at the microscope (the Python scan program, `python -m flakeid.scope`) is described in
[scan_guide.md](scan_guide.md).

## Overview

FlakeIdentifier finds exfoliated 2D flakes (graphite/graphene, hBN) in low-magnification optical scans of SiO2/Si
chips: stitched overview images (Olympus Stream JPEG/TIFF/VSI exports) or Micro-Manager tile folders. For each flake
it estimates the thickness from the optical contrast.

```
python find_flakes.py "example images/g31.jpg" graphite 90 --annotate
python find_flakes.py "example images/bn24.jpg" hbn 285 --annotate --json
python find_flakes.py "example images/"g3*.jpg graphite 90 --annotate      # batch
python find_flakes.py tiles_v2_1 hbn 285 --flat flat_6.3x.npz --viewer    # Micro-Manager tile folder
```

Outputs are written next to the image, or to `-o DIR`:

| file | content |
|---|---|
| `<name>_flakes.csv` | one row per **flake** (a whole connected region of material; hBN: split where touching parts differ 2x in thickness), ranked by `score`; flake 1 = best candidate |
| `<name>_terraces.csv` | one row per **terrace**: a uniform-thickness region of interest inside a flake (`flake_id` links them) |
| `<name>_viewer/index.html` | with `--viewer`: zoomable multi-resolution viewer (tile pyramid, as in Jessen et al., Sci. Rep. 8, 6381 (2018), Fig. 5). Scan / enhanced-contrast layers, flake outlines coloured by thickness, sortable and filterable table, click-to-zoom, scale bar, cursor position in µm and stage coordinates. Open the file in a browser; it works offline. `summary.jpg` in the same folder is a reduced annotated overview |
| `<name>_flakes.jpg` | with `--annotate`: annotated overview JPEG only |
| `<name>_flakes.json` | with `--json`: the same flakes plus all run metadata (pixel size, oxide used, parameters, stage origin, ...) |
| `<name>_flake_crops/` | with `--crops`: a full-resolution crop of every flake, with the target point marked |

Requirements: Python ≥ 3.9, `numpy scipy opencv-python scikit-image` (`pip install -r requirements.txt`).
A 11k × 11k px scan takes about 30–40 s.

## How it works

1. **Load:** JPEG/TIFF/PNG export, `.vsi`, or a Micro-Manager tile folder. If a `.vsi` with the same name sits next
   to an export, the pixel size and stage origin are read from it. Otherwise the defaults are used: 1.005 µm/px for
   the BX51 at 6.3x, or `--pixel-size`.
2. **Flatten the background:** the chip is found and the bare SiO2 is fitted with a smooth
   surface, iteratively ignoring anything that is not substrate. Line-shaped kernels then
   remove the tile vignetting and the stitching seams. The contrast
   `C = I / I_substrate − 1` is computed per RGB channel. Per-channel white balance
   cancels out in this ratio.
3. **Thin-film model:** a transfer-matrix (Fresnel) calculation of
   air / flake / SiO2 / Si. It averages over the NA 0.30 illumination cone and is converted to camera RGB with
   a halogen spectrum, broad R/G/B responses, a colour-correction matrix and the tone curve. This gives the
   expected RGB contrast versus thickness for the chosen material and oxide. The oxide thickness is measured from
   the image itself: on raw scans from the bare substrate's colour (see the live analysis below), on other images
   fitted to the flakes' colours within ±10 %; the value used is reported.
4. **Segmentation:** in the foreground (contrast above the noise), flat "cores" (low local gradient) are
   grown into the connected pixels whose contrast matches the core within max(0.045, 8 %). Each terrace of
   a flake becomes its own region. Step edges and blurred pixels stay unassigned instead of being absorbed.
   Regions of one physical flake share a `parent_id`.
5. **Thickness:** each region's median contrast is matched to the model curve. The fit residual, the
   acceptable thickness range and any alternative thicknesses (thin-film colours repeat) are reported.
   Clipped (saturated) channels are treated as lower bounds.
6. **Quality checks** (flag names in the `quality` column):
   * `out_of_range`: outside the thickness window.
   * `poor_colour_match`: colour off the model curve; the tolerance is max(0.2, 0.25·|C|).
   * `low_contrast`: \|C\| below the detection limit (graphite 0.06 ≈ 1L; hBN 0.15 ≈ 3.5 nm).
   * `blurry_edges`: boundary-averaged 20–80 % edge width > 5 µm (smudges, defocus).
   * `not_flat` / `irregular_shape`: re-checked on the full-resolution pixels. `fill` is the fraction of
     the interior matching the region colour; `fullres_solidity` is the compactness of the matching pixels.
     The limits are stricter for faint regions (\|C\| < 0.4), where residue is the main risk: fill ≥ 0.9,
     solidity ≥ 0.75.
   * `in_haze`: faint region (\|C\| < 0.6) whose surroundings have the same hue and strength, e.g. a patch
     of residue haze.
   * `colour_mismatch`: thin (< 5 nm) graphite candidate that is neutral grey (B/G > 0.8), e.g. a facet
     of a thick crystal or dirt; few-layer graphene is green-dominated.
   * `enclosed_in_thick`: faint region inside a flake dominated by thick material.
   * `non_uniform`, `indistinct`: other rejections.
   * Soft flags (still reported): `saturated`, `chip_edge`, `low_confidence` (\|C\| < 0.25), and
     `ambiguous_thickness`. For hBN, `ambiguous_thickness` also marks regions whose best fit is outside the
     window but that have an in-window solution with residual < 0.25; that solution is listed in
     `alt_thickness_nm`.

   Use `--all` to keep rejected regions too.

## Flakes, terraces and the score

A **flake** is a whole connected region of one material. The flake mask is strict enough that residue haze
does not join neighbouring hBN flakes. **Terraces** are the uniform-thickness regions inside it, the regions
of interest that carry the thickness. For hBN, touching flakes of very different thickness are separate flakes: a
connected region's terraces, sorted by thickness, stay one flake while each is less than twice the next thinner one
(steps such as 74 / 78 nm or 41 / 68 nm); a 3 nm area touching a 66 nm one is two flakes, each pixel of the region
going to the nearer one (`flakeid.flakes.split_by_thickness`, `SPLIT_RATIO` 2).

`score` is a learned probability that a terrace is worth imaging. It is a logistic model over terrace and flake
features, fitted to 391 labelled terraces (`tools/fit_scorer.py` -> `flakeid/profiles/scorer.json`). In
leave-one-scan-out validation it reaches AUC 0.72 for "useful or maybe" and 0.84 for "useful"; 84 % of its top 50
were labelled useful or maybe, against a 47 % base rate. A flake's score is that of its best terrace.

Scans from the Micro-Manager RAW16 camera (`bx51_xc10_mm_raw16`, also `_afm`) of hBN use their own scorer,
`flakeid/profiles/scorer_bx51_xc10_mm_raw16_hbn.json`, fitted with `tools/fit_scorer_mm.py` to 399 labels on
bigbntest_2 (56 useful, 132 maybe, 211 not) with the measured contrasts. In 5-fold cross-validation it
reaches AUC 0.87 for "useful or maybe" (the Stream scorer: 0.75 on the same flakes) and 0.85 for "useful"
(0.81). 96 % of its top 25 were useful or maybe (Stream scorer: 64 %). Large, unambiguous, high-contrast
terraces on small flakes score highest; flakes thicker than the window (> 100 nm) were all labelled not
useful. Graphite and Stream exports keep `scorer.json`. The scorer used is recorded in the result's
`params.scorer`. To refit after more labelling, run
`python tools/fit_scorer_mm.py --flakes <scan>_flakes.json --labels <folder> -o flakeid/profiles/scorer_bx51_xc10_mm_raw16_hbn.json`.

Outlines in the viewer are redrawn at the real edge on full-resolution pixels (`flakeid/outline.py`): markers inside
each terrace, inside the rest of the flake and on the substrate around it are grown over the contrast gradient
(watershed), so neighbours meet at the middle of each edge, and the outline is traced between inside and outside
pixels. The segmentation masks behind the thickness fits sit 1-3 µm inside the edge (half-resolution pixels, edge
band left out); on biggraphite_1 the terrace outlines moved out by a median 1.6 µm.

Flake table columns:

| column | meaning |
|---|---|
| `flake_id`, `score` | rank (1 = best) and score |
| `n_terraces`, `terrace_ids`, `thicknesses_nm` | terraces of the flake, best first |
| `best_terrace_id`, `best_thickness_nm`, `best_terrace_d_um` | the best terrace |
| `target_x/y_px/um`, `stage_x/y_um` | centre of the best terrace (where to image) |
| `x/y_px/um`, `bbox_*`, `area_um2`, `inscribed_diameter_um`, `length_um`, `width_um` | the whole flake |
| `focus_um` | edge width of the flake outline at full resolution (focus; ~2 µm is sharp) |
| `crowding` | fraction of the 5–30 µm surroundings covered by other material |
| `frac_window`, `frac_thick` | fractions of the flake inside the thickness window / thicker or bulk |

Terrace table columns: `terrace_id`, `flake_id`, thickness (`thickness_nm`, min/max, `alt_thickness_nm`,
`layers`), geometry, contrast, quality metrics, the flake-level metrics, `score` and `quality` flags.

What the labels showed:
* **Rejection rules the labels confirmed:** `low_contrast`, `colour_mismatch` and `enclosed_in_thick`
  regions were all "not useful". `blurry_edges` (29 % useful or maybe) and `in_haze` (33 %) stay as
  rejections.
* **Rules that are soft flags:** `poor_colour_match` for strongly coloured regions (42 % useful or maybe) and
  `irregular_shape` (45 %) would reject flakes worth keeping. They only feed the score.
* **What the score rewards:** large terraces, thinner material within the window, a good colour fit, and
  clean surroundings (no haze, no thick neighbouring crystal).

## Important options

| option | default | |
|---|---|---|
| `--range MIN MAX` | graphite 0.3–25, hbn 0–100 | thickness window (nm) |
| `--min-area` | 100 | µm² |
| `--min-contrast` | graphite 0.06, hbn 0.15 | minimum \|C\| |
| `--max-edge-width` | 5 | µm; larger edge widths are treated as smudges or defocus |
| `--scale` | 0.5 | processing resolution; 1.0 = full resolution, slower, needs more memory |
| `--stage-origin X Y`, `--stage-sign SX SY` | from `.vsi` | adds stage coordinates to the CSV |
| `--profile` | `bx51_xc10_6.3x`; RAW16 tile folders: `bx51_xc10_mm_raw16` (hBN), `bx51_xc10_mm_raw16_graphite` (graphite) | camera/optics calibration profile |
| `--flat` | | flat field (`.npz` from `python -m flakeid.mm flat`) for Micro-Manager tiles |
| `--all` | | also list rejected regions with reasons |

`material` and `oxide_nm` can be `auto` for a tile folder written by the scan program: the values entered at the
microscope are read from its `scan_params.json`.

## Calibration

`flakeid/profiles/bx51_xc10_6.3x.json` (Stream exports) was fitted without supervision to g31, g32, bn24 and bn26:

```
python -m flakeid.calibrate --same-settings --image g31.jpg graphite 90 --image bn24.jpg hbn 285 ... -o profile.json
```

The fit adjusts the tone curve, the R/G/B spectral responses, a colour-correction matrix and the graphite
optical constants until the model curve passes through the observed contrasts of flat regions.
`--same-settings` adds a strong constraint: images with different oxides taken with the same exposure
must reproduce the ratio of their substrate colours. The model curve is clipped where the camera saturates, both in
the fit and when estimating thickness. The fitted oxides are 84–85 nm (graphite chips) and 288–290 nm (hBN chips).
The first, narrow-band profile is kept as `bx51_xc10_6.3x_v1.json`; it was wrong for 8–20 nm graphite.

Profiles for the linear RAW16 data of Micro-Manager scans:
* `bx51_xc10_mm_raw16`: unsupervised fit (`--tone flare --no-ccm`, no colour matrix) to hBN on nominal 285 nm SiO2.
* `bx51_xc10_mm_raw16_afm`: the same, refined with 15 AFM-measured hBN terraces on bigbntest_2
  (`tools/calibrate_afm.py`); thickness rms 2.0 -> 1.0 nm.
* `bx51_xc10_mm_raw16_graphite`: refitted for graphite on 90 nm SiO2, with the measured 1–4 layer colours
  (`flakeid/profiles/graphite_layers.json`) as anchors. Thicker graphite is not yet checked against AFM.
* `bx51_xc10_mm_raw16_graphite_100x`: graphite at 100x (stray light and NA fitted to terrace colours).

**Validation against AFM or Raman is limited** to the hBN terraces above and the graphite layer anchors. More
measured flakes (1–3L graphene, 5–25 nm graphite, 5–60 nm hBN) are the most valuable next step.

## Known limitations

* **Thin hBN:** ≥ ~5–6 nm is identified reliably. Between
  3.5 and 5 nm a fraction of the detections are residue patches (flag `low_confidence`). Below 3.5 nm hBN is not
  reported by default (`--min-contrast`).
* **hBN interference order:** with the Stream calibration, cyan / green hBN fits the second order
  (~150–190 nm) better than the first (20–60 nm). Such flakes are kept, flagged `ambiguous_thickness`,
  with the first-order value in `alt_thickness_nm`. The 100x thickness (below) helps decide.
* **Thin graphite** thicknesses are good to about one layer (0.34 nm) until calibrated against Raman/AFM. The
  unsupervised calibration predicts a slightly too neutral monolayer hue.
* **Saturation:** in the hBN Stream scans the substrate sits at R ≈ 145/255, so the red channel clips at a
  contrast of +0.76. Green, yellow and orange hBN (~40–110 nm) is often clipped; the fit then relies on
  the other channels (`saturated` flag).
* Flakes in out-of-focus parts of a scan are rejected as `blurry_edges`.
* **VSI input:** `.vsi` files are read natively, including the full-resolution 14-bit data in the companion
  `_name_/stack1/frame_t_0.ets` folder, which must sit next to the `.vsi`. Outputs are named `<name>_vsi_*`.
  On fullchipbn23 the VSI and the cellSens JPEG give the same result: the JPEG is the 14-bit data scaled
  linearly to 8 bits, contrasts agree to 0.5 %, and 99 % of matched regions agree in thickness within 1 nm
  (`tools/compare_sources.py`). The JPEG is sufficient; its 8-bit quantisation is well below the camera noise.
* A JPEG export with a `.vsi` next to it gets the stage origin from the VSI. The origin is corrected for the
  export's crop offset (1, 93 px for fullchipbn23), found by image registration.
* `tools/contrast_report.py` builds the PDF reference / edge-case report from a normal run and an `--all`
  run.

## Ground-truth tools

* `tools/select_afm.py`: chooses AFM sites (on g34, g32, fullchipbn23 and bn25: 26 sites, 39 terraces) and writes
  `AFM_targets.pdf` / `.csv` with locator maps and crops. Enter the step heights in `afm_thickness_nm`.
* `tools/make_label_set.py` + `tools/build_label_page.py`: builds the crop set (`label_set/`) behind the
  labelling page. `items.json` maps each crop id to its scan, position and CSV flake id.
* `tools/compare_sources.py`: region-by-region comparison of two runs of the same scan, e.g. JPEG vs VSI.
* Camera profiles: `bx51_xc10_6.3x` (tone curve; default) and `bx51_xc10_6.3x_flare` (linear camera plus
  stray light). They fit the unlabelled data equally well and differ for 5–25 nm graphite and cyan/green hBN;
  AFM data decides between them.
* AFM step heights: see [AFM step heights](#afm-step-heights-toolsafm_stepspy-flakeidafmpy) below.

## Stack designer

The vdW Stack Designer, which builds heterostructure designs from the flake catalogue, is in this repository
(`flakeid_designer/`, see [its README](../flakeid_designer/README.txt)). It builds each run's `designer/` folder from the
scan results with [tools/make_designer_bundle.py](../tools/make_designer_bundle.py) and the catalogue builders
(`tools/make_catalog_100x.py` and the scripts it uses). The scan program is developed in
[afylab/Flake-Finder](https://github.com/afylab/Flake-Finder).

## Micro-Manager acquisitions (scan tiles in, 100x targets out)

Micro-Manager 2 (XIMEA XC10 + Oasis stage) saves every tile with its stage position, so the scan can be analysed
without Stream's stitching and the flakes sent back as a position list. The scan program
([scan_guide.md](scan_guide.md)) writes such tile folders, one per piece, and sets the camera.

**Camera data.** Data format RAW16 (12-bit Bayer, R at the top-left pixel; demosaiced here), gain 0, auto exposure
and auto white balance off. White balance and colour matrix don't matter in RAW16. Flakes brighter than 4.5x the
substrate clip. The XC10 black level is about 128 counts: give dark frames (light blocked) to `flat --dark`, which
stores them in the flat-field file. An older dark frame is scaled to the scan's bit depth, with a warning, but a flat
and dark taken with the scan settings are better.

```
# 1. flat field from >= 9 frames of bare substrate (dust is fine; optional dark frames with the light blocked)
python -m flakeid.mm flat tiles_blank_1 -o flat_6.3x.npz --dark dark_1
# 2. identify flakes directly on the tile folder (stitched here, flat-field corrected)
python find_flakes.py tiles_v2_1 hbn 285 --flat flat_6.3x.npz --viewer
#    (python -m flakeid.mm stitch tiles_v2_1 --flat flat_6.3x.npz: mosaic preview + stage map only)
# 3. position list for high magnification: best flakes, short stage path
python -m flakeid.targets tiles_v2_1_flakes.csv --top 30 --offset DX DY -o tiles_v2_1_targets.pos
```

**Flat fields.** The scan program records the frames (start screen → Take a flat field, see
[scan_guide.md](scan_guide.md#take-a-flat-field)) and prints the command that builds the flat, e.g.
`python -m flakeid.mm flat <NAME>_flat_100x --dark <NAME>_dark_100x -o flat_100x.npz`. It takes a per-pixel median,
then masks whatever in each frame differs from that median by more than 4 % (things on the sample move between
frames; dirt on the optics does not) and takes the median again. Pass the result as `--flat` to `find_flakes.py` /
`flakeid.live` (10x), or as `--flat100` to `flakeid.live` / `--flat` to `flakeid.hundred` (100x). In the simulator
(`tools/mm_mock/run_mock_flat.py`) on substrate with 20 % flake cover, the flat matches the simulated vignetting
within 0.4 % rms at 10x and 100x.

By default, with `--viewer`, the tile TIFFs of a Micro-Manager folder are deleted after the flake tables, JSON and
the viewer pyramid have been written and checked (about 1.5 GB per 550-tile scan). Each tile's `metadata.txt` is
kept, so stage positions and Z stay available, and a `TILES_DELETED.txt` note is left in the folder. Nothing is
deleted if the analysis fails. Once the TIFFs are gone the scan cannot be re-analysed (new flat field, retrained
score, …): use `--keep-tiles` while the models are still being developed.

* **Stitching.** Neighbouring tiles are registered by template matching of their overlaps. The
  pixel → stage matrix (pixel size, camera rotation, image flip) is fitted from the registrations and
  stage positions, and the tiles are placed by least squares. Every mosaic pixel comes from the tile
  whose centre is nearest, with no blending, so there are no doubled features. The pixel size is
  measured on every scan (1.0906 µm/px, rotation 179.65° on the first test), so the Micro-Manager
  pixel-size setting is not used. The fitted matrix is stored as `stage_map` in `<scan>_stagemap.json`
  (`flakeid.mm stitch`) and in the `_flakes.json` metadata; this is where to read it after the camera has been
  remounted. The analysis reads the tiles twice (registration, then mosaic) instead of holding them all.
* **Camera glitch frames.** The XC10 occasionally delivers a garbage frame (about 1 in 100–200): random values,
  or scrambled data in which the two green sites of the 2x2 Bayer cell differ by tens of percent (it shows as a
  magenta tile), sometimes in part of the frame only, so the green sites are compared in 8 strips across and 8
  down. The scan program snaps such frames again; the analysis checks every raw tile the same way and leaves
  garbage tiles out of the mosaic (`flakeid.mm.raw_glitch`; listed in the result metadata as `glitch_tiles`).
* **Stage coordinates** in the CSV are the stage position that centres each flake's best terrace. They
  are computed from the tile the point was imaged in, so systematic stage errors (about 16 µm in X
  between columns on the test grid) stay local. Checked against an independent grid: 0.7 px scatter.
  The viewer shows pixel, µm and stage coordinates under the cursor.
* **Targets** (`flakeid.targets`): select with `--top`, `--min-score`, `--thickness MIN MAX`,
  `--min-d`, or `--ids 3,7,12`. The output is `<name>_targets.pos` (load it in Micro-Manager's Stage
  Position List → Load) plus a `.csv` with the same order and labels (`01_F003_35nm`, ...).
  * `--offset DX DY`: offset of the 100x objective relative to the 10x one. Measure it from two snaps of
    one roughly centred flake, taken without moving the stage:
    `python -m flakeid.mm offset offset_10x offset_100x --stagemap <scan>_flakes.json --flat flat_6.3x.npz`
    (BX51 10x → 100x: `--offset 24.0 -18.3`, magnification ratio 9.54, i.e. 0.114 µm/px at 100x). Micro-Manager
    does not see joystick moves, so snap metadata keeps the last position it commanded; move only with
    Micro-Manager, or not at all, between the two snaps.
  * Focus: `--z Z` for one Z, or `--focus X Y Z` repeated 3 or more times for a focus plane.
    Without either, Z is left alone (use autofocus). The `z10_um` column is the 10x focus at each flake,
    interpolated between the scan's tile Z (`--z10-scale` converts the drive units of Z).
  * Chip removed and replaced: `--ref OLD_X OLD_Y NEW_X NEW_Y` for 1–3 features (stage µm).
* **Scans from other software** (Stream JPEG/VSI): add `--ref X_PX Y_PX STAGE_X STAGE_Y` for 3–4
  features spread over the chip. X_PX, Y_PX is the pixel shown in the viewer; STAGE_X, STAGE_Y is the
  Micro-Manager stage position with that feature centred. The fit is a least-squares similarity with
  mirror detection, and it reports the residual. Accuracy is limited by the stitching of the
  source image (Stream misplaced tiles by up to 42 µm on fullchipbn23).

### Live analysis during the scan ([flakeid/live.py](../flakeid/live.py))

Run this on the analysis computer while the scan program is scanning. It needs read access to the
scan's output folder, either a network share or a synced folder:

```
python -m flakeid.live "<scan folder>/chipA_tiles" hbn 285 --flat flat_10x.npz -o chipA_results
```

* `--flat` is the path to the flat-field file, relative to the folder you run the command from (or absolute).
  The command checks it before starting. If the first blocks of a piece all fail with the same error, it stops
  and writes no target list for that piece, so the microscope keeps waiting instead of skipping it.
* The scan is split into blocks of 4×4 tiles (`--block`). A block is analysed as soon as it and the ring of tiles
  around it have arrived, in parallel worker processes. Only finished tiles are read, because the
  scan program writes `metadata.txt` after the TIFF.
* The scan plan `<NAME>_scan.pos` next to the tiles folder tells the watcher which tiles to wait for
  and when the scan is finished. Without it, a block waits until the scan has passed its last row, and
  the watcher stops after `--idle` seconds (default 120) without new tiles.
* Each block is stitched on its own and cropped to the block plus 400 µm (`--margin`). Each flake is
  reported once, by the block that owns the core tile nearest its target point. Flakes cut by the crop
  are flagged `block_edge`.
* **Oxide from the substrate colour** (every material): the first blocks get a quick pass that only measures the
  bare substrate's level. Once three are in (or at the end of a small piece), their median is the substrate level,
  and its R/G and B/G give the SiO2 thickness through the thin-film model, as an offset from an anchor wafer
  of nominal 285 or 90 nm (`thickness.SUBSTRATE_ANCHORS`, `thickness.oxide_from_substrate`). That thickness is used for every
  thickness estimate (hBN, graphite, and the rhombohedral colour paths).
  * On 75 chips of 37 runs it gave the same value on every chip of a wafer within ±2 nm. Fitting the oxide to the
    flakes' own colours instead was off by up to 50 nm on 285 nm graphite and found nothing on chips without flat
    material.
  * **The analysis stops** (no target list; the scan program shows "Analysis stopped") if the oxide is more
    than 20 % from the one entered, or if no oxide explains the colour (misfit above 0.004: the light is not as
    calibrated, e.g. B/G 0.86 instead of 0.94). `--no-refine-oxide` analyses with the entered oxide.
  * The scan program's substrate colour check shows the same estimate after the start autofocus.
  * With it, a block is chip wherever it is brighter than half the substrate level.
  * Blocks analysed before calibration are re-run.
* `<name>_live_flakes.csv` / `_terraces.csv` / `.json` are rewritten after every block, and
  `<name>_live_status.json` and `_live_log.txt` show progress. At the end, the 100x target list
  `<name>_live_targets.pos/.csv` is written with `--min-score` (default 0.75), `--offset` (default
  24.0 −18.3) and `z10_um` from the tile Z. Further target-list options (`--max-targets`, `--min-targets`,
  `--min-d`, `--thickness`, `--target-order`, the graphite options, ...) are listed by `python -m flakeid.live --help`.
* **The zoomable viewer is built during the live analysis** (`<name>_viewer/`; `flakeid/livecanvas.py`).
  With the scan plan, a canvas of the whole piece is created on disk at the start; every block writes its own
  tiles into it (the cell of pixels nearest each tile's planned centre, so blocks never overlap) together with
  the edge-accurate outlines of the flakes it owns. At the end the pyramid is written from the canvas in bands.
  The tiles are read once and memory stays at a few blocks. On bigbntest_2 (546 tiles, replayed fast):
  the whole piece took 303 s, the viewer 7 s after the last block, peak memory 2.7 GB per process (the
  full-mosaic batch pass needed 13–15 GB and ~10 more minutes). `--no-viewer` turns it off.
  **The tile TIFFs are deleted by default** once the viewer and the tables are written (metadata.txt files,
  with the stage positions, Z and camera settings, are kept). `--keep-tiles` keeps them: use it while the
  models are being developed (a new material or substrate, calibration work), since deleted tiles cannot be
  analysed again. With no viewer (`--no-viewer`, or no scan plan) nothing is deleted.
* **100x images in the viewer.** In a multi-piece run folder, once the microscope run has ended (run file state
  `done`, after the 100x and stacking-domain imaging), every piece's 100x images go into its viewer
  (`flakeid/hundred.py`): full-resolution JPEGs white-balanced to the demo 100x look of the material (white
  point twice the demo scale) plus 480 px previews in `<viewer>/100x/`; graphite stacking-domain images as
  lossless 16-bit PNG (the domain contrast is ~1 %) with a contrast-stretched JPEG.
  Selecting a flake shows its images in the side panel (click for full resolution); flakes with images are
  marked `100x` in the table, and "with 100x" lists only those. The 100x TIFFs are then deleted unless
  `--keep-tiles`. `--flat100` gives a 100x flat field, `--no-100x` skips the step, `--wait-100x-h` (6) limits
  the wait. For a piece imaged earlier: `python -m flakeid.hundred <piece>_100x --viewer <piece>_tiles_viewer`.
  On bigbntest_2 the 72 100x TIFFs (206 MB) became 9 MB of JPEGs.
  * **Boosted contrast and debris.** Each 100x image also gets two boosted-contrast versions relative to the
    terrace at the frame centre (`_c8.jpg`: each channel stretched to the terrace colour ±8 %; `_sg.jpg`: grey,
    flattened green relative to the terrace ±4 %, thicker darker) and a debris check of that terrace: spots that
    differ from its colour by more than 2.5 % or show up in the fine-detail image are debris (0.1–10 µm²), larger
    deviating areas are another thickness (steps), and the clean area is the terrace minus both grown by 0.5 µm.
    `_debris.jpg` shows the terrace (white), debris (magenta) and steps (cyan); the outlines, in frame pixels and
    in stage µm, and the figures (terrace, debris and clean area, largest clean disc) are in `_regions.json`. The
    viewer switches each image between natural, colour ±8 %, green ±4 % and debris, and lists the figures.
  * **Three positions per flake.** With the scan program's `hundred_shifts_um` setting (default: +10 µm in X,
    +10 µm in Y) each flake gets two more frames at shifted stage positions (same focus), saved as
    `<file>_s1.tif`, `_s2.tif` and listed in the index. The export aligns them on the flake (the commanded shift,
    refined by phase correlation) and takes the per-pixel median: anything that stays at the same pixel (dust,
    camera specks) lands in a different place in each aligned frame and drops out, if it is smaller than the
    shift. In a synthetic test a 2.9 % dust shadow went to 0.07 %. Pixels not covered by all three frames keep
    the first.
  * **Optics artifacts.** Dust in the light path and camera specks stay at the same pixel in every frame while
    each frame shows a different flake, so the run's own frames give them: the terrace-relative brightness
    averaged over all frames (at least 20) is divided out first (`<viewer>/100x/optics_pattern.jpg`). On
    bigbntest_2 this pattern (soft dark blobs and a lattice of specks, 0.6–0.7 % rms) was most of the texture on
    the terraces (single frames correlate 0.78 with it). It is weak near the frame corners, where few terraces
    fall; a measured 100x flat field covers the whole frame.
* **Thickness at 100x and glue, every target** (`flakeid/thick100.py`). Each 100x picture's target terrace gets a
  thickness from its 100x colour (median of the terrace without its outer 1.5 µm against the frame's bare
  substrate off the terrace, else the run's; the thin-film model at NA 0.9 and the run's oxide), found without the
  10x one. At 100x the colours repeat (hBN on ~290 nm oxide: ~125 nm apart), so readings that fit nearly as well
  are listed too, and the 10x thickness of that target's terrace is compared: agrees (within max(5 nm, 20 %)),
  fits only another reading (ambiguous), or fits none. Glue: the share of the terrace covered by tape residue, 0
  (none) to 1 (all), from the residue detector (`flakeid.defects100.residue`) with its yellow seeds above 8x their
  noise; a terrace with a region of another thickness inside its outline can still read some glue. Stored in each
  `_regions.json` (`t100`, `glue`) and shown in the viewer's table (columns t 10x, t 100x, glue; t 100x orange: the
  10x thickness fits only another reading, red: none, * more readings; hover for them) and in the flake's details.
  For example, ~25 targets of one hBN chip that read 38–51 nm at 10x fit ~175–190 nm at 100x. For a viewer made
  before: `python -m flakeid.thick100 <viewer>`.
  **The run's own 100x oxide:** when the 100x pictures are attached to the viewer, the model's oxide is shifted
  (−12 to +24 nm) to where all the run's 100x colours fit best (median misfit; at least 12 pictures and a misfit
  under 0.85x the unshifted one, else no shift), and every picture's 100x thickness is refitted there
  (`t100.oxide_shift`, the log line "the run's 100x oxide"). On one hBN run (+10 nm, median misfit 0.30 -> 0.086)
  the green targets the 10x read at ~90 nm read 209–219 nm at 100x; the yellow ones stayed 80–92 nm.
* **Remeasured coordinates.** For chips that moved after their run (knocked, taken off and put back, before the
  AFM), the scan program's *Remeasure flake coordinates* job finds the flakes again at the microscope and changes
  the old run's results in place (viewer, tables, target lists, `.pos`, 100x regions and mosaics, AFM centres),
  saving every changed file first to `<run>/old_coords/<time>/`. Its main scan also runs a coordinate check that
  re-finds every 100x picture before the results are copied. See
  [scan_guide.md](scan_guide.md#remeasure-flake-coordinates).
* `--then-batch` additionally runs the full-mosaic analysis (`find_flakes --viewer --json`), only needed to
  compare with the batch analysis. The batch analysis ranks flakes independently,
  so its flake numbers differ from the live ones; match the two by stage position.
* On bigbntest_2 (546 tiles replayed at 0.9 s per tile, 4 workers), each block took 15–25 s
  and the watcher kept up with the scan. Peak memory was 2.4 GB; the batch run needs 13 GB. Oxide came
  out at 303.4 nm (batch 303.4). Of the 392 batch flakes scoring ≥ 0.75, 383 are found and 355 also
  score ≥ 0.75 live (the rest sit near the cut). Thicknesses of matched flakes agree within 5 % for 99 %.
* Test without the microscope: `--replay FINISHED_TILES_FOLDER --interval 0.9` copies a finished scan
  into the watched folder tile by tile at the scan rate.
* **Multi-piece runs:** give the run folder instead of a tiles folder:
  `python -m flakeid.live "<scan folder>/runs" --flat flat_10x.npz -o results [--keep-tiles]`.
  It waits for the newest run file (`<name>_run_<n>.json`, written from 10 min before the command on;
  `--run FILE` picks one). It then analyses the pieces in order as they are scanned, each into
  `results/<piece>/`, and writes `<piece>_targets.csv` into the run folder for the scan program. Each piece's
  material and oxide, as entered at the microscope, come from the run file; values given on the command line are
  used only for pieces without stored ones. A piece with no qualifying flakes gets a header-only list, so the
  microscope does not wait for it. `--then-batch` builds the viewers after the last piece. It exits when the run
  is finished; with `--continuous` it keeps watching the folder and follows every new run file written after it
  started, one after another, until you press Ctrl-C. Pieces that already have a `<piece>_targets.csv` are
  skipped, so a restarted watcher continues where it stopped (`--redo` analyses them again). Tested with
  bigbntest_2 split into two pieces: each piece calibrated itself (oxide 303.4 and 303.2 nm), and each target list
  appeared once that piece's last tiles were analysed.

### AFM step heights ([tools/afm_steps.py](../tools/afm_steps.py), [flakeid/afm.py](../flakeid/afm.py))

```
python tools/afm_steps.py <folder of .spm files> -o <results> --sites "<AFM set>/data.json"
```

It reads Bruker NanoScope files and measures every flat level's step above the substrate. For each file it:
- masks bad data (clipped values, spikes, tip-loss streaks);
- finds the flat regions, which are bounded by edges;
- levels everything in one fit: a per-line offset, a tilt or bow background, and one offset per region.

It writes `steps.csv` with step, error, background used and the matched flake/terrace with its predicted
thickness, plus a diagnostic figure per scan. With `--sites`, the n-th file is the n-th site of the
programmed-move list (`--first N` if the run started later). On the AFM calibration page
([tools/afm_page.html](../tools/afm_page.html)), **Import AFM results** reads that `steps.csv` and fills in the measured heights. `tools/calibrate_afm.py` refines a camera profile with the measured heights.

## Python API / pipeline integration

```python
from flakeid import find_flakes
res = find_flakes("g31.jpg", material="graphite", oxide_nm=90, thickness_range_nm=(0.3, 10))
for f in res.flakes:              # ranked flakes (FlakeSummary); res.terraces: the terraces
    print(f.flake_id, f.best_thickness_nm, f.area_um2, f.stage_x_um, f.stage_y_um)
res.to_csv("g31_flakes.csv"); res.to_json("g31_flakes.json")
```

Modules: `io` (loading, VSI metadata), `mm` (Micro-Manager tiles, flat field, stitching, stage map),
`targets` (flake table → Micro-Manager `.pos`), `background` (flat field, contrast), `optics`
(thin-film model, materials), `thickness` (contrast → thickness, profiles), `segment`
(uniform regions), `flakes` (flakes from terraces), `outline` (edge-accurate outlines), `scorer` (the learned score),
`pipeline` (orchestration, quality cuts, output), `annotate`, `calibrate`, `cli`, `live` (analysis during the scan),
`hundred`, `thick100` (100x images and thickness), `afm` (AFM step heights). To add a material,
add its refractive index to `optics.MATERIALS` and a default window to `pipeline.DEFAULT_RANGE_NM`.
