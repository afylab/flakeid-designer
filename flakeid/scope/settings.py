"""Scan settings: the settings block of tools/mm_chip_scan.bsh, with the same names (lower case) and defaults.

Override any of them with a JSON file (python -m flakeid.scope --settings my_settings.json): {"exposure_ms": 3.0, ...}.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Settings:
    # camera
    set_camera: bool = True
    exposure_ms: float = 3.0                # 10x; hBN on 285 nm: substrate green ~25 % of full scale
    auto_exposure: bool = False
    target_level: float = 0.25
    exposure_range: tuple = (0.05, 500.0)
    camera_props: tuple = (("Data format", "RAW16"), ("Device output data bit depth", "12"),
                           ("Image data bit depth", "12", "optional"), ("AEAG", "Off"), ("Auto white balance", "Off"),
                           ("Gain", "0"), ("Luminosity gamma", "1"), ("Chromaticity gamma", "1"), ("Sharpness", "0"),
                           ("Downsampling", "1x1"), ("Binning", "1"))
    verbose: bool = True
    arrive_tol_um: float = 10.0
    stage_timeout_s: float = 60.0         # Micro-Manager waits this long for a stage / focus move (its default 5 s is too short for a long move at 100x)
    selftest_um: float = 100.0
    mode: str = "multi"                     # multi | chip | targets | flat | capture | retarget
    ask_mode: bool = False                  # start screen: choose scan / flat field / capture (set when started without --mode)
    settings_file: str = ""                 # the JSON file these settings came from (a new flat field can be written into it)
    focus_mode: str = "auto"                # auto | manual
    out_dir: str = "C:/scans"
    name: str = "chip"
    run_scan: bool = False                  # chip mode: acquire the tiles after the plan
    show_plan: bool = False                 # show each piece's scan-plan picture in a window (it is written to the results either way)
    edge_points: int = 5
    edge_spread: float = 0.40
    arc_r_range: tuple = (15000.0, 200000.0)
    overlap: float = 0.15
    margin_um: float = 100.0
    max_travel_um: float = 30000.0
    # the stage's travel, [x min, x max, y min, y max] in stage um (null = not known); limits are also learned when the
    # stage stops short at one (kept in <local_root>/stage_limits.json): a chip partly beyond them is scanned up to them
    stage_limits_um: object = None
    stage_limit_learn: bool = True
    stage_limit_margin_um: float = 20.0
    step_frac: float = 0.80
    dark: float = 128.0                     # black level at 12 bit (RAW16), scaled to the bit depth
    M: tuple = ((-1.0905, -0.0067), (0.0066, -1.0917))   # 10x pixel -> stage um (stage that centres pixel p = stage + M (p - c))
    sample: int = 16
    chroma_tol: float = 0.10
    start_min_substrate: float = 0.20
    substrate_colour: object = None         # None, or (R/G, B/G) of bare SiO2
    bright_min: float = 0.45
    bright_max: float = 1.60
    bin_um: float = 20.0
    gap_um: float = 80.0
    gap_fraction: float = 0.03
    surround_dark: bool = True
    dark_level: float = 0.6
    edge_any_um: float = 1500.0
    # chips close together (2026-10-20: a lab member's chips were not found): the walk to an edge stops before the next
    # chip (its outline once scanned, else its marked position); before that a narrower dark gap is taken as the edge,
    # and with none (touching chips) the user is asked to put the crosshair on the edge (edge_ask; off: the piece fails)
    narrow_gap_um: float = 20.0
    edge_ask: bool = True
    focus_points: int = 4                   # manual focus
    focus_points_auto: int = 9              # autofocus: 4 or 9
    # small pieces (both sides below small_piece_mm, from the first edge search): fewer edge points per side and
    # autofocus points (a tilted plane is enough over ~1 cm; the 9-point quadratic surface is for inch-size pieces)
    small_piece_mm: float = 15.0
    small_edge_points: int = 3              # points per side, the first one included (3 = 2 more)
    small_focus_points: int = 4
    focus_inset: float = 0.15
    # chips mounted with tape over their edges (2026-10-20): a focus point must be on bare substrate (else it moves towards
    # the chip's centre, focus_moves times); one that disagrees with the plane through the others by more than
    # focus_outlier_um is refocused further in, and left out of the focus surface if it still disagrees
    focus_min_substrate: float = 0.3
    focus_moves: int = 3
    focus_outlier_um: float = 8.0
    max_dz_um: float = 150.0
    # the focus drive read back short of where it was sent by more than this: at a limit of its travel (the Oasis
    # controller's lower / upper limit; BN_15_auto 20261025_run_4 stopped at -181.9 um, 33 of 58 100x pictures out of
    # focus): logged, and one message (Slack / email) per direction and job
    z_reach_tol_um: float = 1.0
    focus_limit_stop: bool = True          # an autofocus whose best focus is at the focus drive's lower limit stops the
                                           # scan with a message (Scanner.focus_limit_check; passing the limit on the way: fine)
    z_stepsize_um: float = 0.0250 / 12.0
    z_scale: float = 12.0
    focus_surface: str = "quadratic"
    max_bow_um: float = 15.0
    max_tilt: float = 10.0                  # um/mm
    max_focus_rms: float = 5.0
    af10: tuple = (30.0, 5.0, 6.0, 1.0, 0.8)
    af100_first: tuple = (15.0, 1.0, 1.0, 0.25, 0.9)
    # 100x after the first flakes: the prediction was within 1.3 um of the focus for 95 % of 148 flakes (microscope PC,
    # 2026-10-03; max 3.5 um, found by the sweep's extension when the peak is at an end): +-2.4 um coarse, +-0.5 fine
    af100: tuple = (2.4, 0.8, 0.5, 0.25, 0.9)
    # ... and that sweep alone (its peak and two neighbours, a parabola through log sharpness) instead of the fine sweep
    # after it, when the peak is clear (not at an end of the sweep, curving down); otherwise the fine sweep as before.
    # 7 frames instead of 12-17 (20261014_run_1: 445 autofocus calls x 4.2 s)
    af100_fit: bool = True
    # the focus drive slipped by 40-60 um during 20261014_run_1 (outside the +-15 um af100_first search): every flake
    # after it failed, and the first flake of each new piece waited focus_dialog_timeout_s for an answer.  When the
    # usual searches fail on af_long_after flakes in a row (or before a focus question would be asked), one long coarse
    # search (+-80 um, 3 um steps, no extension; then the fine sweep) is made, once per run of failures; a focus found far
    # from the prediction is confirmed as a focus jump (image_flake)
    af_long_after: int = 2                  # 0 = never
    af_long: tuple = (80.0, 3.0, 1.5, 0.25, 0.9)
    retake_failed: bool = True              # flakes not focused are imaged again at the end of their piece (when a flake
                                            # was focused after them: the prediction knows more), replacing the first image
    # the retakes search harder (2026-10-27): after the usual searches, af_retake (+-40 um coarse, 2 um steps), moved on
    # up to af_retake_extend times when the peak is at an end (+-120 um); the 100x window's "Manual focus (retakes)"
    # button: the current retake (if not focused) or the next one focused by hand (live view, fine knob)
    af_retake: tuple = (40.0, 2.0, 1.0, 0.25, 0.9)
    af_retake_extend: int = 2
    af_max_dev_um: tuple = (5.0, 3.0)
    af_dev_per_mm: float = 0.5
    af_max_dev_cap: float = 8.0
    af_agree_um: float = 1.5
    # Pause / Resume (the main window's Pause button; 2026-10-27): at Resume the focus where the scan paused is measured
    # again (autofocus) and a change beyond pause_refocus_tol_um (10x, 100x) shifts every later Z by it (the coarse knob
    # is not seen by the drive); above pause_refocus_warn_um a message too.  pause_refocus False: no focus check
    pause_refocus: bool = True
    pause_refocus_tol_um: tuple = (1.5, 0.5)
    pause_refocus_warn_um: float = 10.0
    af_extend: int = 2
    af_min_ratio: float = 1.10
    af_backlash_um: float = 5.0
    spike_factor: float = 3.0
    # wait after every Z move before the next frame (the drive reports "done" when the motor stops; 20 ms let the
    # objective settle).  A 100x autofocus makes ~16 Z moves, so each 10 ms here is ~0.16 s per flake.  Not lowered:
    # the simulator has no settling to test it against; try 10 at the microscope and compare the focus repeatability
    z_settle_ms: float = 20.0
    overlap_z_move: bool = True            # tiles: the focus backlash approach is sent during the XY move
    tile_readback_every: int = 10          # tiles: the stage position read back (and checked against arrive_tol_um) on every
                                           # Nth tile, the last and any that raised; the others are recorded where they were
                                           # sent (two controller round trips saved per tile; 1 = every tile, as before)
    background_save: bool = True           # tiles: written by a thread while the stage moves on
    ui_theme: str = "dark"                 # windows: "dark" / "light" (Sun Valley, sv-ttk) or "native" (Windows' own look)
    af_retries: int = 2
    # targets / 100x
    targets_csv: str = ""
    chip_json: str = ""
    plane_min_spread_um: float = 1500.0
    target_exposure_ms: float = 27.0
    target_auto_exposure: bool = False
    target_box_margin_um: float = 2000.0
    focus_offset_um: float = -42.0          # 100x focus Z minus 10x focus Z (real um); NaN = measured at the first flake
    hundred_shifts_um: object = ((10.0, 0.0), (0.0, 10.0))
    hundred_save_frames: bool = False      # also save every frame averaged into a flake's 100x image (<file>_f1.tif ...)
    pattern_frames: int = 0                # targets mode: then this many frames around the first target, for the optics
                                           # pattern (dust, specks: fixed to the camera, found by comparing positions)
    pattern_radius_um: float = 250.0       # ... at up to this distance from it
    hundred_frames: int = 4                 # frames averaged for each flake's main 100x image (pixel noise ~2 % -> ~1 %);
                                            # saved as rounded 16-bit; the shifted frames are single (the analysis weights them)
    # each 100x target checked against its 10x picture (<piece>_targets_crops) after focusing, and the stage moved onto
    # the flake when the shot is off (2026-10-14: 214 of 1574 100x frames had bare substrate at the centre)
    # mosaics of large terraces (flakeid/fieldmosaic.py): when a target's chosen terrace (its outline from the analysis:
    # terrace_poly_um in the target list) reaches beyond the main 100x picture, the fields it reaches into are imaged
    # too (<main>_m02.tif ..., <piece>_100x_mosaic.csv) and stitched by the export (<main>_mosaic.jpg in the viewer, key W).
    # Target lists without the column (before 2026-10-19): the single picture as before
    hundred_mosaic: bool = True             # (off: the single centred picture only, 100x and IR)
    hundred_mosaic_max_tiles: int = 8       # at most this many pictures per flake, the main one included (the nearest
                                            # to it kept when the terrace needs more); 30 before 2026-10-19 (Jordan: 5-10)
    hundred_mosaic_overlap: float = 0.15    # overlap between neighbouring fields (fraction of the field)
    hundred_mosaic_budget_min: float = 20.0  # extra 100x imaging time per run for mosaics; when it is used up the
                                            # remaining flakes get their single picture (logged)
    hundred_mosaic_frames: int = 0          # frames averaged per mosaic field (0 = hundred_frames); no shifted frames
    # each field's focus: predicted from the main picture's focus + the chip's tilt (the fields are within a few hundred
    # um of the focused main picture); a tuple (half range, step, roi) adds one short sweep per field
    # (Scanner.autofocus_fit), e.g. (1.2, 0.4, 0.9) (the default until 2026-10-19: ~7 frames a field)
    hundred_mosaic_af: tuple = ()
    hundred_mosaic_af_min_rise: float = 1.05  # ... its sharpest point this much above both ends of the sweep, else the
                                            # predicted Z is used (focus "predicted" in the mosaic index)
    hundred_mosaic_max_dev_um: float = 2.0  # a sweep result further than this from the prediction is not used
    domain_mosaic_budget_min: float = 10.0  # the same for the IR (stacking-domain) pictures, their own budget per run
                                            # (0 = no IR mosaics); same max tiles and overlap
    domain_mosaic_af: tuple = ()            # IR fields' focus: the main IR picture's (its depth of field is several um);
                                            # a tuple (half range, step, roi) adds a sweep per field, e.g. (4.0, 0.8, 0.9)
    # the IR camera's pixel -> stage matrix (um per px, as its frames are saved: after domain_flip), for its mosaic fields;
    # the Domain camera test prints it ("IR camera pixel -> stage") and saves it here.  null = the 100x camera's
    # directions at 0.358 IR px per 100x px (measured 2026-10-15; the ~6 deg between the cameras is left to the export's
    # overlap matching)
    domain_px_um_matrix: object = None
    xy_approach_um: float = 100.0           # 100x and domain targets are reached from -x / -y: first this far short of
                                            # them, then a short +x / +y move (same backlash every time; 0 = direct)
    xy_approach_clear_um: float = 20.0      # ... an axis on which the move to the target is already at least this far in +
                                            # needs no pre-move (it arrives from - anyway, its backlash taken up: the median
                                            # miss was 2 um after +y moves against 8 um after -y moves); 0 = always pre-move
    hundred_shift_approach_um: float = 20.0  # the shifted 100x frames: a shift move back in -x / -y goes this far short
                                            # first, so every shifted frame is also reached from -x / -y
    hundred_recentre: bool = True
    recentre_before_focus: bool = True      # re-centre at the predicted Z, then autofocus on the centred flake (2026-10-14:
                                            # ~14 % of 100x frames had bare substrate at the centre: no focus signal, a wide
                                            # retry); off, or when the focus is not known yet: after the autofocus as before
    recentre_min_peak: float = 0.5          # least correlation (100x frame against the 10x picture) to trust a match ...
    recentre_max_second: float = 0.85       # ... and the best match elsewhere at most this fraction of it
    recentre_min_um: float = 4.0            # smaller offsets are left alone
    recentre_max_um: float = 150.0          # larger ones are not trusted (a different place that looks alike)
    recentre_search: bool = True            # no confident match: look around (a ring of 6 frames) for the flake
    recentre_carry: int = 5                 # the next targets start corrected by the median of the last N corrections
                                            # (0 = off); slow drift of the stage or of the 10x mapping is absorbed
    af_manual_fallback: str = "first"       # first | always | never
    focus_dialog_timeout_s: float = 120.0
    # target_mode "avram_new1" only: at the start of each piece's 100x imaging, int(its 10x focus points / this) of its
    # targets (spread over the piece, a short path) are focused by hand (0 = none).  They calibrate 10x -> 100x
    # (scan.offset_model: a plane of the 100x - 10x offset, or its median); every flake then starts at the 10x focus
    # map + that calibration (corrected by the flakes focused since), then autofocus.  Skip: that point; Stop, or no
    # answer within focus_dialog_timeout_s: no more points
    hundred_manual_focus_divisor: int = 3
    focus_dialog_repeat: bool = False       # after a focus question went unanswered, ask again on that piece (false: the
                                            # rest of the piece's failures are imaged at the predicted Z and retaken at the end)
    operator_sound: bool = True             # a sound (Windows) when the operator is needed: objective change, IR setup, a
                                            # focus question
    # messages to you (flakeid/scope/notify.py: python -m flakeid.scope.notify --test): Slack and / or email when the
    # operator is needed and when a job ends; the SMTP password from FLAKEID_SMTP_PASSWORD or notify_smtp_password_file
    ask_chip_notes: bool = True             # multi / chip scans: a window beside the scan for each chip's exfoliation details
                                            # (crystal batch, tape, ...), saved as <chip>_notes.json and in the run file
    notify_slack_webhook: str = ""          # a Slack incoming-webhook URL
    notify_email_to: str = ""               # address(es), comma-separated
    notify_email_from: str = ""
    notify_smtp_host: str = ""
    notify_smtp_port: int = 587             # 587 STARTTLS, 465 SSL, 25 plain
    notify_smtp_user: str = ""
    notify_smtp_password_file: str = ""
    notify_slack_token_file: str = ""       # Slack bot (app "FlakeID Scanner"): a file with its xoxb- token (or FLAKEID_SLACK_TOKEN)
    # program updates from GitHub (the start screen's "Update...", run_flakeid.bat update; flakeid/scope/update.py)
    update_repo: str = "afylab/Flake-Finder"
    update_branch: str = "main"
    update_token_file: str = "C:/FlakeID/github_token.txt"   # only for a private update_repo: a token that can read it (or FLAKEID_GITHUB_TOKEN)
    notify_to: str = ""                     # who runs the scan: Slack email, name or member ID (the start screen asks; remembered)
    notify_slack_channel: str = ""          # optional: a channel ("#microscope") that gets each run's thread too
    notify_on: tuple = ("operator", "scanned", "finished", "stopped", "failed")   # (+ "progress": each chip's plan)
    notify_threads: bool = False            # Slack: each run's messages in one thread (replies do not notify: off, every
                                            # message a new one, 2026-10-19)
    notify_min_job_min: float = 5.0         # end-of-job messages only for jobs at least this long (not a quick flat field)
    # multi
    piece_centers: object = None            # None: asked (live viewer); or [[x, y(, z)], ...]
    piece_materials: object = None          # [["graphite", 90], ["hbn", 285], ...] with piece_centers
    # what is on the stage (asked in the Chips window after Start scan, remembered in <local_root>/chips.json):
    #  "large"        larger pieces: the edge walk (find_outline), as before 2026-10-21
    #  "loose_1cm"    1 cm chips (diced by hand), each on its own: a nominal square checked by one probe per side
    #  "mounted_1cm"  the same, rigidly mounted on a glass slide (often taped): up to 2 blocks per scan, each with 3
    #                 reference corners marked at the start (one stage -> AFM mapping for all its chips)
    #  "polygon"      scan areas drawn by hand: no edge search; at the start each area's corners are marked one by one
    #                 (a polygon, as many points as wanted; several polygons), and every field that overlaps it is scanned
    chip_type: str = "large"
    chip_nominal_mm: float = 10.0           # 1 cm chips: the nominal side
    chip_size_tol: float = 0.30             # ... cut by hand: any side within +-30 % of it
    chip_centre_err_mm: float = 1.0         # ... how far the marked centre may be off the true one
    chip_focus_inset_mm: float = 1.5        # ... focus points this far in from the chip's edges (tape over the edges)
    chip_edge_any_um: float = 150.0         # ... this long without substrate across the whole field is the edge (tape, glue:
                                            # not dark, so the usual edge_any_um (1.5 mm) let the probe cross ~1 mm of tape)
    piece_mounts: object = None             # with piece_centers: the block of each piece, e.g. [1, 1, 2, 2] (mounted_1cm)
    mount_corners: object = None            # with piece_centers: {"1": [[x, y], [x, y], [x, y]], "2": [...]}
    piece_polygons: object = None           # chip_type "polygon": None = marked at the start (live viewer); or each area's
                                            # points [[[x, y(, z)], ...], ...] (piece_materials as with piece_centers)
    polygon_focus_mode: str = ""            # ... how its 10x focus points are focused: "auto" | "manual" (the polygon window's
                                            # choice); "" = focus_mode.  The 100x imaging keeps focus_mode
    polygon_focus_per_cm2: float = 4.0      # ... focus points per cm2 of a polygon (at least 4), spread uniformly over it
    polygon_focus_smoothing: float = 1.0    # ... its focus surface: a smooth one through all the points (thin-plate spline);
                                            # larger = smoother (towards a plane), 0 = through every point exactly
    run_dir: str = ""
    run_name: str = ""
    targets_wait_s: float = 3600.0
    objective_prompt: bool = True
    material: str = "hbn"
    oxide_nm: float = 285.0
    # stacking domains (graphite)
    domain_imaging: str = "ask"             # ask | yes | no
    domain_min_layers: int = 3
    domain_max_nm: float = 0.0
    domain_camera: str = "Raptor Owl Camera 640"
    # the Raptor Owl 640 through Micro-Manager's RaptorEPIX driver (tests 2026-10-09, tools/win7/ir_camera_test.py):
    #  - "Frame Average" > 1 is the driver snapping and averaging in software, and a timeout in it leaves the camera stuck
    #    "live": kept at 1 (the scan enforces it), the frames are averaged by the scan instead (domain_frames)
    #  - the exposure setting does not change the picture (not in the camera's own software either): its auto exposure
    #    stays on, domain_exposure_ms is sent but has no effect on this camera
    #  - High Gain Off: darker, but more signal-to-noise per frame than On (53 against 35 at the same exposure)
    domain_camera_props: tuple = (("Exposure: Auto", "On", "optional"), ("High Gain", "Off", "optional"), ("Frame Average", "1", "optional"))
    domain_exposure_ms: float = 28.0
    domain_frames: int = 16                 # frames averaged for each domain image (from the camera's stream; ~1.3 s at 25 Hz)
    domain_wake: bool = True                # the domain camera live for a moment, then stopped, before its settings (it
                                            # gave empty frames until it had been live once)
    domain_flip: str = "xy"                 # the Raptor's picture is mirrored in X and Y against the colour camera: its
                                            # frames are flipped back ("x", "y", "xy", "" = as the camera gives them)
    flat_domain: str = ""                   # IR flat field (.npz from Flat field, objective "IR (domain camera)"): the domain
                                            # images are divided by it in the export ("" = none)
    domain_xy_offset_um: tuple = (0.0, 0.0)
    domain_focus_offset_um: float = float("nan")
    domain_search: tuple = (30.0, 1.0, 1.5, 0.25, 0.9)
    domain_af_first: tuple = (15.0, 1.0, 1.5, 0.25, 0.9)
    domain_af: tuple = (4.0, 0.8, 1.5, 0.25, 0.9)
    domain_max_dev_um: float = 2.0
    domain_af_fit: tuple = (4.0, 0.8, 0.9)  # IR autofocus at a predicted focus: one sweep (half range, step, roi), focus from
                                            # a fit (Scanner.autofocus_fit; empty: the coarse + fine sweeps of domain_af)
    domain_af_min_rise: float = 1.03        # ... its sharpest point at least this much above both ends of the sweep
    domain_restore_camera: bool = True
    # flat
    flat_objective: str = "10x"
    flat_frames: int = 25
    flat_step_um: float = 0.0
    flat_min_substrate: float = 0.6
    flat_focus: str = "fieldstop"           # fieldstop: plane through the field-stop edge focused at 5 points; fixed; auto (each position)
    flat_dark_frames: int = 20
    flat_oxide_nm: float = 285.0            # 100x flat fields are per oxide: in the file name and the flat100 entry
    flats_dir: str = ""                     # where new flat fields are saved; "" = <program folder>/flats
    flat_measured_oxide_nm: float = 0.0     # Filmetrics value of the flat-field piece (0 = none): recorded as the oxide calibration
    flat_gui: bool = False
    # re-image the 100x targets of a piece scanned before (start screen): find the chip again from its flakes
    retarget_piece: str = ""                # piece name (its <piece>_targets.csv and _chip.json are in local_root)
    retarget_reloaded: bool = False         # taken off and put back: search widely (shift and rotation)
    retarget_only_failed: bool = False      # only the targets without a focused image last time (<piece>_retake.csv)
    retarget_manual_points: int = 3         # the user centres this many flakes at 100x (spread over the piece) and the
                                            # positions are converted from those; 0 = found automatically (edges, then
                                            # 10x spots against the first scan's mosaic; less reliable)
    retarget_update_only: bool = False      # no imaging: the re-images done so far into the viewer, designer run and the
                                            # results folder (start screen)
    retarget_spots: int = 7                 # 10x reference spots for the registration (spots that disagree are left out)
    retarget_find_edges: bool = True        # first find the piece's edges again (as the scan does): its shift and rotation
    retarget_min_peak: float = 0.2          # 10x spot against the first scan's mosaic: least correlation to use it
    retarget_check100: bool = True          # check (and correct) the 100x position on the first targets against the mosaic
    retarget_min_peak100: float = 0.3       # ... least correlation for that
    retarget_recentre: bool = True          # re-centre each target at 100x against the mosaic before imaging it
    edge_tol_um: float = 60.0
    lead: float = 1.5
    # analysis on this PC (multi mode): python -m flakeid.live follows the run folder and writes the target lists
    local_root: str = "C:/FlakeScans"       # multi mode: tiles, 100x images and the run's files go here (local disk)
    ask_results_folder: bool = True         # multi mode: ask where the results are copied at the end (copy_results_to)
    local_analysis: bool = True
    results_dir: str = ""                   # analysis results; "" = <local_root>/results
    analysis_workers: int = 2               # leave the processor room for the acquisition
    flat10: str = ""                        # 10x flat field .npz for the analysis
    flat100: tuple = ()                     # ("285=C:/FlakeID/flats/flat_100x_285nm.npz", "90=...")
    keep_tiles: bool = False
    copy_results_to: str = ""               # where the results (viewers, records) are copied at the end; "" = asked
    upload_each_chip: bool = True           # each piece copied there as soon as its 100x export ends (the designer adds
                                            # it while the next pieces are imaged); the rest at the end
    min_targets: int = 10                   # 100x: at least this many flakes per piece, the best-scoring ones even if they
                                            # score below the cut (e.g. graphite on 285 nm oxide); 0 = only those above it
    # 100x targets per piece in proportion to its area (found by the edge walk): targets_per_cm2 x area, at least
    # targets_min_piece, at most targets_max_piece; asked in the Messages popup after Start scan (remembered in
    # <local_root>/targets.json).  0 = the analysis's fixed --max-targets (80 per piece, before 2026-10-21).  25/cm2:
    # Jordan's ~4.5 cm2 pieces had 145-306 qualifying flakes and got 80 (2026-10-20).  100 since 2026-10-27 (Avram: 100
    # targets per cm2, every one imaged at 100x)
    targets_per_cm2: float = 100.0
    targets_min_piece: int = 10
    targets_max_piece: int = 0              # 0 = no upper limit (Avram, 2026-10-27: was 400; a 74 x 10 mm polygon at
                                            # 100 per cm2 wants ~740)
    # hBN 100x targets (the analysis on this PC, python -m flakeid.live --target-order / --target-rank; 2026-10-22):
    #   target_order "size": imaged largest best terrace first (groups of 10, a short stage path within each), so a run
    #     stopped or a piece skipped early still has the biggest flakes; "path": the shortest stage path (before)
    #   target_rank "score": the best-scoring flakes are targets (as before); "size": ranked by logit(score) +
    #     2 ln(best terrace d) instead: the trial, a run of one chip (C:\FlakeID\run_flakeid.bat scope --set target_rank=size;
    #     not in scan_settings.json, or every later run would be a trial too).
    #     Each hBN piece also gets <piece>_targets_alt.csv: the flakes the other ranking would add / drop (comparison)
    #   graphite and other materials are not affected
    # which flakes are 100x targets (the Chips window asks; flakeid.live --target-mode): "standard" = the learned score and
    # the quality flags (everything below); "avram_new1" = every flake whose clean area (its largest one-colour region x
    # the fraction of it matching that colour) is at least target_min_clean_um2, largest first
    target_mode: str = "standard"
    target_min_clean_um2: float = 400.0
    target_clean_thickness_nm: tuple = ()   # avram_new1: the clean areas' thickness (min, max) nm; () = the material's
                                            # window (hBN 0-100, graphite 0.3-25)
    target_order: str = "size"
    target_rank: str = "score"
    # hBN, target_order "size": of each piece's targets only this many with the largest best terrace are imaged (the
    # same targets as before, the smallest left out: 50 % of the 100x time, ~79 % of the keeps and all the flakes used
    # in designs in a replay of 21 pieces; flakeid.live HBN_MAX_TARGETS); 0 = all.  48 since 2026-10-25: the inclusive
    # thickness (thick_context etc. no longer exclude) put ~7 likely-thick flakes into a piece's 40 (multitest chip_1);
    # 8 more slots (~2-3 min) keep the thin ones in, the thick ones are told apart at 100x
    # 0 since 2026-10-27 (Avram: every target is imaged at 100x; the Chips window's targets per cm2 sets how many)
    hbn_max_targets: int = 0
    # hBN: of each piece's targets, hbn_imaged_per_cm2 x its area (cm2) with the largest best terrace are imaged (at least
    # targets_min_piece; a 74 x 10 mm polygon at 30: 222); 0 = every target (the default since 2026-10-27, when the
    # Chips window stopped asking for it).  Pieces of unknown area: hbn_max_targets
    hbn_imaged_per_cm2: float = 0.0
    # graphite: "model" = targets ranked and imaged by the kept-at-100x model (flakeid.target_model, fitted on past runs'
    # designer decisions), within the ribbon / 1-4 layer / thicker groups; "tiers" = layer tier then score, shortest path
    graphite_rank: str = "model"
    # graphite: at most this many imaged per piece (the first in graphite_rank order, ribbon places and the thick fraction
    # kept; replay of 10 pieces: 40 = 56 % of the 100x shots, 111 of 156 keeps); 0 = all
    graphite_max_targets: int = 0
    hundred_s_per_target: float = 7.6       # 100x time per target, for the popup's estimate (bn_20261020_run_1: 40.6 min / 320)
    designer_on_pc: bool = False            # build the designer run on this PC at the end of the analysis; off: the vdW
                                            # Stack Designer builds it from the results on the share (faster computers)
    export_in_background: bool = True       # the 100x export (and the copy to the share) finishes in the background: the
                                            # start screen is back as soon as the microscope work ends
    analysis_wait_h: float = 2.0            # after the run: wait this long for the analysis (100x export) to finish
    # live view Autofocus button (10x): a wide coarse sweep (another holder can put the sample tens of um away), then a
    # short sweep around its best point; at 100x the af100_first search
    live_af_range_um: float = 150.0
    live_af_step_um: float = 8.0
    live_af_refine: tuple = (12.0, 3.0)
    # frames from the camera's continuous stream (the newest frame after each move, as the live view) instead of a snap
    # per frame (~0.2 s each with the XC10); false = snaps
    stream_frames: bool = True
    # ... and which frame of the stream: false = the buffer cleared after the move and the second new frame used (the
    # first may have been exposing during the move; ~1.5 frame periods, ~105 ms at 14.2 fps).  true = the first frame
    # whose exposure started after the move ended, from its time stamp (Micro-Manager's ElapsedTime-ms) taken as
    # exposure + frame_readout_ms before it reached the buffer; the old way whenever the camera gives no time stamps.
    # Off until measured at the microscope: with a readout of a whole frame period it is no faster than the old way
    frame_timestamps: bool = False
    frame_readout_ms: float = 70.0          # time from the end of a frame's exposure to its arrival in the buffer
    # coded nosepiece (Olympus U-CBS box): which objective is in.  objective_positions maps "10x", "100x" (and
    # "domain") to nosepiece positions 1..7; unknown ones are not checked (the position is still recorded)
    # device photos (start screen "Device photos", flakeid/scope/devicephotos.py): frames averaged per picture, and at 100x
    # / 50x the same at shifted stage positions (um at 100x; doubled at 50x) to average dust in the light path away
    device_name: str = ""                   # (with a stack folder: the folder's device name, see device_stack_dir)
    # the stack (device) folder the photos go to, as a fabrication step NN_<step>/ (docs/stack_workflow.md, "Stack
    # folders"); "" = the old way: a folder <device>_<date> copied to device_photos_to.  The start screen asks for all
    # three (the folder is remembered in <local_root>/device_photos.json)
    device_stack_dir: str = ""
    device_step: str = ""                   # the step's short name (contacts, SETs, tgetch, ...): part of the file names
    device_step_description: str = ""       # what was done in this step (step.json "description")
    device_frames: int = 16                 # 100x / 50x main picture
    device_shift_frames: int = 8            # each shifted picture
    device_frames_low: int = 4              # 10x / 5x (one position)
    device_shifts_um: object = ((10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0))
    # objective offsets (start screen "Objective offsets", flakeid/scope/objoffsets.py): per objective, where the same
    # feature is relative to the reference objective (stage um, focus um) and its pixel matrix (stage um per pixel)
    # {"50x": {"dx": .., "dy": .., "dz": .., "M": [[..], [..]], "ref": "100x", "date": ..}, ...}
    objective_offsets: object = None
    offsets_reference: str = "100x"         # the objective the others are measured from
    device_photos_to: str = ""              # without a stack folder: where the device folders are copied; "" =
                                            # <copy_results_to>/../device_photos
    nosepiece_port: str = ""                # e.g. "COM1"; "" = no nosepiece reading
    objective_positions: object = None      # e.g. {"10x": 2, "100x": 5}
    # display
    live_scale: float = 0.5                 # live view at this fraction of the camera resolution
    # how the live views (10x and 100x) and the pictures beside them are shown: flipped left-right, flipped up-down.
    # (True, True) = the camera's image turned 180 deg ("O1", as the stage axes and the scan plans); display only (the
    # saved images stay as the camera takes them).  The live view's "Flip" boxes change it (remembered in
    # <local_root>/view_flip.json)
    view_flip: tuple = (True, True)
    # the main window's camera panel: the last picture the scan took (tiles, autofocus, 100x), what it is doing; display
    # only (no pictures of its own), at most ~2 a second.  Its "Show the camera" box changes it (remembered in
    # <local_root>/show_camera.json)
    show_camera: bool = True
    display_level: float = 0.31             # substrate brightness in the live view (as the chip viewers)


def load(path=None, **overrides) -> Settings:
    s = Settings()
    d = {}
    if path:
        d.update(json.loads(Path(path).read_text()))
    d.update({k: v for k, v in overrides.items() if v is not None})
    names = {f.name for f in fields(Settings)}
    bad = [k for k in d if k not in names]
    if bad:
        raise ValueError(f"unknown setting(s): {', '.join(bad)}")
    for k, v in d.items():
        if isinstance(getattr(s, k), tuple) and isinstance(v, list):
            v = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        setattr(s, k, v)
    return s


def dump(s: Settings) -> dict:
    return {k: (v if v == v else None) if isinstance(v, float) else v for k, v in asdict(s).items()}
