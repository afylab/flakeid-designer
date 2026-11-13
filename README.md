# vdW Stack Designer (FlakeIdentifier)

The **vdW Stack Designer** places exfoliated 2D flakes found by optical scans into van der Waals heterostructure
designs: it loads the scan results, lets you screen the flakes, plans stacks, checks overlaps and shorts, follows a
device through its fabrication steps and exports to KLayout. It runs on your own computer in the browser (or as a
desktop app); its installer sets up Python and the packages it needs. Developed in the Young Lab, UCSB.

This repository holds the designer (`flakeid_designer/`) and a copy of the FlakeIdentifier analysis code it builds its
runs with. The scan program itself, which finds and images the flakes at the microscope, is developed in
[afylab/Flake-Finder](https://github.com/afylab/Flake-Finder).

The parts of FlakeIdentifier:

1. **Scan** (microscope PC): an Olympus BX51 with an XIMEA XC10 camera and an Oasis stage, driven from Python through
   Micro-Manager's device drivers. It finds each chip's edges, focuses, scans it at 10x, and images the best flakes at
   100x (optionally the stacking domains of graphite with an IR camera).
2. **Analyse** (on the same PC, while it scans): flakes and terraces from the 10x tiles, thickness from calibrated
   optical contrast, a learned score, the 100x target list, and a zoomable chip viewer with the 100x images.
3. **Design** (any computer): the vdW Stack Designer places flakes from the scans into stacks, checks overlaps and
   shorts (this repository).

## Repository layout

| path | what |
|---|---|
| `flakeid/` | the analysis package: identification (`pipeline`, `segment`, `thickness`), live analysis during a scan (`live`), 100x export and screening (`hundred`, `screen100`), chip viewer (`zoomview`, `livecanvas`), calibration profiles (`profiles/`) |
| `flakeid/scope/` | the scan program for the microscope PC (`python -m flakeid.scope`): devices, chip finding, focus, tiles, 100x, window UI |
| `flakeid_designer/` | the vdW Stack Designer (Python standard library only; see its README.txt) |
| `tools/` | calibration, labelling, model fitting, catalog and bundle builders, repairs of built runs (`patch_thickness.py`, `reattach_100x.py`), Micro-Manager simulator (`mm_mock/`) |
| `tools/win7/` | the microscope PC (Windows 7, Python 3.8): offline install kit, launcher, Micro-Manager and nosepiece tests |
| `tools/designer_app/` | builds of the designer as a desktop app (`build_mac.sh`, `build_windows.bat`) and its copy to the share |
| `tools/archive/` | one-off analyses and labelling-set makers from development, kept for reference (see its README) |
| `tests/` | tests of the designer and its builders (`python -m pytest tests`) |
| `flats/` | flat fields (not in git; see `flats/README.md`) |
| `docs/scan_guide.md` | **how to scan**: step-by-step use of the scan program, common mistakes, setup and calibration |
| `docs/stack_workflow.md` | the designer's stack folders and fabrication steps (formats) |
| `docs/manual.md` | the detailed original manual (partly out of date) |

Data (scans, images, results, label sets) is not in the repository.

## Quick starts

**Analysis on a workstation** (Python 3.9+):

```sh
pip install -r requirements.txt
python find_flakes.py IMAGE hbn 285 --annotate          # one stitched overview image
python -m flakeid.live RUN_FOLDER -o RESULTS            # a Micro-Manager scan (tiles), live or afterwards
python -m flakeid.hundred RUN/<piece>_100x --viewer RESULTS/<piece>/<piece>_tiles_viewer   # 100x images into a viewer
```

**Microscope PC**: build the offline kit on a Mac or Linux machine with the flat fields in `flats/`
(`sh tools/win7/make_kit.sh`), copy `scope_pc_kit/` to the PC and run `setup_scope_pc.bat` there. It installs Python 3.8
into `C:\FlakeID`, creates a desktop shortcut and pins it to the taskbar. Settings: `C:\FlakeID\scan_settings.json`
(see `flakeid/scope/settings.py`). Useful commands: `C:\FlakeID\run_flakeid.bat check`, `... mmtest`, `... objtest`, `... screenshots`. Using it: [docs/scan_guide.md](docs/scan_guide.md).
Later versions install from GitHub with **Update...** on the start screen (or `run_flakeid.bat update`); only new
packages need a new kit. A private fork as `update_repo` needs a read-only token in `C:\FlakeID\github_token.txt`.

**Designer**: double-click `flakeid_designer/Install shortcut (Windows).bat` or `Install shortcut (Mac).command`, then
open the icon and add your own scan-results folder in *Scan results*. They work on a fresh computer: without Python
3.10-3.13 they install Python 3.12 from python.org (Windows: for this user, no admin; Mac: it asks for an administrator's
password), then pip and the packages (`requirements.txt`) in the designer's own environment `~/.flakeid_designer_venv`,
then the shortcut; run again after an update that changes `requirements.txt`. The log: `~/.flakeid_designer_install_log.txt`.
On a Mac that says the file cannot be opened: right-click it, Open (a downloaded file); or, if it "does not have
access privileges", type `bash ` in Terminal, drag the file in and press Return. Each run's `designer/` folder (written by the scan
program) is a run you can load. See `flakeid_designer/README.txt`.

**Device design in KLayout**: place microscope pictures of a finished stack onto the chip layout by its alignment
marks (any objective, no scale needed; pictures are assumed upright, within 45 deg), then open the session in KLayout
(*File > Restore Session*) and draw. In the designer: *Export to KLayout*. From the command line:

```sh
python3 flakeid_designer/marks_to_klayout.py <folder of pictures> \
    --gds <your layout>.gds [--add-to existing.lys] [--lyp layers.lyp]
```

It writes `<folder>/<name>_klayout/<name>.lys` (layout + placed pictures, with your `.lyp` layer properties, remembered
in `~/.flakeid_designer.json`), `<name>.gds` (the layout plus found marks and picture frames on layers 250-251, which
start greyed out (invalid) in the layer list; flake outlines on 252 only with `--outline-max-um-px`), `<name>_check.jpg` and `<name>_fit.json`; `--add-to` also adds it as a new tab to an existing session
(after a backup copy). **Update the pictures of a session** (the dialog's second choice, `--replace-pictures SESSION.lys`)
swaps an existing session's pictures for the latest fabrication step's (re-aligned): the session's layout, layers and
view stay, no `.gds` is written. A picture taken another way round: `name.jpg@180`. Needs `numpy`, `opencv-python`, `gdstk`.

## Development

- **Microscope PC compatibility.** Everything the microscope PC runs (`flakeid/`, `flakeid/scope/`, `tools/win7/`,
  `tools/make_designer_bundle.py`, `tools/make_catalog_100x.py`) must work on Windows 7 with Python 3.8: check the syntax
  with `ast.parse(source, feature_version=(3, 8))` (no `X | Y` types at run time, no f-strings reusing their own quote
  character, no `match`), pass `str` (not `Path`) to `np.memmap`, write text as UTF-8 explicitly, and expect Windows file
  locking (virus scanner, indexer): retry renames, never let a status or log write stop a run.
- **Rebuild the kit** after such changes: `sh tools/win7/make_kit.sh` (needs the flat fields in `flats/`), then
  `python3 tools/win7/check_kit.py scope_pc_kit/wheels tools/win7/requirements-py38.txt tools/win7/requirements-mm-py38.txt`
  must report nothing missing.
- **Designer.** `flakeid_designer/` uses the Python standard library only. Its `index.html` is served by the local
  server (`window.FLAKEID_LOCAL`); the same page also runs as a claude.ai artifact (`window.claude`), the lab's own
  hosted copy, which outside users do not need. Test both modes after changing it.
- **Tests.** The Micro-Manager simulator tests the scan program without hardware:
  `python tools/mm_mock/run_mock_py.py` (options: `--graphite`, `--mounted`, `--nosepiece`, `--partial`, `--mosaic`,
  `--stage-timeout`, ...; it needs the example chip pictures as `example images/` beside the repository's code). The
  designer: `python -m pytest tests`; web pages (viewer, designer) are checked in real browsers with Playwright.
- **Data** (scans, pictures, results, label sets, flat fields) stays outside the repository.

## License

MIT (see [LICENSE](LICENSE)). `flakeid/vendor/` contains OpenSeadragon (BSD-3-Clause) and `flakeid/scope/theme_sv/` the
Sun Valley ttk theme (MIT), each with its own license.
