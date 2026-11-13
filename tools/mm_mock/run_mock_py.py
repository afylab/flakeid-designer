"""Run the Python scan (flakeid.scope) in multi mode against the Python simulator (flakeid.scope.mock).

The scenario and the scoring are those of run_mock_multi.py (the BeanShell script's multi-piece test): three real chip
scans (bn24/25/26) on the stage at their own positions, rotations and heights, on a tilted and bowed surface; demo 100x
images placed on each chip as the "flakes"; a fake analysis (this harness) writes <piece>_targets.csv once a piece's
tiles are on disk (the last one late).  The simulated user focuses roughly, switches to 100x when asked, and (with
--graphite) puts in the stacking-domain configuration.  PASS needs: every chip covered by its tiles, tile focus within
4 um of the true surface, every flake imaged at 100x within 0.6 um of focus, (graphite) every flake of 3+ layers
imaged with the domain camera within 0.6 um, piece names continuing the folder's numbering, run file state "done".

    python tools/mm_mock/run_mock_py.py [--graphite [--domain-offset-known]] [--materials graphite:90,hbn:285] [--glitch N]

--mosaic: each chip also gets a large flake (3 x 2 demo 100x pictures, ~470 x 240 um) whose terrace outline
(terrace_poly_um, ~380 x 200 um) is in the target list, and the other targets small outlines (one field): PASS also
needs the large ones' mosaics (fields as planned, within hundred_mosaic_max_tiles, focus within 0.6 um; with --graphite
their IR mosaics too) and none for the small ones, then flakeid.hundred.export of each piece into a small viewer
stitching them (<main>_mosaic.jpg attached as "mos", IR "irmos").  --mosaic-budget MIN: the run's mosaic budget
(hundred_mosaic_budget_min and domain_mosaic_budget_min): PASS then needs it respected (the budget said used up).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from flakeid.background import detect_chip  # noqa: E402
from flakeid.scope.mock import MockCore  # noqa: E402
from flakeid.scope.scan import Scanner  # noqa: E402
from flakeid.scope.settings import load  # noqa: E402
from flakeid.scope.ui import AutoUI  # noqa: E402
from run_mock_multi import M10, OFF100, OFFD, UM, UM100, ZOFF100, ZOFFD, img_to_stage, stage_to_img  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/multi_py"))
    ap.add_argument("--tilt", nargs=2, type=float, default=(2.0, -1.5), help="um/mm")
    ap.add_argument("--bow", type=float, default=1.5, help="um of bow 5 mm from the bow centre")
    ap.add_argument("--patches", type=int, default=3)
    ap.add_argument("--late", type=float, default=25.0)
    ap.add_argument("--chips", default="bn24,bn25,bn26")
    ap.add_argument("--tape", choices=("amber", "clear"), help="tape over the first chip's +x +y corner, 40 um above it "
                    "(amber: its own colour; clear: the substrate's): focus points must move off it or be left out")
    ap.add_argument("--gap", type=float, help="the second chip this many um to the +x side of the first (both square to the "
                    "stage; 0: touching): the edge search must stop at the next chip (a narrow gap, or the user marks the edge)")
    ap.add_argument("--mounted", action="store_true", help="mounted 1 cm chips (chip_type mounted_1cm, scaled to a 4 mm nominal "
                    "side): 4 plain chips cut by hand (sides 3.3-4.7 mm, turned up to 2 deg) on 2 blocks, a few hundred um "
                    "apart, amber tape over the outer edge of each block; centres marked up to ~0.4 mm off.  PASS also needs "
                    "every chip covered (less the taped band) and the run file's blocks")
    ap.add_argument("--graphite", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="the scan's detailed log (each autofocus)")
    ap.add_argument("--ir-blur", type=float, help="the domain camera's defocus blur (px per um; default the simulator's 2.5): "
                    "0.6 is the real IR camera's (its pictures change little over +-4 um, graphite_20261019_run_1)")
    ap.add_argument("--domain-offset-known", action="store_true")
    ap.add_argument("--glitch", type=int, default=-1, help="a camera garbage frame in the Nth glitch-checked image (tiles, 100x)")
    ap.add_argument("--materials")
    ap.add_argument("--mosaic", action="store_true", help="large terraces: 100x (and IR) mosaics, then the export stitches them")
    ap.add_argument("--mosaic-budget", type=float, help="hundred_mosaic_budget_min / domain_mosaic_budget_min for the run (minutes)")
    ap.add_argument("--mosaic-max", type=int, help="hundred_mosaic_max_tiles for the run")
    ap.add_argument("--real-time", action="store_true", help="keep the focus settle wait after every Z move (20 ms; the "
                                                             "simulated camera does not need it, so by default it is 0)")
    ap.add_argument("--stage-timeout", action="store_true", help="every stage move longer than 5 mm raises a Micro-Manager "
                                                                 "style timeout once (the stage still arrives)")
    ap.add_argument("--partial", action="store_true", help="fake analysis also writes provisional lists (<piece>_targets_partial.csv, "
                                                           "half the targets) and the final list --late s later, for every piece")
    ap.add_argument("--nosepiece", action="store_true", help="simulated coded nosepiece (10x at 2, 100x at 5)")
    ap.add_argument("--miss", type=float, default=0.0, help="every other target's stage position is wrong by 20..MISS um "
                    "(a mapping error); the scan must re-centre on the flake from its 10x picture (PASS: every 100x image "
                    "within 15 um of its flake)")
    ap.add_argument("--no-crops", action="store_true", help="the fake analysis writes no 10x target pictures")
    ap.add_argument("--stage-short", action="store_true", help="at 100x, a -y move longer than 1 mm stops 100 um short and "
                    "any -y move leaves 6 um (as the real stage did, 2026-10-15); the stage reports where it was sent")
    ap.add_argument("--approach", type=float, help="xy_approach_um for the scan (default: its setting)")
    ap.add_argument("--no-recentre", action="store_true", help="hundred_recentre off (to test the approach alone)")
    ap.add_argument("--z-slip", type=float, default=0.0, help="the focus drive slips by this much (um) before the --slip-at-th "
                    "100x flake (as 20261014_run_1: 40-60 um, outside the +-15 um wide search); scored against the slipped focus")
    ap.add_argument("--slip-at", type=int, default=0, help="100x flake (1 = the first of the run) before which the drive slips "
                    "(default: the first flake of the second piece)")
    ap.add_argument("--z-floor", type=float, help="the focus drive stops at this Z (um above the simulated focus at the first chip's centre; negative: below it), as the Oasis drive at -181.9 um in BN_15_auto 20261025_run_4 (PASS: the limit logged and reported)")
    ap.add_argument("--absent", action="store_true", help="nobody at the microscope: every focus question goes unanswered (each "
                    "counted as focus_dialog_timeout_s of microscope time in the estimate)")
    ap.add_argument("--stage-xmax", type=float, help="the stage's travel ends this far (um) in +x from the right-most chip's centre "
                    "(it stops there, as the real stage at its limit): that chip is scanned up to it (PASS: the reachable part "
                    "covered, no move beyond, the flakes in reach imaged)")
    ap.add_argument("--slack", action="store_true", help="Slack bot messages to a stand-in Slack server (tools/mm_mock/slack_standin.py): "
                    "PASS also needs the run's thread, its replies (each chip's plan picture, the end) in the DM and the channel")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    run_dir = out / "run"
    run_dir.mkdir(parents=True)
    today = datetime.date.today().strftime("%Y%m%d")
    (run_dir / f"{today}_1_tiles").mkdir()            # an earlier scan: the new pieces must start at index 2

    names = a.chips.split(",")
    poses = [(50000.0, 20000.0, 2.0, 0.0), (66000.0, 21000.0, -3.0, 25.0), (58000.0, 35000.0, 4.0, -15.0)][: len(names)]
    SIZES, TAPES = {}, []                              # --mounted: chip -> (w, h) um; tape rectangles (x0, x1, y0, y1)
    if a.mounted:
        names = [names[0]] * 4
        SIZES = {0: (4300.0, 3600.0), 1: (3500.0, 4100.0), 2: (4700.0, 4500.0), 3: (3800.0, 3300.0)}
        poses = [(50000.0, 20000.0, 1.5, 0.0), (50000.0 + 2150 + 400 + 1750, 20100.0, -1.0, 3.0),     # block 1
                 (62000.0, 21000.0, 0.5, -4.0), (62000.0 + 2350 + 550 + 1900, 20800.0, -2.0, 2.0)]     # block 2
        TAPES = [(50000 - 2150 - 900.0, 50000 - 2150 + 250.0, 17000.0, 23000.0),        # over chip 1's -x edge
                 (62000 + 2350 + 550 + 3800 - 250.0, 62000 + 2350 + 550 + 3800 + 900.0, 18000.0, 24000.0)]   # chip 4's +x edge
    if a.gap is not None and len(names) >= 2:          # (two plain rectangular chips, `gap` um apart along x: exact edges)
        CW, CH = 4400.0, 3900.0                         # um
        p0 = (50000.0, 20000.0, 0.0, 0.0)
        poses = [p0, (p0[0] + CW + a.gap, 20000.0, 0.0, 0.0)] + poses[2:len(names)]
        GAPX = p0[0] + CW / 2.0                         # (the first chip's +x edge: what the user marks)
        print(f"chips {a.gap:g} um apart: the first ends at x = {GAPX:.0f}, the second starts at {GAPX + a.gap:.0f}")
    rng = np.random.default_rng(7)
    demo = [f for f in sorted(glob.glob(str(ROOT / "example images/demo_high_magnification/*/*_100x.jpg"))) if " X" not in f]
    chips, centres, patches, masks, images = [], [], [], [], []
    big = {}                                           # --mosaic: patch index -> its (large) picture
    k_demo = 0
    for i_ch, (nm, pose) in enumerate(zip(names, poses)):
        path = ROOT / "example images" / f"{nm}.jpg"
        im = cv2.imread(str(path))
        if a.mounted:                                  # (a rectangle cut from the real chip's middle: its texture and flakes,
            CW, CH = SIZES[i_ch]                        # straight edges; each chip a different part of it)
            w_, h_ = int(CW / UM), int(CH / UM)
            x_, y_ = im.shape[1] // 2 - w_ // 2 + (i_ch - 1) * 700, im.shape[0] // 2 - h_ // 2 + (i_ch % 2) * 500 - 250
            im = np.ascontiguousarray(im[y_:y_ + h_, x_:x_ + w_])
        if a.gap is not None and i_ch < 2:             # (a plain chip: the real one's substrate colour, edge to edge)
            sub = np.median(im[im.shape[0] // 3: 2 * im.shape[0] // 3, im.shape[1] // 3: 2 * im.shape[1] // 3].reshape(-1, 3), axis=0)
            im = np.clip(np.full((int(CH / UM), int(CW / UM), 3), sub) + np.random.default_rng(i_ch).normal(0, 3, (int(CH / UM), int(CW / UM), 3)), 0, 255).astype(np.uint8)
        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sc = 0.125
        small = cv2.cvtColor(cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        m = detect_chip(small.astype(np.float32), 0)
        masks.append((m, sc, im.shape))
        ys, xs = np.nonzero(m)
        centres.append(img_to_stage(np.array([xs.mean(), ys.mean()]) / sc, im.shape, pose)[0])
        chips.append((path, pose))
        er = cv2.erode(m.astype(np.uint8), np.ones((90, 90), np.uint8))
        yy, xx = np.nonzero(er)
        for _ in range(a.patches):
            j = rng.integers(len(yy))
            pc = img_to_stage(np.array([xx[j], yy[j]]) / sc, im.shape, pose)[0]
            patches.append((nm, pc, demo[k_demo % len(demo)])); k_demo += 1
        if a.mosaic:                                   # one large flake per chip (its picture: 3 x 2 demo pictures)
            j = rng.integers(len(yy))
            pc = img_to_stage(np.array([xx[j], yy[j]]) / sc, im.shape, pose)[0]
            big[len(patches)] = np.vstack([np.hstack([cv2.resize(cv2.imread(demo[(k_demo + 3 * r + c) % len(demo)]), (1376, 1038))
                                                      for c in range(3)]) for r in range(2)])
            k_demo += 6
            patches.append((nm, pc, None))
    tx, ty = a.tilt[0] / 1000, a.tilt[1] / 1000
    bowc = np.mean(centres, 0)
    bowK = a.bow / 5000.0 ** 2
    z0 = -90.0 - tx * centres[0][0] - ty * centres[0][1]

    def chip_dz(x, y):
        for (path, pose), (m, sc, shape) in zip(chips, masks):
            u = stage_to_img(np.array([x, y]), shape, pose)[0]
            if 0 <= u[0] < shape[1] and 0 <= u[1] < shape[0]:
                return pose[3]
        return 0.0

    def true_focus(x, y, obj=10):
        sx, sy = (x - OFF100[0], y - OFF100[1]) if obj == 100 else (x - OFFD[0], y - OFFD[1]) if obj == "domain" else (x, y)
        z = z0 + tx * x + ty * y + bowK * ((x - bowc[0]) ** 2 + (y - bowc[1]) ** 2) + chip_dz(sx, sy)
        return z + (ZOFF100 if obj == 100 else ZOFFD if obj == "domain" else 0.0)
    thick = {}
    pcs = [[c[0] + o[0], c[1] + o[1], true_focus(c[0] + o[0], c[1] + o[1]) + rng.normal(0, 3)]
           for c, o in zip(centres, rng.normal(0, 200 if a.mounted else 300, (len(centres), 2)))]

    # the simulated microscope
    first = centres[0]
    core = MockCore(UM, M10, first[0], first[1], seed=1)
    MMD = np.asarray(core.MD, float)
    if a.ir_blur:
        core.blurD = a.ir_blur
    for img, (path, pose) in zip(images, chips):
        core.add_chip(img, pose[0], pose[1], pose[2], pose[3])
    if a.tape:                                         # (first in the list: it covers the chip and sets the focus there)
        m0, sc0, shp0 = masks[0]
        ys0, xs0 = np.nonzero(m0)
        Pm = img_to_stage(np.c_[xs0, ys0] / sc0, shp0, chips[0][1])
        xa, xb, ya, yb = Pm[:, 0].min(), Pm[:, 0].max(), Pm[:, 1].min(), Pm[:, 1].max()
        tx0, ty0 = xb - 0.35 * (xb - xa), yb - 0.35 * (yb - ya)
        tw, th = (xb + 1000 - tx0), (yb + 1000 - ty0)
        sub0 = np.median(images[0][images[0].shape[0] // 3: 2 * images[0].shape[0] // 3, images[0].shape[1] // 3: 2 * images[0].shape[1] // 3].reshape(-1, 3), axis=0)
        col = np.array([205.0, 150.0, 60.0]) if a.tape == "amber" else np.clip(sub0 * 1.02, 0, 255)
        timg = np.clip(np.full((int(th / UM), int(tw / UM), 3), col) + np.random.default_rng(5).normal(0, 3, (int(th / UM), int(tw / UM), 3)), 0, 255).astype(np.uint8)
        core.chips.insert(0, (timg, tx0 + tw / 2.0, ty0 + th / 2.0, 0.0, 40.0))
        TAPE = (tx0, ty0)
        print(f"{a.tape} tape over the first chip's corner: x > {tx0:.0f}, y > {ty0:.0f} (chip {xa:.0f}..{xb:.0f}, {ya:.0f}..{yb:.0f}), 40 um high")
    for (x0_, x1_, y0_, y1_) in TAPES:               # (--mounted: amber tape over a block's outer edge, 40 um above)
        timg = np.clip(np.full((int((y1_ - y0_) / UM), int((x1_ - x0_) / UM), 3), (205.0, 150.0, 60.0)) +
                       np.random.default_rng(6).normal(0, 3, (int((y1_ - y0_) / UM), int((x1_ - x0_) / UM), 3)), 0, 255).astype(np.uint8)
        core.chips.insert(0, (timg, (x0_ + x1_) / 2.0, (y0_ + y1_) / 2.0, 0.0, 40.0))
        print(f"amber tape x {x0_:.0f}..{x1_:.0f}, y {y0_:.0f}..{y1_:.0f}")
    for j, (_, pc, f) in enumerate(patches):
        core.add_patch(cv2.cvtColor(big[j] if j in big else cv2.imread(f), cv2.COLOR_BGR2RGB), pc[0], pc[1])
    core.patch_um_per_px = UM100
    core.zoff100, core.off100 = ZOFF100, OFF100
    core.tilt, core.z0, core.bowK, core.bowc = np.array([tx, ty]), z0, bowK, bowc
    if a.gap is not None:
        edges_asked = []

        def mock_edge(sx, sy, dx, dy):                  # the simulated user puts the cross on the asked edge
            edges_asked.append((dx, dy))
            if dx > 0:
                core.setXYPosition(GAPX - 10.0, sy)
            elif dx < 0:
                core.setXYPosition(GAPX + a.gap + 10.0, sy)
        core.mockEdge = mock_edge
    core.z = true_focus(*first) + 2.0
    xmax = None
    if a.stage_xmax is not None:
        xmax = max(q[0] for q in centres) + a.stage_xmax     # (beyond the right-most chip: the others are reachable)
        _set_lim = core.setXYPosition

        def set_xy_lim(*args):                         # the stage stops at the end of its travel
            _set_lim(args[0], min(float(args[-2]), xmax), float(args[-1]))
        core.setXYPosition = set_xy_lim
    if a.stage_timeout:
        _set, _wait, last = core.setXYPosition, core.waitForDevice, {"d": 0.0}

        def set_xy(dev, x, y):
            last["d"] = float(np.hypot(x - core.getXPosition(dev), y - core.getYPosition(dev)))
            _set(dev, x, y)

        def wait_dev(dev):
            _wait(dev)
            if last["d"] > 5000:
                last["d"] = 0.0
                core.n_timeouts = getattr(core, "n_timeouts", 0) + 1
                raise RuntimeError("Error in device \"XYStage\": Wait for device \"XYStage\" timed out after 5000ms")
        core.setXYPosition, core.waitForDevice = set_xy, wait_dev

    if a.z_floor is not None:
        _setz0, zfloor = core.setPosition, -90.0 + a.z_floor     # (the simulated focus is -90 um at the first chip's centre)

        def set_z_floor(*args):
            _setz0(*args[:-1], max(float(args[-1]), zfloor * core.z_scale))
        core.setPosition = set_z_floor
        print(f"HARNESS focus drive floor {a.z_floor:+.1f} um from the simulated focus at the first chip's centre", flush=True)

    actual = {}                                       # commanded (rounded) -> where the simulated stage really went
    if a.stage_short:
        _set0 = core.setXYPosition
        cmd = {"x": core.x, "y": core.y}

        def set_xy_short(*args):
            x, y = float(args[-2]), float(args[-1])
            dy = y - cmd["y"]
            err = (100.0 if dy < -1000.0 else 6.0 if dy < 0 else 0.0) if core.objective == 100 else 0.0
            cmd.update(x=x, y=y)
            _set0(*args[:-2], x, y + err)
            actual[(round(x, 1), round(y, 1))] = (x, y + err)
        core.setXYPosition = set_xy_short
        core.getXPosition = lambda *args: cmd["x"]
        core.getYPosition = lambda *args: cmd["y"]

    # each flake's 10x picture as the analysis writes it (live.write_target_crops: 360 px, centred on the flake, the
    # 100x field's box and a cross drawn on it), rendered from the simulated sample before the scan starts
    crop_img = {}
    if not a.no_crops:
        x_, y_, ob_ = core.x, core.y, core.objective
        core.objective = 10
        for j, (nm_, pc, f) in enumerate(patches):
            core.x, core.y = float(pc[0]), float(pc[1])
            im = cv2.cvtColor(np.clip(core._render_affine(360, 360, np.asarray(M10, float)), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            fw, fh = core.W0 / 9.54, core.H0 / 9.54
            cv2.rectangle(im, (int(180 - fw / 2), int(180 - fh / 2)), (int(180 + fw / 2), int(180 + fh / 2)), (0, 190, 255), 2)
            cv2.drawMarker(im, (180, 180), (0, 190, 255), cv2.MARKER_CROSS, 14, 1)
            crop_img[j] = im
        core.x, core.y, core.objective = x_, y_, ob_
    true100 = {}                                       # label -> the stage position that centres the flake at 100x

    # fake analysis: target lists once a piece's tiles are complete
    stop = threading.Event()
    written = {}
    polys, bigl = {}, set()

    def thk(lab, k):
        return 12.0 if lab in bigl else [0.67, 3.4, 12.0][k % 3]      # (the large flakes: thick enough for IR pictures)

    def analysis():
        while not stop.is_set():
            for mf in run_dir.glob("*_run_*.json"):
                try:
                    man = json.loads(mf.read_text())
                except Exception:
                    continue
                for i, p in enumerate(man["pieces"]):
                    nm = p["name"]
                    if nm in written:
                        continue
                    pos, tiles = run_dir / f"{nm}_scan.pos", run_dir / p["tiles"]
                    if not pos.exists() or not tiles.exists():
                        continue
                    n_plan = len(json.loads(pos.read_text())["map"]["StagePositions"]["array"])
                    if len(list(tiles.glob("*/metadata.txt"))) < n_plan:
                        continue
                    c = np.array(p["center"])
                    chip = names[int(np.argmin([np.hypot(*(c - q)) for q in centres]))]
                    mine = [(j, q) for j, q in enumerate(patches) if q[0] == chip]
                    rows = []
                    for k, (j, (n_, pc, f)) in enumerate(mine):
                        lab = f"{k + 1:02d}_F{k + 1:03d}_{chip}"
                        true100[lab] = pc + OFF100
                        err = np.random.default_rng(k + 100).normal(0, 3, 2)
                        if a.miss > 0 and k % 2 == 0:             # a mapping error: the target is off the flake
                            g_ = np.random.default_rng(k + 200)
                            ang, r_ = g_.uniform(0, 2 * np.pi), g_.uniform(20.0, a.miss)
                            err = err + r_ * np.array([np.cos(ang), np.sin(ang)])
                        rows.append((lab, pc + OFF100 + err))
                        if a.mosaic:                   # the terrace outline (stage um, as live.write_targets writes it)
                            from flakeid.fieldmosaic import format_poly
                            ab = (190.0, 100.0) if j in big else (40.0, 30.0)
                            tt = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                            polys[lab] = format_poly(pc + OFF100 + np.c_[ab[0] * np.cos(tt), ab[1] * np.sin(tt)])
                            if j in big:
                                bigl.add(lab)
                        if j in crop_img:
                            (run_dir / f"{nm}_targets_crops").mkdir(exist_ok=True)
                            cv2.imwrite(str(run_dir / f"{nm}_targets_crops" / f"{lab}.jpg"), crop_img[j], [cv2.IMWRITE_JPEG_QUALITY, 88])
                    if a.partial and "_part_" + nm not in written:
                        tmp = run_dir / f"{nm}_targets_partial.csv.tmp"
                        with open(tmp, "w", newline="") as fh:
                            w = csv.writer(fh)
                            w.writerow(["order", "label", "stage_x_um", "stage_y_um", "z10_um", "thickness_nm"] + (["terrace_poly_um"] if a.mosaic else []))
                            for k, (lab, xy) in enumerate(rows[:(len(rows) + 1) // 2]):
                                w.writerow([k + 1, lab, f"{xy[0]:.2f}", f"{xy[1]:.2f}", "", thk(lab, k)] + ([polys.get(lab, "")] if a.mosaic else []))
                        tmp.replace(run_dir / f"{nm}_targets_partial.csv")
                        written["_part_" + nm] = time.time()
                        print(f"HARNESS wrote {nm}_targets_partial.csv ({(len(rows) + 1) // 2} targets)", flush=True)
                    if (a.partial or i == len(man["pieces"]) - 1) and a.late > 0:
                        written.setdefault("_late_" + nm, time.time())
                        if time.time() - written["_late_" + nm] < a.late:
                            continue
                    rows = [(lab, xy + rng.normal(0, 1, 2)) for lab, xy in rows]      # (re-detected: slightly different centres)
                    tmp = run_dir / f"{nm}_targets.csv.tmp"
                    with open(tmp, "w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["order", "label", "stage_x_um", "stage_y_um", "z10_um", "thickness_nm"] + (["terrace_poly_um"] if a.mosaic else []))
                        for k, (lab, xy) in enumerate(rows):
                            th = thk(lab, k)
                            thick[lab] = th
                            w.writerow([k + 1, lab, f"{xy[0]:.2f}", f"{xy[1]:.2f}", "", th] + ([polys.get(lab, "")] if a.mosaic else []))
                    tmp.rename(run_dir / f"{nm}_targets.csv")
                    written[nm] = time.time()
                    print(f"HARNESS wrote {nm}_targets.csv ({len(rows)} targets)", flush=True)
            time.sleep(2)

    threading.Thread(target=analysis, daemon=True).start()

    pm = [(m.split(":")[0], float(m.split(":")[1])) for m in a.materials.split(",")] if a.materials else None
    if pm:
        a.graphite = any(m == "graphite" for m, _ in pm)
    s = load(None, mode="multi", out_dir=str(out), run_dir=str(run_dir), run_name="", piece_centers=pcs, af_manual_fallback="never",
             targets_wait_s=900.0, focus_mode="auto", material="graphite" if a.graphite else "hbn", domain_camera="DomainCam",
             domain_camera_props=(), domain_exposure_ms=4.0, piece_materials=[list(q) for q in pm] if pm else None,
             domain_xy_offset_um=tuple(OFFD - OFF100), verbose=a.verbose, local_analysis=False)
    if a.mounted:                                      # (mounted 1 cm chips, scaled: 4 mm nominal, centres within 0.4 mm)
        s.chip_type, s.chip_nominal_mm, s.chip_size_tol, s.chip_centre_err_mm, s.chip_focus_inset_mm = "mounted_1cm", 4.0, 0.3, 0.4, 0.6
        s.piece_mounts = [1, 1, 2, 2]
        corner = lambda i, sx, sy: img_to_stage(np.array([(0 if sx < 0 else images[i].shape[1] - 1), (0 if sy < 0 else images[i].shape[0] - 1)], float),
                                                images[i].shape, chips[i][1])[0].tolist()     # noqa: E731
        s.mount_corners = {"1": [corner(0, -1, -1), corner(0, -1, 1), corner(1, 1, 1)], "2": [corner(2, -1, -1), corner(3, 1, -1), corner(3, 1, 1)]}
    if not a.real_time:
        s.z_settle_ms = 0.0
    if a.approach is not None:
        s.xy_approach_um = a.approach
    if a.no_recentre:
        s.hundred_recentre = False
    if a.mosaic:                                       # (the mock IR camera, frames flipped in x and y: as the domain test measures it)
        s.domain_px_um_matrix = (-MMD).tolist()
    if a.mosaic_budget is not None:
        s.hundred_mosaic_budget_min = s.domain_mosaic_budget_min = a.mosaic_budget
    if a.mosaic_max:
        s.hundred_mosaic_max_tiles = a.mosaic_max
    if a.domain_offset_known:
        s.domain_focus_offset_um = float(ZOFFD - ZOFF100)
    slack = None
    if a.slack:
        sys.path.insert(0, str(ROOT / "tools" / "mm_mock"))
        import os
        import slack_standin
        from flakeid.scope import notify
        slack = slack_standin.start(8814, users={"op@lab.example": "U01OPERATR"}, names={"U01OPERATR": "Test Operator"})
        notify.SLACK_API = "http://127.0.0.1:8814/api/"
        os.environ["FLAKEID_SLACK_TOKEN"] = "xoxb-test"
        s.notify_to, s.notify_slack_channel, s.local_root = "Test Operator", "#microscope", str(out)
    ui = AutoUI(core, analysis=False)
    t_start = time.time()
    sc = Scanner(core, s, ui)
    # what the operations would cost on the microscope (20261014_run_1: a frame 117 ms, an XY move 442 ms with its
    # readback, a Z move 38 ms + z_settle_ms; an unanswered focus question focus_dialog_timeout_s): the simulator has no
    # time of its own, so this estimate is what to compare between versions
    cnt = dict(xy=0, z=0, rd=0, ask=0, flakes=0)
    _sxy, _sz, _gx, _gy = core.setXYPosition, core.setPosition, core.getXPosition, core.getYPosition

    def c_sxy(*args):
        cnt["xy"] += 1
        return _sxy(*args)

    def c_sz(*args):
        cnt["z"] += 1
        return _sz(*args)

    def c_gx(*args):
        cnt["rd"] += 1
        return _gx(*args)
    core.setXYPosition, core.setPosition, core.getXPosition = c_sxy, c_sz, c_gx
    if a.absent:
        s.af_manual_fallback = "first"               # (the scan's default: asked on a piece's first flakes)

        def ask_absent(*args, **kw):
            cnt["ask"] += 1
            return 3
        ui.ask_focus = ask_absent
    slip = {"on": False, "labels": set()}
    if a.z_slip:
        _imf = sc.image_flake

        def image_flake(P, t, tag, **kw):
            cnt["flakes"] += 1
            at = a.slip_at or (a.patches + 1)
            if not slip["on"] and cnt["flakes"] >= at:
                slip["on"] = True
                core.z0 += a.z_slip                  # (the drive reads the same; the sample is now this much further)
                print(f"HARNESS focus drive slipped by {a.z_slip:+.1f} um before 100x flake {cnt['flakes']} ({t[4]})", flush=True)
            if slip["on"]:
                slip["labels"].add((sc.s.name, t[4]))
            return _imf(P, t, tag, **kw)
        sc.image_flake = image_flake
    if a.glitch >= 0:
        # the camera garbage frame goes into the glitch-checked image number a.glitch (a tile or a 100x frame), so the
        # check and the retake are exercised (a fixed frame number lands in autofocus / edge frames, which are not checked)
        _sg, n_sg = sc.snap_good, {"n": 0}

        def snap_good(what):
            n_sg["n"] += 1
            if n_sg["n"] == a.glitch:
                core.glitch_at = core.snaps + 1
            return _sg(what)
        sc.snap_good = snap_good
    if a.nosepiece:
        from flakeid.scope.nosepiece import MockNosepiece
        s.objective_positions = {"10x": 2, "100x": 5, "domain": 3}
        sc.nose = MockNosepiece(core, s.objective_positions)
    err = None
    try:
        sc.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        err = e
    stop.set()
    if slack is not None:                              # (the scan program sends the end of each job: notify.job_end)
        from flakeid.scope import notify
        notify.job_end(sc, err is None, False, 99.0)
        notify.flush(60)
    print(f"stage timeouts simulated: {getattr(core, 'n_timeouts', 0)}")
    print(f"run time {time.time() - t_start:.0f} s, {core.snaps} snaps{'' if err is None else ', ERROR ' + str(err)}")
    from flakeid.scope.settings import Settings
    zst = Settings().z_settle_ms                       # (the microscope's setting; the simulation runs without the wait)
    est = sc.n_snaps * 0.117 + cnt["xy"] * 0.442 + cnt["z"] * (0.038 + zst / 1000.0) + cnt["ask"] * s.focus_dialog_timeout_s
    print(f"ESTIMATE microscope time {est / 60:.1f} min: {sc.n_snaps} frames, {cnt['xy']} XY moves, {cnt['z']} Z moves, "
          f"{cnt['rd']} X readbacks, {cnt['ask']} unanswered focus questions, autofocus {sc.opn.get('autofocus', 0)} calls "
          f"(at 117 ms / frame, 442 ms / XY move, 38 ms + {zst:g} ms / Z move, {s.focus_dialog_timeout_s:g} s / question)")

    # ---- score (as run_mock_multi.py)
    s_margin = 20.0                                    # (stage_limit_margin_um)
    man = json.loads(next(run_dir.glob("*_run_*.json")).read_text())
    print("manifest state:", man["state"], [(p["name"], p["state"]) for p in man["pieces"]])
    ok = err is None
    for p in man["pieces"]:
        nm = p["name"]
        c = np.array(p["center"])
        ci = int(np.argmin([np.hypot(*(c - q)) for q in centres]))
        m, scl, shape = masks[ci]
        pose = chips[ci][1]
        pos = json.loads((run_dir / f"{nm}_scan.pos").read_text())["map"]["StagePositions"]["array"]
        T = np.array([q["DevicePositions"]["array"][0]["Position_um"]["array"] for q in pos])
        chipj = json.loads((run_dir / f"{nm}_chip.json").read_text())
        fov = np.array(chipj["field_um"])
        cover = np.zeros(m.shape, np.uint8)
        for t in T:
            cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov / 2 + t
            cv2.fillPoly(cover, [np.round(stage_to_img(cs, shape, pose) * scl).astype(np.int32)], 1)
        if xmax is not None:                           # only what the stage can reach (tile centres up to the limit, less the margin)
            yy_, xx_ = np.nonzero(m)
            st_ = img_to_stage(np.c_[xx_, yy_] / scl, shape, pose)
            reach_ = st_[:, 0] <= xmax - s_margin + fov[0] / 2 * 0.95
            covered = float((cover[yy_[reach_], xx_[reach_]] > 0).mean()) if reach_.any() else 1.0
            beyond = float(T[:, 0].max()) > xmax + 0.01
            print(f"{nm}: stage limit x {xmax:.1f}: {int((~reach_).sum())} of {len(reach_)} chip pixels beyond reach; tiles up to x "
                  f"{float(T[:, 0].max()):.1f}" + ("  TILE BEYOND THE LIMIT" if beyond else ""))
            ok &= not beyond
        else:
            mm_ = m.astype(bool).copy()
            if TAPES:                                  # (--mounted: the band under tape need not be scanned)
                yy_, xx_ = np.nonzero(mm_)
                st_ = img_to_stage(np.c_[xx_, yy_] / scl, shape, pose)
                under = np.zeros(len(yy_), bool)
                for (x0_, x1_, y0_, y1_) in TAPES:
                    under |= (st_[:, 0] >= x0_) & (st_[:, 0] <= x1_) & (st_[:, 1] >= y0_) & (st_[:, 1] <= y1_)
                mm_[yy_[under], xx_[under]] = False
            covered = float((cover[mm_] > 0).mean())
        zt = []
        for md in (run_dir / f"{nm}_tiles").glob("*/metadata.txt"):
            d = json.loads(md.read_text())
            fm = next(v for k, v in d.items() if k.startswith("Metadata"))
            u = stage_to_img(np.array([fm["XPositionUm"], fm["YPositionUm"]]), shape, pose)[0] * scl
            if not (0 <= u[0] < m.shape[1] and 0 <= u[1] < m.shape[0] and m[int(u[1]), int(u[0])]):
                continue
            zt.append(fm["ZPositionUm"] - true_focus(fm["XPositionUm"], fm["YPositionUm"]))
        zt = np.array(zt)
        idx = run_dir / f"{nm}_100x" / f"{nm}_100x_index.csv"
        rows = list(csv.DictReader(open(idx))) if idx.exists() else []
        e100 = np.array([float(r["z_um"]) - true_focus(float(r["stage_x_um"]), float(r["stage_y_um"]), 100)
                         - (a.z_slip if (nm, r["label"]) in slip["labels"] else 0.0) for r in rows])
        if a.z_slip:                                   # (unfocused images are expected after a slip; the retakes must fix them)
            e100 = np.array([e for e, r in zip(e100, rows) if r["focus"] in ("auto", "manual")])
        n_t = len([q for q in patches if q[0] == names[ci] and (xmax is None or q[1][0] + OFF100[0] <= xmax - s_margin)])
        print(f"{nm} ({names[ci]}): {len(T)} tiles, chip covered {covered:.3f}, surface {chipj.get('focus_surface', {}).get('type', 'plane')}, "
              f"tile focus error max {np.abs(zt).max():.2f} um (rms {np.sqrt((zt ** 2).mean()):.2f}); 100x {len(rows)}/{n_t} imaged "
              f"{[r['focus'] for r in rows]}, focus error " + " ".join(f"{e:+.2f}" for e in e100))
        ok &= covered > 0.98 and (len(rows) == n_t if xmax is None else n_t - 1 <= len(rows) <= n_t + 1) and (len(e100) == 0 or np.abs(e100).max() < 0.6) and np.abs(zt).max() < 4.0
        if a.z_slip:                                   # every flake focused in the end (the slip found, the failed ones retaken)
            ok &= all(r["focus"] in ("auto", "manual") for r in rows)
        # centring: where each 100x image was taken against the stage position that centres its flake
        def where(r):
            xy = (float(r["stage_x_um"]), float(r["stage_y_um"]))
            return np.array(actual.get((round(xy[0], 1), round(xy[1], 1)), xy))
        cen = [float(np.hypot(*(where(r) - true100[r["label"]]))) for r in rows if r["label"] in true100]
        if cen:
            print(f"  100x centring: off by " + " ".join(f"{e:.1f}" for e in cen) + " um; re-centred: "
                  + " ".join(r.get("recentre_um") or "-" for r in rows))
            if a.miss > 0 or a.stage_short:
                ok &= max(cen) < 15.0
    if a.graphite:
        nd, ed, n_want = 0, [], 0
        for p in man["pieces"]:
            nm = p["name"]
            f100 = run_dir / f"{nm}_100x" / f"{nm}_100x_index.csv"
            r100 = list(csv.DictReader(open(f100))) if f100.exists() else []
            if p.get("material", "graphite") == "graphite":
                n_want += sum(1 for r in r100 if thick.get(r["label"], 0) >= 0.84)
            di = run_dir / f"{nm}_domains" / f"{nm}_domains_index.csv"
            rows = list(csv.DictReader(open(di))) if di.exists() else []
            nd += len(rows)
            for r in rows:
                e = float(r["z_um"]) - true_focus(float(r["stage_x_um"]), float(r["stage_y_um"]), "domain") - (a.z_slip if slip["on"] else 0.0)
                ed.append(e)
                print(f"  domain {nm} {r['label']} {r['thickness_nm']} nm: focus {r['focus']}, error {e:+.2f} um")
        ed = np.array(ed)
        print(f"domain images {nd} of {n_want} flakes of 3+ layers; focus error max {np.abs(ed).max() if len(ed) else float('nan'):.2f} um")
        ok &= nd == n_want and n_want > 0 and np.abs(ed).max() < 0.6
    if a.mosaic:
        ok &= score_mosaics(a, s, sc, man, run_dir, out, polys, bigl, true_focus, patches, big, thick)
    if a.mounted:                                      # the run file: each chip's block, each block's corners and chips
        mt = man.get("mounts") or []
        good = (man.get("chip_type") == "mounted_1cm" and [p.get("mount") for p in man["pieces"]] == [1, 1, 2, 2]
                and [len(q["corners"]) for q in mt] == [3, 3] and [len(q["pieces"]) for q in mt] == [2, 2])
        print("blocks in the run file:", [(q["mount"], q["pieces"], len(q["corners"])) for q in mt], "OK" if good else "WRONG")
        ok &= good
        logt = next(run_dir.glob("*_run_*_log.txt")).read_text()
        n_tape = logt.count("found (tape or glue")
        print(f"edges stopped at the tape: {n_tape} (2 taped sides)", "OK" if n_tape == 2 else "WRONG")
        ok &= n_tape == 2
    names_ok = [p["name"] for p in man["pieces"]] == [f"{today}_{i + 2}" for i in range(len(names))]
    print("piece names", [p["name"] for p in man["pieces"]], "OK" if names_ok else "WRONG")
    if slack is not None:
        srv_, calls, st_ = slack
        posts = [d for m, d in calls if m == "chat.postMessage"]
        ups = [d for m, d in calls if m == "files.completeUploadExternal"]
        tops = [d for d in posts if not d.get("thread_ts")]
        chans = sorted({d["channel"] for d in tops})
        print(f"Slack: {len(tops)} thread start(s) in {chans}; {len(posts) - len(tops)} replies; {len(ups)} pictures "
              f"({len(st_['files'])} uploaded, {sum(st_['files'].values()) // 1000} kB); lookups {[m for m, _ in calls if m.startswith('users.')]}")
        for d in posts:
            print("   ", "  reply:" if d.get("thread_ts") else "THREAD:", d["channel"], d["text"].splitlines()[0][:90], "(broadcast)" if d.get("reply_broadcast") else "")
        for d in ups:
            print("      picture:", d["channel_id"], d["files"][0]["title"], "in thread" if d.get("thread_ts") else "NOT in a thread", "|", d["initial_comment"].splitlines()[0][:70])
        # only what needs the operator, each a new message (no threads: replies do not notify), to the DM and the channel
        ok &= (len(tops) == len(posts) and set(chans) == {"DU01OPERATR", "#microscope"} and not ups
               and not any("started" in d["text"] for d in posts) and any("you are needed" in d["text"] for d in posts)
               and sum(1 for d in posts if "finished" in d["text"]) == 2)
    print("RESULT", "PASS" if ok and names_ok and man["state"] == "done" else "FAIL")


def score_mosaics(a, s, sc, man, run_dir, out, polys, bigl, true_focus, patches, big, thick):
    """--mosaic: the mosaics planned / imaged / focused as they should be, the budget and cap kept, then the export."""
    from flakeid.fieldmosaic import parse_poly, plan_fields
    ok = True
    M100 = np.asarray(s.M, float) / 9.54
    n_fields = {"100x": 0, "IR": 0}
    for p in man["pieces"]:
        nm = p["name"]
        for kind, fol, csvn, obj in (("100x", f"{nm}_100x", f"{nm}_100x_mosaic.csv", 100), ("IR", f"{nm}_domains", f"{nm}_domains_mosaic.csv", "domain")):
            mp = run_dir / fol / csvn
            rows = list(csv.DictReader(open(mp))) if mp.exists() else []
            groups = {}
            for r in rows:
                groups.setdefault(r["main_file"], []).append(r)
            for main, rs in groups.items():
                lab = rs[0]["label"]
                fe = [float(r["z_um"]) - true_focus(float(r["stage_x_um"]), float(r["stage_y_um"]), obj) for r in rs[1:]]
                miss = [r["file"] for r in rs if not (run_dir / fol / r["file"]).is_file()]
                M = M100 if kind == "100x" else -np.asarray(sc.core.MD, float)
                wh = (int(rs[0]["width"]), int(rs[0]["height"]))
                pg = parse_poly(polys.get(lab))
                o = np.zeros(2) if kind == "100x" else np.asarray(s.domain_xy_offset_um, float)   # (IR: its stage offset)
                want = plan_fields(pg + o - [float(rs[0]["stage_x_um"]), float(rs[0]["stage_y_um"])], M, wh, s.hundred_mosaic_overlap,
                                   s.hundred_mosaic_max_tiles)[0] if pg is not None else []
                n_fields[kind] += len(rs) - 1
                good = (lab in bigl and not miss and len(rs) <= s.hundred_mosaic_max_tiles and (len(fe) == 0 or max(abs(e) for e in fe) < 0.6)
                        and (abs(len(rs) - 1 - len(want)) <= 1 or a.mosaic_budget is not None))
                # (+-1: the scan anchors the outline at the target as it found it (re-centred, carried corrections), the
                # plan here at the outline's own place: a field at the terrace's edge may come or go)
                print(f"  {kind} mosaic {nm} {lab}: {len(rs)} pictures (plan from where the main one was taken: {len(want) + 1}), "
                      f"focus {[r['focus'] for r in rs[1:]]}, error " + " ".join(f"{e:+.2f}" for e in fe)
                      + (f", MISSING {miss}" if miss else "") + ("" if good else "  <-- WRONG"))
                ok &= good
    want_big = len(bigl)
    n_main = {"100x": 0, "IR": 0}
    for p in man["pieces"]:
        for kind, fol, csvn in (("100x", f"{p['name']}_100x", f"{p['name']}_100x_mosaic.csv"), ("IR", f"{p['name']}_domains", f"{p['name']}_domains_mosaic.csv")):
            mp = run_dir / fol / csvn
            n_main[kind] += sum(1 for r in csv.DictReader(open(mp)) if r["k"] == "1") if mp.exists() else 0
    print(f"mosaics: 100x {n_main['100x']} of {want_big} large terraces ({n_fields['100x']} extra fields), IR {n_main['IR']} "
          f"({n_fields['IR']} extra fields); mosaic time {sc.mosaic_s['100x']:.1f} s (100x), {sc.mosaic_s['IR']:.1f} s (IR); budget used up: "
          f"{sorted(sc.mosaic_over) or 'no'}")
    if a.mosaic_budget is not None:
        # the budget: once used up no new mosaic starts (the one running stops at the next field)
        ok &= "100x" in sc.mosaic_over and n_main["100x"] < want_big
    else:
        ok &= n_main["100x"] == want_big and (not a.graphite or n_main["IR"] == want_big)
    # the export: each piece into a small viewer, the mosaics stitched and attached
    from flakeid.hundred import export
    for p in man["pieces"]:
        nm = p["name"]
        f100 = run_dir / f"{nm}_100x"
        if not f100.is_dir():
            continue
        vdir = out / "viewers" / nm
        make_viewer(vdir, nm, p.get("material", "graphite" if a.graphite else "hbn"), [(int(l.split("_F")[1][:3]), q[1]) for l, q in label_patches(patches, nm, man, p)])
        export(f100, vdir, material=p.get("material") or ("graphite" if a.graphite else "hbn"), keep_tiles=True, workers=1, verbose=False)
        html = (vdir / "index.html").read_text()
        import re
        H = json.loads(re.search(r"const H100=(.*?);\n", html, re.S).group(1))
        ents = [e for v in H.values() for e in v if not e.get("domain")]
        nm_mos = [e for e in ents if e.get("mos")]
        nm_ir = [e for e in ents if e.get("irmos")]
        n_exp100 = sum(1 for r in csv.DictReader(open(f100 / f"{nm}_100x_mosaic.csv")) if r["k"] == "1") if (f100 / f"{nm}_100x_mosaic.csv").exists() else 0
        dm = run_dir / f"{nm}_domains" / f"{nm}_domains_mosaic.csv"
        n_expir = sum(1 for r in csv.DictReader(open(dm)) if r["k"] == "1") if dm.exists() else 0
        files_ok = all((vdir / e["mos"]).is_file() for e in nm_mos) and all((vdir / e["irmos"]).is_file() for e in nm_ir)
        print(f"export {nm}: {len(ents)} 100x pictures attached, {len(nm_mos)} with a mosaic (of {n_exp100}), {len(nm_ir)} with an IR mosaic "
              f"(of {n_expir}) -> {vdir / 'index.html'}")
        ok &= len(nm_mos) == n_exp100 and len(nm_ir) == n_expir and files_ok and "'mos','mosaic'" in html
    return ok


def label_patches(patches, nm, man, p):
    """[(label, patch)] of piece p as the fake analysis labelled them (nearest chip, then patch order)."""
    c = np.array(p["center"])
    chip_names = sorted({q[0] for q in patches}, key=lambda n_: [q[0] for q in patches].index(n_))
    chs = {n_: np.mean([q[1] for q in patches if q[0] == n_], axis=0) for n_ in chip_names}
    chip = min(chs, key=lambda n_: np.hypot(*(chs[n_] - c)))
    mine = [q for q in patches if q[0] == chip]
    return [(f"{k + 1:02d}_F{k + 1:03d}_{chip}", q) for k, q in enumerate(mine)]


def make_viewer(vdir, name, material, flakes):
    """A small chip viewer (flakeid.zoomview's page; its scan image a grey picture) with flakes at the given 10x stage
    positions, for the export to attach the 100x pictures to."""
    from flakeid.zoomview import _HTML, _write_pyramid
    vdir = Path(vdir)
    if vdir.exists():
        shutil.rmtree(vdir)
    (vdir / "tiles").mkdir(parents=True)
    img = np.full((600, 800, 3), 90, np.uint8)
    info = {"image": _write_pyramid(img, vdir / "tiles" / "image", 85)}
    fl = []
    for k, (fid, xy) in enumerate(flakes):
        x, y = 80 + 150 * (k % 5), 100 + 150 * (k // 5)
        fl.append(dict(id=fid, score=0.9, t=12.0, n=1, terr=[fid], area=5000.0, L=100.0, W=50.0, d=40.0, bd=40.0, x=x, y=y, tx=x, ty=y,
                       bx=x - 30, by=y - 20, bw=60, bh=40, sx=round(float(xy[0]), 1), sy=round(float(xy[1]), 1), q="good", focus=0.0,
                       crowd=0.0, fw=1.0, fthick=0.0, poly=[[x - 30, y - 20, x + 30, y - 20, x + 30, y + 20, x - 30, y + 20]]))
    terr = {str(f["id"]): dict(id=f["id"], fl=f["id"], t=12.0, tlo=11.0, thi=13.0, alt="", layers=36, area=4000.0, L=90.0, W=45.0, d=40.0,
                               tx=f["tx"], ty=f["ty"], sx=f["sx"], sy=f["sy"], q="good", score=0.9, c=[0, 0, 0],
                               poly=[[f["tx"] - 20, f["ty"] - 15, f["tx"] + 20, f["ty"] - 15, f["tx"] + 20, f["ty"] + 15, f["tx"] - 20, f["ty"] + 15]])
            for f in fl}
    data = dict(name=name, material=material, oxide=90.0, pixel_um=1.09, tmin=0, tmax=100, info=info, flakes=fl, terraces=terr,
                stage_origin=None, stage_sign=[1, 1], stage_map=None)
    vend = ROOT / "flakeid" / "vendor"
    shutil.copy(vend / "openseadragon.min.js", vdir / "openseadragon.min.js")
    (vdir / "index.html").write_text(_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TITLE__", f"{name} flakes"))


if __name__ == "__main__":
    main()
