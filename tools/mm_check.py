#!/usr/bin/env python3
"""Micro-Manager hardware check for the BX51 / XC10 flake-scanning setup.

Run on the microscope PC with Stream and the Micro-Manager GUI CLOSED
(only one program can own the camera and stage):

    pip install pymmcore-plus numpy opencv-python tifffile
    python tools/mm_check.py --mm "C:\\Program Files\\Micro-Manager-2.0" --cfg "C:\\...\\bx51.cfg" --out mm_check

Put the stage over a flake-covered area of a chip (not bare substrate:
the image-shift tests need features), in focus, at 5x/6.3x.

Steps (each prints PASS / WARN / FAIL):
  1  load the configuration, list devices
  2  camera: snap, image size, bit depth, colour, exposure, frame rate
  3  XY stage: read position, move +/-200 um, read back, timing
  4  calibration: move the stage and measure the image shift -> pixel size,
     camera-to-stage rotation and axis signs (what the pipeline needs)
  5  repeatability: return to one spot from 4 directions -> backlash/repeat error
  6  focus (Z), if motorised: move +/-5 um, sharpness vs Z
  7  mini scan: 3x3 tiles with 15 % overlap, saved as 16-bit TIFF + positions.csv

Everything is written to --out; send me report.json (and the tiles folder).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

REPORT = {}


def say(step, status, msg, **data):
    print(f"[{status:4s}] {step}: {msg}")
    REPORT.setdefault(step, []).append(dict(status=status, msg=msg, **data))


def to_array(core, img):
    """Return (H, W) or (H, W, 3) numpy array whatever the camera delivers."""
    a = np.asarray(img)
    if a.ndim == 3:
        return a[..., :3]
    ncomp = core.getNumberOfComponents()
    if ncomp == 4:                         # RGB32 packed as uint32 BGRA
        if a.dtype == np.uint32:
            b = a.view(np.uint8).reshape(a.shape[0], a.shape[1], 4)
        else:
            b = a.reshape(core.getImageHeight(), core.getImageWidth(), 4)
        return b[..., [2, 1, 0]]
    return a


def snap(core):
    core.snapImage()
    return to_array(core, core.getImage())


def gray(a):
    g = a.astype(np.float32)
    return g.mean(axis=2) if g.ndim == 3 else g


def shift_px(a, b):
    """Translation of image b relative to a (pixels) and match quality."""
    ga, gb = gray(a), gray(b)
    ga = (ga - ga.mean()) / (ga.std() + 1e-6)
    gb = (gb - gb.mean()) / (gb.std() + 1e-6)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return np.array([dx, dy]), resp


def sharpness(a):
    return float(cv2.Laplacian(gray(a), cv2.CV_32F).var())


def wait(core):
    core.waitForSystem()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", required=True, help="Micro-Manager install folder (device adapters)")
    ap.add_argument("--cfg", required=True, help="hardware configuration .cfg saved by the Micro-Manager wizard")
    ap.add_argument("--out", default="mm_check")
    ap.add_argument("--step-um", type=float, default=200.0, help="stage move for the calibration test")
    ap.add_argument("--tile-overlap", type=float, default=0.15)
    ap.add_argument("--expected-um-per-px", type=float, default=1.005, help="6.3x: 1.005; 100x with 0.63x adapter: ~0.102")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from pymmcore_plus import CMMCorePlus
    core = CMMCorePlus()
    core.setDeviceAdapterSearchPaths([a.mm])

    # 1 ------------------------------------------------------------ config
    try:
        core.loadSystemConfiguration(a.cfg)
    except Exception as e:
        say("1 config", "FAIL", f"could not load {a.cfg}: {e}")
        return finish(out)
    devs = list(core.getLoadedDevices())
    cam, xy, z = core.getCameraDevice(), core.getXYStageDevice(), core.getFocusDevice()
    say("1 config", "PASS", f"loaded {len(devs)} devices; camera={cam!r} xy={xy!r} focus={z!r}", devices=devs)
    if not cam:
        say("1 config", "FAIL", "no default camera set (wizard: 'Select default devices')")
        return finish(out)

    # 2 ------------------------------------------------------------ camera
    try:
        core.setExposure(core.getExposure() or 10)
        t = time.time()
        img = snap(core)
        dt = time.time() - t
        bits = core.getImageBitDepth()
        colour = img.ndim == 3
        say("2 camera", "PASS", f"snap {img.shape} {img.dtype}, {bits}-bit, {'colour' if colour else 'mono'}, "
            f"exposure {core.getExposure():.1f} ms, snap {dt * 1000:.0f} ms, mean {img.mean():.1f}, max {img.max()}",
            shape=list(img.shape), dtype=str(img.dtype), bit_depth=bits)
        full = 2 ** bits - 1
        if img.max() >= full:
            say("2 camera", "WARN", f"{(img >= full).mean() * 100:.1f} % of pixels clipped: lower the exposure")
        if colour and bits <= 8:
            say("2 camera", "WARN", "colour arrives as 8-bit RGB: fine for flake finding; for 14-bit colour look for a "
                                    "raw/Bayer output-format property on the camera (list below)")
        props = {p: core.getProperty(cam, p) for p in core.getDevicePropertyNames(cam)}
        say("2 camera", "INFO", "camera properties saved to report.json", properties=props)
        cv2.imwrite(str(out / "snap.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if colour else img)
        t = time.time()
        for _ in range(5):
            snap(core)
        say("2 camera", "INFO", f"{5 / (time.time() - t):.1f} snaps per second")
    except Exception as e:
        say("2 camera", "FAIL", f"snap failed: {e}")
        return finish(out)

    if not xy:
        say("3 stage", "FAIL", "no XY stage: set it as default in the wizard")
        return finish(out)

    # 3 ------------------------------------------------------------ stage
    x0, y0 = core.getXPosition(), core.getYPosition()
    say("3 stage", "PASS", f"position x={x0:.2f} y={y0:.2f} um")
    try:
        for dx, dy in ((a.step_um, 0), (0, a.step_um)):
            t = time.time()
            core.setXYPosition(x0 + dx, y0 + dy); wait(core)
            dt = time.time() - t
            x1, y1 = core.getXPosition(), core.getYPosition()
            err = np.hypot(x1 - x0 - dx, y1 - y0 - dy)
            say("3 stage", "PASS" if err < 2 else "WARN",
                f"move ({dx:g},{dy:g}) um: read back error {err:.2f} um, {dt * 1000:.0f} ms")
            core.setXYPosition(x0, y0); wait(core)
    except Exception as e:
        say("3 stage", "FAIL", f"move failed: {e}")
        return finish(out)

    # 4 ------------------------------------------------------------ calibration
    ref = snap(core)
    shifts = {}
    for name, (dx, dy) in {"x": (a.step_um, 0), "y": (0, a.step_um)}.items():
        core.setXYPosition(x0 + dx, y0 + dy); wait(core); time.sleep(0.2)
        im = snap(core)
        s, q = shift_px(ref, im)
        shifts[name] = (s, q)
        core.setXYPosition(x0, y0); wait(core)
    sx, qx = shifts["x"]; sy, qy = shifts["y"]
    if min(qx, qy) < 0.05:
        say("4 calibration", "WARN", f"weak image match (quality {qx:.2f}, {qy:.2f}): move to an area with more flakes "
                                     f"or reduce --step-um below half the field of view")
    # image displacement per um of stage motion; image moves opposite to the sample view
    M = np.column_stack([sx, sy]) / a.step_um                # px per um, columns = stage x, y
    px_per_um = np.sqrt(abs(np.linalg.det(M)))
    um_per_px = 1 / px_per_um
    angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    sign = [int(np.sign(-M[0, 0])) or 1, int(np.sign(-M[1, 1])) or 1]
    status = "PASS" if abs(um_per_px / a.expected_um_per_px - 1) < 0.05 else "WARN"
    say("4 calibration", status,
        f"{um_per_px:.4f} um/px (expected {a.expected_um_per_px}); camera rotated {angle:+.2f} deg vs stage x; "
        f"stage-sign (image +x/+y -> stage) = {sign}; match quality {qx:.2f}/{qy:.2f}",
        um_per_px=um_per_px, angle_deg=angle, stage_sign=sign, matrix_px_per_um=M.tolist())
    if abs(angle) > 0.5:
        say("4 calibration", "INFO", "rotation > 0.5 deg: rotate the camera on its C-mount to align with the stage, "
                                     "or keep it: the pipeline can correct it")
    cal = dict(um_per_px=um_per_px, angle_deg=angle, stage_sign=sign)

    # 5 ------------------------------------------------------------ repeatability
    errs = []
    d = 100.0
    for ox, oy in ((d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, -d)):
        core.setXYPosition(x0 + ox, y0 + oy); wait(core)
        core.setXYPosition(x0, y0); wait(core); time.sleep(0.2)
        s, q = shift_px(ref, snap(core))
        errs.append(np.hypot(*s) * um_per_px)
    e = np.array(errs)
    say("5 repeatability", "PASS" if e.max() < 3 else "WARN",
        f"return error from 6 directions: median {np.median(e):.2f} um, max {e.max():.2f} um "
        f"(<3 um is fine for relocating flakes)", errors_um=e.round(2).tolist())

    # 6 ------------------------------------------------------------ focus
    if z:
        try:
            z0 = core.getPosition(z)
            sh = []
            for dz in (-10, -5, 0, 5, 10):
                core.setPosition(z, z0 + dz); wait(core)
                sh.append((dz, sharpness(snap(core))))
            core.setPosition(z, z0); wait(core)
            best = max(sh, key=lambda t: t[1])[0]
            say("6 focus", "PASS", f"Z moves; sharpness peak at {best:+d} um of the current focus "
                                   f"(check the sign/units of the focus drive)", sharpness=sh)
        except Exception as e:
            say("6 focus", "WARN", f"focus drive present but move failed: {e}")
    else:
        say("6 focus", "INFO", "no motorised focus: scans will rely on a flat, level chip")

    # 7 ------------------------------------------------------------ mini scan
    try:
        import tifffile
        tiles = out / "tiles"
        tiles.mkdir(exist_ok=True)
        h, w = ref.shape[:2]
        stepx = w * um_per_px * (1 - a.tile_overlap)
        stepy = h * um_per_px * (1 - a.tile_overlap)
        rows = []
        prev = {}
        for j in range(3):
            for i in range(3):
                sx_, sy_ = x0 + sign[0] * i * stepx, y0 + sign[1] * j * stepy
                core.setXYPosition(sx_, sy_); wait(core); time.sleep(0.1)
                im = snap(core)
                fn = f"tile_r{j}_c{i}.tif"
                tifffile.imwrite(tiles / fn, im)
                rx, ry = core.getXPosition(), core.getYPosition()
                rows.append(dict(file=fn, row=j, col=i, stage_x_um=round(rx, 2), stage_y_um=round(ry, 2)))
                prev[(i, j)] = im
        core.setXYPosition(x0, y0); wait(core)
        with open(tiles / "positions.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
        # check the overlap of neighbours against the expected step
        res = []
        ov = int(w * a.tile_overlap)
        for j in range(3):
            for i in range(2):
                A, B = prev[(i, j)], prev[(i + 1, j)]
                s, q = shift_px(A[:, -2 * ov:], B[:, :2 * ov])
                res.append(min(np.hypot(*(s - [ov, 0])), np.hypot(*(s + [ov, 0]))) if q > 0.05 else np.nan)   # sign convention-free
        r = np.array(res, float)
        say("7 scan", "PASS", f"9 tiles saved with stage positions in {tiles}; neighbour overlap agrees with the "
                              f"stage within {np.nanmedian(r) * um_per_px:.1f} um (median)", overlap_error_um=(r * um_per_px).round(2).tolist())
    except Exception as e:
        say("7 scan", "FAIL", f"scan failed: {e}")

    REPORT["calibration"] = cal
    finish(out)


def finish(out):
    (out / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    bad = [k for k, v in REPORT.items() if isinstance(v, list) and any(x["status"] == "FAIL" for x in v)]
    print("\nreport:", out / "report.json", "|", "FAILED: " + ", ".join(bad) if bad else "no failures")


if __name__ == "__main__":
    main()
