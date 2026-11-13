"""Run tools/mm_chip_scan.bsh in MODE = "targets" (100x) against the simulated microscope.

The "sample" is a mosaic of the demo 100x images (example images/demo_high_magnification);
each target is one of them, with a position error, on a tilted chip.  The mock camera blurs
with defocus (100x: ~3 px per um).  Scores the focus the script found at every flake.

    python tools/mm_mock/run_mock_targets.py --java .../bin/java --jars DIR [--focus-mode auto]
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M10 = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])
RATIO = 9.54                                   # 10x -> 100x magnification ratio (measured)
M100 = M10 / RATIO
UM100 = 1.0894 / RATIO
O = np.array([50000.0, 20000.0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True)
    ap.add_argument("--jars", required=True)
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/targets"))
    ap.add_argument("--focus-mode", default="auto")
    ap.add_argument("--tilt", nargs=2, type=float, default=(3.0, -2.0), help="um/mm")
    ap.add_argument("--z-error", type=float, default=6.0, help="true focus at the first flake minus the start Z (um)")
    ap.add_argument("--pos-error", type=float, default=8.0, help="target position error (um, rms)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--z10-offset", type=float, default=None,
                    help="true 100x - 10x focus offset (um): writes a z10_um column and starts at the 10x focus")
    ap.add_argument("--z10-tilt-error", type=float, default=0.0, help="error of the 10x map's Y tilt (um/mm): the offset drifts")
    ap.add_argument("--offset-known", action="store_true", help="set FOCUS_OFFSET_UM to the true offset")
    ap.add_argument("--chip-json", action="store_true", help="give the script the true chip tilt (as from a 10x chip scan)")
    ap.add_argument("--spread-um", type=float, default=3000.0, help="spacing of the flakes on the chip (um)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(glob.glob(str(ROOT / "example images/demo_high_magnification/*/*_100x.jpg"))) if " X" not in f][: a.n]
    ims = [cv2.imread(f) for f in files]
    # sample: the 100x images placed spread-um apart on the stage (patch list for the mock)
    rng = np.random.default_rng(3)
    cols = 3
    truth = [O + np.array([(k % cols) * a.spread_um, (k // cols) * a.spread_um]) for k in range(len(files))]
    cpath = out / "patches.txt"
    cpath.write_text("".join(f"{t[0]} {t[1]} {Path(f).resolve().as_posix()}\n" for t, f in zip(truth, files)))
    targets = [t + rng.normal(0, a.pos_error, 2) for t in truth]
    tcsv = out / "targets.csv"
    with open(tcsv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["order", "label", "stage_x_um", "stage_y_um", "z10_um"])
        tx_, ty_ = a.tilt[0] / 1000, a.tilt[1] / 1000
        z0_ = -90.0 + (a.z10_offset if a.z10_offset is not None else a.z_error) - tx_ * targets[0][0] - ty_ * targets[0][1]
        for k, t in enumerate(targets):
            z10 = (z0_ + tx_ * t[0] + ty_ * t[1] - a.z10_offset + a.z10_tilt_error / 1000 * (t[1] - targets[0][1])) if a.z10_offset is not None else None
            wr.writerow([k + 1, f"{k + 1:02d}_{Path(files[k]).stem}", f"{t[0]:.3f}", f"{t[1]:.3f}", "" if z10 is None else f"{z10:.3f}"])
    start = targets[0] + np.array([40.0, -25.0])
    tx, ty = a.tilt[0] / 1000, a.tilt[1] / 1000
    z0 = -90.0 + (a.z10_offset if a.z10_offset is not None else a.z_error) - tx * targets[0][0] - ty * targets[0][1]

    if a.chip_json:   # 10x plane with the true tilt (its offset does not matter)
        (out / "chip.json").write_text('{"focus_plane": [%r, %r, -12.0], "focus_rms_um": 0.4}' % (tx + 0.0002, ty - 0.0003))
    script = (ROOT / "tools/mm_chip_scan.bsh").read_text()
    for k, v in (("MODE", '"targets"'), ("TARGETS_CSV", f'"{tcsv.as_posix()}"'), ("FOCUS_MODE", f'"{a.focus_mode}"'),
                 ("OUT_DIR", f'"{out.as_posix()}"'), ("NAME", '"mock"'),
                 ("CHIP_JSON", f'"{(out / "chip.json").as_posix()}"' if a.chip_json else '""'),
                 ("FOCUS_OFFSET_UM", f"{a.z10_offset}" if (a.offset_known and a.z10_offset is not None) else "Double.NaN")):
        script = re.sub(rf'^{k}\s*=.*$', f'{k} = {v};', script, flags=re.M)
    prelude = (f'mmc = new MockCore("{cpath.as_posix()}", {UM100}, {O[0]}, {O[1]}, 0.0, '
               f'{M100[0, 0]}, {M100[0, 1]}, {M100[1, 0]}, {M100[1, 1]}, {start[0]}, {start[1]}, 1L);\n'
               f'mm = new MockStudio();\nmmc.blurPerUm = 3.0; mmc.tiltX = {tx}; mmc.tiltY = {ty}; mmc.z0 = {z0}; mmc.setPosition("Z", -90.0);\n'
               f'mmc.focusNoise = 0.1;\nt0 = System.currentTimeMillis();\n')
    post = ('\nprint("MOCK snaps " + mmc.getSnaps() + " seconds " + (System.currentTimeMillis() - t0) / 1000.0);\n')
    run = out / "run.bsh"
    run.write_text(prelude + script + post)
    jars = Path(a.jars)
    cp = ":".join(str(p) for p in (jars / "bsh.jar", jars / "ij.jar", jars / "classes"))
    r = subprocess.run([a.java, "-Xmx6g", "-Djava.awt.headless=true", "-cp", cp, "bsh.Interpreter", str(run)],
                       capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if "move to" not in l))
    if r.returncode:
        print(r.stderr[-2000:])
    idx = list(csv.DictReader(open(out / "mock_100x" / "mock_100x_index.csv")))
    errs = []
    for row in idx:
        x, y, z = float(row["stage_x_um"]), float(row["stage_y_um"]), float(row["z_um"])
        errs.append(z - (z0 + tx * x + ty * y))
        assert (out / "mock_100x" / row["file"]).exists()
    errs = np.array(errs)
    print(f"RESULT {len(idx)}/{len(targets)} imaged; focus error (um): " + " ".join(f"{e:+.2f}" for e in errs) +
          f"; max |error| {np.abs(errs).max():.2f} um")


if __name__ == "__main__":
    main()
