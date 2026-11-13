"""Run tools/mm_autofocus_test.bsh against the simulated microscope (10x chip image or 100x demo image).

    python tools/mm_mock/run_mock_aftest.py --objective 100x --z-error 4 --java .../bin/java --jars DIR
"""
from __future__ import annotations

import argparse
import glob
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M10 = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True); ap.add_argument("--jars", required=True)
    ap.add_argument("--objective", default="10x")
    ap.add_argument("--z-error", type=float, default=5.0, help="true focus minus the start Z (um)")
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/aftest"))
    ap.add_argument("--glitch", type=int, default=-1, help="snap number that returns a garbage frame")
    ap.add_argument("--step-readonly", action="store_true", help="the drive's StepSize_um cannot be set")
    ap.add_argument("--flags", action="store_true", help="script flags Z_STEPSIZE_UM = 0, Z_SCALE = 12 instead")
    ap.add_argument("--z-scale", type=float, default=12.0, help="drive units per um (mock) = Z_SCALE in the script")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.objective == "100x":
        f = [x for x in sorted(glob.glob(str(ROOT / "example images/demo_high_magnification/*/*_100x.jpg"))) if " X" not in x][2]
        (out / "p.txt").write_text(f"50000 20000 {Path(f).resolve().as_posix()}\n")
        img, um, M, blur = out / "p.txt", 1.0894 / 9.54, M10 / 9.54, 3.0
    else:
        img, um, M, blur = ROOT / "example images/g34.jpg", 1.09, M10, 0.15
    s = (ROOT / "tools/mm_autofocus_test.bsh").read_text()
    subs = [("OUT_DIR", f'"{out.as_posix()}"'), ("OBJECTIVE", f'"{a.objective}"'), ("Z_SCALE", str(a.z_scale))]
    if a.flags:
        subs.append(("Z_STEPSIZE_UM", "0"))
    for k, v in subs:
        s = re.sub(rf'^{k}\s*=.*$', f'{k} = {v};', s, flags=re.M)
    pre = (f'mmc = new MockCore("{Path(img).resolve().as_posix()}", {um}, 50000.0, 20000.0, 0.0, {M[0,0]}, {M[0,1]}, {M[1,0]}, {M[1,1]}, '
           f'50000.0, 20000.0, 1L);\nmm = new MockStudio();\nmmc.blurPerUm = {blur}; mmc.exposure = 3.0; mmc.z0 = -90.0 + {a.z_error}; mmc.setPosition("Z", -90.0 * {a.z_scale});\n' + ('mmc.stepReadOnly = true;\n' if a.step_readonly else '') + f'mmc.glitchAt = {a.glitch};\n')
    (out / "run.bsh").write_text(pre + s)
    jars = Path(a.jars)
    cp = ":".join(str(p) for p in (jars / "bsh.jar", jars / "ij.jar", jars / "classes"))
    r = subprocess.run([a.java, "-Xmx4g", "-Djava.awt.headless=true", "-cp", cp, "bsh.Interpreter", str(out / "run.bsh")],
                       capture_output=True, text=True)
    print(r.stdout); print(r.stderr[-1500:] if r.returncode else "")
    print(f"TRUE focus: Z {-90.0 + a.z_error:.3f}")


if __name__ == "__main__":
    main()
