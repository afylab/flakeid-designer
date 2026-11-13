"""Run tools/mm_chip_scan.bsh in MODE = "flat" against the simulated microscope, build the flat field with
flakeid.mm and compare it with the simulator's vignetting (1 - 0.38 r^2, r = 1 at the frame corner).

    python tools/mm_mock/run_mock_flat.py --java .../bin/java --jars DIR [--objective 100x]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
M10 = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])
O = np.array([50000.0, 20000.0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True)
    ap.add_argument("--jars", required=True)
    ap.add_argument("--objective", default="10x")
    ap.add_argument("--image", default=str(ROOT / "example images/bn24.jpg"))
    ap.add_argument("--start", nargs=2, type=float, default=(-1500.0, 1200.0), help="start relative to the image centre (um)")
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/flat"))
    a = ap.parse_args(argv)
    out = Path(a.out) / a.objective
    out.mkdir(parents=True, exist_ok=True)
    script = (ROOT / "tools/mm_chip_scan.bsh").read_text()
    for k, v in (("MODE", '"flat"'), ("OUT_DIR", f'"{out.as_posix()}"'), ("NAME", '"mock"'), ("FLAT_OBJECTIVE", f'"{a.objective}"'),
                 ("FLAT_FRAMES", "12"), ("FLAT_DARK_FRAMES", "6")):
        script, n = re.subn(rf'^{k}\s*=.*$', f'{k} = {v};', script, flags=re.M)
        assert n == 1, k
    start = O + np.array(a.start)
    prelude = (f'mmc = new MockCore("{Path(a.image).resolve().as_posix()}", 1.09, {O[0]}, {O[1]}, 0.0, '
               f'{M10[0, 0]}, {M10[0, 1]}, {M10[1, 0]}, {M10[1, 1]}, {start[0]}, {start[1]}, 1L);\nmm = new MockStudio();\n'
               + ('mmc.mockObjective(100); mmc.offX100 = 0; mmc.offY100 = 0; mmc.zOff100 = 0; mmc.signal = 2200.0 / 9.0;\n' if a.objective == "100x" else ""))
    run = out / "run.bsh"
    run.write_text(prelude + script)
    cp = ":".join(str(Path(a.jars) / p) for p in ("bsh.jar", "ij.jar", "classes"))
    r = subprocess.run([a.java, "-Xmx4g", "-Djava.awt.headless=true", "-cp", cp, "bsh.Interpreter", str(run)], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if "move to" not in l]
    print("\n".join(lines[-22:]))
    if r.returncode:
        print(r.stderr[-2000:])
    flat_dir = next(out.glob("mock_flat_*"))
    dark_dir = next(out.glob("mock_dark_*"), None)
    npz = out / "flat.npz"
    cmd = [sys.executable, "-m", "flakeid.mm", "flat", str(flat_dir), "-o", str(npz)] + (["--dark", str(dark_dir)] if dark_dir else [])
    print(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout[-600:])
    from flakeid.mm import FlatField
    ff = FlatField.load(npz)
    H, W = ff.flat.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    true = 1 - 0.38 * r2
    true /= true.max()
    centre = r2 < 0.02
    g = ff.flat[..., 1] / ff.flat[..., 1][centre].mean()        # both 1 at the centre (a max is set by noise)
    err = g[8:-8:4, 8:-8:4] - true[8:-8:4, 8:-8:4]
    print(f"RESULT {a.objective}: flat vs simulated vignetting: rms {np.sqrt((err ** 2).mean()) * 100:.2f} %, max {np.abs(err).max() * 100:.2f} %; "
          f"dark {np.round(np.asarray(ff.dark).reshape(-1, 3).mean(0), 1) if np.ndim(ff.dark) else ff.dark} (simulated 128 at 12 bit)")


if __name__ == "__main__":
    main()
