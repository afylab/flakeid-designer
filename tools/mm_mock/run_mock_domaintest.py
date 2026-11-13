"""Run tools/mm_domain_test.bsh against the simulated microscope.

Starts at 100x on a demo 100x image; the simulated user then puts in the domain configuration: camera
"DomainCam" (1024 x 1024 mono 16 bit, 0.108 um/px, rotated 90 deg against the scan camera), XY offset
(12.5, -7.0) um and focus offset +6.3 um from 100x (MockCore defaults).  The script must find those.

    python tools/mm_mock/run_mock_domaintest.py --java .../bin/java --jars DIR [--z-error 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M10 = [[-1.0905, -0.0067], [0.0066, -1.0917]]
OFF100, ZOFF100 = (24.0, -18.3), -41.0
OFFD, ZOFFD = (36.5, -25.3), -34.7          # MockCore defaults for the domain configuration


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True); ap.add_argument("--jars", required=True)
    ap.add_argument("--z-error", type=float, default=2.0, help="100x: true focus minus the start Z (um)")
    ap.add_argument("--camera", default="DomainCam", help="DOMAIN_CAMERA in the script ('' = the mock selects it)")
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/domaintest"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = [x for x in sorted(glob.glob(str(ROOT / "example images/demo_high_magnification/*/*_100x.jpg"))) if " X" not in x][2]
    (out / "p.txt").write_text(f"50000 20000 {Path(f).resolve().as_posix()}\n")
    s = (ROOT / "tools/mm_domain_test.bsh").read_text()
    for k, v in (("OUT_DIR", f'"{out.as_posix()}"'), ("DOMAIN_CAMERA", f'"{a.camera}"'), ("DOMAIN_CAMERA_PROPS", "new String[][] { }"),
                 ("DOMAIN_AUTO_EXPOSURE", "true")):
        s, n = re.subn(rf'^{k}\s*=.*$', f'{k} = {v};', s, flags=re.M)
        assert n == 1, k
    um = 1.0894 / 9.54
    x0, y0 = 50000 + OFF100[0], 20000 + OFF100[1]          # the 100x view is centred on the image centre
    pre = (f'mmc = new MockCore("{(out / "p.txt").as_posix()}", {um}, 50000.0, 20000.0, 0.0, {M10[0][0]}, {M10[0][1]}, {M10[1][0]}, {M10[1][1]}, '
           f'{x0}, {y0}, 1L);\nmm = new MockStudio();\nmmc.mockObjective(100); mmc.offX100 = {OFF100[0]}; mmc.offY100 = {OFF100[1]}; '
           f'mmc.zOff100 = {ZOFF100}; mmc.exposure = 27.0; mmc.signal = 2200.0 / 9.0;\n'
           f'mmc.z0 = -90.0; mmc.setPosition("Z", (-90.0 + {ZOFF100} - {a.z_error}) * 12.0);\n'
           f't0 = System.currentTimeMillis();\n')
    post = '\nprint("MOCK snaps " + mmc.getSnaps() + " seconds " + (System.currentTimeMillis() - t0) / 1000.0);\nSystem.exit(0);\n'
    (out / "run.bsh").write_text(pre + s + post)
    jars = Path(a.jars)
    cp = ":".join(str(p) for p in (jars / "bsh.jar", jars / "ij.jar", jars / "classes"))
    r = subprocess.run([a.java, "-Xmx4g", "-Djava.awt.headless=true", "-cp", cp, "bsh.Interpreter", str(out / "run.bsh")],
                       capture_output=True, text=True)
    print(r.stdout); print(r.stderr[-2500:] if r.returncode or "Error" in r.stderr else "")
    want_xy = (OFFD[0] - OFF100[0], OFFD[1] - OFF100[1])
    want_dz = ZOFFD - ZOFF100
    print(f"TRUE: XY offset {want_xy[0]:.3f}, {want_xy[1]:.3f} um; focus offset {want_dz:+.3f} um; "
          f"pixel matrix [0, 0.108; -0.108, 0]")
    try:
        d = json.loads((out / "domaintest_domain.json").read_text())
    except Exception as e:
        print("RESULT FAIL (no json:", e, ")"); return
    exy = (d["xy_offset_um"][0] - want_xy[0], d["xy_offset_um"][1] - want_xy[1])
    edz = d["focus_offset_um"] - want_dz
    m = d["um_per_px_matrix"]
    em = max(abs(m[0]), abs(m[1] - 0.108), abs(m[2] + 0.108), abs(m[3]))
    print(f"errors: XY {exy[0]:+.3f}, {exy[1]:+.3f} um; focus {edz:+.3f} um; matrix {em:.4f} um/px; "
          f"AF {d['af_runs_passed']}/{d['af_runs']} worst {d['af_worst_um']:.3f} um")
    ok = max(map(abs, exy)) < 0.5 and abs(edz) < 0.5 and em < 0.003 and d["af_runs_passed"] == d["af_runs"]
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
