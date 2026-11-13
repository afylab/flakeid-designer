"""The domain camera test (flakeid.scope mode "domaintest", the Python port of tools/mm_domain_test.bsh) on the simulated
microscope: a demo 100x image as the sample, the simulator's domain configuration (a 1024 x 1024 monochrome camera turned
90 deg, 0.108 um/px, XY offset 12.5, -7.0 um and focus offset +6.3 um from the 100x objective).

    python tools/mm_mock/run_mock_domaintest_py.py [--z-error UM] [--out DIR]

PASS: the XY offset and the focus offset within 0.5 um, the pixel matrix within 0.003 um/px, every autofocus run passed.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.scope.mock import MockCore  # noqa: E402
from flakeid.scope.scan import Scanner  # noqa: E402
from flakeid.scope.settings import load  # noqa: E402
from flakeid.scope.ui import AutoUI  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z-error", type=float, default=2.0, help="100x: true focus minus the start Z (um)")
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/domaintest_py"))
    ap.add_argument("--demo", default=str(ROOT / "example images/demo_high_magnification"), help="folder of the demo 100x images")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    demo = [x for x in sorted(glob.glob(str(Path(a.demo) / "*" / "*_100x.jpg"))) if " X" not in x]
    if not demo:
        print(f"no demo 100x images in {a.demo} (give the folder with --demo)")
        return 2
    core = MockCore(1.09, x0=50000.0, y0=20000.0)
    core.add_patch(cv2.cvtColor(cv2.imread(demo[2]), cv2.COLOR_BGR2RGB), 50000.0, 20000.0)
    core.patch_um_per_px = 1.0894 / 9.54
    off100, offD = core.off100.copy(), core.offD.copy()
    core.mockObjective(100)
    x0, y0 = 50000.0 + off100[0], 20000.0 + off100[1]              # the 100x view centred on the image centre
    core.setXYPosition("XY", x0, y0)
    core.z = core.true_focus(x0, y0) - a.z_error
    (out / "settings.json").write_text("{}")
    s = load(None, mode="domaintest", local_root=str(out), domain_camera="DomainCam", settings_file=str(out / "settings.json"),
             z_settle_ms=0.0, verbose=False, af_manual_fallback="never", target_exposure_ms=27.0)
    sc = Scanner(core, s, AutoUI(core, analysis=False))
    sc.run()
    want_xy = (offD[0] - off100[0], offD[1] - off100[1])
    want_dz = core.zoffD - core.zoff100
    fl = (s.domain_flip or "").lower()                             # flipped frames: a flipped axis's matrix column negated
    MD = [[core.MD[r][0] * (-1 if "x" in fl else 1), core.MD[r][1] * (-1 if "y" in fl else 1)] for r in (0, 1)]
    print(f"TRUE: XY offset {want_xy[0]:.3f}, {want_xy[1]:.3f} um; focus offset {want_dz:+.3f} um; pixel matrix "
          f"[{MD[0][0]:g}, {MD[0][1]:g}; {MD[1][0]:g}, {MD[1][1]:g}]" + (f" (frames flipped in {fl})" if fl else ""))
    js = sorted(out.glob("domaintest_*/domaintest_domain.json"))
    if not js:
        print("RESULT FAIL (no json)")
        return 1
    d = json.loads(js[-1].read_text())
    exy = (d["xy_offset_um"][0] - want_xy[0], d["xy_offset_um"][1] - want_xy[1])
    edz = d["focus_offset_um"] - want_dz
    m = d["um_per_px_matrix"]
    em = max(abs(m[0] - MD[0][0]), abs(m[1] - MD[0][1]), abs(m[2] - MD[1][0]), abs(m[3] - MD[1][1]))
    saved = json.loads((out / "settings.json").read_text())
    print(f"errors: XY {exy[0]:+.3f}, {exy[1]:+.3f} um; focus {edz:+.3f} um; matrix {em:.4f} um/px; AF {d['af_runs_passed']}/{d['af_runs']} "
          f"worst {d['af_worst_um']:.3f} um; saved into the settings: {saved}")
    ok = max(map(abs, exy)) < 0.5 and abs(edz) < 0.5 and em < 0.003 and d["af_runs_passed"] == d["af_runs"] and "domain_xy_offset_um" in saved
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
