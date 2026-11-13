"""Simulated objective offsets tool (start screen "Objective offsets"): the simulator's 100x, 50x, 10x and 5x objectives
have their own XY and focus offsets (flakeid.scope.mock); the tool should find them, and the pixel sizes.

    python tools/mm_mock/run_mock_objoffsets.py --image <a 10x picture of flakes (RGB JPEG / PNG / raw TIFF)>
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.scope.mock import MockCore  # noqa: E402
from flakeid.scope.scan import Scanner  # noqa: E402
from flakeid.scope.settings import load  # noqa: E402
from flakeid.scope.ui import AutoUI  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="a 10x picture of flakes on a chip (1.09 um per px)")
    ap.add_argument("--dx", type=float, default=0.0, help="the feature: um from the picture's centre (put a flake edge there)")
    ap.add_argument("--dy", type=float, default=0.0)
    a = ap.parse_args(argv)
    out = ROOT / "tools" / "mm_mock" / "out" / "objoffsets"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    img = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:                                            # (a raw Bayer tile)
        from flakeid.mm import BAYER_CODE
        img = cv2.cvtColor(img, BAYER_CODE).astype(np.float32)
        img = np.clip(img / np.percentile(img, 99.5) * 230, 0, 255).astype(np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    core = MockCore(1.09, x0=50000.0, y0=20000.0)
    core.add_chip(img, 50000.0, 20000.0, 0.0, 0.0)
    core.mockObjective(100)
    core.setXYPosition("XY", 50000.0 + core.off100[0] + a.dx, 20000.0 + core.off100[1] + a.dy)
    core.z = core.true_focus(core.x, core.y)
    (out / "settings.json").write_text(json.dumps({"objective_positions": {"10x": 2, "100x": 5}}))
    s = load(None, mode="offsets", local_root=str(out), settings_file=str(out / "settings.json"), z_settle_ms=0.0, verbose=False,
             af_manual_fallback="never", target_exposure_ms=27.0)
    sc = Scanner(core, s, AutoUI(core, analysis=False))
    res = sc.run() or {}
    saved = json.loads((out / "settings.json").read_text()).get("objective_offsets") or {}
    offs = {100: core.off100, 50: core.off_obj[50], 10: np.zeros(2), 5: core.off_obj[5]}
    zoffs = {100: core.zoff100, 50: core.zoff_obj[50], 10: 0.0, 5: core.zoff_obj[5]}
    ok = bool(saved) and set(saved) == {"100x", "50x", "10x", "5x"}
    for o, e in sorted(saved.items(), key=lambda kv: -int(kv[0][:-1])):
        m = int(o[:-1])
        want = offs[m] - offs[100]
        wz = zoffs[m] - zoffs[100]
        err = math.hypot(e["dx"] - want[0], e["dy"] - want[1])
        px = math.sqrt(abs(np.linalg.det(np.asarray(e["M"]))))
        wpx = 1.09 / (9.54 if m == 100 else m / 10.0)
        print(f"{o:>5}: found {e['dx']:+8.2f}, {e['dy']:+8.2f}, dz {e['dz']:+7.2f}; true {want[0]:+8.2f}, {want[1]:+8.2f}, dz {wz:+7.2f}; "
              f"XY error {err:.2f} um; pixel {px:.4f} (true {wpx:.4f})")
        tol = max(1.0, 3 * wpx)
        ok = ok and err <= tol and abs(e["dz"] - wz) <= max(1.5, 0.1 * abs(wz) + 1.0) and abs(px / wpx - 1) < 0.03
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
