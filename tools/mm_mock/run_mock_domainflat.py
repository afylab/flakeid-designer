"""Simulated IR flat field (scan: Flat field, objective "IR (domain camera)") and its use in the domain export.

    python tools/mm_mock/run_mock_domainflat.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.hundred import load_domain_flat  # noqa: E402
from flakeid.scope.mock import MockCore  # noqa: E402
from flakeid.scope.scan import Scanner  # noqa: E402
from flakeid.scope.settings import load  # noqa: E402
from flakeid.scope.ui import AutoUI  # noqa: E402


def main():
    out = ROOT / "tools" / "mm_mock" / "out" / "domainflat"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    core = MockCore(1.09, x0=50000.0, y0=20000.0)
    core.mockObjective(100)
    (out / "settings.json").write_text("{}")
    s = load(None, mode="flat", flat_objective="domain", local_root=str(out), domain_camera="DomainCam", settings_file=str(out / "settings.json"),
             flat_frames=9, flat_dark_frames=0, domain_frames=4, z_settle_ms=0.0, verbose=False, af_manual_fallback="never")
    sc = Scanner(core, s, AutoUI(core, analysis=False))
    core.setCameraDevice("DomainCam"); wh_ir = (core.getImageWidth(), core.getImageHeight())
    core.setCameraDevice("MockCam"); wh_scan = (core.getImageWidth(), core.getImageHeight())
    sc.run()
    fs = sorted((out / "flat_fields").glob("flat_domain_*.npz"))
    saved = json.loads((out / "settings.json").read_text())
    ok = bool(fs) and saved.get("flat_domain") == str(fs[-1])
    if fs:
        ff = load_domain_flat(fs[-1])
        flat = ff[0]
        print(f"flat {flat.shape[1]} x {flat.shape[0]}, mean {flat.mean():.4f}, 1-99 % {np.percentile(flat, 1):.4f} .. {np.percentile(flat, 99):.4f}; "
              f"camera {core.getCameraDevice()} after the run; settings {saved}")
        print(f"IR camera {wh_ir[0]} x {wh_ir[1]} px, scan camera {wh_scan[0]} x {wh_scan[1]} px")
        ok = ok and (flat.shape[1], flat.shape[0]) == tuple(wh_ir) and abs(float(flat.mean()) - 1) < 1e-3 and np.isfinite(flat).all() and core.getCameraDevice() != "DomainCam"
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
