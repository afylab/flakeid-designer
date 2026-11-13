"""Scale bar calibration (start screen "Scale bar calibration", or --mode scalecal; Avram, 2026-10-27): each objective's
um per camera pixel from a slide or chip with known distances.

For each objective in turn (5x, 10x, 50x, 100x: those of objective_positions, else these): the program asks for it (the
nosepiece reader waits for it), the person puts a feature of known length under it, draws a line between its two ends in
the live view (the Measure tool) and types the length in um.  um per pixel = length / the line's length in camera pixels.

Saved in <local_root>/scale_calibration.json ({objective: dict(um_per_px, length_um, px, measured)}), merged with the
objectives calibrated before.  The live views' scale bar and Measure tool use it for the objective in use (the pause
window's picture tools); without it the stage calibration M (10x) and the magnification (100x: the measured 9.54, others
10 / magnification).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .scan import f3

SCALE_FILE = "scale_calibration.json"
OBJECTIVES = ("5x", "10x", "50x", "100x")


def load_scale_calibration(local_root):
    """{objective: um per camera pixel} from <local_root>/scale_calibration.json, or {}."""
    try:
        d = json.loads((Path(local_root) / SCALE_FILE).read_text(encoding="utf-8"))
        return {k: float(v["um_per_px"]) for k, v in d.items() if isinstance(v, dict) and float(v.get("um_per_px") or 0) > 0}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def nominal_um_per_px(sc, name):
    """um per camera pixel of objective `name` without a calibration: the stage calibration M (10x) x the objective
    (100x: the measured ratio 9.54, others 10 / magnification)."""
    try:
        mag = float(str(name).lower().rstrip("x"))
    except ValueError:
        return float("nan")
    k = 1.0 / 9.54 if abs(mag - 100) < 1e-6 else 10.0 / mag
    return float(np.sqrt(abs(np.linalg.det(np.asarray(sc.M, float))))) * k


def save_scale_calibration(local_root, name, rec):
    f = Path(local_root) / SCALE_FILE
    try:
        d = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    except (OSError, ValueError):
        d = {}
    d[name] = rec
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=1), encoding="utf-8")


def run_scale_calibration(sc):
    """The scale bar calibration, objective after objective (see the module's docstring)."""
    s = sc.s
    sc.set_phase("scale bar calibration")
    sc.open_nosepiece()
    names = [o for o in OBJECTIVES if o in (s.objective_positions or {})] or list(OBJECTIVES)
    if s.set_camera:
        sc.apply_camera(s.exposure_ms)
    sc.DARK = s.dark * (sc.full_scale() + 1) / 4096.0
    cal = load_scale_calibration(s.local_root)
    sc.say(f"scale bar calibration: {', '.join(names)}; saved in {Path(s.local_root) / SCALE_FILE}")
    done = []
    for i, name in enumerate(names):
        if sc.nose is not None and (s.objective_positions or {}).get(name):
            if sc.wait_objective(name, "Scale bar calibration: put it in (the window closes by itself when it is in).") is False:
                break
        else:
            a = sc.ui.confirm(f"Scale bar calibration: {name}", f"Put in the {name} objective, then a slide or chip with "
                              "distances you know (a stage micrometer, marks of known spacing), and focus.",
                              ("It is in", "Skip this objective", "Finish"))
            sc.ui.mock_hook(sc, "mockObjective", float(name.rstrip("x")))
            if a == 1:
                continue
            if a != 0:
                break
        cur = cal.get(name, nominal_um_per_px(sc, name))
        r = sc.ui.scale_calibrate(sc, dict(objective=name, current=cur, calibrated=name in cal, next=names[i + 1] if i + 1 < len(names) else None))
        if r == "finish" or r is None:
            break
        if r == "skip":
            continue
        rec = dict(um_per_px=round(float(r["um_per_px"]), 6), length_um=float(r["length_um"]), px=round(float(r["px"]), 2),
                   measured=time.strftime("%Y-%m-%d %H:%M"), before=round(float(cur), 6))
        save_scale_calibration(s.local_root, name, rec)
        cal[name] = rec["um_per_px"]
        done.append(name)
        sc.say(f"  {name}: {f3(rec['um_per_px'])} um per pixel ({rec['length_um']:g} um over {rec['px']:.1f} px; before "
               f"{f3(cur)}, {100 * (rec['um_per_px'] / cur - 1):+.2f} %)")
    sc.say(f"scale bar calibration: {', '.join(done) if done else 'nothing'} calibrated")
    sc.scale_cal = load_scale_calibration(s.local_root)
