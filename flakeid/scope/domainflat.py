"""Flat field of the stacking-domain (IR) camera (start screen: Flat field, objective "IR (domain camera)").

On a bare chip at 100x with the IR configuration in place, the domain camera's averaged picture (domain_frames frames,
as the domain imaging takes them, flipped by domain_flip) is taken at flat_frames positions around the start; each is
divided by its own median (the camera's automatic level control changes the brightness from picture to picture), and
the per-pixel median of them is the flat (mean 1): dust and residue at one position do not survive the median.
No dark picture: with the camera's automatic level control a picture with the light blocked is read at another gain.  The flat goes to <local_root>/flat_fields/flat_domain_<date>.npz
(flat, dark, info) with a PNG preview; the export of the domain images divides by it (setting flat_domain, analysis
option --flat-domain), so their grey levels are contrast to a flat substrate whatever covers the frame.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from .scan import f1, f3

MAX_RING = 12


def run_domain_flat(sc):
    s, c = sc.s, sc.core
    out = Path(s.local_root) / "flat_fields"
    out.mkdir(parents=True, exist_ok=True)
    name = "flat_domain" + datetime.datetime.now().strftime("_%Y%m%d_%H%M")
    sc.open_log(out / f"{name}_log.txt")
    say = sc.say
    cams = None
    try:
        if not s.domain_camera:
            raise sc.fail("no domain camera in the settings (domain_camera): the IR flat field needs it")
        cams = sc.setup_domain_camera()                  # the IR camera from the start: every picture below is its own
        camD = cams[3]
        if camD != s.domain_camera:
            raise sc.fail(f"the camera selected is {camD}, not the domain camera {s.domain_camera}")
        say(f"  camera {camD} selected for the flat field ({sc.W} x {sc.H} px)")
        if sc.ui.can_ask:
            a = sc.ask_focus(f"Flat field: IR camera ({camD})",
                             f"This is the IR camera's picture ({camD}).\n\nPut the IR configuration in place (100x objective, IR "
                             "illumination and filter), move to a clean area of a bare chip (no flakes, no tape residue; the "
                             "pictures are taken around this point), focus on the substrate, then press \"Use this focus\"."
                             + ("\n\nNext the focus plane is found on the field-stop edge: you are asked to close the field stop."
                                if s.flat_focus == "fieldstop" else ""))
            if a != 0:
                raise sc.fail("flat field cancelled")
        X0, Y0 = c.getXPosition(sc.xy), c.getYPosition(sc.xy)
        sc.X0, sc.Y0 = X0, Y0
        if sc.has_z:
            sc.zref = sc.z_read()
        nfr = max(1, s.domain_frames)
        fov = max(sc.FOVX, sc.FOVY) / 9.54                     # the 100x field (the IR field is about the same)
        step = s.flat_step_um if s.flat_step_um > 0 else 1.1 * fov
        say(f"IR flat field ({camD}): {s.flat_frames} positions {f1(step)} um apart around {f1(X0)}, {f1(Y0)}, {nfr} frames each, "
            f"focus {s.flat_focus} -> {out}")
        # focus: the closed field stop's edge at the centre and 4 corners -> a tilted plane (bare substrate has nothing
        # else to focus on), as the colour flat fields; "fixed": the Z focused by hand; "auto": autofocus at each position
        plane = None
        if sc.has_z and s.flat_focus == "fieldstop":
            sc.ui.mock_hook(sc, "mockFieldStop", True)
            plane = sc.fieldstop_plane(step)
            sc.move_to(X0, Y0)
        zf = sc.z_read() if sc.has_z else 0.0

        def avg():
            if c.getCameraDevice() != camD:              # (never the scan camera's pictures in the IR flat)
                raise sc.fail(f"the camera changed to {c.getCameraDevice()} during the IR flat field")
            fr = [np.asarray(sc.frame(), np.float32) for _ in range(nfr)]
            g = np.mean(fr, axis=0)
            return g if g.ndim == 2 else g.mean(axis=2)

        pics, ring = [], 0
        while len(pics) < s.flat_frames and ring <= MAX_RING:
            pts = [(0, 0)] if ring == 0 else ([(i, -ring) for i in range(-ring, ring + 1)] + [(i, ring) for i in range(-ring, ring + 1)]
                                              + [(-ring, j) for j in range(-ring + 1, ring)] + [(ring, j) for j in range(-ring + 1, ring)])
            for gx, gy in pts:
                if len(pics) >= s.flat_frames or sc.stop_all:
                    break
                px, py = X0 + gx * step, Y0 + gy * step
                if ((px - X0) ** 2 + (py - Y0) ** 2) ** 0.5 > s.max_travel_um:
                    continue
                sc.move_to(px, py)
                how = ""
                if sc.has_z and plane:
                    zf = plane[0] * px + plane[1] * py + plane[2]
                    sc.set_z(zf - s.af_backlash_um); sc.set_z(zf)
                    how = f", Z {f3(zf)} (field-stop plane)"
                elif sc.has_z and s.flat_focus == "auto":
                    r = sc.autofocus(zf, s.af100)
                    if r is not None and r[0] is not None:
                        zf = r[0]
                    how = f", Z {f3(zf)} (autofocus)"
                g = avg()
                med = float(np.median(g))
                full = float(np.max(g))
                if med <= 0:
                    say(f"  {len(pics) + 1}: no signal (median {f1(med)}): skipped")
                    continue
                pics.append(g / med)
                say(f"  {len(pics)}/{s.flat_frames} at {f1(px)}, {f1(py)}{how}: median {f1(med)}, max {f1(full)} counts")
            ring += 1
        sc.move_to(X0, Y0)
        if sc.has_z and plane:
            z0 = plane[0] * X0 + plane[1] * Y0 + plane[2]
            sc.set_z(z0 - s.af_backlash_um); sc.set_z(z0)
        if len(pics) < 3:
            raise sc.fail(f"only {len(pics)} IR pictures for the flat field (need 3 or more)")
        st = np.asarray(pics, np.float32)
        flat = np.median(st, axis=0)
        flat = flat / float(np.mean(flat))
        spread = float(np.median(np.abs(st - flat[None] * np.median(st, axis=(1, 2))[:, None, None])))
        # no dark picture: the IR camera's automatic level control (Exposure: Auto) raises the gain when the light is
        # blocked, so a "dark" is read at another gain (2026-10-13: 30 % of the substrate level, with a band in its top
        # third that the pictures do not have); subtracting it put that band into every domain image
        dark = None
        lo, hi = np.percentile(flat, [1, 99])
        say(f"  flat: {len(pics)} pictures; illumination {f3(float(lo))} .. {f3(float(hi))} of the mean (1-99 %), "
            f"pictures differ from it by {f3(spread * 100)} % (median)")
        info = dict(camera=camD, positions=len(pics), frames=nfr, x_um=X0, y_um=Y0, step_um=step, flip=getattr(sc, "frame_flip", ""),
                    focus=s.flat_focus, plane=(list(map(float, plane)) if plane else None),
                    time=datetime.datetime.now().isoformat(timespec="seconds"))
        f = out / f"{name}.npz"
        np.savez_compressed(f, flat=flat.astype(np.float32), dark=(dark.astype(np.float32) if dark is not None else np.zeros(0, np.float32)),
                            info=json.dumps(info))
        try:
            import cv2
            v = np.clip((flat - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / f"{name}.png"), v)
        except Exception:
            pass
        say(f"  saved {f}")
        save_setting(sc, f)
        return f
    finally:
        if cams is not None:
            sc.restore_scan_camera(cams[0], cams[1], cams[2], cams[3])


def save_setting(sc, f):
    """flat_domain into the settings file, if the user agrees (a dated backup is kept)."""
    s = sc.s
    sf = Path(s.settings_file) if s.settings_file else None
    if sf is None or not sf.is_file():
        sc.say(f"  no settings file to update: set flat_domain = \"{f}\" in your scan settings")
        return
    if sc.ui.can_ask and sc.ui.confirm("Flat field (IR domain camera): save", f"Use this flat field for the domain images from now on?\n\n"
                                       f"   flat_domain = {f}\n\n(Saved into {sf.name}; a copy of the old file is kept.)",
                                       ("Use it", "Do not use it")) != 0:
        sc.say("  not set in the settings")
        return
    try:
        cfg = json.loads(sf.read_text(encoding="utf-8"))
        bak = sf.with_name(sf.stem + datetime.datetime.now().strftime("_before_flat_domain_%Y%m%d_%H%M") + sf.suffix)
        bak.write_text(sf.read_text(encoding="utf-8"), encoding="utf-8")
        cfg["flat_domain"] = str(f)
        tmp = sf.with_name(sf.name + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        tmp.replace(sf)
        sc.say(f"  flat_domain set in {sf} (old version: {bak.name})")
    except Exception as e:
        sc.say(f"  WARNING: the settings were not saved ({type(e).__name__}: {e}); set flat_domain = \"{f}\" by hand")

