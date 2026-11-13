"""Objective offsets (start screen "Objective offsets", or --mode offsets): where the same feature is with each
objective (5x, 10x, 50x, 100x) relative to a reference objective, in stage XY and in focus, and each objective's
pixel size and orientation.  Used by the device photos (the stage and focus follow when the objective changes, and the
photos' scale bars use the measured pixel sizes); written into the settings as objective_offsets.

  0  with the reference objective (offsets_reference, default 100x) you centre a small, sharp feature on the cross (a
     flake corner, a marker, dust: something with edges in X and Y) and focus
  1  its picture, focus and pixel calibration (stage moves of ~20 % of the field in X and in Y)
  2  the other objectives in a chain, each measured against the one before it (100x -> 50x -> 10x -> 5x; a reference
     in the middle measures down, then up): you change the objective (the stage and the focus untouched), the program
     focuses (a wide search: parfocality is often off by tens of um), calibrates the pixel, and matches its picture to the
     previous one on a common stage-aligned grid (edges only); the stage goes to the offset, the result is checked
     there, and you confirm or correct it by centring the feature on the cross
  3  the results in <local_root>/objoffsets_<date>_<time>/ (log, objoffsets.json, pictures) and, if you agree,
     objective_offsets in the settings file

Offsets: (dx, dy, dz) = the stage position and focus of the feature with that objective minus with the reference.
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import numpy as np

from ..devicephoto import MAG
from .domaintest import MIN_MATCH, gray, pow2, prep, to_grid, xcorr
from .scan import f1, f3

FRAMES = 4
CAL_FRACTION = 0.2
SEARCH = {"100x": (40.0, 1.0, 1.0, 0.25, 0.9), "50x": (60.0, 2.0, 2.0, 0.5, 0.9), "10x": (150.0, 6.0, 6.0, 1.0, 0.8),
          "5x": (250.0, 12.0, 12.0, 2.0, 0.8)}          # autofocus: (range, step, fine range, fine step, roi)


# ---------------------------------------------------------------------------------------------- used elsewhere
def update_settings_file(sc, updates):
    """Write updates (key -> value) into the settings file, a dated copy of the old one kept.  True if written."""
    s = sc.s
    sf = Path(s.settings_file) if s.settings_file else None
    if not sf or not sf.is_file():
        sc.say(f"  no settings file to update: set {', '.join(updates)} in your scan settings")
        return False
    try:
        txt = sf.read_text(encoding="utf-8")
        cfg = json.loads(txt)
        cfg.update(updates)
        sf.with_name(sf.stem + "_before_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + sf.suffix).write_text(txt, encoding="utf-8")
        sf.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return True
    except (OSError, ValueError) as e:
        sc.say(f"  (settings file not updated: {e})")
        return False


def offset_between(s, a, b):
    """(dx, dy, dz): stage and focus change from objective a to objective b for the same feature, or None if either is
    not measured."""
    T = s.objective_offsets or {}
    if a not in T or b not in T:
        return None
    return tuple(float(T[b].get(k, 0.0)) - float(T[a].get(k, 0.0)) for k in ("dx", "dy", "dz"))


def pixel_um(s, obj):
    """Measured um per camera pixel of objective obj (objective offsets tool), or None."""
    e = (s.objective_offsets or {}).get(obj) or {}
    M = e.get("M")
    if not M:
        return None
    try:
        return math.sqrt(abs(np.linalg.det(np.asarray(M, float))))
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------- the tool
def nominal_M(M10, obj):
    """um per camera pixel from the scan's 10x matrix M10, scaled by magnification (100x: the measured 9.54)."""
    k = 9.54 if obj == "100x" else MAG[obj] / 10.0
    return np.asarray(M10, float) / k


def run_objective_offsets(sc):
    s, c = sc.s, sc.core
    out = Path(s.local_root) / ("objoffsets_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    sc.open_log(out / "objoffsets_log.txt")
    say = sc.say
    ref = getattr(s, "offsets_reference", "100x") or "100x"
    todo = [o for o in (getattr(s, "offsets_objectives", None) or ("100x", "50x", "10x", "5x")) if o in MAG and o != ref]
    say(f"objective offsets: reference {ref}, measured {', '.join(todo)} -> {out}")
    M_scan = np.asarray(sc.M, float).copy()
    if s.set_camera:
        sc.apply_camera(s.target_exposure_ms)
    sc.DARK = s.dark * (sc.full_scale() + 1) / 4096.0
    sc.open_nosepiece()
    sc.move_box = None
    res = {}

    def avg(n=FRAMES):
        fr = [sc.frame() for _ in range(n)]
        return fr[0] if n == 1 else np.mean(np.asarray(fr, np.float32), axis=0)

    def picture(path, a, bayer):
        try:
            import cv2
            g, _ = gray(a, bayer)
            lo, hi = np.percentile(g, (0.5, 99.5))
            v = cv2.cvtColor(np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            hh, ww = v.shape[:2]
            cv2.drawMarker(v, (ww // 2, hh // 2), (200, 80, 255), cv2.MARKER_CROSS, max(hh, ww) // 12, 2)
            cv2.imwrite(str(path), v, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return str(path)
        except Exception:
            return None

    def objective_in(obj, first=False):
        from .devicephotos import remember_objective
        r = sc.wait_objective(obj, f"Put in the {obj} objective (the window closes by itself when it is in)."
                              + ("" if first else "  Do not move the stage or the focus."))
        if r is False:
            raise sc.fail("cancelled")
        if r is None:
            if sc.ui.confirm(f"Objective: {obj}", f"Put in the {obj} objective" + ("" if first else " (do not move the stage or the focus)")
                             + ", then press Continue.", ("Continue", "Cancel")) != 0:
                raise sc.fail("cancelled")
            sc.ui.mock_hook(sc, "mockObjective", int(MAG[obj]))
            if sc.nose is not None:
                try:
                    remember_objective(sc, obj, sc.nose.position())
                except Exception:
                    pass
        sc.objective_now = obj
        sc.M, sc.view_scale = nominal_M(M_scan, obj), 1.0      # (the live view's stage steps; the measured one once known)
        sc.stream_off()
        try:
            sc.check_exposure(True)
        except Exception as e:
            say(f"    exposure check: {e}")

    def focus(obj, zc, by_hand_text):
        r = sc.autofocus(zc, SEARCH[obj])
        if r[2]:
            return sc.z_read(), "autofocus"
        a = sc.ask_focus(f"Objective offsets: {obj} focus", by_hand_text, None)
        if a != 0:
            raise sc.fail(f"no {obj} focus")
        return sc.z_read(), "by hand"

    def calibrate(obj, X, Y):
        """um per camera pixel of the objective in (2 x 2 matrix) from stage moves; the nominal one if that fails."""
        raw = avg()
        bayer = np.asarray(raw).ndim == 2
        g0, gb = gray(raw, bayer)
        nC = min(512, pow2(min(g0.shape)))
        a0 = prep(g0, nC)
        Mn = nominal_M(M_scan, obj)
        D = CAL_FRACTION * nC * gb * math.sqrt(abs(np.linalg.det(Mn)))
        sc.move_to(X + D, Y)
        sx = xcorr(a0, prep(gray(avg(), bayer)[0], nC))
        sc.move_to(X, Y + D)
        sy = xcorr(a0, prep(gray(avg(), bayer)[0], nC))
        sc.move_to(X, Y)
        s0 = xcorr(a0, prep(gray(avg(), bayer)[0], nC))
        mi = np.array([[-sx[0] / D, -sy[0] / D], [-sx[1] / D, -sy[1] / D]])
        ok = min(sx[2], sy[2]) >= MIN_MATCH and abs(np.linalg.det(mi)) > 1e-12
        if not ok:
            say(f"    pixel calibration failed (match {f1(sx[2])}, {f1(sy[2])}): the nominal pixel used; use a feature with more structure")
            return Mn, False, (sx[2], sy[2])
        M = np.linalg.inv(mi) / gb
        sc.M = M
        px = math.sqrt(abs(np.linalg.det(M)))
        rot = (math.degrees(math.atan2(M[1, 0], M[0, 0]) - math.atan2(M_scan[1, 0], M_scan[0, 0])) + 180.0) % 360.0 - 180.0
        say(f"    pixel: {f3(px)} um per camera px (nominal {f3(math.sqrt(abs(np.linalg.det(Mn))))}), turned {f1(rot)} deg against the "
            f"10x scan camera" + (", MIRRORED" if np.linalg.det(M) * np.linalg.det(M_scan) < 0 else "")
            + f"; calibration moves {f1(D)} um, back at the start {f3(s0[0])}, {f3(s0[1])} px")
        return M, True, (sx[2], sy[2])

    def shot(obj, M, tag):
        raw = avg()
        bayer = np.asarray(raw).ndim == 2
        g, gb = gray(raw, bayer)
        sc.save_image(out / f"objoffsets_{tag}.tif", np.asarray(raw).astype(np.float32),
                      {"objective": obj, "stage_x_um": c.getXPosition(sc.xy), "stage_y_um": c.getYPosition(sc.xy), "z_um": sc.z_read()})
        return dict(g=g, M=np.asarray(M, float) * gb, bayer=bayer, pic=picture(out / f"objoffsets_{tag}.jpg", raw, bayer))

    def ext(sh):
        g, M = sh["g"], sh["M"]
        h, w = g.shape
        return min(w * abs(M[0, 0]) + h * abs(M[0, 1]), w * abs(M[1, 0]) + h * abs(M[1, 1]))

    try:
        # ---- 0, 1: the reference
        objective_in(ref, first=True)
        a = sc.ask_focus(f"Objective offsets: the feature ({ref})",
                         "Centre a small, sharp feature on the cross (a flake corner, a marker, dust: something with edges in X and "
                         "Y that is also visible at low magnification, e.g. the corner of a large flake), with the stage, and focus.  "
                         f"Then press \"Use this position\".\n\nThe program then asks you to change to the other objectives "
                         f"({', '.join(todo)}) in turn without moving the stage or the focus.", None, button="Use this position", cross=True)
        if a != 0:
            raise sc.fail("cancelled")
        X0, Y0 = c.getXPosition(sc.xy), c.getYPosition(sc.xy)
        z0, how = focus(ref, sc.z_read(), "No focus found: focus by hand, then press \"Use this focus\".")
        say(f"  {ref}: feature at {f1(X0)}, {f1(Y0)} um, Z {f3(z0)} ({how})")
        M, okc, mt = calibrate(ref, X0, Y0)
        res[ref] = dict(dx=0.0, dy=0.0, dz=0.0, M=M.tolist(), calibrated=okc, focus=how, ref=ref)
        base = {ref: (shot(ref, M, ref), X0, Y0, z0)}

        lower = sorted([o for o in todo if MAG[o] < MAG[ref]], key=lambda o: -MAG[o])
        higher = sorted([o for o in todo if MAG[o] > MAG[ref]], key=lambda o: MAG[o])
        for chain in (lower, higher):
            prev = ref
            for obj in chain:
                Pshot, PX, PY, Pz = base[prev]
                if (c.getXPosition(sc.xy), c.getYPosition(sc.xy)) != (PX, PY):
                    sc.move_to(PX, PY)
                    sc.set_z(Pz)
                objective_in(obj)
                say(f"  {obj}: measured against {prev}")
                zO, how = focus(obj, Pz, f"No focus found with the {obj} objective: focus by hand on the feature, then press \"Use this focus\".")
                say(f"    focus Z {f3(zO)} ({how}); {obj} - {prev} = {f3(zO - Pz)} um")
                MO, okc, mt = calibrate(obj, PX, PY)
                Oshot = shot(obj, MO, f"{obj}_at_{prev}")
                nG = 512
                gG = max(math.sqrt(abs(np.linalg.det(Pshot["M"]))), math.sqrt(abs(np.linalg.det(Oshot["M"]))), min(ext(Pshot), ext(Oshot)) / nG)
                gP = prep(to_grid(Pshot["g"], Pshot["M"], nG, gG), nG)
                m = xcorr(gP, prep(to_grid(Oshot["g"], Oshot["M"], nG, gG), nG))
                dx, dy = m[0] * gG, m[1] * gG
                say(f"    match: {obj} - {prev} = {f3(dx)}, {f3(dy)} um (match {f1(m[2])}; grid {f3(gG)} um per px)")
                if m[2] < MIN_MATCH:
                    say(f"    WARNING: weak match ({f1(m[2])} < {MIN_MATCH}); check by hand below")
                sc.move_to(PX + dx, PY + dy)
                ck = xcorr(gP, prep(to_grid(gray(avg(), Oshot["bayer"])[0], Oshot["M"], nG, gG), nG))
                if math.hypot(ck[0] * gG, ck[1] * gG) > max(2.0, 3 * gG) and ck[2] >= MIN_MATCH:
                    dx, dy = dx + ck[0] * gG, dy + ck[1] * gG
                    sc.move_to(PX + dx, PY + dy)
                    say(f"    corrected once: {f3(dx)}, {f3(dy)} um")
                a = sc.ask_focus(f"Objective offsets: {obj}", f"The feature that was on the cross with the {prev} objective (the picture "
                                 "on the right) should now be on the cross.  If it is not, centre it with the stage (and focus).\n\n"
                                 "\"Use this position\": the position now.  \"Skip\": keep the automatic result.", None,
                                 image=Pshot["pic"], button="Use this position", cross=True)
                xy_how = "image match"
                if a == 2:
                    raise sc.fail("cancelled")
                if a == 0:
                    hx, hy = c.getXPosition(sc.xy) - PX, c.getYPosition(sc.xy) - PY
                    if math.hypot(hx - dx, hy - dy) > 0.05:
                        say(f"    centred by hand: {f3(hx)}, {f3(hy)} um ({f3(hx - dx)}, {f3(hy - dy)} from the automatic result)")
                        xy_how = "by hand"
                    dx, dy = hx, hy
                    zO = sc.z_read()
                Pr = res[prev]
                res[obj] = dict(dx=round(Pr["dx"] + dx, 3), dy=round(Pr["dy"] + dy, 3), dz=round(Pr["dz"] + zO - Pz, 3), M=MO.tolist(),
                                calibrated=okc, match=round(float(m[2]), 2), xy=xy_how, focus=how, via=prev, ref=ref)
                base[obj] = (shot(obj, MO, obj), PX + dx, PY + dy, zO)
                prev = obj
    finally:
        sc.objective_now = ""
        sc.M = M_scan
        sc.stream_off()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for v in res.values():
        v["date"] = stamp
    (out / "objoffsets.json").write_text(json.dumps(res, indent=1))
    say("objective offsets (stage um and focus um relative to " + ref + "):")
    for o in sorted(res, key=lambda o: -MAG[o]):
        e = res[o]
        say(f"  {o:>5}: dx {e['dx']:+9.3f}  dy {e['dy']:+9.3f}  dz {e['dz']:+9.3f}   {pixel_um_of(e):.4f} um/px"
            + ("" if e["calibrated"] else " (nominal)") + (f"   via {e['via']}, {e['xy']}" if "via" in e else ""))
    sc.report_times()
    if sc.ui.confirm("Objective offsets", "Measured (stage and focus, relative to " + ref + "):\n\n"
                     + "\n".join(f"{o}: {res[o]['dx']:+.2f}, {res[o]['dy']:+.2f} um, focus {res[o]['dz']:+.2f} um, {pixel_um_of(res[o]):.4f} um/px"
                                 for o in sorted(res, key=lambda o: -MAG[o]))
                     + f"\n\nWrite them into the settings (objective_offsets)?\n(The results are in {out}.)", ("Write", "Do not write")) == 0:
        T = dict(s.objective_offsets or {})
        T.update(res)
        s.objective_offsets = T
        if update_settings_file(sc, {"objective_offsets": T}):
            say("  written into the settings file (objective_offsets)")
    return res


def pixel_um_of(e):
    return math.sqrt(abs(np.linalg.det(np.asarray(e["M"], float))))
