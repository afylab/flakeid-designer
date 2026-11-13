"""Minimal test of the IR camera (the stacking-domain camera, Raptor Owl 640) on the microscope PC: loads the scan
program's hardware configuration, selects the camera, wakes it up the way it needs in Micro-Manager (live once, stop,
then the settings, then live again), reads frames from the live stream and as single snaps, saves them and reports what
came back.  Nothing moves: no stage or focus commands.

    C:\\FlakeID\\run_flakeid.bat irtest                         (the scan program's configuration and settings)
    C:\\FlakeID\\run_flakeid.bat irtest --live 60               (then a live window for 60 s, to focus on a flake)
    C:\\FlakeID\\run_flakeid.bat irtest --exposure 50 --prop "High Gain=On"
    C:\\FlakeID\\run_flakeid.bat irtest --no-wake               (the settings without the first live: for comparison)
    python tools/win7/ir_camera_test.py --fake                  (a simulated camera: checks the script, no hardware)

Before: close the Micro-Manager GUI and the camera's own software (only one program can own a camera); the camera
powered on (and given a few minutes to cool if it has a cooler).  For a picture with something in it: the IR objective,
the IR filter and the lamp, over a graphite flake; for the camera alone, any field (even dark) will do.

Steps (each printed as OK / INFO / WARN / FAIL):
  1  load the hardware configuration (--cfg); Micro-Manager versions
  2  the cameras in it; select the IR camera (--camera, default domain_camera of the settings)
  3  every property of the camera (name = value, read-only, allowed values) and any temperature reading
  4  wake-up (as in Micro-Manager, where the settings only take hold after the camera has been live once): live for
     --wake-s seconds, then stopped (--no-wake: left out)
  5  the settings: the scan's domain_camera_props (High Gain = Off, Frame Average = 4; --no-scan-props: not), --prop, the
     exposure (--exposure, default domain_exposure_ms); read back
  6  live again: frames per second, and the levels of the last --frames frames (against full scale, contrast across the
     frame, saturation, frame-to-frame noise, frames that repeat)
  7  exposure response, still live: the median at half and twice the exposure; then the live window (--live)
  8  the stop (asked once, up to 20 s; if it does not stop: Frame Average set to 1, to see if that frees it), then single
     snaps (what the scan program's domain imaging takes): the same numbers, against the stream's
  9  files: live_frame.tif (raw, as read), live_mean.tif (mean, 32-bit), snap_frame.tif, live_preview.png and
     snap_preview.png (contrast-stretched)
The camera selected before is selected again at the end.  Writes everything to --out (default
C:\\FlakeScans\\ir_test_<date>_<time>); send report.json and the previews.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORT = {"python": sys.version.split()[0], "platform": platform.platform(), "steps": []}
OUT = {"dir": None}
CAMERA_TYPE = 2                       # MMCore DeviceType: CameraDevice


def step(name, status, msg, **data):
    print(f"[{status:4s}] {name}: {msg}", flush=True)
    REPORT["steps"].append(dict(step=name, status=status, msg=msg, **data))
    if OUT["dir"] is not None:                       # (saved at once: a run that hangs still leaves what it found)
        try:
            (OUT["dir"] / "report.json").write_text(json.dumps(REPORT, indent=1, default=str))
        except OSError:
            pass


def is_camera(core, dev):
    try:
        return int(core.getDeviceType(dev)) == CAMERA_TYPE
    except Exception:
        return "cam" in dev.lower()


def to_array(core, img):
    a = np.asarray(img)
    if a.ndim == 1:
        a = a.reshape(core.getImageHeight(), core.getImageWidth())
    return a


def snap(core):
    core.snapImage()
    return to_array(core, core.getImage())


def stretch(a):
    """8-bit preview: 0.5 - 99.5 % of the values over the full range."""
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    v = a[::2, ::2]
    v = v[v > 0] if (v > 0).mean() > 0.5 else v.ravel()         # (zero pixels: dead pixels, the border)
    lo, hi = np.percentile(v, (0.5, 99.5))
    return np.clip((a - lo) / max(float(hi - lo), 1e-6) * 255.0, 0, 255).astype(np.uint8), float(lo), float(hi)


STOP_WAIT = {"s": 0.0, "stuck": False}


def stop_live(core, timeout=20.0):
    """Stop the live stream and wait until the camera says it has stopped (the Raptor's driver can keep reporting
    "running" after the stop; a snap or a new start before then fails).  The stop is asked once, and once more half way.
    True when stopped; the longest wait is recorded."""
    t0, again = time.time(), False
    try:
        core.stopSequenceAcquisition()
    except Exception:
        pass
    while core.isSequenceRunning() and time.time() - t0 < timeout:
        time.sleep(0.1)
        if not again and time.time() - t0 > timeout / 2:
            again = True
            try:
                core.stopSequenceAcquisition()
            except Exception:
                pass
    dt = time.time() - t0
    STOP_WAIT["s"] = max(STOP_WAIT["s"], dt)
    if core.isSequenceRunning():
        STOP_WAIT["stuck"] = True
        return False
    return True


def grab(core, seconds, keep=1, skip=2):
    """Frames of the running live stream for `seconds`: (the last `keep`, copies; frames/s); the first `skip` frames are
    not kept (they can be from before a change)."""
    got, n, t0 = [], 0, time.time()
    while time.time() - t0 < seconds:
        k = core.getRemainingImageCount()
        if not k:
            time.sleep(0.005)
            continue
        a = np.array(to_array(core, core.getLastImage()), copy=True)
        core.clearCircularBuffer()
        n += k
        if n > skip:
            got.append(a)
            got = got[-keep:]
    return got, n / max(time.time() - t0, 1e-6)


def camera_only_cfg(cfg, cam, out):
    """A hardware configuration with only the camera `cam` (its Device and Property lines from `cfg`, the core
    initialised with it as the camera): for testing the camera without the other devices."""
    dev, pre, post, inited = [], [], [], False
    for ln in Path(cfg).read_text(errors="replace").splitlines():
        f = [x.strip() for x in ln.split(",")]
        if not f or ln.startswith("#"):
            continue
        if f[:3] == ["Property", "Core", "Initialize"] and len(f) > 3 and f[3] == "1":
            inited = True                                # (properties after this line exist only once initialised)
        elif f[0] == "Device" and len(f) > 1 and f[1] == cam:
            dev.append(ln)
        elif f[0] == "Property" and len(f) > 1 and f[1] == cam:
            (post if inited else pre).append(ln)
    if not dev:
        raise RuntimeError(f"no Device line for {cam!r} in {cfg}")
    txt = "# made by ir_camera_test.py --camera-only from " + str(cfg) + "\n" + "\n".join(
        ["Property,Core,Initialize,0"] + dev + pre + ["Property,Core,Initialize,1"] + post + [f"Property,Core,Camera,{cam}"]) + "\n"
    path = Path(out) / "camera_only.cfg"
    path.write_text(txt)
    return path


def stream(core, seconds, keep=1):
    """Live (continuous sequence acquisition) for `seconds`, then stopped: (the last `keep` frames, copies; frames/s).
    The first two frames after the start are not kept (they can be from before)."""
    if core.isSequenceRunning():
        stop_live(core)
    core.startContinuousSequenceAcquisition(0)
    got, n, t0 = [], 0, time.time()
    try:
        while time.time() - t0 < seconds:
            k = core.getRemainingImageCount()
            if not k:
                time.sleep(0.005)
                continue
            a = np.array(to_array(core, core.getLastImage()), copy=True)
            core.clearCircularBuffer()
            n += k
            if n > 2:
                got.append(a)
                got = got[-keep:]
    finally:
        dt = time.time() - t0
        stop_live(core)
    return got, n / max(dt, 1e-6)


def frame_stats(f, full, frames=None):
    """Levels of frame f (and frame-to-frame numbers over `frames`)."""
    f = np.asarray(f)
    g = f.astype(np.float64)
    if g.ndim == 3:
        g = g[..., :3].mean(axis=2)
    p1, p50, p99 = np.percentile(g, (1, 50, 99))
    # contrast across the frame: the spread of a smoothed copy (pixel noise left out), against the level
    h, w = g.shape
    b = g[: h // 8 * 8, : w // 8 * 8].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
    d = dict(dtype=str(f.dtype), shape=list(f.shape), min=float(g.min()), p1=float(p1), median=float(p50), p99=float(p99), max=float(g.max()),
             spread_8px=float(np.percentile(b, 99) - np.percentile(b, 1)), saturated_frac=float((g >= full).mean()),
             zero_frac=float((g == 0).mean()))
    if frames is not None and len(frames) > 1:
        st = np.asarray(frames, np.float64)
        d["noise_counts"] = float(np.std(st[1:] - st[:-1]) / math.sqrt(2))
        d["repeated_frames"] = sum(1 for i in range(1, len(frames)) if np.array_equal(frames[i], frames[i - 1]))
    else:
        d["noise_counts"], d["repeated_frames"] = float("nan"), 0
    return d


def describe(d, full):
    return (f"median {d['median']:.0f} of {full:g} ({d['median'] / full * 100:.1f} %), 1-99 % {d['p1']:.0f}..{d['p99']:.0f}, "
            f"contrast across the frame {d['spread_8px']:.0f}, saturated {d['saturated_frac'] * 100:.2f} %, zeros {d['zero_frac'] * 100:.1f} %"
            + (f", noise {d['noise_counts']:.1f} per frame" if d['noise_counts'] == d['noise_counts'] else "")
            + (f", {d['repeated_frames']} repeated frame(s)" if d.get("repeated_frames") else ""))


def judge(d, full):
    """(status, why): FAIL for no picture at all; WARN for dark, saturated, flat (no contrast across the frame) or
    repeated frames; else OK."""
    if d["max"] == d["min"] or d["zero_frac"] > 0.99:
        return "FAIL", "  -> NO PICTURE: every pixel the same (or zero)"
    noise = d["noise_counts"] if d["noise_counts"] == d["noise_counts"] else 0.0
    why = []
    if d["median"] < 0.02 * full:
        why.append("very dark: lamp, IR filter, shutter, settings?")
    if d["saturated_frac"] > 0.01:
        why.append("saturated: shorter exposure")
    if d["spread_8px"] < max(4.0 * noise, 0.002 * full):
        why.append("flat: no contrast across the frame beyond the pixel noise (no signal, or a blank field)")
    if d.get("repeated_frames"):
        why.append("repeated frames: the camera may be returning an old frame")
    return ("WARN", "  -> " + "; ".join(why)) if why else ("OK", "")


def diagnose(core, cam, s, full, out, exp):
    """--diagnose: which order of operations gives pictures, and where the stop hangs.  Snaps and the stop that works
    come first; the second live (whose stop hung on the Owl) last.  Every step is saved at once, with a preview
    (D<n>_<what>.png)."""
    import cv2

    def save(tag, f):
        try:
            cv2.imwrite(str(out / (tag.split(" ")[0] + ".png")), stretch(f)[0])
        except Exception:
            pass

    def props(tag):
        names = ("Exposure: Auto", "High Gain", "Frame Average", "Exposure", "Frame Rate", "Frame Rate (User)")
        vals = {}
        for n in names:
            try:
                vals[n] = core.getProperty(cam, n)
            except Exception:
                pass
        step(tag, "INFO", "; ".join(f"{k} = {v}" for k, v in vals.items()) + f"; core exposure {core.getExposure():g} ms", values=vals)

    def snaps(tag, n=3):
        fr = []
        for _ in range(n):
            try:
                fr.append(snap(core))
            except Exception as e:
                step(tag, "FAIL", f"snap failed: {type(e).__name__}: {e}")
                return None
        st = frame_stats(fr[-1], full, fr)
        jv = judge(st, full)
        step(tag, jv[0], f"snaps: {describe(st, full)}{jv[1]}", **st)
        save(tag, fr[-1])
        return st

    def live(tag, sec=2.5):
        try:
            if core.isSequenceRunning():
                stop_live(core)
            core.startContinuousSequenceAcquisition(0)
        except Exception as e:
            step(tag, "FAIL", f"live did not start: {type(e).__name__}: {e}")
            return False
        fr, fps = grab(core, sec, keep=3)
        if fr:
            st = frame_stats(fr[-1], full, fr)
            jv = judge(st, full)
            step(tag, jv[0], f"live {fps:.1f} frames/s: {describe(st, full)}{jv[1]}", fps=fps, **st)
            save(tag, fr[-1])
        else:
            step(tag, "WARN", "live, but no frames")
        return True

    def stop(tag, timeout=15.0):
        t0 = time.time()
        ok = stop_live(core, timeout)
        step(tag, "OK" if ok else "WARN", f"stopped in {time.time() - t0:.1f} s" if ok else f"still live {timeout:.0f} s after the stop")
        return ok

    def exposure_try(tag, how):
        """Set the exposure one way, read it back, snap at it and at twice it: does the level follow?"""
        res = {}
        for e in (exp, 2 * exp):
            try:
                if how == "property":
                    core.setProperty(cam, "Exposure", f"{e:g}")
                else:
                    core.setExposure(e)
            except Exception as ex:
                step(tag, "WARN", f"{e:g} ms not set ({how}): {type(ex).__name__}: {ex}")
                return None
            got = float(core.getExposure())
            try:
                gp = core.getProperty(cam, "Exposure")
            except Exception:
                gp = "?"
            try:
                snap(core)                               # (the first frame after a change can be from before)
                f = snap(core)
                res[e] = (got, gp, float(np.median(f)))
            except Exception as ex:
                step(tag, "FAIL", f"snap failed: {type(ex).__name__}: {ex}")
                return None
        (e1, r1), (e2, r2) = sorted(res.items())
        follows = r2[2] > 1.4 * r1[2] or r2[2] >= 0.98 * full
        step(tag, "OK" if follows else "WARN", f"asked {e1:g} ms: core {r1[0]:g}, property {r1[1]}, median {r1[2]:.0f};  asked {e2:g} ms: "
             f"core {r2[0]:g}, property {r2[1]}, median {r2[2]:.0f}" + ("" if follows else "  -> the level does not follow the exposure"),
             tries={f"{k:g}": v for k, v in res.items()})
        return follows

    def setp(tag, nm, val):
        try:
            core.setProperty(cam, nm, val)
            return True
        except Exception as e:
            step(tag, "WARN", f"{nm} = {val} not set: {type(e).__name__}: {e}")
            return False

    def exposures(tag, values):
        """The exposure set (core) to each value, read back, two snaps: does the level follow?"""
        res = {}
        for e in values:
            try:
                core.setExposure(e)
                got = float(core.getExposure())
                snap(core)                               # (the first frame after a change can be from before)
                res[e] = (got, float(np.median(snap(core))))
            except Exception as ex:
                step(tag, "FAIL", f"{e:g} ms: {type(ex).__name__}: {ex}")
                return None
        m = [res[k][1] for k in sorted(res)]
        follows = all(m[i + 1] > 1.3 * m[i] or m[i + 1] >= 0.98 * full for i in range(len(m) - 1))
        step(tag, "OK" if follows else "WARN", ", ".join(f"asked {k:g} ms: camera {v[0]:g} ms, median {v[1]:.0f}" for k, v in sorted(res.items()))
             + ("" if follows else "  -> the level does not follow the exposure"), tries={f"{k:g}": v for k, v in res.items()})
        return follows

    # 2026-10-09 (ir_test_20261009_165716 and Raptor's own software): the exposure changes nothing at all, at any frame rate,
    # also in the camera's own software.  Suspect: the camera's level control (auto exposure / AGC) still running though
    # "Exposure: Auto" reads Off (the driver repeats the value it set unless "Force Update" is On; it reads the auto-exposure
    # state from the camera only then).  Tests: the real state with Force Update; the brightness when the auto-exposure
    # target ("Xpert: AE Set Level") changes with auto exposure Off (a running control follows it; a stopped one does
    # not), the same with it On as a check; the digital gain.
    def levels(tag, nm, values, note=""):
        res = {}
        for v in values:
            if not setp(tag, nm, str(v)):
                return None
            try:
                time.sleep(0.5)                          # (a level control needs a few frames to settle)
                snap(core)
                res[v] = float(np.median(snap(core)))
            except Exception as ex:
                step(tag, "FAIL", f"{type(ex).__name__}: {ex}")
                return None
        vals = [res[v] for v in values]
        changes = max(vals) > 1.3 * min(vals)
        step(tag, "INFO", ", ".join(f"{nm} {v}: median {res[v]:.0f}" for v in values) + (f"  -> the brightness FOLLOWS {nm}" if changes
             else f"  -> the brightness does not follow {nm}") + note, medians={str(v): res[v] for v in values}, follows=changes)
        return changes

    step("D0", "INFO", "diagnose: is the camera's level control (auto exposure / AGC) still running with auto exposure off?")
    props("D0 as loaded")
    snaps("D1 snaps as loaded")
    live("D2 live as loaded")
    if not stop("D2 stop"):
        step("D2", "WARN", "the first stop already hangs: the rest needs a stopped camera")
        return
    setp("D3", "Frame Average", "1")
    setp("D3", "Force Update", "On")                     # (from here the driver reads the auto-exposure state from the camera)
    props("D3 Force Update On (the states read from the camera)")
    setp("D4", "Exposure: Auto", "Off")
    time.sleep(0.5)
    props("D4 auto exposure set Off: the camera's state now")
    exposures("D5 exposure, auto exposure Off, High Gain On", (2.0, 4.0, 8.0, 16.0))
    try:
        target0 = core.getProperty(cam, "Xpert: AE Set Level")
    except Exception:
        target0 = None
    if target0 is not None:
        levels("D6 auto-exposure target, auto exposure OFF", "Xpert: AE Set Level", (4000, 14000),
               "  (with the control really off this should not matter)")
        setp("D7", "Exposure: Auto", "On")
        props("D7 auto exposure On")
        levels("D7 auto-exposure target, auto exposure ON (the check)", "Xpert: AE Set Level", (4000, 14000),
               "  (with the control on this should change the brightness)")
        setp("D7", "Xpert: AE Set Level", str(target0))
        setp("D7", "Exposure: Auto", "Off")
    try:
        g0 = core.getProperty(cam, "Gain")
        levels("D8 digital gain, auto exposure Off", "Gain", (1, 2))
        setp("D8", "Gain", str(g0))
    except Exception as e:
        step("D8", "WARN", f"Gain not tried: {type(e).__name__}: {e}")
    # back as loaded
    setp("D9", "Exposure: Auto", "On")
    setp("D9", "Force Update", "Off")
    props("D9 end (auto exposure on again)")


def fake_core():
    """The simulated microscope's IR camera (flakeid.scope.mock: DomainCam, 1024 x 1024, 16 bit) over a few flakes."""
    import cv2
    from flakeid.scope.mock import MockCore
    core = MockCore(1.09, x0=30000.0, y0=20000.0)
    img = np.full((2000, 2000, 3), (150, 120, 190), np.uint8)
    rng = np.random.default_rng(2)
    for _ in range(40):
        u, v = rng.uniform(900, 1100, 2)
        cv2.circle(img, (int(u), int(v)), int(rng.uniform(3, 12)), (60, 160, 230), -1)
    core.add_chip(img, 30000.0, 20000.0, 0.0, 0.0)
    core.mockDomain()
    core.setCameraDevice("MockCam")
    core.getDeviceType = lambda d: CAMERA_TYPE if "cam" in d.lower() else 0
    core.z = core.true_focus(core.x, core.y)
    # as the Raptor's driver: still "running" for a moment after a stop; snaps and starts refused until then
    t_stop, real_stop, real_run, real_snap, real_start = [0.0], core.stopSequenceAcquisition, core.isSequenceRunning, core.snapImage, \
        core.startContinuousSequenceAcquisition
    busy = "This operation can not be executed while sequence acquisition is running."

    import os
    stuck = os.environ.get("FLAKEID_FAKE_STUCK") == "1"     # (as the Owl did: a stop ignored while Frame Average is 4)
    dprops = {"Exposure: Auto": "On", "High Gain": "On", "Frame Average": "1"}
    real_get, real_set, real_names = core.getProperty, core.setProperty, core.getDevicePropertyNames
    core.getProperty = lambda d, n: dprops[n] if d == "DomainCam" else real_get(d, n)
    core.setProperty = lambda d, n, v: dprops.__setitem__(n, str(v)) if d == "DomainCam" else real_set(d, n, v)
    core.getDevicePropertyNames = lambda d: list(dprops) if d == "DomainCam" else real_names(d)

    def stop(*a_):
        if stuck and dprops["Frame Average"] != "1":
            return
        if real_run():
            t_stop[0] = time.time()
        real_stop()

    def running(*a_):
        return real_run() or time.time() - t_stop[0] < 0.8

    def snap_():
        if running():
            raise RuntimeError(busy)
        real_snap()

    def start(*a_):
        if running():
            raise RuntimeError(busy)
        real_start(*a_)
    real_last = core.getLastImage

    def last():                                          # (the simulator renders a stream frame with its own snap)
        core.snapImage = real_snap
        try:
            return real_last()
        finally:
            core.snapImage = snap_
    core.stopSequenceAcquisition, core.isSequenceRunning, core.snapImage, core.startContinuousSequenceAcquisition = stop, running, snap_, start
    core.getLastImage = last
    return core


def live_window(core, cam, seconds):
    """The camera's stream in a window for `seconds` (auto contrast; closing the window ends it)."""
    import tkinter as tk
    from PIL import Image, ImageTk
    root = tk.Tk()
    root.title(f"IR camera {cam}: live")
    lbl = tk.Label(root, bg="black")
    lbl.pack()
    info = tk.Label(root, font=("Consolas", 9), anchor="w", justify="left")
    info.pack(fill="x")
    streaming, mine = True, False
    if not core.isSequenceRunning():                     # (else the running stream is shown, and left running)
        try:
            core.startContinuousSequenceAcquisition(0)
            mine = True
        except Exception as e:
            print(f"  (live window: the stream did not start: {e}; single snaps instead)")
            streaming = False
    t_end, n = time.time() + seconds, [0]

    def tick():
        left = t_end - time.time()
        if left <= 0:
            root.destroy()
            return
        try:
            if streaming:
                if core.getRemainingImageCount() == 0:
                    root.after(10, tick)
                    return
                a = to_array(core, core.getLastImage())
                core.clearCircularBuffer()
            else:
                a = snap(core)
            d, lo, hi = stretch(a)
            h, w = d.shape
            k = min(1.0, 900.0 / max(h, w)) if max(h, w) > 900 else (2.0 if max(h, w) <= 520 else 1.0)
            im = Image.fromarray(d)
            if k != 1.0:
                im = im.resize((int(w * k), int(h * k)), Image.NEAREST if k > 1 else Image.BILINEAR)
            ph = ImageTk.PhotoImage(im)
            lbl.configure(image=ph)
            lbl.image = ph
            n[0] += 1
            info.configure(text=f"{w} x {h}  min {int(a.min())}  median {float(np.median(a)):.0f}  max {int(a.max())}   "
                                f"(shown {lo:.0f} .. {hi:.0f})   exposure {core.getExposure():g} ms   frame {n[0]}   "
                                f"closes in {int(left)} s (or close the window)")
        except Exception as e:
            info.configure(text=f"error: {type(e).__name__}: {e}")
        root.after(20, tick)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    tick()
    root.mainloop()
    if mine:
        stop_live(core)
    return n[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--mm", default=r"C:\Program Files\Micro-Manager-2.0", help="the Micro-Manager install folder")
    ap.add_argument("--cfg", default=r"C:\Program Files\Micro-Manager-2.0\autoscan_withir.cfg", help="hardware configuration")
    ap.add_argument("--settings", default=r"C:\FlakeID\scan_settings.json", help="the scan program's settings (domain_camera ...)")
    ap.add_argument("--camera", help="Micro-Manager camera device (default: domain_camera of the settings)")
    ap.add_argument("--exposure", type=float, help="ms (default: domain_exposure_ms of the settings)")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--prop", action="append", default=[], help='a camera property to set first, "Name=Value" (repeatable)')
    ap.add_argument("--no-scan-props", action="store_true", help="do not set the scan's domain_camera_props (High Gain = Off, "
                    "Frame Average = 4)")
    ap.add_argument("--no-wake", action="store_true", help="no first live before the settings (for comparison)")
    ap.add_argument("--wake-s", type=float, default=2.0, help="seconds of the first live (default 2)")
    ap.add_argument("--diagnose", action="store_true", help="instead of steps 4-8: is the camera's level control (auto exposure / "
                    "AGC) still running with auto exposure off? (the real state with Force Update, the exposure, the auto-exposure "
                    "target with auto exposure off and on, the digital gain)")
    ap.add_argument("--camera-only", action="store_true", help="load a configuration with only the IR camera (made from --cfg): "
                    "the camera without the other devices")
    ap.add_argument("--live", type=float, default=0.0, help="then a live window for this many seconds (0: none)")
    ap.add_argument("--out", help="output folder (default C:\\FlakeScans\\ir_test_<date>_<time>; with --fake a temporary folder)")
    ap.add_argument("--fake", action="store_true", help="the simulated camera (no hardware)")
    a = ap.parse_args(argv)

    from flakeid.scope.settings import load
    sp = Path(a.settings)
    s = load(str(sp) if sp.is_file() and not a.fake else None)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if a.out:
        out = Path(a.out)
    elif a.fake:
        import tempfile
        out = Path(tempfile.gettempdir()) / f"ir_test_{stamp}"
    else:
        out = Path(s.local_root) / f"ir_test_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    OUT["dir"] = out
    REPORT.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), cfg=a.cfg, settings=str(sp) if sp.is_file() else None, out=str(out))

    def finish(code):
        (out / "report.json").write_text(json.dumps(REPORT, indent=1, default=str))
        bad = [x["step"] for x in REPORT["steps"] if x["status"] == "FAIL"]
        warn = [x["step"] for x in REPORT["steps"] if x["status"] == "WARN"]
        print("")
        print(("FAILED: " + ", ".join(bad)) if bad else ("done, with warnings: " + ", ".join(warn)) if warn else "done: the IR camera works")
        print(f"report and pictures: {out}  (send report.json)")
        return code

    # 1 configuration
    if a.fake:
        core = fake_core()
        step("1 load", "OK", "simulated microscope (--fake)")
    else:
        try:
            from flakeid.scope.__main__ import load_core
            if a.camera_only:
                cam0 = a.camera or s.domain_camera
                a.cfg = str(camera_only_cfg(a.cfg, cam0, out))
                step("1 load", "INFO", f"camera only: {a.cfg} (the lines of {cam0} from the configuration)")
            core = load_core(a.mm, a.cfg, log=lambda m: step("1 load", "INFO", m))
        except Exception as e:
            step("1 load", "FAIL", f"the configuration {a.cfg} did not load: {type(e).__name__}: {e}",
                 hints=["Is the Micro-Manager GUI closed?  Is the camera's own software (e.g. XCAP) closed?",
                        "Is the camera powered on and its cable connected?  (Micro-Manager stops at the first device that fails.)",
                        "Does the configuration load in the Micro-Manager GUI (Devices > Hardware Configuration Wizard)?",
                        "--cfg: the configuration with the IR camera; --mm: the Micro-Manager folder."])
            return finish(1)
        step("1 load", "OK", f"{a.cfg}: {core.getVersionInfo()}; {core.getAPIVersionInfo()}")

    # 2 cameras
    devs = list(core.getLoadedDevices())
    cams = [d for d in devs if is_camera(core, d)]
    orig = core.getCameraDevice()
    cam = a.camera or s.domain_camera
    if a.fake and not a.camera:
        cam = "DomainCam"
    step("2 cameras", "OK" if cams else "FAIL", f"cameras in the configuration: {', '.join(cams) or 'none'} (selected now: {orig or 'none'})",
         devices=devs, cameras=cams)
    if cam not in cams:
        step("2 cameras", "FAIL", f"{cam!r} is not a camera of this configuration: give the IR camera's name with --camera "
                                  f"(one of: {', '.join(cams)}), and put it in domain_camera of the settings")
        return finish(1)
    try:
        core.setCameraDevice(cam)
    except Exception as e:
        step("2 cameras", "FAIL", f"cannot select {cam}: {type(e).__name__}: {e}")
        return finish(1)
    step("2 cameras", "OK", f"selected {cam}")

    try:
        # 3 properties
        props = {}
        names = list(core.getDevicePropertyNames(cam))
        for p in names:
            try:
                v = core.getProperty(cam, p)
            except Exception as e:
                v = f"(not readable: {e})"
            ro = bool(core.isPropertyReadOnly(cam, p)) if hasattr(core, "isPropertyReadOnly") else None
            try:
                allowed = list(core.getAllowedPropertyValues(cam, p)) if hasattr(core, "getAllowedPropertyValues") else []
            except Exception:
                allowed = []
            props[p] = dict(value=v, read_only=ro, allowed=allowed[:40])
            print(f"         {p} = {v}" + ("   (read-only)" if ro else "")
                  + (f"   [{', '.join(allowed[:8])}{', ...' if len(allowed) > 8 else ''}]" if allowed else ""))
        step("3 properties", "OK" if names else "INFO", f"{len(names)} properties of {cam}", properties=props)
        temps = {p: props[p]["value"] for p in props if "temp" in p.lower() or "tec" in p.lower()}
        if temps:
            step("3 temperature", "INFO", ", ".join(f"{k} = {v}" for k, v in temps.items()))
        W, H = int(core.getImageWidth()), int(core.getImageHeight())
        bpp, bits = int(core.getBytesPerPixel()), int(core.getImageBitDepth())
        full = 255.0 if bpp == 4 else 2.0 ** bits - 1

        if a.diagnose:
            diagnose(core, cam, s, full, out, float(a.exposure if a.exposure else s.domain_exposure_ms))
            return finish(0 if not any(x["status"] == "FAIL" for x in REPORT["steps"]) else 1)

        # 4 wake-up: live once, then stopped
        if not a.no_wake:
            fr, fps = stream(core, a.wake_s, keep=1)
            if fr:
                st0 = frame_stats(fr[-1], full)
                empty = st0["max"] == st0["min"]
                step("4 wake-up", "INFO", f"live {a.wake_s:g} s: {fps:.1f} frames/s; last frame " + (f"all {st0['max']:g} (empty, as expected before "
                     "the settings take hold)" if empty else describe(st0, full)) + f"; stopped (took {STOP_WAIT['s']:.1f} s)", fps=fps, **st0)
            else:
                step("4 wake-up", "WARN", f"live {a.wake_s:g} s: no frames came (the settings may not take hold)")
        else:
            step("4 wake-up", "INFO", "left out (--no-wake)")

        # 5 settings (the camera stopped)
        todo = ([tuple(x[:2]) for x in s.domain_camera_props] if not a.no_scan_props else []) + [tuple(x.split("=", 1)) for x in a.prop if "=" in x]
        for nm, val in todo:
            nm, val = nm.strip(), str(val).strip()
            try:
                core.setProperty(cam, nm, val)
                step("5 settings", "OK", f"{nm} = {core.getProperty(cam, nm)} (asked {val})")
            except Exception as e:
                step("5 settings", "WARN", f"{nm} = {val} not set: {type(e).__name__}: {e}")
        exp = float(a.exposure if a.exposure else s.domain_exposure_ms)
        try:
            core.setExposure(exp)
            got = float(core.getExposure())
            close = abs(got - exp) <= max(0.05 * exp, 0.1)
            step("5 exposure", "OK" if close else ("INFO" if abs(got - exp) <= 0.3 * exp else "WARN"), f"exposure {got:g} ms (asked {exp:g})"
                 + ("" if close else ": the camera's nearest setting (its steps or frame-rate limit)"), exposure_ms=got)
        except Exception as e:
            step("5 exposure", "WARN", f"exposure {exp:g} ms not set: {type(e).__name__}: {e}")
        W, H = int(core.getImageWidth()), int(core.getImageHeight())
        bpp, bits = int(core.getBytesPerPixel()), int(core.getImageBitDepth())
        full = 255.0 if bpp == 4 else 2.0 ** bits - 1
        step("5 format", "OK" if W and H else "FAIL", f"{W} x {H} px, {bpp} bytes/px, {bits} bit; full scale {full:g}",
             width=W, height=H, bytes_per_px=bpp, bit_depth=bits)

        # 6 live again: one session for the frames, the exposure response and the live window (no stop in between)
        live_frames, stl, meds = [], None, {}
        try:
            if core.isSequenceRunning():
                stop_live(core)
            core.startContinuousSequenceAcquisition(0)
            live_frames, fps = grab(core, 3.0, keep=max(2, a.frames))
        except Exception as e:
            step("6 live", "FAIL", f"the live stream did not start after the settings: {type(e).__name__}: {e}")
        if live_frames:
            stl = frame_stats(live_frames[-1], full, live_frames)
            stl["fps"] = fps
            jv = judge(stl, full)
            step("6 live", jv[0], f"{fps:.1f} frames/s; {describe(stl, full)}{jv[1]}", **stl)
        elif core.isSequenceRunning():
            step("6 live", "FAIL", "live, but no frames came in 3 s")
        for nm, _ in todo:                               # (did the settings hold through the live?)
            try:
                step("6 settings now", "INFO", f"{nm.strip()} = {core.getProperty(cam, nm.strip())}")
            except Exception:
                pass

        # 7 exposure response, while live (the exposure changed during the stream; frames after the change)
        if live_frames:
            try:
                wait = max(1.5, 6.0 / max(stl["fps"], 0.5))
                for e in (exp / 2.0, exp, exp * 2.0):
                    core.setExposure(e)
                    fr, _ = grab(core, wait, keep=1, skip=3)
                    meds[e] = (float(np.median(fr[-1])) if fr else float("nan"), float(core.getExposure()))
                core.setExposure(exp)
                m = [meds[k][0] for k in sorted(meds)]
                rises = all(np.isfinite(m)) and (m[0] < m[1] < m[2] or m[2] >= 0.98 * full)
                step("7 exposure response", "OK" if rises else "WARN", "live median at " + ", ".join(
                     f"{k:g} ms (camera: {v[1]:g}): {v[0]:.0f}" for k, v in sorted(meds.items()))
                     + ("" if rises else "  -> the level does not follow the exposure (auto exposure on, the setting not reaching the "
                        "camera, or no light)"), medians={f"{k:g}": v[0] for k, v in meds.items()})
            except Exception as e:
                step("7 exposure response", "WARN", f"not measured: {type(e).__name__}: {e}")
        if a.live > 0 and core.isSequenceRunning():
            nl = live_window(core, cam, a.live)
            step("live window", "OK" if nl else "WARN", f"{nl} frames shown")

        # 8 stop, and single snaps (what the scan program's domain imaging takes)
        stopped = stop_live(core) if core.isSequenceRunning() else True
        if stopped:
            step("8 stop", "OK", f"stopped in {STOP_WAIT['s']:.1f} s")
        else:
            step("8 stop", "WARN", f"still live {STOP_WAIT['s']:.0f} s after the stop")
            if "Frame Average" in props:                 # does it stop without the frame averaging?
                try:
                    core.setProperty(cam, "Frame Average", "1")
                    t0 = time.time()
                    while core.isSequenceRunning() and time.time() - t0 < 6.0:
                        time.sleep(0.1)
                    if core.isSequenceRunning():
                        core.stopSequenceAcquisition()
                        while core.isSequenceRunning() and time.time() - t0 < 12.0:
                            time.sleep(0.1)
                    stopped = not core.isSequenceRunning()
                    step("8 stop", "INFO" if stopped else "WARN", "Frame Average set to 1 while live: " + (
                         f"then it stopped ({time.time() - t0:.1f} s)" if stopped else "still live"))
                except Exception as e:
                    step("8 stop", "WARN", f"Frame Average = 1 while live not possible: {type(e).__name__}: {e}")
        snaps, times = [], []
        if stopped:
            for i in range(max(1, a.frames)):
                t0 = time.time()
                try:
                    snaps.append(snap(core))
                except Exception as e:
                    step("8 snaps", "FAIL", f"snap {i + 1} failed: {type(e).__name__}: {e}")
                    break
                times.append(time.time() - t0)
        else:
            step("8 snaps", "WARN", "not tried: the camera does not stop (the scan would have to take its pictures from the live stream)")
        if snaps:
            sts = frame_stats(snaps[-1], full, snaps)
            sts["snap_ms"] = [round(t * 1000, 1) for t in times]
            cmp_ = ""
            if stl:
                ml, ms_ = stl["median"], sts["median"]
                cmp_ = f"; the stream's median {ml:.0f}" + ("  -> snaps differ from the stream" if abs(ms_ - ml) > max(0.1 * max(ml, ms_), 3 * max(stl["noise_counts"], 1)) else "")
            jv = judge(sts, full)
            step("8 snaps", jv[0], f"{len(snaps)} snaps, {np.median(times) * 1000:.0f} ms each; {describe(sts, full)}{cmp_}{jv[1]}", **sts)

        # 9 files
        try:
            import cv2
            import tifffile
            saved = []
            if live_frames:
                tifffile.imwrite(str(out / "live_frame.tif"), live_frames[-1])
                tifffile.imwrite(str(out / "live_mean.tif"), np.mean(np.asarray(live_frames, np.float64), axis=0).astype(np.float32))
                cv2.imwrite(str(out / "live_preview.png"), stretch(np.mean(np.asarray(live_frames, np.float64), axis=0))[0])
                saved += ["live_frame.tif", "live_mean.tif", "live_preview.png"]
            if snaps:
                tifffile.imwrite(str(out / "snap_frame.tif"), snaps[-1])
                cv2.imwrite(str(out / "snap_preview.png"), stretch(snaps[-1])[0])
                saved += ["snap_frame.tif", "snap_preview.png"]
            step("9 files", "OK", f"{', '.join(saved)} in {out}")
        except Exception as e:
            step("9 files", "WARN", f"pictures not saved: {type(e).__name__}: {e}")

        step("stop times", "INFO" if not STOP_WAIT["stuck"] else "WARN", f"stopping the live took up to {STOP_WAIT['s']:.1f} s"
             + ("; once it did not stop at all" if STOP_WAIT["stuck"] else ""), stop_s=STOP_WAIT["s"])
    finally:
        try:
            if core.isSequenceRunning():
                stop_live(core)
            if orig:
                core.setCameraDevice(orig)
        except Exception:
            pass
    return finish(0 if not any(x["status"] == "FAIL" for x in REPORT["steps"]) else 1)


if __name__ == "__main__":
    sys.exit(main())
