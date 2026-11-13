"""Can the scan script (tools/mm_chip_scan.bsh) run in Python on the microscope PC?  A hardware test.

    C:\\FlakeID\\run_flakeid.bat mmtest --mm "C:\\Program Files\\Micro-Manager-2.0" --cfg "C:\\Program Files\\Micro-Manager-2.0\\autoscan_withir.cfg"
    (or: python tools\\win7\\mm_python_test.py --fake  : the same steps on a simulated microscope, no hardware)

Close the Micro-Manager GUI first: Python loads the same device drivers and only one program can own the camera and
stage.  Put a chip under the 10x objective, in focus, over an area with flakes (not bare substrate).

It uses pymmcore-plus 0.11.1 / pymmcore 11.1.1.71 (the last versions for Python 3.8; device interface 71, the version of
Micro-Manager 2.0.3's drivers) and makes every Micro-Manager call the BeanShell scan makes:
  1  load the configuration; versions; camera, XY stage and focus devices
  2  camera settings of the scan (CAMERA_PROPS of mm_chip_scan.bsh: RAW16, 12 bit, no auto exposure / white balance,
     gamma 1, ...), exactly as the script sets them; the old values are put back at the end
  3  RAW16 snap: 16-bit Bayer frame, size, bit depth, demosaiced colour; substrate colour vs the 10x blank-chip
     reference (flakeid.bgcheck); snap time
  4  XY stage: moves of +-200 um, read back, timing; waitForDevice
  5  focus: read the drive, move by a few drive steps and back (no more than ~2 um)
  6  mini scan: 3 x 3 tiles at the scan's 10x step written exactly like mm_chip_scan.bsh (<name>_tiles/Pos-col_row/
     metadata.txt + TIFF), read back with flakeid.mm.read_tile and stitched by the analysis' own registration;
     seconds per tile against the BeanShell scan
  7  dialogs and live view: the prompts of the scan (focus check, objective change) as Tk dialogs, a 3 s live view
     window (what replaces the Micro-Manager GUI's live mode)
Writes everything to --out (report.json, snap.png, the tiles); send report.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# mm_chip_scan.bsh settings this test mirrors
CAMERA_PROPS = [("Data format", "RAW16", False), ("Device output data bit depth", "12", False),
                ("Image data bit depth", "12", True), ("AEAG", "Off", False), ("Auto white balance", "Off", False),
                ("Gain", "0", False), ("Luminosity gamma", "1", False), ("Chromaticity gamma", "1", False),
                ("Sharpness", "0", False), ("Downsampling", "1x1", False), ("Binning", "1", False)]
EXPOSURE_10X_MS = 3.0
OVERLAP = 0.15
UM_PER_PX_10X = 1.09              # 10x with the XC10 (flakeid.mm.DEFAULT_M); the mini scan measures it
Z_SCALE = 12.0                    # focus drive units per um (mm_chip_scan.bsh)
BSH_S_PER_TILE = 1.4              # the BeanShell scan on this PC writing to the share (multitest chip_1, 567 tiles)

REPORT = {"python": sys.version.split()[0], "platform": platform.platform(), "steps": []}


def say(step, status, msg, **data):
    print(f"[{status:4s}] {step}: {msg}", flush=True)
    REPORT["steps"].append(dict(step=step, status=status, msg=msg, **data))


# ---------------------------------------------------------------------------------------------------- simulated core
class FakeCore:
    """The handful of CMMCore calls the scan uses, on a synthetic chip (for running the test without hardware)."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.world = (rng.random((3000, 3000)) < 0.0004).astype(np.float32)
        import cv2
        self.world = cv2.GaussianBlur(cv2.dilate(self.world, np.ones((25, 25), np.uint8)), (0, 0), 3)
        self.x = self.y = 0.0
        self.z = 120000.0
        self.exp = 10.0
        self.props = {p: "?" for p, _, _ in CAMERA_PROPS}
        self.props.update({"Data format": "RGB24", "Device output data bit depth": "8"})
        self._img = None

    def getVersionInfo(self): return "FakeCore"
    def getAPIVersionInfo(self): return "Device API version 71, Module API version 10 (simulated)"
    def getLoadedDevices(self): return ["FakeCam", "FakeXY", "FakeZ"]
    def getCameraDevice(self): return "FakeCam"
    def getXYStageDevice(self): return "FakeXY"
    def getFocusDevice(self): return "FakeZ"
    def hasProperty(self, dev, p): return p in self.props
    def getProperty(self, dev, p): return self.props[p]
    def setProperty(self, dev, p, v): self.props[p] = str(v)
    def getDevicePropertyNames(self, dev): return list(self.props)
    def getExposure(self): return self.exp
    def setExposure(self, v): self.exp = float(v)
    def getImageBitDepth(self): return 12
    def getBytesPerPixel(self): return 2
    def getImageWidth(self): return 1376
    def getImageHeight(self): return 1038
    def getXPosition(self, dev=None): return self.x
    def getYPosition(self, dev=None): return self.y
    def setXYPosition(self, *a): self.x, self.y = float(a[-2]), float(a[-1]); time.sleep(0.05)
    def getPosition(self, dev=None): return self.z
    def setPosition(self, *a): self.z = float(a[-1]); time.sleep(0.02)
    def waitForDevice(self, dev): pass
    def waitForSystem(self): pass

    def snapImage(self):
        from flakeid.mm import DEFAULT_M                    # stage um = M @ pixel offset (the real camera's orientation)
        q = np.linalg.inv(np.asarray(DEFAULT_M, float)) @ np.array([self.x, self.y])
        ix, iy = int(1000 + q[0]), int(1000 + q[1])
        f = self.world[iy:iy + 1038, ix:ix + 1376]
        f = np.pad(f, ((0, 1038 - f.shape[0]), (0, 1376 - f.shape[1])))
        img = np.zeros((1038, 1376), np.float32)
        base = np.array([[1300, 2400], [2400, 2300]], np.float32)            # RGGB substrate levels (12-bit)
        img[:] = np.tile(base, (519, 688))
        img *= (1 - 0.4 * f)
        self._img = (np.clip(img + np.random.default_rng().normal(0, 15, img.shape), 0, 4095) + 128).astype(np.uint16)
        time.sleep(self.exp / 1000)

    def getImage(self): return self._img


# ---------------------------------------------------------------------------------------------------- DLL search
XIMEA_DIRS = [r"C:\XIMEA\API\x64", r"C:\XIMEA\API\xiAPI", r"C:\XIMEA\API", r"C:\XIMEA\bin",
              r"C:\Program Files\XIMEA\API\x64", r"C:\Program Files\XIMEA\API", r"C:\Program Files (x86)\XIMEA\API\x64"]


def check_mm_dir(mm, cfg):
    """The Micro-Manager folder holding the device drivers (mmgr_dal_*.dll).  If --mm is wrong, the configuration
    file's folder is used when it has the drivers (configs usually sit in the install folder)."""
    import glob
    import os
    has = lambda d: os.path.isdir(d) and bool(glob.glob(os.path.join(d, "mmgr_dal_*.dll")))
    if has(mm):
        return mm
    alt = os.path.dirname(os.path.abspath(cfg))
    if has(alt):
        say("1 load", "WARN", f"--mm {mm!r} has no Micro-Manager device drivers; using the configuration's folder {alt!r}")
        return alt
    raise RuntimeError(f"no Micro-Manager device drivers (mmgr_dal_*.dll) in {mm!r} or {alt!r}: give the Micro-Manager "
                       "install folder with --mm")


def prepare_dll_search(mm):
    """Make Windows find the libraries Micro-Manager's device drivers load (e.g. the XIMEA API, xiapi64.dll): the GUI
    finds them because it runs from the Micro-Manager folder; here python.exe does.  The Micro-Manager folder, the
    XIMEA API folders that exist and the user's original PATH (without Anaconda) go on the search path; the working
    directory becomes the Micro-Manager folder (Windows searches it too)."""
    import os
    orig = [p for p in os.environ.get("FLAKEID_ORIG_PATH", os.environ.get("PATH", "")).split(os.pathsep)
            if p and "anaconda" not in p.lower()]
    dirs = [mm] + [d for d in XIMEA_DIRS if os.path.isdir(d)]
    os.environ["PATH"] = os.pathsep.join(dirs + [p for p in os.environ.get("PATH", "").split(os.pathsep) if p] + orig)
    for d in dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    os.chdir(mm)
    return dirs + ([f"+ {len(orig)} folders of the original PATH"] if orig else [])


def find_libs(mm):
    """Where xiapi*.dll (the XIMEA camera library) can be found: the Micro-Manager folder, PATH, C:\XIMEA, Program Files."""
    import os
    roots = [mm, r"C:\XIMEA", r"C:\Program Files\XIMEA", r"C:\Program Files (x86)\XIMEA", r"C:\Windows\System32"]
    roots += [p for p in os.environ.get("FLAKEID_ORIG_PATH", "").split(os.pathsep) if p]
    hits = []
    for r in dict.fromkeys(roots):
        if not os.path.isdir(r):
            continue
        depth0 = r.rstrip("\\/").count(os.sep)
        for dp, dn, fn in os.walk(r):
            if dp.count(os.sep) - depth0 >= (0 if "System32" in r else 3):
                dn[:] = []
            hits += [os.path.join(dp, f) for f in fn if f.lower().startswith("xiapi") and f.lower().endswith(".dll")]
    return ", ".join(dict.fromkeys(hits)) if hits else ("no xiapi*.dll found: install XIMEA's software package (it puts "
                                                         "the API in C:\\XIMEA), the one the Micro-Manager GUI uses")


# ---------------------------------------------------------------------------------------------------- helpers
def to_array(core, img):
    a = np.asarray(img)
    if a.ndim == 1:
        a = a.reshape(core.getImageHeight(), core.getImageWidth())
    return a


def snap(core):
    core.snapImage()
    return to_array(core, core.getImage())


def demosaic(raw):
    import cv2
    from flakeid.mm import BAYER_CODE
    return cv2.cvtColor(raw, BAYER_CODE).astype(np.float32) - 128.0


def write_tile(folder, col, row, idx, raw, core, cam, xy):
    """One tile as mm_chip_scan.bsh writes it: <folder>/Pos-CCC_RRR/metadata.txt + img_channel000_positionNNN_...tif."""
    import tifffile
    lab = f"Pos-{col:03d}_{row:03d}"
    d = folder / lab
    d.mkdir(parents=True, exist_ok=True)
    fn = f"img_channel000_position{idx:03d}_time000000000_z000.tif"
    tifffile.imwrite(d / fn, raw)
    z = core.getPosition()
    fm = {"Label": lab, "GridCol": col, "GridRow": row, "XPositionUm": core.getXPosition(), "YPositionUm": core.getYPosition(),
          "ZPositionUm": z / Z_SCALE, "ZPositionDrive": z, "Exposure-ms": core.getExposure(), "Camera": cam,
          "Width": int(raw.shape[1]), "Height": int(raw.shape[0])}
    for p in core.getDevicePropertyNames(cam):
        try:
            fm[f"{cam}-{p}"] = core.getProperty(cam, p)
        except Exception:
            pass
    md = {"Summary": {"Source": "mm_python_test", "Width": int(raw.shape[1]), "Height": int(raw.shape[0]),
                      "PixelType": "GRAY16", "Z_SCALE": Z_SCALE}, f"Metadata-Default/{fn}": fm}
    (d / "metadata.txt").write_text(json.dumps(md, indent=1))


# ---------------------------------------------------------------------------------------------------- steps
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm", help="Micro-Manager install folder (its device drivers)")
    ap.add_argument("--cfg", help="the hardware configuration .cfg the Micro-Manager GUI uses")
    ap.add_argument("--out", default="mm_python_test")
    ap.add_argument("--fake", action="store_true", help="simulated microscope (no hardware)")
    ap.add_argument("--no-gui", action="store_true", help="skip the dialog / live view step")
    ap.add_argument("--z-steps", type=float, default=24.0, help="focus test move in drive units (default 24 = 2 um)")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()              # (absolute: the load step changes the working directory)
    out.mkdir(parents=True, exist_ok=True)

    # 1 ------------------------------------------------------------ load
    if a.fake:
        core = FakeCore()
        say("1 load", "INFO", "simulated microscope (--fake)")
    else:
        if not (a.mm and a.cfg):
            ap.error("--mm and --cfg are needed (or --fake)")
        try:
            try:
                from pymmcore_plus import CMMCorePlus
                core = CMMCorePlus()
                lib = "pymmcore-plus " + __import__("pymmcore_plus").__version__
            except ImportError:
                import pymmcore
                core = pymmcore.CMMCore()
                lib = "pymmcore " + pymmcore.__version__
            a.mm = check_mm_dir(a.mm, a.cfg)
            dll_dirs = prepare_dll_search(a.mm)
            say("1 load", "INFO", "library folders: " + "; ".join(dll_dirs))
            core.setDeviceAdapterSearchPaths([a.mm])
            core.loadSystemConfiguration(a.cfg)
        except Exception as e:
            say("1 load", "FAIL", f"could not load {a.cfg}: {e}  (Micro-Manager GUI still open?  A 'device interface "
                                  "version' error means the drivers in --mm are not Micro-Manager 2.0.3's)")
            if "could not be found" in str(e) or "specified module" in str(e):
                say("1 load", "INFO", "a library the driver needs was not found; where camera libraries are: " + find_libs(a.mm))
            return finish(out)
        say("1 load", "PASS", f"{lib}; {core.getVersionInfo()}; {core.getAPIVersionInfo()}")
    cam, xy, zdev = core.getCameraDevice(), core.getXYStageDevice(), core.getFocusDevice()
    say("1 load", "PASS" if cam and xy and zdev else "FAIL", f"camera {cam!r}, XY stage {xy!r}, focus {zdev!r}",
        devices=list(core.getLoadedDevices()))
    if not (cam and xy):
        return finish(out)

    # 2 ------------------------------------------------------------ camera settings
    old = {}
    problems = []
    for name, want, optional in CAMERA_PROPS:
        if not core.hasProperty(cam, name):
            if not optional:
                problems.append(f"{name}: not a property")
            continue
        old[name] = core.getProperty(cam, name)
        try:
            core.setProperty(cam, name, want)
        except Exception as e:
            if not optional:
                problems.append(f"{name}={want}: {e}")
    core.waitForDevice(cam)
    got = {n: core.getProperty(cam, n) for n, _, _ in CAMERA_PROPS if core.hasProperty(cam, n)}
    say("2 camera", "FAIL" if problems else "PASS", "scan settings applied" + ("; " + "; ".join(problems) if problems else ""),
        settings=got, previous=old)
    old_exp = core.getExposure()
    core.setExposure(EXPOSURE_10X_MS)

    try:
        # 3 -------------------------------------------------------- snap
        t0 = time.time()
        raw = snap(core)
        dt = time.time() - t0
        ok = raw.dtype == np.uint16 and raw.ndim == 2 and core.getBytesPerPixel() == 2
        say("3 snap", "PASS" if ok else "FAIL", f"{raw.shape} {raw.dtype}, {core.getBytesPerPixel()} bytes/px, "
            f"{core.getImageBitDepth()}-bit, max {int(raw.max())}, snap {dt * 1000:.0f} ms"
            + ("" if ok else "  (expected a 16-bit Bayer frame: is Data format RAW16?)"))
        rgb = demosaic(raw)
        from flakeid.hundred import _mode_rgb
        sub, frac = _mode_rgb(rgb[::2, ::2])
        try:
            from flakeid import bgcheck
            res = bgcheck.check(sub, "10x", 285.0, exposure_ms=core.getExposure(), bit_depth=12)
            say("3 snap", "INFO", f"substrate {np.round(sub).tolist()} ({frac:.0%} of the frame): R/G {res['rg']}, "
                f"B/G {res['bg']}, " + ("; ".join(res["warnings"]) if res["warnings"] else "matches the 10x reference"))
        except Exception as e:
            say("3 snap", "INFO", f"substrate {np.round(sub).tolist()} (reference check skipped: {e})")
        import cv2
        v = np.clip(rgb / max(float(np.percentile(rgb, 99.5)), 1) * 230, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "snap.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
        t0 = time.time()
        for _ in range(10):
            snap(core)
        say("3 snap", "INFO", f"{10 / (time.time() - t0):.1f} snaps per second at {core.getExposure():.0f} ms")

        # 4 -------------------------------------------------------- XY stage
        x0, y0 = core.getXPosition(), core.getYPosition()
        times = []
        for dx, dy in ((200, 0), (-200, 0), (0, 200), (0, -200)):
            t0 = time.time()
            core.setXYPosition(x0 + dx, y0 + dy)
            core.waitForDevice(xy)
            times.append(time.time() - t0)
            err = float(np.hypot(core.getXPosition() - x0 - dx, core.getYPosition() - y0 - dy))
            if err > 2:
                say("4 stage", "WARN", f"move ({dx},{dy}): read back {err:.1f} um off")
            core.setXYPosition(x0, y0)
            core.waitForDevice(xy)
        say("4 stage", "PASS", f"moves of 200 um: {np.mean(times) * 1000:.0f} ms each (incl. settling), back at "
            f"({core.getXPosition():.1f}, {core.getYPosition():.1f})")

        # 5 -------------------------------------------------------- focus
        if zdev:
            z0 = core.getPosition()
            t0 = time.time()
            core.setPosition(z0 + a.z_steps)
            core.waitForDevice(zdev)
            z1 = core.getPosition()
            core.setPosition(z0)
            core.waitForDevice(zdev)
            say("5 focus", "PASS" if abs(z1 - z0 - a.z_steps) < max(2, 0.2 * a.z_steps) else "WARN",
                f"drive {z0:.1f} -> {z1:.1f} -> {core.getPosition():.1f} ({(time.time() - t0) * 1000:.0f} ms for both moves)")

        # 6 -------------------------------------------------------- mini scan
        h, w = raw.shape
        step_x, step_y = w * UM_PER_PX_10X * (1 - OVERLAP), h * UM_PER_PX_10X * (1 - OVERLAP)
        folder = out / "pytest_tiles"
        t0 = time.time()
        k = 0
        for row in range(3):
            for col in range(3):
                core.setXYPosition(x0 + col * step_x, y0 + row * step_y)
                core.waitForDevice(xy)
                write_tile(folder, col, row, k, snap(core), core, cam, xy)
                k += 1
        per = (time.time() - t0) / 9
        core.setXYPosition(x0, y0)
        core.waitForDevice(xy)
        say("6 scan", "PASS", f"9 tiles in the scan's format, on the local disk: {per:.2f} s per tile (BeanShell scan "
            f"{BSH_S_PER_TILE:.1f} s per tile to the share, incl. its focus tracking)", s_per_tile=per)
        from flakeid.mm import read_tile
        tiles = [read_tile(m) for m in sorted(folder.glob("*/metadata.txt"))]
        ok = len(tiles) == 9 and all(t is not None and t.grid is not None for t in tiles)
        say("6 scan", "PASS" if ok else "FAIL", f"read back by the analysis (flakeid.mm.read_tile): {len(tiles)} tiles"
            + (f", grid {sorted(t.grid for t in tiles)[-1]}" if ok else ""))
        try:
            from flakeid.mm import _prep, fit_layout, read_frame
            smap, stats = fit_layout(tiles, [_prep(read_frame(t)) for t in tiles], verbose=False)
            say("6 scan", "PASS", f"stitched by the analysis' registration: {smap.pixel_um:.4f} um/px from "
                f"{stats['used']}/{stats['pairs']} tile pairs", um_per_px=smap.pixel_um)
        except Exception as e:
            say("6 scan", "WARN", f"registration not checked ({type(e).__name__}: {e}); an area with flakes is needed")

        # 7 -------------------------------------------------------- dialogs and live view
        if not a.no_gui:
            gui(core)
    except Exception as e:
        say("error", "FAIL", f"{type(e).__name__}: {e}", trace=traceback.format_exc())
    finally:
        for name, v in old.items():                     # camera back as it was
            try:
                core.setProperty(cam, name, v)
            except Exception:
                pass
        core.setExposure(old_exp)
    finish(out)


def gui(core):
    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageTk
    except Exception as e:
        say("7 gui", "FAIL", f"Tk / Pillow not available: {e}")
        return
    root = tk.Tk()
    root.title("FlakeIdentifier: live view test (3 s)")
    lbl = tk.Label(root)
    lbl.pack()
    t_end = time.time() + 3.0
    n = [0]

    def tick():
        rgb = demosaic(snap(core))
        v = np.clip(rgb[::2, ::2] / max(float(np.percentile(rgb, 99.5)), 1) * 230, 0, 255).astype(np.uint8)
        im = ImageTk.PhotoImage(Image.fromarray(v))
        lbl.configure(image=im)
        lbl.image = im
        n[0] += 1
        if time.time() < t_end:
            root.after(1, tick)
        else:
            root.destroy()
    root.after(10, tick)
    root.mainloop()
    say("7 gui", "PASS", f"live view: {n[0] / 3:.1f} frames per second (half resolution, demosaiced)")
    r = tk.Tk()
    r.withdraw()
    ans = messagebox.askyesno("FlakeIdentifier scan", "Dialog test (the scan's prompts, e.g. 'Is the first tile in focus?').\n\nClick Yes.")
    r.destroy()
    say("7 gui", "PASS" if ans else "WARN", "scan prompt dialog " + ("answered" if ans else "answered No"))


def finish(out):
    (out / "report.json").write_text(json.dumps(REPORT, indent=1, default=str))
    bad = [s["step"] for s in REPORT["steps"] if s["status"] == "FAIL"]
    print(f"\nreport: {out / 'report.json'}  |  " + ("FAILED: " + ", ".join(sorted(set(bad))) if bad else "no failures"))


if __name__ == "__main__":
    main()
