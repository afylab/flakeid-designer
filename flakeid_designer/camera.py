"""Live camera for the stacker version of the designer: the transfer station's microscope camera as a live picture
under the design (the stacking view).

Sources, tried in this order unless the station's settings name one:
  - "ic4":    The Imaging Source cameras through IC Imaging Control 4 (pip package imagingcontrol4; IC Capture's maker;
              the transfer stations' DFK 33UP1300, USB3 colour, 1280 x 1024)
  - "opencv": any camera Windows / macOS shows as a normal camera (OpenCV VideoCapture; many Imaging Source USB
              cameras too, without their SDK)
  - "sim":    a simulated camera (a chip with alignment marks, a target flake and a movable, defocused stamp flake),
              for development and for trying the stacking view without a camera

A background thread keeps the newest frame as a JPEG; the page fetches it (GET /api/camera/frame.jpg) as fast as
it draws.  Needs OpenCV (cv2) for JPEG encoding and the opencv source: the office build leaves it out and the
designer then shows no live view.
"""
from __future__ import annotations

import math
import threading
import time

try:
    import numpy as np
    import cv2
except ImportError:                                  # office build: no camera
    np = cv2 = None

MAX_W = 1600                                         # frames are shrunk to this width before they are sent
SLOW_S = 20.0                                        # no picture after this long: say so (the camera is not answering)


def say(msg):
    """A step of opening / running the camera: to the log (the app's stderr goes to ~/.flakeid_designer_log.txt)."""
    import sys
    try:
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} camera: {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def available():
    return cv2 is not None


def sources():
    """What can be used on this computer: [{"id", "name"}]."""
    out = []
    if cv2 is None:
        return out
    try:
        import imagingcontrol4 as ic4                # noqa: F401
        out.append(dict(id="ic4", name="The Imaging Source camera (IC Imaging Control 4)"))
    except Exception:
        pass
    out.append(dict(id="opencv", name="System camera (OpenCV)"))
    out.append(dict(id="sim", name="Simulated camera"))
    return out


# ---------------------------------------------------------------- simulated camera
def is_grey(f):
    """A 3-channel picture whose channels are all the same (a monochrome camera format shown as BGR)."""
    if f is None or f.ndim != 3 or f.shape[2] < 3:
        return f is not None
    s = f[::7, ::7, :3].astype(np.int16)
    return int(np.abs(s[..., 0] - s[..., 1]).max()) <= 1 and int(np.abs(s[..., 1] - s[..., 2]).max()) <= 1


def render_marks(polys, w, h, um_px, rot_deg=0.0, centre_um=(0.0, 0.0), mirror=False, bg=(150, 92, 118), gold=(90, 190, 235)):
    """A chip's alignment marks as a camera would see them (BGR float32): layout polygons (um) drawn in gold on the oxide
    colour, um_px um per pixel, the picture turned rot_deg (counterclockwise) against the layout and centred on
    centre_um; mirror: the picture mirrored left-right.  (The simulated camera, and tests of the calibration.)"""
    im = np.zeros((h, w, 3), np.float32)
    im[:] = bg
    a = um_px * complex(math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg)))
    for poly in polys:
        z = ((np.asarray(poly, float) - centre_um) @ np.array([1, 1j])) / a
        u = -z.real if mirror else z.real
        px = np.stack([u + w / 2.0 - 0.5, h / 2.0 - 0.5 - z.imag], axis=-1)
        if np.all((px[:, 0] < -20) | (px[:, 0] > w + 20) | (px[:, 1] < -20) | (px[:, 1] > h + 20)):
            continue
        cv2.fillPoly(im, [np.round(px * 16).astype(np.int32)], gold, cv2.LINE_AA, shift=4)
    return im


SIM_MARKS = None      # [polygon (um), ...]: the simulated camera shows these marks (the designer sets them from its template)
SIM_MARKS_VIEW = dict(um_px=0.45, rot_deg=2.5, centre_um=(0.0, 0.0))


class SimCamera:
    """A transfer-station view: SiO2 chip with gold alignment crosses and a target flake (in focus), and the stamp's
    flake above it (defocused, semi-transparent) that the page can move (jog) like a micromanipulator.  Scale
    ~0.45 um/px (a 20x objective), slow drift and noise like a live camera."""
    name = "Simulated camera"

    def __init__(self, w=1280, h=960):
        self.w, self.h = w, h
        self.stamp = [140.0, -90.0, 12.0]            # stamp flake offset (px) and rotation (deg) from its target place
        self.focus = 0.0                             # 0 = the chip in focus (stamp defocused); 1 = the stamp in focus
        self.t0 = time.time()
        rng = np.random.default_rng(7)
        bg = np.zeros((h, w, 3), np.float32)
        bg[:] = (150, 92, 118)                       # ~285 nm SiO2 (BGR)
        if SIM_MARKS:                                # the chip's real marks (the designer's template layout)
            bg = render_marks(SIM_MARKS, w, h, **SIM_MARKS_VIEW)
        else:
            for x0, y0 in ((260, 220), (w - 260, 220), (260, h - 220), (w - 260, h - 220)):
                for dx, dy in ((1, 0), (0, 1)):      # alignment crosses (gold)
                    cv2.rectangle(bg, (x0 - 60 * dx - 8 * dy, y0 - 60 * dy - 8 * dx), (x0 + 60 * dx + 8 * dy, y0 + 60 * dy + 8 * dx), (90, 190, 235), -1)
        target = np.array([[0, -150], [190, -90], [230, 70], [80, 190], [-160, 150], [-210, -40]], np.float32) + [w / 2 - 40, h / 2 + 30]
        cv2.fillPoly(bg, [target.astype(np.int32)], (205, 160, 120))       # target hBN flake (bluish)
        cv2.polylines(bg, [target.astype(np.int32)], True, (215, 175, 140), 2)
        for _ in range(140):                         # residue specks
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            cv2.circle(bg, (int(x), int(y)), int(rng.uniform(1, 4)), tuple(float(v) for v in rng.uniform(60, 200, 3)), -1)
        self.bg = bg
        self.stamp_poly = np.array([[-120, -60], [60, -110], [150, 20], [40, 120], [-110, 90]], np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        self.vign = (1.0 - 0.28 * r2).astype(np.float32)[..., None]
        self.noise = [rng.normal(0, 3.0, (h, w, 3)).astype(np.float32) for _ in range(6)]
        self.k = 0

    def jog(self, dx=0.0, dy=0.0, drot=0.0):
        self.stamp[0] += dx
        self.stamp[1] += dy
        self.stamp[2] += drot

    def grab(self):
        t = time.time() - self.t0
        drift = (3.0 * math.sin(t / 7.0), 2.0 * math.cos(t / 9.0))      # a slow wobble of the whole view
        M = np.float32([[1, 0, drift[0]], [0, 1, drift[1]]])
        sharp = 1.0 + 3.0 * self.focus
        img = cv2.warpAffine(self.bg, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)
        if self.focus > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), sharp)
        # the stamp's flake: rotated, offset, blurred unless focused on it, seen through the stamp
        a = math.radians(self.stamp[2])
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
        P = self.stamp_poly @ R.T + [self.w / 2 + self.stamp[0] + drift[0], self.h / 2 + self.stamp[1] + drift[1]]
        sg = 1.0 + 7.0 * (1 - self.focus)
        m = int(3 * sg) + 2
        x0, y0 = np.floor(P.min(0)).astype(int) - m
        x1, y1 = np.ceil(P.max(0)).astype(int) + m
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.w), min(y1, self.h)
        if x1 > x0 and y1 > y0:                      # only around the stamp's flake
            layer = np.zeros((y1 - y0, x1 - x0), np.float32)
            cv2.fillPoly(layer, [(P - [x0, y0]).astype(np.int32)], 1.0)
            layer = cv2.GaussianBlur(layer, (0, 0), sg)[..., None]
            roi = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = roi * (1 - 0.35 * layer) + np.array((120, 200, 150), np.float32) * 0.35 * layer
        self.k = (self.k + 1) % len(self.noise)
        img = img * self.vign + self.noise[self.k]
        return np.clip(img, 0, 255).astype(np.uint8)

    def close(self):
        pass


# ---------------------------------------------------------------- real cameras
class OpenCVCamera:
    def __init__(self, index=0, step=say):
        import sys
        apis = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF)] if sys.platform == "win32" else [("default", cv2.CAP_ANY)]
        self.cap = None
        for nm, api in apis:
            step(f"opening system camera {index} ({nm})")
            cap = cv2.VideoCapture(int(index), api)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
            step(f"system camera {index} did not open with {nm}")
        if self.cap is None:
            raise RuntimeError(f"system camera {index} could not be opened (another program using it? a different camera number?)")
        # the camera's full size (DirectShow often starts at 640 x 480): the stations' DFK 33UP1300 is 1280 x 1024
        for w, h in ((1280, 1024), (1920, 1200), (1920, 1080), (1600, 1200), (1280, 960), (1280, 720)):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w and int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == h:
                break
        step(f"system camera {index}: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        self.fmt = self._fourcc()
        if sys.platform == "win32" and is_grey(self._test()):
            # (an Imaging Source camera through DirectShow starts in Y800, monochrome: its Bayer format BY8 is
            # debayered by OpenCV; then the usual colour formats)
            orig = self.cap.get(cv2.CAP_PROP_FOURCC)
            for fc in ("BY8 ", "YUY2", "RGB3", "MJPG"):
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fc))
                    self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    f = self._test()
                except Exception:
                    f = None
                if f is not None and not is_grey(f):
                    step(f"system camera {index}: grey in {self.fmt}, colour in {fc.strip()}")
                    self.fmt = fc.strip()
                    break
            else:
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, orig)
                except Exception:
                    pass
                step(f"WARNING: system camera {index} gives a grey picture ({self.fmt}) and no colour format worked: use The Imaging "
                     "Source camera (IC Imaging Control 4) instead, or set a colour format in Camera settings")
        self.name = f"System camera {index}"
        self.has_settings = sys.platform == "win32"

    def _fourcc(self):
        v = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        t = "".join(chr((v >> 8 * k) & 0xFF) for k in range(4)).strip()
        return t if t.isprintable() and t else str(v)

    def _test(self):
        f = None
        for _ in range(3):                           # (the first frames after a format change can be stale)
            ok, f = self.cap.read()
            f = f if ok else None
        return f

    def settings(self):
        """The driver's own settings window (Windows DirectShow: exposure, gain, white balance ...)."""
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def grab(self):
        ok, f = self.cap.read()
        return f if ok else None

    def close(self):
        self.cap.release()


class IC4Camera:
    """The Imaging Source camera through IC Imaging Control 4 (the first device unless `serial` is given)."""

    def __init__(self, serial=None, step=say):
        import imagingcontrol4 as ic4
        from pathlib import Path
        step("starting IC Imaging Control 4")
        try:
            tgt = getattr(ic4.LogTarget, "FILE", None)
            if tgt is not None:
                ic4.Library.init(api_log_level=ic4.LogLevel.INFO, internal_log_level=ic4.LogLevel.WARNING, log_targets=tgt,
                                 log_file=str(Path.home() / ".flakeid_ic4_log.txt"))
            else:
                ic4.Library.init()
        except RuntimeError:
            pass                                     # (already initialised)
        step("looking for Imaging Source cameras")
        devs = ic4.DeviceEnum.devices()
        step(f"found {len(devs)}: " + ", ".join(f"{d.model_name} ({d.serial})" for d in devs))
        if not devs:
            raise RuntimeError("no Imaging Source camera found by IC Imaging Control 4 (is its USB3 driver installed? "
                               "the camera plugged in?)")
        dev = next((d for d in devs if serial and d.serial == serial), devs[0])
        self.ic4 = ic4
        self.grabber = ic4.Grabber()
        step(f"opening {dev.model_name} ({dev.serial})")
        self.grabber.device_open(dev)
        pm = self.grabber.device_property_map
        for prop, val in ((ic4.PropId.TRIGGER_MODE, "Off"), (ic4.PropId.EXPOSURE_AUTO, "Continuous"),
                          (ic4.PropId.BALANCE_WHITE_AUTO, "Continuous")):
            try:
                ok = pm.try_set_value(prop, val)
                step(f"{prop} = {val}" + ("" if ok else " (not available)"))
            except Exception as e:
                step(f"{prop}: {e}")
        self.bayer = None
        self._colour_format(pm, step)
        step("starting the stream")
        try:                                         # colour cameras (e.g. DFK 33UP1300): debayered to BGR8 by the library
            self.sink = ic4.SnapSink(accepted_pixel_formats=[ic4.PixelFormat.BGR8])
        except Exception:
            self.sink = ic4.SnapSink()
        self.grabber.stream_setup(self.sink)
        self.name = f"{dev.model_name} ({dev.serial})"
        step("stream started, waiting for the first picture")

    # the camera's own pixel format: a colour (Bayer) one, so the library's conversion to BGR8 has colour to convert
    # (a camera left in Mono8 by another program, e.g. IC Capture, gave a grey picture at the stacker, 2026-10-14)
    COLOUR_FORMATS = ("BayerRG8", "BayerGR8", "BayerBG8", "BayerGB8", "BGR8", "RGB8", "BGRa8",
                      "BayerRG16", "BayerGR16", "BayerBG16", "BayerGB16")

    def _colour_format(self, pm, step):
        ic4 = self.ic4
        try:
            cur = str(pm.get_value_str(ic4.PropId.PIXEL_FORMAT))
        except Exception as e:
            step(f"pixel format not readable ({e})")
            return
        names = []
        try:
            names = [str(e.name) for e in pm.find_enumeration(ic4.PropId.PIXEL_FORMAT).entries if getattr(e, "is_available", True)]
        except Exception:
            pass
        step(f"camera pixel format {cur}" + (f" (available: {', '.join(names)})" if names else ""))
        self.fmt = cur
        if cur.startswith(("Bayer", "BGR", "RGB")):
            self.bayer = cur if cur.startswith("Bayer") else None
            return
        for want in self.COLOUR_FORMATS:
            if names and want not in names:
                continue
            try:
                if pm.try_set_value(ic4.PropId.PIXEL_FORMAT, want):
                    step(f"pixel format {cur} -> {want} (colour)")
                    self.fmt = want
                    self.bayer = want if want.startswith("Bayer") else None
                    return
            except Exception:
                continue
        step(f"WARNING: pixel format {cur}: no colour format could be set; the picture is grey")

    # GenICam Bayer names (first row: R G ...) -> OpenCV's names (taken from the second row): RG -> BG, etc.
    _CV_BAYER = {"RG": "COLOR_BayerBG2BGR", "GR": "COLOR_BayerGB2BGR", "BG": "COLOR_BayerRG2BGR", "GB": "COLOR_BayerGR2BGR"}

    def grab(self):
        buf = self.sink.snap_single(2000)
        a = buf.numpy_copy()
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        if a.dtype != np.uint8:                     # 16-bit modes: shown as 8 bit
            a = (a >> 8).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8)
        if a.ndim == 2 and self.bayer:              # (the sink gave the raw mosaic: debayered here)
            code = getattr(cv2, self._CV_BAYER.get(self.bayer[5:7], ""), None)
            if code is not None:
                return cv2.cvtColor(a, code)
        return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.ndim == 2 else a[..., :3]

    def close(self):
        try:
            self.grabber.stream_stop()
            self.grabber.device_close()
        except Exception:
            pass


# ---------------------------------------------------------------- the running camera
class Live:
    """One camera at a time.  Opening runs in the background (a camera can take seconds, or hang when another program
    holds it); `phase` says what it is doing, `error` what went wrong; a grab thread keeps the newest frame (JPEG)."""

    def __init__(self):
        self.cam = None
        self.src = None
        self.jpeg = None
        self.raw = None                              # the latest frame itself (BGR, as sent to the page): calibration
        self.shape = None
        self.seq = 0
        self.fps = 0.0
        self.error = None
        self.phase = None
        self.t_start = 0.0
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.thread = None
        self.last_req = 0.0
        self.gen = 0
        self.want_settings = False

    def start(self, src="sim", index=0, serial=None):
        """Starts opening `src` in the background and returns at once (status() follows it)."""
        self.stop()
        self.gen += 1
        gen = self.gen
        self.error, self.phase, self.src, self.t_start = None, "opening the camera", src, time.time()
        self.last_req = time.time()
        self.raw = None
        self.stop_ev = threading.Event()
        self.thread = threading.Thread(target=self._open_and_run, args=(gen, src, index, serial, self.stop_ev), daemon=True)
        self.thread.start()
        return True

    def _step(self, gen, msg):
        if gen == self.gen:
            self.phase = msg
        say(msg)

    def _open_and_run(self, gen, src, index, serial, stop_ev):
        step = lambda m: self._step(gen, m)          # noqa: E731
        try:
            if src == "sim":
                step("starting the simulated camera")
                cam = SimCamera()
            elif src == "opencv":
                cam = OpenCVCamera(index, step=step)
            elif src == "ic4":
                cam = IC4Camera(serial, step=step)
            else:
                raise RuntimeError(f"unknown camera source {src!r}")
        except Exception as e:
            if gen == self.gen:
                self.error, self.phase = f"{type(e).__name__}: {e}" if not str(e).startswith(("no ", "system")) else str(e), None
            say(f"failed: {e}")
            return
        if gen != self.gen or stop_ev.is_set():      # (stopped / another camera chosen meanwhile)
            _close(cam)
            return
        self.cam = cam
        step("running")
        self._run(cam, stop_ev)
        _close(cam)

    def _run(self, cam, stop_ev):
        n, t0, errs = 0, time.time(), 0
        while not stop_ev.is_set():
            if time.time() - self.last_req > 30:     # nobody is looking: idle
                time.sleep(0.2)
                continue
            if self.want_settings:                   # (driver dialogs from the camera's own thread)
                self.want_settings = False
                try:
                    cam.settings()
                except Exception as e:
                    say(f"settings window: {e}")
            try:
                f = cam.grab()
                errs = 0
            except Exception as e:
                errs += 1
                if errs in (1, 10) or errs % 100 == 0:
                    say(f"no picture: {e}")
                self.error = f"no picture from the camera: {e}"
                time.sleep(0.3)
                continue
            if f is None:
                time.sleep(0.05)
                continue
            if self.error and self.error.startswith("no picture"):
                self.error = None
            if f.shape[1] > MAX_W:
                k = MAX_W / f.shape[1]
                f = cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.lock:
                    first = self.jpeg is None
                    self.jpeg, self.raw, self.shape, self.seq = buf.tobytes(), f, f.shape[:2], self.seq + 1
                if first:
                    self.grey = is_grey(f)
                    say(f"first picture {f.shape[1]} x {f.shape[0]} after {time.time() - self.t_start:.1f} s, format "
                        f"{getattr(cam, 'fmt', '?')}" + (": GREY (the camera sends a monochrome format)" if self.grey else ", colour"))
            n += 1
            if time.time() - t0 >= 2:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
            if self.src == "sim":
                time.sleep(1 / 30)

    def frame(self):
        self.last_req = time.time()
        with self.lock:
            return self.jpeg, self.seq

    def average(self, n=6, timeout=15.0):
        """The mean of the next n frames (BGR uint8, the size sent to the page), or None (less noise for a calibration)."""
        got, seq0, t0 = [], -1, time.time()
        while len(got) < n and time.time() - t0 < timeout:
            self.last_req = time.time()
            with self.lock:
                f, sq = self.raw, self.seq
            if f is not None and sq != seq0:
                got.append(f.astype(np.float32))
                seq0 = sq
            else:
                time.sleep(0.01)
        if not got:
            return None
        return np.clip(np.mean(got, axis=0), 0, 255).astype(np.uint8)

    def status(self):
        slow = (self.src is not None and self.jpeg is None and not self.error and time.time() - self.t_start > SLOW_S)
        return dict(available=available(), sources=sources(), running=self.cam is not None, source=self.src,
                    name=getattr(self.cam, "name", None), size=None if self.shape is None else [int(self.shape[1]), int(self.shape[0])],
                    fps=round(self.fps, 1), error=self.error, phase=self.phase, waiting_s=round(time.time() - self.t_start, 1) if self.src else 0,
                    slow=slow, jog=self.src == "sim", settings=bool(getattr(self.cam, "has_settings", False)),
                    fmt=getattr(self.cam, "fmt", None), grey=bool(getattr(self, "grey", False)))

    def open_settings(self):
        if not getattr(self.cam, "has_settings", False):
            return "this camera has no settings window here"
        self.want_settings = True
        return None

    def jog(self, dx=0.0, dy=0.0, drot=0.0, focus=None):
        if isinstance(self.cam, SimCamera):
            self.cam.jog(dx, dy, drot)
            if focus is not None:
                self.cam.focus = min(1.0, max(0.0, float(focus)))

    def stop(self):
        """Stops at once; the camera is closed by its own thread (a hung camera must not hang the designer)."""
        self.gen += 1
        self.stop_ev.set()
        with self.lock:
            self.jpeg, self.shape = None, None
        self.cam, self.thread, self.src, self.phase, self.error = None, None, None, None, None


def _close(cam):
    try:
        cam.close()
    except Exception:
        pass


LIVE = Live()
