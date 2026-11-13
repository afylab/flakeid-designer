"""Windows of the Python scan (Tk): the run window (log, progress, Stop), a colour live view in every focus question and
in the window where the pieces are added, the dialogs; and AutoUI, which answers everything itself (simulated
microscope, tests).

Threads: Tk runs in the main thread; the scan runs in a worker thread and calls these methods, which run their windows
in the Tk thread and wait for the answer.  While a live view is open the scan thread is waiting, so only the Tk thread
uses the camera and stage.

Live view colour: the RAW16 Bayer frame is demosaiced and each channel divided by the bare-substrate level (the scan's
reference colour, or the most common colour of the frame: "Balance"), so the substrate shows neutral grey and flakes
their contrast colours, as in the chip viewers; "Enhanced" stretches the contrast 4x around the substrate.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .scan import Stop, f1, f3, polygon_area_um2, polygon_crossing, polygon_interior_point

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:                     # (no Tk: headless runs only)
    tk = ttk = None

OK_FG, ERR_FG = "#2e9d4f", "#d9534f"     # status colours readable on the light and the dark theme

def dpi_aware():
    """Crisp text on high-resolution screens (Windows scales programs that do not declare this, blurring them)."""
    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)         # Windows 8.1 and later
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()              # Windows 7
        except Exception:
            pass


ASSETS = __import__("pathlib").Path(__file__).parent / "assets"


def set_icon(root):
    """The logo (assets/logo.ico, logo_*.png) as the icon of every window; returns the PhotoImages (keep a reference)."""
    import sys
    imgs = []
    try:
        imgs = [tk.PhotoImage(master=root, file=str(ASSETS / f"logo_{n}.png")) for n in (256, 64, 48, 32, 16)]
        root.iconphoto(True, *imgs)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            root.iconbitmap(default=str(ASSETS / "logo.ico"))       # (crisp at every size in the title bar and taskbar)
        except Exception:
            pass
    return imgs


def use_sun_valley(root, theme):
    """The Sun Valley theme in the icon's violet (assets made by tools/make_purple_theme.py, in theme_sv/; the sv-ttk
    package's blue original if that folder is missing).  False if neither can be loaded."""
    tcl = ASSETS.parent / "theme_sv" / "sv.tcl"
    try:
        if tcl.is_file():
            if not getattr(root, "_sv_loaded", False):
                root.tk.call("source", str(tcl))
                root._sv_loaded = True
            ttk.Style(root).theme_use(f"sun-valley-{theme}")
            # the theme sets its colours (ttk defaults, and tk_setPalette for plain Tk windows) on <<ThemeChanged>>,
            # which only runs once Tk processes its events: until then the background reads as "" and the windows
            # made before that got the system colour (white on Windows, 2026-10-04).  Set them now.
            root.tk.call("configure_colors")
        else:
            import sv_ttk
            sv_ttk.set_theme(theme, root)
        return True
    except Exception:
        return False


def apply_theme(root, theme="dark"):
    """ttk theme: "dark" / "light" (Sun Valley, the sv-ttk package) or "native" (Windows' own look).  Returns the log
    text colours (foreground, background)."""
    style = ttk.Style(root)
    if theme in ("dark", "light") and use_sun_valley(root, theme):
        return ("#e8e8e8", "#1c1c1c") if theme == "dark" else ("#1c1c1c", "#fafafa")
    names = style.theme_names()
    style.theme_use("vista" if "vista" in names else ("aqua" if "aqua" in names else "clam"))
    return ("black", "white")

# --------------------------------------------------------------------------------------------- display of a frame


def smooth_background(rgb):
    """Slowly varying substrate level (vignetting), relative to its median: per channel, repeated 5x5 medians on a 1/32
    image (flakes smaller than ~1/4 of the frame drop out), then smoothed and scaled back up."""
    import cv2
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb.astype(np.float32), (max(8, w // 32), max(8, h // 32)), interpolation=cv2.INTER_AREA)
    for _ in range(4):
        small = cv2.medianBlur(small, 5)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg / np.maximum(np.median(bg.reshape(-1, 3), axis=0), 1e-6), 0.2)


def bayer_half(a):
    """RAW16 Bayer frame (R at the top-left pixel) -> half-resolution RGB float32: one pixel per 2x2 cell (R, mean of
    the two G, B).  Several times faster than demosaicing at full resolution and then shrinking (the live view)."""
    a = a.astype(np.float32)
    return np.dstack([a[0::2, 0::2], 0.5 * (a[0::2, 1::2] + a[1::2, 0::2]), a[1::2, 1::2]])


def half_flat(flat):
    """A FlatField (flakeid.mm) averaged over 2x2 pixels, for bayer_half frames."""
    from ..mm import FlatField
    f = flat.flat
    f = f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2].reshape(f.shape[0] // 2, 2, f.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    d = np.asarray(flat.dark, np.float32)
    if d.ndim == 3:
        d = d[: d.shape[0] // 2 * 2, : d.shape[1] // 2 * 2].reshape(d.shape[0] // 2, 2, d.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    return FlatField(flat=f.astype(np.float32), dark=d)


def to_display(frame, dark, sub=None, level=0.31, enhanced=False, scale=0.5, flat=None, mono=False):
    """RAW16 Bayer (or mono / RGB) frame -> (8-bit RGB image, substrate RGB used).  flat: a FlatField (flakeid.mm) for
    the vignetting; without one, enhanced mode divides by a smooth background estimate instead.  mono: a 16-bit frame
    of a monochrome camera (the domain camera), not a colour mosaic."""
    import cv2
    from ..mm import BAYER_CODE
    a = np.asarray(frame)
    bayer = a.ndim == 2 and a.dtype == np.uint16 and a.shape[0] % 2 == 0 and a.shape[1] % 2 == 0 and not mono
    if bayer and scale == 0.5:
        rgb = bayer_half(a)                                      # (flat: a half_flat)
        if flat is not None and flat.flat.shape[:2] == rgb.shape[:2]:
            rgb = flat.apply(rgb)
            flat_done = True
        else:
            rgb = rgb - dark
            flat_done = False
        scale = 1.0
    elif bayer and flat is not None and flat.flat.shape[:2] == a.shape:
        rgb = flat.apply(cv2.cvtColor(a, BAYER_CODE).astype(np.float32))
        flat_done = True
    elif bayer:
        rgb = cv2.cvtColor(a, BAYER_CODE).astype(np.float32) - dark
        flat_done = False
    elif a.ndim == 2:
        g = a.astype(np.float32) - dark
        rgb = np.dstack([g, g, g])
    else:
        rgb = a[..., :3].astype(np.float32)
    if a.ndim == 2 and not bayer:
        flat_done = False
    elif a.ndim != 2:
        flat_done = False
    if scale != 1.0:
        rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    if sub is None or sub is False:
        from ..hundred import _mode_rgb
        # the most common colour of the lit part: with the field stop closed (flat-field focus) most of the frame is the
        # dark surround, whose level stretched the noise to full scale and saturated the lit disc (2026-10-04)
        px = np.maximum(rgb[::2, ::2], 0).reshape(-1, 3)
        g = px[:, 1]
        lit = px[g > 0.3 * float(np.percentile(g, 99.5))] if g.size else px
        sub, _ = _mode_rgb((lit if len(lit) >= 100 else px)[:, None, :], step=4)
    s = np.maximum(np.asarray(sub, np.float32), 1.0)
    r = rgb / s
    if enhanced and not flat_done:
        r = r / smooth_background(rgb)
    if enhanced:
        out = 128.0 * (1.0 + 3.0 * (r - 1.0))
    else:
        out = r * (level * 255.0)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), s


# --------------------------------------------------------------------------------------------- headless answers
def remembered_notify(s):
    """(shown, sent) for the start screen's "Messages to" given last (<local_root>/notify.json): the name shown and
    what the messages go to (a Slack member ID when picked from the list); else the settings' notify_to."""
    try:
        f = Path(s.local_root) / "notify.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            return d.get("name") or d.get("to", ""), d.get("to", "")
    except (OSError, ValueError):
        pass
    return s.notify_to or "", s.notify_to or ""


def remembered_notify_to(s):
    return remembered_notify(s)[0]


def remember_notify_to(s, to, name=None, via=None, email=None, handle=None, shown=None):
    """Remember the Messages answer: via "slack" (to: a name / member ID, name: as shown, handle: @handle), "email"
    (email) or "none" (shown: the name on the Scan in progress screen); the other way's entry is kept for the next time."""
    try:
        Path(s.local_root).mkdir(parents=True, exist_ok=True)
        f = Path(s.local_root) / "notify.json"
        d = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        if via != "email" and (to or via == "slack"):
            d.update(to=to, name=name or to, handle=handle or "")
        if shown is not None:                       # (also when emptied: the next time shows what was typed last)
            d["shown"] = shown
        if email is not None:
            d["email"] = email
        d["via"] = via or ("slack" if to else "none")
        f.write_text(json.dumps(d), encoding="utf-8")
    except (OSError, ValueError):
        pass


def remembered_results_folder(s):
    """The folder the last scan's results were saved to on this PC (<local_root>/results_folder.json), else the
    settings' out_dir: where "Where to save the results" opens."""
    try:
        f = Path(s.local_root) / "results_folder.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8")).get("folder")
            if d:
                return d
    except (OSError, ValueError):
        pass
    return s.out_dir


def remember_results_folder(s, folder):
    try:
        Path(s.local_root).mkdir(parents=True, exist_ok=True)
        (Path(s.local_root) / "results_folder.json").write_text(json.dumps(dict(folder=str(folder))), encoding="utf-8")
    except OSError:
        pass


def remembered_messages(s):
    """dict(via, name, to, email) given last (notify.json; else the settings: notify_to, notify_email_to)."""
    name, to = remembered_notify(s)
    d = {}
    try:
        f = Path(s.local_root) / "notify.json"
        d = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    except (OSError, ValueError):
        pass
    email = d.get("email") or getattr(s, "notify_email_to", "") or ""
    via = d.get("via") or ("slack" if to else "email" if email else "none")
    return dict(via=via, name=name, to=to, email=email, handle=d.get("handle", ""), shown=d.get("shown", ""))


def remembered_targets_density(s):
    """100x targets per cm2 given last on this PC (<local_root>/targets.json), else the settings' targets_per_cm2."""
    try:
        f = Path(s.local_root) / "targets.json"
        if f.is_file():
            return float(json.loads(f.read_text(encoding="utf-8"))["per_cm2"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return float(getattr(s, "targets_per_cm2", 100.0) or 0.0)


def remember_targets_density(s, v):
    try:
        f = Path(s.local_root) / "targets.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"per_cm2": float(v)}), encoding="utf-8")
    except OSError:
        pass


CHIP_TYPES = (("mounted_1cm", "Mounted 1 cm chips", "chips rigidly mounted together on a block (e.g. a glass slide, often "
                "taped): the chips of a block share one stage frame, for the AFM"),
              ("loose_1cm", "Loose 1 cm chips", "each chip on its own"),
              ("large", "Larger pieces", "any size: the edges are walked and fitted, as before"),
              ("polygon", "Scan areas drawn by hand (polygons)", "no chip detection: you mark the corners of each area to scan "
               "(as many as you want, several areas), and the whole of each area is scanned"))


def remembered_flip(s):
    """(flip left-right, flip up-down) of the live views and pictures: the live view's boxes set last on this PC
    (<local_root>/view_flip.json), else the settings' view_flip."""
    try:
        f = Path(s.local_root) / "view_flip.json"
        d = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        return bool(d["flip"][0]), bool(d["flip"][1])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        v = tuple(getattr(s, "view_flip", (True, True)) or (False, False)) + (False, False)
        return bool(v[0]), bool(v[1])


def remember_flip(s, h, v):
    try:
        f = Path(s.local_root) / "view_flip.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"flip": [bool(h), bool(v)]}), encoding="utf-8")
    except OSError:
        pass


def flipped(img, flip):
    """A picture (array, or PIL image) as the live views show it: flip = (left-right, up-down)."""
    if hasattr(img, "transpose") and not isinstance(img, np.ndarray):          # (PIL)
        from PIL import Image
        if flip[0]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if flip[1]:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    if flip[0]:
        img = img[:, ::-1]
    if flip[1]:
        img = img[::-1]
    return np.ascontiguousarray(img)


def remembered_show_camera(s):
    try:
        f = Path(s.local_root) / "show_camera.json"
        return bool(json.loads(f.read_text(encoding="utf-8"))["show"]) if f.is_file() else bool(getattr(s, "show_camera", True))
    except (OSError, ValueError, KeyError, TypeError):
        return bool(getattr(s, "show_camera", True))


def remember_show_camera(s, on):
    try:
        f = Path(s.local_root) / "show_camera.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"show": bool(on)}), encoding="utf-8")
    except OSError:
        pass


def remembered_chips(s):
    """The Chips window's answers given last on this PC (<local_root>/chips.json), else the settings."""
    d = {}
    try:
        f = Path(s.local_root) / "chips.json"
        d = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    except (OSError, ValueError):
        d = {}
    ct = d.get("chip_type") if d.get("chip_type") in [c[0] for c in CHIP_TYPES] else s.chip_type
    tm = d.get("target_mode") if d.get("target_mode") in ("standard", "avram_new1") else getattr(s, "target_mode", "standard")
    return dict(chip_type=ct, mounts=[int(x) for x in (d.get("mounts") or [4])][:2],
                focus_inset_mm=float(d.get("focus_inset_mm", s.chip_focus_inset_mm)),
                targets_per_cm2=remembered_targets_density(s), target_mode=tm,
                min_clean_um2=float(d.get("min_clean_um2", getattr(s, "target_min_clean_um2", 400.0))),
                clean_thickness=[float(v) for v in (d.get("clean_thickness") or getattr(s, "target_clean_thickness_nm", ()) or ())][:2])


def remember_chips(s, r):
    try:
        f = Path(s.local_root) / "chips.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({k: r[k] for k in ("chip_type", "mounts", "focus_inset_mm", "target_mode", "min_clean_um2", "clean_thickness")
                                 if k in r}),
                     encoding="utf-8")
    except OSError:
        pass
    if "targets_per_cm2" in r:
        remember_targets_density(s, r["targets_per_cm2"])


def apply_chips(s, r):
    """The Chips window's answer -> this job's settings."""
    s.chip_type = r["chip_type"]
    s.targets_per_cm2 = float(r["targets_per_cm2"])
    if r.get("target_mode") in ("standard", "avram_new1"):
        s.target_mode = r["target_mode"]
        s.target_min_clean_um2 = float(r.get("min_clean_um2", s.target_min_clean_um2))
        s.target_clean_thickness_nm = tuple(float(v) for v in (r.get("clean_thickness") or ()))[:2]
    if r["chip_type"] in ("loose_1cm", "mounted_1cm"):
        s.chip_focus_inset_mm = float(r["focus_inset_mm"])


def targets_estimate(s, per_cm2):
    """The Chips window's line: targets (every one imaged at 100x) and 100x time for a few piece sizes at per_cm2."""
    sec = float(getattr(s, "hundred_s_per_target", 7.6) or 7.6)
    out = []
    for a, b in ((1.0, 1.0), (2.5, 1.9), (7.4, 1.0)):
        hi = int(getattr(s, "targets_max_piece", 0) or 0)
        n = 80 if per_cm2 <= 0 else int(max(s.targets_min_piece, round(per_cm2 * a * b)))
        n = min(n, hi) if hi > 0 and per_cm2 > 0 else n
        out.append(f"{a:g} x {b:g} cm: {n} targets (~{n * sec / 60:.0f} min)")
    return "; ".join(out)


def apply_messages(s, r):
    """The Messages answer -> this job's settings (Slack: notify_to; email: notify_email_to), remembered."""
    via = r.get("via")
    if via == "slack":
        s.notify_to, s.notify_email_to = r.get("to", ""), ""
    elif via == "email":
        s.notify_to, s.notify_email_to = "", r.get("email", "")
    else:
        s.notify_to, s.notify_email_to = "", ""
    remember_notify_to(s, r.get("to", "") if via == "slack" else "", r.get("name"), via, r.get("email") if via == "email" else None,
                       r.get("handle"), r.get("shown"))


def contact_of(r):
    """Who runs this job, for the Scan in progress screen, from the Messages answer: dict(name, how) where how is
    "Slack: @handle" / "Slack: name" / "Email: address" / "" (no messages: only the name typed for the screen)."""
    r = r or {}
    via = r.get("via")
    if via == "slack":
        h = r.get("handle") or ""
        return dict(name=r.get("name") or r.get("to") or "", how=f"Slack: @{h}" if h else f"Slack: {r.get('name') or r.get('to')}")
    if via == "email":
        return dict(name=r.get("shown") or "", how=f"Email: {r.get('email', '')}")
    return dict(name=r.get("shown") or r.get("name") or "", how="")


class AutoUI:
    """No windows: answers every question itself.  With a simulated microscope (flakeid.scope.mock) the mock* methods
    stand in for the user (focusing, changing the objective, blocking the light), as in the BeanShell simulator runs."""
    headless_analysis_off = False

    def __init__(self, core, analysis=True):
        self.core = core
        self.can_ask = hasattr(core, "mockFocus")
        self.headless_analysis_off = not analysis

    def log(self, msg): pass
    def message(self, title, msg): print(f"[{title}] {msg}")
    def ask_piece_info(self, scanner, title, what, mat, ox): return (mat, ox)

    def ask_pieces(self, scanner):
        raise Stop("multi mode without a screen: set piece_centers")

    def ask_mounted(self, scanner, plan):
        raise Stop("mounted chips without a screen: set piece_centers, piece_mounts and mount_corners")

    def ask_polygons(self, scanner):
        raise Stop("scan areas drawn by hand without a screen: set piece_polygons")

    def ask_dir(self, title, initial): return initial
    def ask_text(self, title, prompt, default): return default
    def ask_mode(self, scanner): return None
    def ask_messages(self, scanner): return None
    def ask_chips(self, scanner): return None
    def confirm(self, title, msg, buttons=("OK", "Cancel")): return 0
    def control(self, scanner, title, msg, buttons, once=()): return None
    def show_image(self, scanner, title, path): pass
    def ask_preview(self, scanner, title, msg, path, buttons=("Keep", "Retake", "Stop")): return 0
    def chip_notes(self, scanner, chips, fields, last, save): pass
    def check_stop(self, scanner): pass
    def progress(self, scanner, text): pass
    def analysis_progress(self, scanner, frac, text): pass
    def upload_progress(self, scanner, info): pass
    def pause_window(self, scanner, info): return "resume"
    def pause_check(self, scanner, info): return "ok"
    def scale_calibrate(self, scanner, info): return "finish"
    def busy_banner(self, scanner, contact): pass
    def banner_off(self): pass

    def mock_hook(self, scanner, name, *args):
        fn = getattr(self.core, name, None)
        if fn:
            fn(*args)

    def wait_objective(self, scanner, name, want, msg):
        hook = {"100x": ("mockObjective", 100), "domain": ("mockDomain",), "10x": ("mockObjective", 10), "50x": ("mockObjective", 50),
                "5x": ("mockObjective", 5)}.get(name)
        if hook and hasattr(self.core, hook[0]):
            getattr(self.core, hook[0])(*hook[1:])
        return True

    def ask_focus(self, scanner, title, what, timeout=None, image=None, button="", cross=False):
        if cross and hasattr(self.core, "mockCentre"):
            self.core.mockCentre()
        if hasattr(self.core, "mockFocus"):
            self.core.mockFocus()
            return 0
        return 3


# --------------------------------------------------------------------------------------------- Tk
class _Handle:
    """A non-modal window with buttons; value() = the last button pressed (or None).  One-shot buttons (control's
    once=...): take(name) is True once per press (and clears it); enable(name, on) greys them out or not."""

    def __init__(self, ui):
        self.ui, self._v, self.win = ui, None, None
        self._once, self._btn = {}, {}

    def value(self): return self._v

    def take(self, name):
        v = bool(self._once.get(name))
        self._once[name] = False
        return v

    def enable(self, name, on):
        b = self._btn.get(name)
        if b is not None:
            self.ui._post(lambda: b.state(["!disabled"] if on else ["disabled"]))

    def close(self):
        w = self.win
        if w is not None:
            self.ui._post(lambda: w.destroy())
            self.win = None


PAUSE_TEXT = "Pause scan and take normal image"     # (the main window's Pause button; Avram, 2026-10-27)


class TkUI:
    def __init__(self, title="FlakeIdentifier scan", theme="dark"):
        import tkinter as tk
        self.tk = tk
        dpi_aware()
        self.root = tk.Tk()
        self._icons = set_icon(self.root)
        self.log_fg, self.log_bg = apply_theme(self.root, theme)
        st = ttk.Style(self.root)
        self.bg = str(st.lookup("TFrame", "background") or st.lookup(".", "background") or self.root.cget("bg"))
        self.root.configure(bg=self.bg)
        self.root.title(title)
        self.q = queue.Queue()
        self.can_ask = True
        self.headless_analysis_off = False
        self.stop_requested = False
        self.back_to_start = False               # "Start screen" during a job: stop it, then the start screen
        self.job_running = False
        self.idle = threading.Event()            # after a stopped / failed job: the user's choice (Start screen / Close)
        self.idle_choice = None
        self._start_win = None                   # the start screen while it is open
        self._banner = None                      # the "Scan in progress" window while a job runs (busy_banner)
        self._uploads = {}                       # run label -> its "copying to the share" window (upload_progress)
        self.pause_requested = False             # Pause pressed: the scan pauses at its next safe point (Scanner.pause_point)
        self._build_main()
        self.root.after(30, self._pump)

    # ---- thread plumbing
    def _pump(self):
        try:
            while True:
                fn = self.q.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.root.after(30, self._pump)

    def _post(self, fn):
        self.q.put(fn)

    def _ask(self, make):
        """Run make(done) in the Tk thread; it builds a window and calls done(result) when answered.  Waits."""
        ev, box = threading.Event(), {}

        def done(r):
            box["r"] = r
            ev.set()
        if threading.current_thread() is threading.main_thread():
            make(done)
            while not ev.is_set():
                self.root.update()
                time.sleep(0.01)
        else:
            self._post(lambda: make(done))
            ev.wait()
        return box.get("r")

    # ---- main window
    def _build_main(self):
        from tkinter import scrolledtext
        r = self.root
        top = ttk.Frame(r)
        top.pack(fill="x", padx=8, pady=(8, 4))
        try:
            self._logo = tk.PhotoImage(master=r, file=str(ASSETS / "logo_32.png"))
            ttk.Label(top, image=self._logo).pack(side="left", padx=(0, 8))
        except Exception:
            pass
        self.status = ttk.Label(top, text="starting ...", anchor="w", font=("Segoe UI", 11, "bold"))
        self.status.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(top, text="Stop the scan", command=self._stop)
        self.stop_btn.pack(side="right")
        # Pause: the scan stops at its next safe point (a few seconds) for pictures in between, and goes on with Resume
        try:
            ttk.Style().configure("Pause.TButton", font=("Segoe UI", 14, "bold"), padding=(18, 8))
        except Exception:
            pass
        self.pause_btn = ttk.Button(top, text=PAUSE_TEXT, command=self._pause, state="disabled", style="Pause.TButton")
        self.pause_btn.pack(side="right", padx=(0, 10))
        self.start_btn = ttk.Button(top, text="Start screen", command=self._to_start)
        self.start_btn.pack(side="right", padx=(0, 6))
        self.top_row = top
        self.ana_row = ttk.Frame(r)                       # (the analysis's progress at the end of a run: analysis_progress)
        self.ana_bar = ttk.Progressbar(self.ana_row, orient="horizontal", length=280, maximum=100)
        self.ana_bar.pack(side="left")
        self.ana_text = ttk.Label(self.ana_row, text="", anchor="w")
        self.ana_text.pack(side="left", padx=(10, 0), fill="x", expand=True)
        # the camera panel: the last picture the scan took and what it is doing (preview; display only)
        pv = ttk.Frame(r)
        pv.pack(side="right", fill="y", padx=(0, 8), pady=(0, 8))
        self.pv_show = tk.BooleanVar(value=True)
        self._pv_on, self._pv_latest, self._pv_pending, self._pv_last, self._pv_s = True, None, False, 0.0, None
        ttk.Checkbutton(pv, text="Show the camera", variable=self.pv_show, command=self._pv_toggle).pack(anchor="w")
        self.pv_img = tk.Label(pv, bg="black", borderwidth=0, width=48, height=17)
        self.pv_img.pack(anchor="n", pady=(4, 2))
        self.pv_cap = ttk.Label(pv, text="", font=("Segoe UI", 9), foreground="#9a96b8", wraplength=360, justify="left")
        self.pv_cap.pack(anchor="w")
        self.text = scrolledtext.ScrolledText(r, width=110, height=30, font=("Consolas", 9), fg=self.log_fg, bg=self.log_bg,
                                              insertbackground=self.log_fg, relief="flat", borderwidth=0, padx=6, pady=4)
        self.text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def preview(self, scanner, img, caption):
        """The camera panel (from the scan thread): keeps a reference to the newest picture; the Tk thread shows it at
        most every PREVIEW_S.  Nothing waits: the scan goes on."""
        if self._pv_s is None:                          # (the first picture of the program: the remembered choice)
            self._pv_s = scanner.s
            on = remembered_show_camera(scanner.s)
            self._pv_on = on
            self._post(lambda: self.pv_show.set(on))
        if not self._pv_on:
            return
        self._pv_latest = (img, caption, scanner)
        if not self._pv_pending:
            self._pv_pending = True
            self._post(self._pv_render)

    PREVIEW_S = 0.5

    def _pv_render(self):
        wait = self.PREVIEW_S - (time.time() - self._pv_last)
        if wait > 0:
            self.root.after(int(wait * 1000) + 1, self._pv_render)
            return
        self._pv_pending = False
        got, self._pv_latest = self._pv_latest, None
        if got is None or not self._pv_on:
            return
        img, caption, sc = got
        self._pv_last = time.time()
        try:
            import cv2
            mono = bool(getattr(sc, "mono_view", False))
            dark = getattr(sc, "DARK", None)
            disp, _ = to_display(img, 0.0 if dark is None else dark, None, sc.s.display_level, False, 0.5, None, mono)
            disp = flipped(disp, remembered_flip(sc.s))
            k = min(360.0 / disp.shape[1], 272.0 / disp.shape[0], 1.0)
            w, h = max(1, int(disp.shape[1] * k)), max(1, int(disp.shape[0] * k))
            disp = np.ascontiguousarray(cv2.resize(disp, (w, h), interpolation=cv2.INTER_AREA))
            # (Tk's own PPM photo: no PIL in the way, a few ms for 360 x 272)
            ph = self.tk.PhotoImage(master=self.root, data=f"P6 {w} {h} 255 ".encode() + disp.tobytes(), format="PPM")
            self.pv_img.configure(image=ph, width=w, height=h)
            self.pv_img.image = ph
            self.pv_cap.configure(text=caption)
        except Exception as e:                          # (display only: never costs the scan)
            self.pv_cap.configure(text=f"(camera picture not shown: {type(e).__name__}: {e})")

    def _pv_toggle(self):
        self._pv_on = bool(self.pv_show.get())
        if self._pv_s is not None:
            remember_show_camera(self._pv_s, self._pv_on)
        if not self._pv_on:
            self.pv_img.configure(image="", width=48, height=17)
            self.pv_img.image = None
            self.pv_cap.configure(text="")

    def _stop(self):
        self.stop_requested = True
        self.stop_btn.configure(text="Stopping ...", state="disabled")

    def _pause(self):
        self.pause_requested = True
        self.pause_btn.configure(text="Pausing at the next safe point ...", state="disabled")
        self._banner_pause("Pausing at the next safe point (a few seconds) ...", False)

    def _banner_pause(self, text, enabled):
        """The Scan in progress window's Pause button: its text and state (Tk thread)."""
        b = getattr(self, "_banner", None)
        if b is not None and b.get("pause") is not None:
            try:
                b["pause"].configure(text=text, state="normal" if enabled else "disabled")
            except Exception:
                pass

    def _to_start(self):
        """Back to the start screen: during a job it is stopped first (at its next safe point, as Stop does); after a
        stopped or failed job the next one starts; while the start screen is open it is brought to the front."""
        w = self._start_win
        if w is not None:
            try:
                if w.winfo_exists():
                    w.deiconify(); w.lift(); w.focus_force()
                    return
            except Exception:
                pass
        if self.job_running:
            if self.back_to_start:
                return
            if self.confirm("Start screen", "Stop the current job and go back to the start screen?\n\nIt stops at the next safe "
                            "point, as \"Stop\" does (an open window, e.g. a focus question, has to be answered first).",
                            ("Stop and go back", "Cancel")) != 0:
                return
            self.back_to_start = True
            self._stop()
            self.start_btn.configure(text="Going back ...", state="disabled")
        else:
            self.idle_choice = "start"
            self.idle.set()

    def _close(self):
        self.idle_choice = "close"
        self.idle.set()

    def log(self, msg):
        def f():
            self.text.insert("end", msg + "\n")
            self.text.see("end")
        self._post(f)

    def progress(self, scanner, text):
        self._last_status = text

        def f():
            self.status.configure(text=text)
            b = getattr(self, "_banner", None)
            if b is not None:
                try:
                    b["status"].configure(text="Now: " + text)
                except Exception:
                    pass
        self._post(f)

    # ---- "Scan in progress" screen for people coming to use the microscope
    BANNER_HIDE_MIN = 10

    def busy_banner(self, scanner, contact):
        """A big window while the job runs: SCAN IN PROGRESS, who runs it and how to reach them (contact_of: name, how),
        when it started and what it is doing.  It cannot be closed, only hidden for BANNER_HIDE_MIN minutes (by the
        person running the scan, to work at the PC); banner_off() at the end of the job."""
        tk = self.tk
        contact = contact or {}

        def f():
            self._banner_destroy()
            w = tk.Toplevel(self.root)
            w.title("SCAN IN PROGRESS" + (f" - {contact.get('name')}" if contact.get("name") else ""))
            sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
            W, H = int(sw * 0.72), int(sh * 0.66)
            w.geometry(f"{W}x{H}+{(sw - W) // 2}+{max(0, (sh - H) // 3)}")
            px = lambda frac: -max(12, int(sh * frac))           # (negative: font size in pixels, scaled to the screen)
            bg, fg, muted, hot = "#15121f", "#f4f1ff", "#b9b4d6", "#e8590c"
            w.configure(bg=bg)
            tk.Label(w, text="SCAN IN PROGRESS", bg=hot, fg="white", font=("Segoe UI", px(0.075), "bold"),
                     pady=int(sh * 0.02)).pack(fill="x")
            tk.Label(w, text="The microscope is running an automatic scan. Please don't use it, move the stage, "
                             "or change the objective or the lamp.  To take a few pictures, press PAUSE: the scan stops "
                             "in a few seconds and goes on afterwards.", bg=bg, fg=fg, font=("Segoe UI", px(0.024)),
                     wraplength=int(W * 0.9), justify="center").pack(pady=(int(sh * 0.025), int(sh * 0.012)))
            # (the pause, as the main window's Pause button: someone needs the microscope for a few pictures)
            pb = tk.Button(w, text="PAUSE THE SCAN", font=("Segoe UI", px(0.04), "bold"), bg="#2f9e44", fg="white",
                           activebackground="#37b24d", activeforeground="white", disabledforeground="#d8f5dd",
                           relief="flat", padx=int(sw * 0.03), pady=int(sh * 0.012), cursor="hand2",
                           command=lambda: self._pause() if self.job_running and not self.pause_requested else None)
            pb.pack(pady=(0, int(sh * 0.01)))
            if contact.get("name") or contact.get("how"):
                tk.Label(w, text="Running this scan:", bg=bg, fg=muted, font=("Segoe UI", px(0.022))).pack()
                if contact.get("name"):
                    tk.Label(w, text=contact["name"], bg=bg, fg=fg, font=("Segoe UI", px(0.052), "bold")).pack()
                if contact.get("how"):
                    tk.Label(w, text=contact["how"], bg=bg, fg=fg, font=("Segoe UI", px(0.036))).pack(pady=(int(sh * 0.005), 0))
            started = time.strftime("%H:%M, %a %d %b").replace(" 0", " ")
            tk.Label(w, text=f"Started {started}", bg=bg, fg=muted, font=("Segoe UI", px(0.02))).pack(pady=(int(sh * 0.03), 0))
            status = tk.Label(w, text="Now: " + (getattr(self, "_last_status", "") or "starting"), bg=bg, fg=muted,
                              font=("Segoe UI", px(0.018)), wraplength=int(W * 0.9), justify="center")
            status.pack(pady=(int(sh * 0.006), 0))
            bot = tk.Frame(w, bg=bg)
            bot.pack(side="bottom", fill="x", pady=int(sh * 0.015))

            def hide():
                w.withdraw()
                b = self._banner
                if b is not None:
                    if b.get("timer"):
                        w.after_cancel(b["timer"])
                    b["timer"] = w.after(int(self.BANNER_HIDE_MIN * 60000), show)

            def show():
                if self._banner is not None and self._banner["win"] is w:
                    self._banner["timer"] = None
                    w.deiconify()
                    w.lift()
            ttk.Button(bot, text=f"Hide for {self.BANNER_HIDE_MIN} minutes", command=hide).pack(side="right", padx=16)
            tk.Label(bot, text="(for the person running the scan: it comes back by itself)", bg=bg, fg=muted).pack(side="right")
            w.protocol("WM_DELETE_WINDOW", hide)
            self._banner = dict(win=w, status=status, timer=None, hide=hide, show=show, pause=pb)
            w.lift()
        self._post(f)

    def _banner_destroy(self):
        b, self._banner = getattr(self, "_banner", None), None
        if b is not None:
            try:
                b["win"].destroy()
            except Exception:
                pass

    def banner_off(self):
        self._post(self._banner_destroy)

    def upload_progress(self, scanner, info):
        """A window per run while its results are copied to the share (Scanner.upload, 2026-10-27): the percentage, MB
        and files done, the time left and when it will be done; it stays open with the outcome when the copy ends
        (Close).  info: run, dst, state ("copying" | "done" | "failed"), done / total (bytes), files / n_files, elapsed
        (s), error, local."""
        def f():
            key = info.get("run") or ""
            W = self._uploads.get(key)
            if W is None or not W["w"].winfo_exists():
                w = self._dialog(f"Copying {key} to the share")
                fr = ttk.Frame(w, padding=14)
                fr.pack(fill="both", expand=True)
                ttk.Label(fr, text=f"Copying the results of {key} to the share", font=("Segoe UI", 11, "bold")).pack(anchor="w")
                ttk.Label(fr, text=info.get("dst", ""), foreground="#9a96b8").pack(anchor="w")
                bar = ttk.Progressbar(fr, length=440, mode="determinate", maximum=100)
                bar.pack(fill="x", pady=(10, 4))
                pct = ttk.Label(fr, text="", font=("Segoe UI", 12, "bold"))
                pct.pack(anchor="w")
                det = ttk.Label(fr, text="")
                det.pack(anchor="w")
                eta = ttk.Label(fr, text="", wraplength=440)
                eta.pack(anchor="w", pady=(2, 0))
                btn = ttk.Button(fr, text="Hide (the copy goes on)", command=w.destroy)
                btn.pack(anchor="e", pady=(10, 0))
                W = self._uploads[key] = dict(w=w, bar=bar, pct=pct, det=det, eta=eta, btn=btn)
            done, total = float(info.get("done") or 0), float(info.get("total") or 0)
            k, n, el = int(info.get("files") or 0), int(info.get("n_files") or 0), float(info.get("elapsed") or 0)
            mins = lambda s_: f"{s_ / 60:.0f} min" if s_ >= 90 else f"{s_:.0f} s"   # noqa: E731
            state = info.get("state")
            if state == "done":
                W["bar"].configure(value=100)
                W["pct"].configure(text="Done: 100 %")
                W["det"].configure(text=f"{k} files, {done / 1e6:.0f} MB copied in {mins(el)}")
                W["eta"].configure(text="The results are on the share (the designer adds the run now).")
                W["btn"].configure(text="Close")
            elif state == "failed":
                W["pct"].configure(text="The copy failed", foreground="#e06c75")
                W["det"].configure(text=info.get("error", ""))
                W["eta"].configure(text=f"The results are still on this PC: {info.get('local', '')}")
                W["btn"].configure(text="Close")
            else:
                frac = done / total if total > 0 else 0.0
                W["bar"].configure(value=frac * 100)
                W["pct"].configure(text=f"{frac * 100:.0f} %" if total > 0 else "Starting ...")
                W["det"].configure(text=(f"{done / 1e6:.0f} of {total / 1e6:.0f} MB, {k} of {n} files" if n else "Listing the files ..."))
                if done > 0 and el >= 3 and total > done:
                    left = (total - done) / (done / el)
                    W["eta"].configure(text=f"About {mins(left)} left: done at about "
                                            f"{time.strftime('%H:%M', time.localtime(time.time() + left))}")
                else:
                    W["eta"].configure(text="Working out the time left ...")
        self._post(f)

    def analysis_progress(self, scanner, frac, text):
        """The bar under the status line while the run waits for the analysis (100x export): frac 0..1, None (not known
        yet: moving), "done" (hidden)."""
        def f():
            if frac == "done":
                self.ana_bar.stop()
                self.ana_row.pack_forget()
                return
            if not self.ana_row.winfo_ismapped():
                self.ana_row.pack(fill="x", padx=8, pady=(0, 6), after=self.top_row)
            if frac is None:
                self.ana_bar.configure(mode="indeterminate")
                self.ana_bar.start(40)
            else:
                self.ana_bar.stop()
                self.ana_bar.configure(mode="determinate", value=max(0.0, min(1.0, float(frac))) * 100)
            self.ana_text.configure(text=text + ("" if frac is None else f"   ({float(frac) * 100:.0f} % in all)"))
        self._post(f)

    def check_stop(self, scanner):
        if self.stop_requested:
            scanner.stop_all = True
            raise Stop("stopped by the user")
        if self.pause_requested:                            # (the stop checks are safe points for a pause too)
            scanner.pause_point()

    def mock_hook(self, scanner, name, *args):
        pass

    # ---- simple dialogs
    def message(self, title, msg):
        self.confirm(title, msg, ("OK",))                   # (themed, unlike tkinter's messagebox)

    def confirm(self, title, msg, buttons=("OK", "Cancel")):
        def make(done):
            w = self._dialog(title)
            ttk.Label(w, text=msg, justify="left", wraplength=560).pack(padx=16, pady=12, anchor="w")
            bf = ttk.Frame(w)
            bf.pack(pady=(0, 12))
            for i, b in enumerate(buttons):
                ttk.Button(bf, text=b, width=max(10, len(b) + 2), command=lambda i=i: (w.destroy(), done(i))).pack(side="left", padx=6)
            w.protocol("WM_DELETE_WINDOW", lambda: (w.destroy(), done(None)))
        return self._ask(make)

    def live_viewer(self, scanner):
        """A live view opened from the start screen (look at the chips before choosing what to do): the stage and focus
        move freely; the exposure starts at the 10x scan's (exposure_ms) and can be changed.  Returns its close
        function, or None if the microscope could not be set up."""
        sc, s, c = scanner, scanner.s, scanner.core
        try:
            if not getattr(sc, "xy", None):              # (the scan sets these up after the start screen)
                sc.setup_devices()
                sc.DARK = s.dark * (sc.full_scale() + 1) / 4096.0
            if s.set_camera:
                sc.apply_camera(s.exposure_ms)
            else:
                c.setExposure(s.exposure_ms)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Live view", f"The microscope could not be set up for the live view:\n\n{type(e).__name__}: {e}")
            return None
        sc.view_scale = 1.0
        w = self._dialog("Live view")
        lv = LiveView(self, w, sc, free_z=True)
        lv.frame.pack(padx=8, pady=8)
        ef = ttk.Frame(w)
        ef.pack(padx=8, pady=(0, 10), fill="x")
        ttk.Label(ef, text="Exposure").pack(side="left")
        ex = self.tk.StringVar(value=f"{s.exposure_ms:g}")
        sp = ttk.Spinbox(ef, from_=0.1, to=5000, increment=0.5, textvariable=ex, width=8)
        sp.pack(side="left", padx=4)
        ttk.Label(ef, text="ms").pack(side="left")
        state = {"open": True}

        def set_exp(ms=None, hundred=None):
            if lv.busy():
                return
            if ms is not None:
                ex.set(f"{ms:g}")
            try:
                v = float(ex.get())
                assert 0.01 <= v <= 10000
            except (ValueError, AssertionError):
                lv.readout.configure(text="exposure: a number of ms (0.01 to 10000)")
                return
            if hundred is not None:
                lv.set_objective(hundred)
            lv.set_exposure(v)
        sp.configure(command=set_exp)
        sp.bind("<Return>", lambda e: set_exp())
        ttk.Button(ef, text="Set", width=6, command=set_exp).pack(side="left", padx=(4, 12))
        ttk.Button(ef, text=f"10x ({s.exposure_ms:g} ms)", command=lambda: set_exp(s.exposure_ms, False)).pack(side="left", padx=2)
        ttk.Button(ef, text=f"100x ({s.target_exposure_ms:g} ms)", command=lambda: set_exp(s.target_exposure_ms, True)).pack(side="left", padx=2)
        ttk.Label(ef, text="(these also set the stage steps and flat field)", foreground="#9a96b8").pack(side="left", padx=8)

        def close():
            if not state["open"]:
                return
            state["open"] = False
            lv.stop()
            sc.view_scale = 1.0                          # (the scan sets its own once started)
            try:
                w.destroy()
            except Exception:
                pass
        ttk.Button(ef, text="Close", width=8, command=close).pack(side="right")
        w.protocol("WM_DELETE_WINDOW", close)
        lv.start()
        close.is_open = lambda: state["open"]
        return close

    def ask_mode(self, scanner):
        """Start screen: scan chips, take a flat field, or capture 100x images.  Returns dict(mode=..., flat options)
        or None (quit)."""
        s0 = scanner.s

        def make(done):
            w = self._dialog("FlakeIdentifier")
            self._start_win = w
            w.bind("<Destroy>", lambda e: setattr(self, "_start_win", None) if e.widget is w else None)
            f = ttk.Frame(w, padding=16)
            f.pack(fill="both", expand=True)
            ttk.Label(f, text="What would you like to do?", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
            mode = tk.StringVar(value="multi")
            # the everyday jobs; the calibrations and tests in a sub-list under them ("Calibrations and tests", opened by
            # its button or when one of them is chosen)
            opts = (("multi", "Scan chips", "find and scan the chips at 10x, analyse them, image the best flakes at 100x"),
                    ("retarget", "Re-image 100x targets", "a chip scanned before (moved, or taken off and put back): found again from its flakes"),
                    ("device", "Device photos", "photos of a device at 100x, 50x, 10x and 5x (and any extra ones): frame-averaged, "
                                                "shifted at 100x / 50x, with high-contrast frames of its terraces"))
            cal_opts = (("flat", "Take a flat field", "frames of a bare chip for the 10x or 100x flat-field correction"),
                        ("offsets", "Objective offsets", "where the same feature is with each objective (stage XY and focus) and each "
                                                         "one's pixel size, for the device photos (start on a small, sharp feature)"),
                        ("domaintest", "Domain camera test", "the IR camera's XY and focus offsets from the 100x objective, and its "
                                                             "autofocus (start at 100x on a small, sharp feature)"),
                        ("capture", "Capture 100x images", "raw images of flakes you pick, for developing the 100x processing"),
                        ("scalecal", "Scale bar calibration", "the um per pixel of each objective (5x, 10x, 50x, 100x) from a slide "
                                                              "with known distances: draw a line, type its length"))
            for i, (k, t, d) in enumerate(opts):
                ttk.Radiobutton(f, text=t, value=k, variable=mode).grid(row=1 + 2 * i, column=0, columnspan=3, sticky="w", pady=(6, 0))
                ttk.Label(f, text=d, foreground="#9a96b8").grid(row=2 + 2 * i, column=0, columnspan=3, sticky="w", padx=(28, 0))
            cal_w = []
            for i, (k, t, d) in enumerate(cal_opts):
                rb_ = ttk.Radiobutton(f, text=t, value=k, variable=mode)
                rb_.grid(row=11 + 2 * i, column=0, columnspan=3, sticky="w", padx=(28, 0), pady=(6, 0))
                lb_ = ttk.Label(f, text=d, foreground="#9a96b8")
                lb_.grid(row=12 + 2 * i, column=0, columnspan=3, sticky="w", padx=(56, 0))
                cal_w += [rb_, lb_]
            cal_open = {"on": False}
            cal_b = ttk.Button(f, text="", width=26)
            cal_b.grid(row=10, column=0, columnspan=3, sticky="w", pady=(12, 0))

            def show_cal(on):
                cal_open["on"] = on
                for x in cal_w:
                    x.grid() if on else x.grid_remove()
                cal_b.configure(text=("Calibrations and tests  \u25be" if on else "Calibrations and tests  \u25b8"))
                if not on and mode.get() in [k for k, _, _ in cal_opts]:
                    mode.set("multi")
            cal_b.configure(command=lambda: show_cal(not cal_open["on"]))
            show_cal(False)
            rt = ttk.LabelFrame(f, text=" Re-image ", padding=10)
            rt.grid(row=31, column=0, columnspan=3, sticky="ew", pady=(12, 0))
            root = Path(s0.local_root)
            try:
                found = sorted((p for p in root.glob("*_targets.csv") if "_retake" not in p.name and "_partial" not in p.name
                                and (root / p.name.replace("_targets.csv", "_chip.json")).is_file()),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                found = []
            pieces = [p.name[:-len("_targets.csv")] for p in found]
            ttk.Label(rt, text="Piece").grid(row=0, column=0, sticky="w", padx=(0, 10))
            pc = ttk.Combobox(rt, values=pieces, width=30, state="readonly")
            if pieces:
                pc.set(pieces[0])
            pc.grid(row=0, column=1, sticky="w", pady=2)
            rel = tk.BooleanVar(value=False)
            ttk.Checkbutton(rt, text="The chip was taken off and put back (search wider, with rotation)", variable=rel).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
            onlyf = tk.BooleanVar(value=False)
            ttk.Checkbutton(rt, text="Only the flakes without a focused image last time (the piece's retake list)", variable=onlyf).grid(row=2, column=0, columnspan=2, sticky="w")
            byhand = tk.BooleanVar(value=s0.retarget_manual_points > 0)
            ttk.Checkbutton(rt, text=f"Align by centring {max(s0.retarget_manual_points, 3)} flakes by hand at 100x (recommended; otherwise found automatically)",
                            variable=byhand).grid(row=3, column=0, columnspan=2, sticky="w")
            upd = tk.BooleanVar(value=False)
            ttk.Checkbutton(rt, text="No imaging: only update the viewer, designer run and results folder from the re-images done",
                            variable=upd).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
            fl = ttk.LabelFrame(f, text=" Flat field ", padding=10)
            fl.grid(row=32, column=0, columnspan=3, sticky="ew", pady=(12, 0))
            ttk.Label(fl, text="Objective").grid(row=0, column=0, sticky="w", padx=(0, 10))
            obj = ttk.Combobox(fl, values=("10x", "100x", "IR (domain camera)"), width=18, state="readonly")
            obj.set({"domain": "IR (domain camera)"}.get(s0.flat_objective, s0.flat_objective if s0.flat_objective in ("10x", "100x") else "10x"))
            obj.grid(row=0, column=1, sticky="w", pady=2)
            ttk.Label(fl, text="Oxide (SiO2) thickness, nm").grid(row=1, column=0, sticky="w", padx=(0, 10))
            ox = ttk.Entry(fl, width=8)
            ox.insert(0, f"{s0.flat_oxide_nm:g}")
            ox.grid(row=1, column=1, sticky="w", pady=2)
            ttk.Label(fl, text="Measured oxide (Filmetrics), nm").grid(row=5, column=0, sticky="w", padx=(0, 10), pady=(6, 0))
            mox = ttk.Entry(fl, width=8)
            mox.grid(row=5, column=1, sticky="w", pady=(6, 0))
            ttk.Label(fl, text="optional: the piece's measured thickness, recorded as the calibration for this wafer type",
                      foreground="#9a96b8").grid(row=6, column=0, columnspan=2, sticky="w")
            ttk.Label(fl, text="Frames (positions)").grid(row=2, column=0, sticky="w", padx=(0, 10))
            nfr = ttk.Spinbox(fl, from_=9, to=100, width=6)
            nfr.set(str(s0.flat_frames))
            nfr.grid(row=2, column=1, sticky="w", pady=2)
            dark = tk.BooleanVar(value=s0.flat_dark_frames > 0)
            ttk.Checkbutton(fl, text="Dark frames at the end (you block the light)", variable=dark).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Label(fl, text="Focus").grid(row=4, column=0, sticky="w", padx=(0, 10), pady=(4, 0))
            focus_opts = {"Field stop, tilted plane (recommended)": "fieldstop", "Fixed (focus once by hand)": "fixed",
                          "Autofocus at every position (10x only)": "auto"}
            foc = ttk.Combobox(fl, values=list(focus_opts), width=36, state="readonly")
            foc.set(next((k for k, v in focus_opts.items() if v == s0.flat_focus), "Field stop, tilted plane (recommended)"))
            foc.grid(row=4, column=1, sticky="w", pady=(4, 0))
            # device photos: a fabrication step of a stack (device) folder, written there as NN_<step>/ (devicephotos.py;
            # docs/stack_workflow.md, "Stack folders"); the folder is remembered in <local_root>/device_photos.json
            from .devicephotos import remembered_stack_dir, stack_device_name
            dv = ttk.LabelFrame(f, text=" Device photos ", padding=10)
            dv.grid(row=33, column=0, columnspan=3, sticky="ew", pady=(12, 0))
            ttk.Label(dv, text="Stack folder").grid(row=0, column=0, sticky="w", padx=(0, 10))
            dfold = tk.StringVar(value=remembered_stack_dir(s0))
            dfe = ttk.Entry(dv, textvariable=dfold, width=52)
            dfe.grid(row=0, column=1, sticky="ew", pady=2)

            def browse():
                from tkinter import filedialog
                cur = dfold.get().strip()
                p = filedialog.askdirectory(title="The device's stack folder (with its .stack file)", parent=w,
                                            initialdir=cur if cur and Path(cur).is_dir() else (str(Path(cur).parent) if cur else ""))
                if p:
                    dfold.set(str(Path(p)))
            ttk.Button(dv, text="Browse...", width=10, command=browse).grid(row=0, column=2, sticky="w", padx=(6, 0))
            ddev = ttk.Label(dv, text="", foreground="#9a96b8")
            ddev.grid(row=1, column=1, columnspan=2, sticky="w")

            def show_device(*_):
                # (the device name: the folder's .stack file, else the folder's name; looked up as typed)
                p = dfold.get().strip()
                try:
                    if not p:
                        ddev.configure(text="the device's folder (<share>/<you>/Devices/...)")
                    elif Path(p).is_dir():
                        st = any(Path(p).glob("*.stack"))
                        ddev.configure(text=f"device: {stack_device_name(p)}" + ("" if st else "  (no .stack file: the folder's name)"))
                    else:
                        ddev.configure(text="(folder not found: it is made when you press Start, if you agree)")
                except OSError:
                    ddev.configure(text="(folder not reachable)")
            pend = {"id": None}

            def later(*_):                              # (looked up once typing pauses: a share that is down is slow)
                if pend["id"] is not None:
                    w.after_cancel(pend["id"])
                pend["id"] = w.after(600, show_device)
            dfold.trace_add("write", later)
            show_device()
            ttk.Label(dv, text="Step").grid(row=2, column=0, sticky="w", padx=(0, 10))
            dstep = ttk.Entry(dv, width=20)
            dstep.insert(0, s0.device_step or "")
            dstep.grid(row=2, column=1, sticky="w", pady=2)
            ttk.Label(dv, text="short name, part of the file names: e.g. contacts, SETs, tgetch", foreground="#9a96b8").grid(
                row=3, column=1, columnspan=2, sticky="w")
            ttk.Label(dv, text="What was done").grid(row=4, column=0, sticky="nw", padx=(0, 10), pady=(4, 0))
            ddesc = tk.Text(dv, width=52, height=3, wrap="word", font=("Segoe UI", 10), fg=self.log_fg, bg=self.log_bg,
                            insertbackground=self.log_fg)
            ddesc.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(4, 2))
            ttk.Label(dv, text="Oxide (SiO2) thickness, nm").grid(row=5, column=0, sticky="w", padx=(0, 10))
            dox = ttk.Entry(dv, width=8)
            dox.insert(0, f"{s0.oxide_nm:g}")
            dox.grid(row=5, column=1, sticky="w", pady=2)
            of = ttk.LabelFrame(f, text=" Objective offsets ", padding=10)
            of.grid(row=34, column=0, columnspan=3, sticky="ew", pady=(12, 0))
            ttk.Label(of, text="Reference objective (start with it in)").grid(row=0, column=0, sticky="w", padx=(0, 10))
            oref = ttk.Combobox(of, values=("100x", "50x", "10x", "5x"), width=8, state="readonly")
            oref.set(s0.offsets_reference if s0.offsets_reference in ("100x", "50x", "10x", "5x") else "100x")
            oref.grid(row=0, column=1, sticky="w", pady=2)
            err = ttk.Label(f, text="", foreground=ERR_FG)
            err.grid(row=35, column=0, columnspan=3, sticky="w", pady=(6, 0))

            def sync(*_):
                # only the chosen job's options are shown (all of them made the window taller than the screen, 2026-10-19)
                for fr, m in ((rt, "retarget"), (fl, "flat"), (dv, "device"), (of, "offsets")):
                    if mode.get() == m:
                        fr.grid()
                    else:
                        fr.grid_remove()
                st = ["!disabled"] if mode.get() == "flat" else ["disabled"]
                for c in fl.winfo_children():
                    try:
                        c.state(st)
                    except Exception:
                        pass
                if mode.get() == "flat":
                    obj.state(["readonly"]); foc.state(["readonly"])
                for c in rt.winfo_children():
                    try:
                        c.state(["!disabled"] if mode.get() == "retarget" else ["disabled"])
                    except Exception:
                        pass
                if mode.get() == "retarget":
                    pc.state(["readonly"])
                for c in dv.winfo_children():
                    try:
                        c.state(["!disabled"] if mode.get() == "device" else ["disabled"])
                    except Exception:
                        pass
                for c in of.winfo_children():
                    try:
                        c.state(["!disabled"] if mode.get() == "offsets" else ["disabled"])
                    except Exception:
                        pass
                if mode.get() == "offsets":
                    oref.state(["readonly"])
            mode.trace_add("write", sync)
            sync()

            def ok(*_):
                r = dict(mode=mode.get())
                if r["mode"] == "flat":
                    try:
                        r.update(objective={"IR (domain camera)": "domain"}.get(obj.get(), obj.get()), oxide=float(ox.get()), frames=int(nfr.get()), dark=bool(dark.get()), focus=focus_opts[foc.get()],
                                 measured_oxide=float(mox.get()) if mox.get().strip() else None)
                        assert 50 <= r["oxide"] <= 2000 and 5 <= r["frames"] <= 200
                        assert r["measured_oxide"] is None or abs(r["measured_oxide"] / r["oxide"] - 1) < 0.3
                        if r["focus"] == "auto" and r["objective"] == "100x":
                            err.configure(text="At 100x, autofocus on bare substrate focuses on dust in the light path: use the field stop.")
                            return
                    except (ValueError, AssertionError):
                        err.configure(text="Enter the oxide thickness in nm (e.g. 90 or 285), 5-200 frames, and a measured oxide within 30 % of it (or none).")
                        return
                if r["mode"] == "retarget":
                    if not pc.get():
                        err.configure(text=f"No scanned piece with a target list in {root}.")
                        return
                    r.update(piece=pc.get(), reloaded=bool(rel.get()), only_failed=bool(onlyf.get()),
                             manual_points=max(s0.retarget_manual_points, 3) if byhand.get() else 0, update_only=bool(upd.get()))
                if r["mode"] == "device":
                    import re
                    from tkinter import messagebox
                    folder, step = dfold.get().strip(), " ".join(dstep.get().split())
                    try:
                        r.update(oxide=float(dox.get()))
                        assert 50 <= r["oxide"] <= 2000
                    except (ValueError, AssertionError):
                        err.configure(text="Device photos: enter the oxide thickness in nm (e.g. 285).")
                        return
                    if not folder:
                        err.configure(text="Device photos: choose the device's stack folder (Browse...).")
                        return
                    if not re.search(r"[A-Za-z0-9]", step):
                        err.configure(text="Device photos: enter the step's short name (e.g. contacts): letters and digits.")
                        return
                    try:
                        if not Path(folder).is_dir():
                            if not messagebox.askyesno("Device photos", f"The folder\n{folder}\ndoes not exist.  Make it (a new "
                                                       "device folder)?", parent=w):
                                return
                            Path(folder).mkdir(parents=True)
                    except OSError as e:
                        err.configure(text=f"Device photos: the folder cannot be made or reached ({e}).")
                        return
                    from .devicephotos import remember_stack_dir
                    remember_stack_dir(s0, folder)
                    r.update(stack_dir=folder, step=step, description=ddesc.get("1.0", "end").strip(),
                             device_name=stack_device_name(folder))
                if r["mode"] == "offsets":
                    r["reference"] = oref.get()
                close_live()
                w.destroy()
                done(r)
            viewer = {"close": None}

            def open_live():
                if viewer["close"] is None or not viewer["close"].is_open():
                    viewer["close"] = self.live_viewer(scanner)

            def close_live():
                if viewer["close"] is not None:
                    viewer["close"]()
                    viewer["close"] = None

            def quit_():
                close_live()
                w.destroy()
                done(None)
            lb = ttk.Frame(f)
            lb.grid(row=37, column=0, columnspan=2, sticky="w", pady=(12, 0))
            ttk.Button(lb, text="Live view...", width=14, command=open_live).pack(side="left")
            ttk.Label(lb, text=f"  (exposure {s0.exposure_ms:g} ms, as the 10x scan)", foreground="#9a96b8").pack(side="left")
            ttk.Button(lb, text="Update...", width=10, command=lambda: (close_live(), self.update_window(scanner, w))).pack(side="left", padx=(16, 0))
            b = ttk.Frame(f)
            b.grid(row=37, column=2, sticky="e", pady=(12, 0))
            ttk.Button(b, text="Start", style="Accent.TButton", width=10, command=ok).pack(side="left", padx=6)
            ttk.Button(b, text="Quit", width=10, command=quit_).pack(side="left")
            w.protocol("WM_DELETE_WINDOW", quit_)
        return self._ask(make)

    def update_window(self, scanner, parent):
        """The start screen's "Update...": what is new on GitHub, and installing it (flakeid/scope/update.py), then a
        restart of the program.  Runs in the Tk thread; GitHub is asked in a background thread."""
        from tkinter import messagebox
        from . import update as U
        from .scan import EXPORTING
        s0 = scanner.s
        w = self.tk.Toplevel(parent)
        w.configure(bg=self.bg)
        w.title("Update the scan program")
        w.transient(parent)
        f = ttk.Frame(w, padding=16)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="Update from GitHub", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(f, text=f"{s0.update_repo}, branch {s0.update_branch}; program folder {U.ROOT}", foreground="#9a96b8").pack(anchor="w")
        txt = tk.Text(f, width=92, height=16, wrap="word", font=("Consolas", 9), relief="flat")
        txt.pack(fill="both", expand=True, pady=(10, 6))
        bf = ttk.Frame(f)
        bf.pack(anchor="e")
        inst = ttk.Button(bf, text="Install the update", style="Accent.TButton", state="disabled")
        inst.pack(side="left", padx=6)
        close_b = ttk.Button(bf, text="Close", width=10, command=w.destroy)
        close_b.pack(side="left")
        st = {"busy": False}

        def say(line):                                  # (from any thread)
            def f_():
                if w.winfo_exists():
                    txt.insert("end", line + "\n")
                    txt.see("end")
            self._post(f_)

        def busy_reason():
            bg = [t.name for t in getattr(self, "bg_uploads", []) if t.is_alive()]
            if bg or EXPORTING:
                return ("a run is still being exported or copied in the background (" + ", ".join(bg or sorted(EXPORTING))
                        + "): update when it is done")
            return None

        def checked(c, err):
            if not w.winfo_exists():
                return
            if err:
                say(f"Not checked: {err}")
                return
            for ln in U.describe(c):
                say(ln)
            if not c["up_to_date"]:
                inst.configure(state="normal")
                say("\n\"Install the update\": the new version is downloaded, checked and installed (the current code is "
                    "backed up first; flat fields and settings stay), then the program starts again.")

        def work_check():
            try:
                c, err = U.check(s0), None
            except U.UpdateError as e:
                c, err = None, str(e)
            except Exception as e:
                c, err = None, f"{type(e).__name__}: {e}"
            self._post(lambda: checked(c, err))

        def installed_ok(info, err):
            st["busy"] = False
            if not w.winfo_exists():
                return
            close_b.configure(state="normal")
            if err:
                say(f"\nUPDATE NOT DONE: {err}")
                return
            scanner.say(f"program updated to {info['sha'][:7]} ({info['date']}): {info['message']}")
            say("\nThe update is installed.")
            if messagebox.askyesno("Update installed", f"Installed {info['sha'][:7]}: {info['message']}\n\nRestart the program "
                                   "now?  (No: the new version is used from the next start.)", parent=w):
                self.restart_program()

        def work_install():
            try:
                info, err = U.update(s0, log=say), None
            except U.UpdateError as e:
                info, err = None, str(e)
            except Exception as e:
                info, err = None, f"{type(e).__name__}: {e}"
            self._post(lambda: installed_ok(info, err))

        def start_install():
            why = busy_reason()
            if why:
                say(f"\nNot now: {why}.")
                return
            st["busy"] = True
            inst.configure(state="disabled")
            close_b.configure(state="disabled")
            say("")
            threading.Thread(target=work_install, name="update", daemon=True).start()
        inst.configure(command=start_install)
        w.protocol("WM_DELETE_WINDOW", lambda: None if st["busy"] else w.destroy())
        say("Asking GitHub ...")
        threading.Thread(target=work_check, name="update check", daemon=True).start()

    def restart_program(self):
        """Start the program again (after an update) and end this one: the new one waits until this one has ended."""
        import os
        import subprocess
        import sys
        from .update import ROOT, restart_command
        env = dict(os.environ, FLAKEID_WAIT_PID=str(os.getpid()))
        flags = 0x00000008 if sys.platform == "win32" else 0          # (DETACHED_PROCESS: not ended with this one)
        subprocess.Popen(restart_command(), env=env, cwd=str(ROOT), close_fds=True, creationflags=flags)
        self.root.destroy()

    def ask_messages(self, scanner):
        """Who gets this job's messages (when you are needed, when it ends): Slack (a member of the lab's workspace,
        found as you type), email (anyone), or none.  Returns dict(via, to, name, email) or None (closed: as before)."""
        s0 = scanner.s
        from .notify import slack_token, members as slack_members, email_ready

        def make(done):
            w = self._dialog("Messages")
            f = ttk.Frame(w, padding=16)
            f.pack(fill="both", expand=True)
            ttk.Label(f, text="Who should get messages from this job?", font=("Segoe UI", 12, "bold")).pack(anchor="w")
            ttk.Label(f, text="When you are needed (objective change, a focus question) and when the job is finished.",
                      foreground="#9a96b8").pack(anchor="w", pady=(2, 10))
            R = remembered_messages(s0)
            via = tk.StringVar(value=R["via"] if R["via"] in ("slack", "email", "none") else "slack")
            has_slack, has_mail = bool(slack_token(s0)), email_ready(s0)
            # Slack: a name, completed from the workspace's members (loaded in the background; cached for a day)
            ttk.Radiobutton(f, text="Slack (Young lab workspace)", value="slack", variable=via).pack(anchor="w")
            sl = ttk.Frame(f)
            sl.pack(anchor="w", padx=(24, 0), pady=(2, 8))
            sr = ttk.Frame(sl)
            sr.pack(anchor="w")
            who = ttk.Entry(sr, width=32)
            who.insert(0, R["name"])
            who.pack(side="left")
            ttk.Label(sr, text="  type your name to find it" if has_slack else "  (the Slack bot is not set up on this PC)",
                      foreground="#9a96b8").pack(side="left")
            pick = {"text": R["name"], "id": R["to"] if R["to"] != R["name"] else "", "handle": R.get("handle", "")}  # (a member picked)
            M = []
            if has_slack:
                threading.Thread(target=lambda: M.extend(slack_members(s0)), daemon=True).start()
            lst = tk.Listbox(sl, height=6, width=52, exportselection=False, activestyle="dotbox")
            shown_m = []

            def label(m):
                return m["name"] + (f"  (@{m['handle']})" if m["handle"] and m["handle"].lower() != m["name"].lower() else "")

            def complete(e=None):
                if e is not None and e.keysym in ("Down", "Up", "Return", "Escape", "Tab"):
                    return
                via.set("slack")
                q = " ".join(who.get().lower().split())
                if who.get().strip() != pick["text"]:
                    pick["id"] = ""
                hit = [m for m in M if q and (q in m["name"].lower() or q in m["handle"].lower())][:8] if "@" not in q else []
                if not hit or (len(hit) == 1 and hit[0]["name"] == who.get().strip()):
                    lst.pack_forget()
                    return
                shown_m[:] = hit
                lst.delete(0, "end")
                for m in hit:
                    lst.insert("end", label(m))
                lst.configure(height=len(hit))
                if not lst.winfo_ismapped():
                    lst.pack(anchor="w", pady=(2, 0))

            def choose(e=None):
                sel = lst.curselection()
                if not sel and shown_m:
                    sel = (0,)
                if not sel:
                    return
                m = shown_m[sel[0]]
                who.delete(0, "end")
                who.insert(0, m["name"])
                pick.update(text=m["name"], id=m["id"], handle=m.get("handle") or "")
                lst.pack_forget()
                who.focus_set()
                return "break"

            def down(e=None):
                if lst.winfo_ismapped() and shown_m:
                    lst.focus_set()
                    lst.selection_clear(0, "end")
                    lst.selection_set(0)
                    lst.activate(0)
                    return "break"
            who.bind("<KeyRelease>", complete)
            who.bind("<Down>", down)
            who.bind("<Escape>", lambda e: (lst.pack_forget(), "break")[1])
            lst.bind("<Return>", choose)
            lst.bind("<Double-Button-1>", choose)
            lst.bind("<ButtonRelease-1>", choose)
            lst.bind("<Escape>", lambda e: (lst.pack_forget(), who.focus_set(), "break")[2])
            # email: anyone (not in the lab's Slack)
            ttk.Radiobutton(f, text="Email (not in the lab's Slack)", value="email", variable=via).pack(anchor="w")
            er = ttk.Frame(f)
            er.pack(anchor="w", padx=(24, 0), pady=(2, 8))
            mail = ttk.Entry(er, width=32)
            mail.insert(0, R["email"])
            mail.pack(side="left")
            ttk.Label(er, text="  e.g. name@ucsb.edu" if has_mail else "  (email is not set up on this PC: notify_smtp_host)",
                      foreground="#9a96b8").pack(side="left")
            mail.bind("<KeyRelease>", lambda e: via.set("email") if e.keysym not in ("Tab", "Return") else None)
            mail.bind("<FocusIn>", lambda e: via.set("email"))     # (clicking into a box chooses its way)
            who.bind("<FocusIn>", lambda e: via.set("slack"), add="+")
            ttk.Radiobutton(f, text="No messages", value="none", variable=via).pack(anchor="w")
            # (email / no messages: the name shown on the "Scan in progress" screen; Slack shows the member's name)
            nr = ttk.Frame(f)
            nr.pack(anchor="w", pady=(10, 0))
            ttk.Label(nr, text="Your name (email / no messages), shown on the Scan in progress screen:").pack(anchor="w")
            shown = ttk.Entry(nr, width=32)
            shown.insert(0, R.get("shown", ""))
            shown.pack(anchor="w", pady=(2, 0))
            err = ttk.Label(f, text="", foreground="#e06c75")
            err.pack(anchor="w", pady=(6, 0))
            bf = ttk.Frame(f)
            bf.pack(anchor="e", pady=(8, 0))

            def ok(e=None):
                if e is not None and lst.winfo_ismapped() and e.widget is who:
                    return choose()

                v = via.get()
                typed = " ".join(who.get().split())
                addr = mail.get().strip()
                if v == "slack":
                    if not typed:
                        err.configure(text="Type a Slack name (or choose No messages).")
                        return
                    if not has_slack:
                        err.configure(text="The Slack bot is not set up on this PC: choose Email or No messages.")
                        return
                    same = bool(pick["id"]) and typed == pick["text"]
                    r = dict(via="slack", to=pick["id"] if same else typed, name=typed, handle=pick["handle"] if same else "",
                             shown=" ".join(shown.get().split()))             # (remembered as typed, empty too)
                elif v == "email":
                    import re as _re
                    if not _re.match(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$", addr):
                        err.configure(text="Type one email address, e.g. name@ucsb.edu.")
                        return
                    if not has_mail:
                        err.configure(text="Email is not set up on this PC (notify_smtp_host): choose Slack or No messages.")
                        return
                    if not " ".join(shown.get().split()):
                        err.configure(text="Type your name for the Scan in progress screen.")
                        shown.focus_set()
                        return
                    r = dict(via="email", email=addr, name=addr, shown=" ".join(shown.get().split()))
                else:
                    if not " ".join(shown.get().split()):
                        err.configure(text="Type your name: it is shown on the Scan in progress screen while the job runs.")
                        shown.focus_set()
                        return
                    r = dict(via="none", shown=" ".join(shown.get().split()))
                w.destroy()
                done(r)
            ttk.Button(bf, text="OK", width=10, command=ok, style="Accent.TButton").pack(side="left", padx=6)
            w.bind("<Return>", ok)
            w.protocol("WM_DELETE_WINDOW", lambda: (w.destroy(), done(None)))
            (mail if via.get() == "email" else who).focus_set()
        return self._ask(make)

    def ask_chips(self, scanner):
        """What is on the stage (mounted 1 cm chips / loose 1 cm chips / larger pieces), the blocks and their chips,
        the focus inset of 1 cm chips and the 100x pictures per cm2.  Returns dict(chip_type, mounts, focus_inset_mm,
        targets_per_cm2, ...) or None (closed: as last time), or "cancel": closed when last time was drawn areas with
        Standard targets, whose targets per cm2 must be typed for each job (the job is cancelled)."""
        s0 = scanner.s
        R = remembered_chips(s0)

        def make(done):
            w = self._dialog("Chips")
            f = ttk.Frame(w, padding=16)
            f.pack(fill="both", expand=True)
            ttk.Label(f, text="What is on the stage?", font=("Segoe UI", 12, "bold")).pack(anchor="w")
            ct = tk.StringVar(value=R["chip_type"])
            for key, label, hint in CHIP_TYPES:
                ttk.Radiobutton(f, text=label, value=key, variable=ct).pack(anchor="w", pady=(6, 0))
                ttk.Label(f, text=hint, foreground="#9a96b8").pack(anchor="w", padx=(24, 0))
            # mounted chips: the blocks and their chips
            mf = ttk.Frame(f)
            nm = tk.IntVar(value=max(1, min(2, len(R["mounts"]))))
            mr = ttk.Frame(mf)
            mr.pack(anchor="w")
            ttk.Label(mr, text="Blocks in this scan:").pack(side="left")
            for v in (1, 2):
                ttk.Radiobutton(mr, text=str(v), value=v, variable=nm).pack(side="left", padx=(8, 0))
            cnt = []
            for i in range(2):
                rr = ttk.Frame(mf)
                ttk.Label(rr, text=f"Chips on block {i + 1}:").pack(side="left")
                e = ttk.Entry(rr, width=5)
                e.insert(0, str(R["mounts"][i] if i < len(R["mounts"]) else 4))
                e.pack(side="left", padx=(8, 0))
                cnt.append((rr, e))
            ttk.Label(mf, text="You then mark each chip's centre and 3 corners of each block (the AFM's reference points).",
                      foreground="#9a96b8", wraplength=520).pack(anchor="w", pady=(4, 0))
            # 1 cm chips: focus inset
            ff = ttk.Frame(f)
            fr = ttk.Frame(ff)
            fr.pack(anchor="w")
            ttk.Label(fr, text="Focus points").pack(side="left")
            fin = ttk.Entry(fr, width=5)
            fin.insert(0, f"{R['focus_inset_mm']:g}")
            fin.pack(side="left", padx=(8, 4))
            ttk.Label(fr, text="mm in from the chip edges").pack(side="left")
            ttk.Label(ff, text="Further in keeps focus points off tape over the edges (a chip is ~10 mm: at most ~4 mm).",
                      foreground="#9a96b8").pack(anchor="w")
            # 100x pictures per cm2 of each piece
            tf = ttk.Frame(f)
            ttk.Label(tf, text="100x pictures", font=("Segoe UI", 11, "bold")).pack(anchor="w")
            tr = ttk.Frame(tf)
            tr.pack(anchor="w", pady=(2, 0))
            dens = ttk.Entry(tr, width=6)
            dens_last = f"{R['targets_per_cm2']:g}"         # (drawn areas: no default, typed for each job; 2026-10-27)
            if R["chip_type"] != "polygon":
                dens.insert(0, dens_last)
            dens.pack(side="left")
            ttk.Label(tr, text="  targets per cm\u00b2 of each piece (its area from the edge search or polygon)").pack(side="left")
            ttk.Label(tf, text="every target is imaged at 100x", foreground="#9a96b8").pack(anchor="w")
            est = ttk.Label(tf, text="", foreground="#9a96b8")
            est.pack(anchor="w", pady=(2, 0))

            # which flakes are imaged at 100x: the learned score (standard) or the clean area (avram new1)
            ttk.Label(tf, text="Which flakes", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 0))
            tmode = tk.StringVar(value=R.get("target_mode", "standard"))
            ttk.Radiobutton(tf, text="Standard: the learned score and the quality checks", value="standard",
                            variable=tmode).pack(anchor="w", pady=(2, 0))
            cr = ttk.Frame(tf)
            cr.pack(anchor="w")
            ttk.Radiobutton(cr, text="avram new1: every flake whose clean area is at least", value="avram_new1",
                            variable=tmode).pack(side="left")
            mclean = ttk.Entry(cr, width=7)
            mclean.insert(0, f"{R.get('min_clean_um2', 400.0):g}")
            mclean.pack(side="left", padx=(6, 4))
            ttk.Label(cr, text="um\u00b2, thickness").pack(side="left")
            ct0 = R.get("clean_thickness") or []
            cth = []
            for j, t_ in enumerate(("", "to")):
                if t_:
                    ttk.Label(cr, text=t_).pack(side="left", padx=(4, 0))
                e_ = ttk.Entry(cr, width=6)
                if len(ct0) == 2:
                    e_.insert(0, f"{ct0[j]:g}")
                e_.pack(side="left", padx=(4, 0))
                cth.append(e_)
            ttk.Label(cr, text="nm (empty: hBN 0-100, graphite 0.3-25)").pack(side="left", padx=(4, 0))
            ttk.Label(tf, text="clean area = an area of one colour of the right thickness, found from the colours alone "
                      "(not the flake's outline); every one is imaged (no limit: the flakes per cm\u00b2 above are for Standard), "
                      "largest first, the 100x picture in its middle", foreground="#9a96b8",
                      wraplength=560).pack(anchor="w", padx=(24, 0))
            ttk.Label(tf, text="100x: you first focus by hand one flake per three 10x focus points (spread over the piece); they "
                      "calibrate the 10x focus map to 100x, then the rest use autofocus", foreground="#9a96b8",
                      wraplength=560).pack(anchor="w", padx=(24, 0), pady=(2, 0))
            err = ttk.Label(f, text="", foreground="#e06c75")
            bf = ttk.Frame(f)

            def layout(*_):
                for x in (mf, ff, tf, err, bf):
                    x.pack_forget()
                if ct.get() == "mounted_1cm":
                    mf.pack(anchor="w", padx=(24, 0), pady=(10, 0))
                    for i, (rr, _) in enumerate(cnt):
                        if i < nm.get():
                            rr.pack(anchor="w", pady=(2, 0))
                        else:
                            rr.pack_forget()
                if ct.get() in ("mounted_1cm", "loose_1cm"):
                    ff.pack(anchor="w", padx=(24, 0), pady=(10, 0))
                tf.pack(anchor="w", pady=(14, 0))
                err.pack(anchor="w", pady=(6, 0))
                bf.pack(anchor="e", pady=(8, 0))

            poly = {"on": R["chip_type"] == "polygon", "typed": ""}

            def poly_dens(*_):
                """drawn areas: the box empty (or what was typed for them in this window), to be typed for this job; other
                chip types: their number again"""
                now = ct.get() == "polygon"
                if now != poly["on"]:
                    if now:
                        poly["other"] = dens.get()
                        dens.delete(0, "end")
                        dens.insert(0, poly["typed"])
                    else:
                        poly["typed"] = dens.get()
                        dens.delete(0, "end")
                        dens.insert(0, poly.get("other") or dens_last)
                    poly["on"] = now
                upd()

            def upd(*_):
                try:
                    d = float(dens.get())
                except ValueError:
                    est.configure(text=("Drawn areas: type the number of targets per cm\u00b2 for this job (no default; "
                                        "e.g. 68: a 74 x 10 mm area gets ~500)") if ct.get() == "polygon" and not dens.get().strip()
                                  else "Type a number, e.g. 100.")
                    return
                est.configure(text=targets_estimate(s0, d) if d >= 0 else "Type 0 or more.")
            ct.trace_add("write", layout)
            ct.trace_add("write", poly_dens)
            nm.trace_add("write", layout)
            dens.bind("<KeyRelease>", upd)

            def ok(e=None):
                r = dict(chip_type=ct.get())
                r["target_mode"] = tmode.get()
                try:
                    if not dens.get().strip() and r["chip_type"] == "polygon" and r["target_mode"] == "avram_new1":
                        r["targets_per_cm2"] = float(dens_last)      # (avram new1: not used, no limit)
                    else:
                        r["targets_per_cm2"] = float(dens.get())
                    assert r["targets_per_cm2"] >= 0
                except (ValueError, AssertionError):
                    err.configure(text=("Drawn areas: type the number of targets per cm\u00b2 for this job (e.g. 68: a 74 x 10 mm "
                                        "area gets ~500 targets, all imaged at 100x)." if r["chip_type"] == "polygon" else
                                        "100x pictures: type the number of targets per cm\u00b2 (e.g. 100)."))
                    return
                try:
                    r["min_clean_um2"] = float(mclean.get())
                    assert r["min_clean_um2"] > 0 or r["target_mode"] != "avram_new1"
                except (ValueError, AssertionError):
                    err.configure(text="avram new1: type the smallest clean area in um\u00b2 (e.g. 400).")
                    return
                try:
                    tv = [c_.get().strip() for c_ in cth]
                    r["clean_thickness"] = [] if not any(tv) else [float(tv[0]), float(tv[1])]
                    assert not r["clean_thickness"] or 0 <= r["clean_thickness"][0] < r["clean_thickness"][1]
                except (ValueError, AssertionError):
                    err.configure(text="avram new1: the thickness, e.g. 10 to 80 nm (or both empty: the material's window).")
                    return
                if r["chip_type"] in ("mounted_1cm", "loose_1cm"):
                    try:
                        r["focus_inset_mm"] = float(fin.get())
                        assert 0 <= r["focus_inset_mm"] <= 4.5
                    except (ValueError, AssertionError):
                        err.configure(text="Focus points: type how far in from the chip edges, in mm (0 to 4.5).")
                        return
                if r["chip_type"] == "mounted_1cm":
                    try:
                        r["mounts"] = [int(cnt[i][1].get()) for i in range(nm.get())]
                        assert all(1 <= n <= 30 for n in r["mounts"])
                    except (ValueError, AssertionError):
                        err.configure(text="Chips on each block: type a number from 1 to 30.")
                        return
                w.destroy()
                done(r)
            ttk.Button(bf, text="OK", width=10, command=ok, style="Accent.TButton").pack(side="left", padx=6)
            w.bind("<Return>", ok)
            # closed: as last time, except drawn areas with Standard targets, whose number is typed for each job: cancelled
            last_poly = R["chip_type"] == "polygon" and R.get("target_mode", "standard") != "avram_new1"
            w.protocol("WM_DELETE_WINDOW", lambda: (w.destroy(), done("cancel" if last_poly else None)))
            layout()
            upd()
        return self._ask(make)

    def ask_text(self, title, prompt, default=""):
        def make(done):
            w = self._dialog(title)
            ttk.Label(w, text=prompt, justify="left").pack(padx=16, pady=(12, 4), anchor="w")
            e = ttk.Entry(w, width=40)
            e.insert(0, default)
            e.pack(padx=16, pady=4)
            e.focus_set()
            bf = ttk.Frame(w)
            bf.pack(pady=10)
            def ok(*_):
                v = e.get()
                w.destroy()
                done(v)
            ttk.Button(bf, text="OK", width=10, command=ok, style="Accent.TButton").pack(side="left", padx=6)
            ttk.Button(bf, text="Cancel", width=10, command=lambda: (w.destroy(), done(None))).pack(side="left", padx=6)
            e.bind("<Return>", ok)
        return self._ask(make)

    def ask_dir(self, title, initial):
        from tkinter import filedialog
        return self._ask(lambda done: done(filedialog.askdirectory(title=title, initialdir=initial, parent=self.root) or None))

    def _dialog(self, title):
        w = self.tk.Toplevel(self.root)
        w.configure(bg=self.bg)
        w.title(title)
        w.transient(self.root)
        w.lift()
        w.attributes("-topmost", True)
        w.after(500, lambda: w.attributes("-topmost", False))
        return w

    def ask_piece_info(self, scanner, title, what, mat, ox):
        return self._ask(lambda done: self._piece_info(title, what, mat, ox, done))

    def _piece_info(self, title, what, mat, ox, done, parent=None):
        w = self._dialog(title)
        ttk.Label(w, text=what).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6), sticky="w")
        ttk.Label(w, text="Material:").grid(row=1, column=0, padx=12, sticky="w")
        cb = ttk.Combobox(w, values=["hBN", "graphite"], state="readonly", width=12)
        cb.current(1 if mat == "graphite" else 0)
        cb.grid(row=1, column=1, padx=12, pady=4, sticky="w")
        ttk.Label(w, text="Oxide (SiO2) thickness, nm:").grid(row=2, column=0, padx=12, sticky="w")
        e = ttk.Entry(w, width=10)
        e.insert(0, f"{ox:g}")
        e.grid(row=2, column=1, padx=12, pady=4, sticky="w")
        err = ttk.Label(w, text="", foreground=ERR_FG)
        err.grid(row=3, column=0, columnspan=2)

        def ok(*_):
            try:
                v = float(e.get())
                assert 0 < v < 3000
            except (ValueError, AssertionError):
                err.configure(text="Enter the oxide thickness in nm (for example 90 or 285).")
                return
            mat_ = "graphite" if cb.get() == "graphite" else "hbn"      # (read before the window is destroyed)
            w.destroy()
            done((mat_, v))
        bf = ttk.Frame(w)
        bf.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(bf, text="OK", width=10, command=ok, style="Accent.TButton").pack(side="left", padx=6)
        ttk.Button(bf, text="Cancel", width=10, command=lambda: (w.destroy(), done(None))).pack(side="left", padx=6)
        e.bind("<Return>", ok)

    def notice(self, parent, title, msg):
        """A small message window with OK (Tk thread, does not block)."""
        w = self.tk.Toplevel(parent)
        w.configure(bg=self.bg)
        w.title(title)
        w.transient(parent)
        ttk.Label(w, text=msg, justify="left", wraplength=480).pack(padx=16, pady=12, anchor="w")
        ttk.Button(w, text="OK", width=10, command=w.destroy).pack(pady=(0, 12))
        w.lift()

    def control(self, scanner, title, msg, buttons, once=()):
        """A non-modal window with buttons (the 100x imaging: Skip the rest of this piece, Stop 100x imaging): value() = the
        last one pressed; once: one-shot buttons (h.take), greyed out until h.enable(name, True)."""
        h = _Handle(self)

        def make(done):
            w = self.tk.Toplevel(self.root)
            w.configure(bg=self.bg)
            w.title(title)
            ttk.Label(w, text=msg, justify="left", wraplength=520).pack(padx=14, pady=10, anchor="w")
            bf = ttk.Frame(w)
            bf.pack(pady=(0, 10))
            for b in buttons:
                ttk.Button(bf, text=b, command=lambda b=b: setattr(h, "_v", b)).pack(side="left", padx=6)
            for b in once:
                btn = ttk.Button(bf, text=b, command=lambda b=b: h._once.__setitem__(b, True))
                btn.state(["disabled"])
                btn.pack(side="left", padx=6)
                h._btn[b] = btn
            w.protocol("WM_DELETE_WINDOW", lambda: None)
            h.win = w
            done(True)
        self._ask(make)
        return h

    def show_image(self, scanner, title, path):
        def f():
            from PIL import Image, ImageTk
            w = self.tk.Toplevel(self.root)
            w.configure(bg=self.bg)
            w.title(title)
            im = ImageTk.PhotoImage(Image.open(path))
            lab = tk.Label(w, image=im, borderwidth=0)
            lab.image = im
            lab.pack()
        self._post(f)

    def chip_notes(self, scanner, chips, fields, last, save):
        """A window beside the scan (it does not wait: the scan goes on) for each chip's notes, one chip at a time.
        chips: [dict(name, material, oxide)]; fields: ((key, label), ...), the last one a multi-line box; each chip's
        boxes are filled in from the chip before (the first from `last`, the last chip noted before, if any);
        save(name, {key: text}) is called for each chip saved."""
        def make():
            w = self.tk.Toplevel(self.root)
            w.configure(bg=self.bg)
            w.title("Chip notes (the scan goes on meanwhile)")
            f = ttk.Frame(w, padding=14)
            f.pack(fill="both", expand=True)
            head = ttk.Label(f, text="", font=("Segoe UI", 12, "bold"))
            head.grid(row=0, column=0, columnspan=2, sticky="w")
            src = ttk.Label(f, text="", foreground="#9a96b8")
            src.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
            boxes = {}
            for i, (k, lab) in enumerate(fields):
                ttk.Label(f, text=lab).grid(row=2 + i, column=0, sticky="nw", padx=(0, 10), pady=3)
                if i == len(fields) - 1:
                    b = tk.Text(f, width=56, height=5, wrap="word", font=("Segoe UI", 10))
                else:
                    b = ttk.Entry(f, width=60)
                b.grid(row=2 + i, column=1, sticky="ew", pady=3)
                boxes[k] = b
            st = {"i": 0, "saved": {}}

            def get(k):
                b = boxes[k]
                return b.get("1.0", "end").strip() if isinstance(b, tk.Text) else b.get().strip()

            def put(vals):
                for k, b in boxes.items():
                    v = str((vals or {}).get(k, ""))
                    if isinstance(b, tk.Text):
                        b.delete("1.0", "end"); b.insert("1.0", v)
                    else:
                        b.delete(0, "end"); b.insert(0, v)

            def show():
                i = st["i"]
                c = chips[i]
                head.configure(text=f"Chip {i + 1} of {len(chips)}: {c['name']}  ({c['material']}, {c['oxide']:g} nm oxide)")
                if c["name"] in st["saved"]:
                    put(st["saved"][c["name"]]); src.configure(text="saved; change it and save again if needed")
                elif i > 0 and chips[i - 1]["name"] in st["saved"]:
                    put(st["saved"][chips[i - 1]["name"]]); src.configure(text=f"filled in from {chips[i - 1]['name']}: change what differs")
                elif last:
                    put(last); src.configure(text=f"filled in from {last.get('chip', 'the last chip noted')} (an earlier scan): change what differs")
                else:
                    put({}); src.configure(text="crystal batch, exfoliation details, anything worth knowing later")
                back.state(["!disabled"] if i > 0 else ["disabled"])
                nxt.configure(text="Save" if i == len(chips) - 1 else "Save, next chip")
                boxes[fields[0][0]].focus_set()

            def save_one():
                c = chips[st["i"]]
                vals = {k: get(k) for k, _ in fields}
                st["saved"][c["name"]] = vals
                try:
                    save(c["name"], vals)
                except Exception as e:
                    src.configure(text=f"not saved: {e}")
                    return
                if st["i"] == len(chips) - 1:
                    w.destroy()
                else:
                    st["i"] += 1
                    show()

            def go_back():
                if st["i"] > 0:
                    st["i"] -= 1
                    show()

            def skip():
                if st["i"] == len(chips) - 1:
                    w.destroy()
                else:
                    st["i"] += 1
                    show()
            bf = ttk.Frame(f)
            bf.grid(row=2 + len(fields), column=0, columnspan=2, sticky="e", pady=(10, 0))
            back = ttk.Button(bf, text="Back", width=10, command=go_back)
            back.pack(side="left", padx=4)
            ttk.Button(bf, text="Skip this chip", width=14, command=skip).pack(side="left", padx=4)
            nxt = ttk.Button(bf, text="Save, next chip", width=16, style="Accent.TButton", command=save_one)
            nxt.pack(side="left", padx=4)
            f.columnconfigure(1, weight=1)
            show()
        self._post(make)

    def ask_preview(self, scanner, title, msg, path, buttons=("Keep", "Retake", "Stop")):
        """A picture just taken (fitted to the screen), a message and buttons: the index of the button pressed (closing the
        window = the first button)."""
        def make(done):
            w = self._dialog(title)
            try:
                from PIL import Image, ImageTk
                im = flipped(Image.open(path), remembered_flip(scanner.s))      # (as the live views show it)
                sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
                im.thumbnail((max(400, int(sw * 0.7)), max(300, int(sh * 0.62))))
                ph = ImageTk.PhotoImage(im)
                lab = tk.Label(w, image=ph, borderwidth=0)
                lab.image = ph
                lab.pack(padx=8, pady=8)
            except Exception as e:
                ttk.Label(w, text=f"(the preview could not be shown: {e})").pack(padx=12, pady=8)
            ttk.Label(w, text=msg, justify="left", wraplength=900, font=("Segoe UI", 10)).pack(padx=12, anchor="w")
            bf = ttk.Frame(w)
            bf.pack(pady=10)

            def fin(v):
                w.destroy()
                done(v)
            for i, t in enumerate(buttons):
                ttk.Button(bf, text=t, width=max(10, len(t) + 2), style="Accent.TButton" if i == 0 else "TButton",
                           command=lambda i=i: fin(i)).pack(side="left", padx=6)
            w.protocol("WM_DELETE_WINDOW", lambda: fin(0))
        return self._ask(make)

    def wait_objective(self, scanner, name, want, msg):
        """A window that waits for the nosepiece to reach position `want` and closes by itself once it has been there
        for ~1 s.  True = in place (or "Go on anyway"), False = Cancel."""
        def make(done):
            w = self._dialog(f"Objective: {name}")
            ttk.Label(w, text=msg, justify="left", wraplength=520).pack(padx=16, pady=(12, 6), anchor="w")
            now = ttk.Label(w, text="", font=("Segoe UI", 11, "bold"))
            now.pack(padx=16, pady=6, anchor="w")
            bf = ttk.Frame(w)
            bf.pack(pady=(4, 12))
            state = {"n": 0, "open": True}

            def fin(v):
                if state["open"]:
                    state["open"] = False
                    w.destroy()
                    done(v)
            ttk.Button(bf, text="Go on anyway", command=lambda: fin(True)).pack(side="left", padx=6)
            ttk.Button(bf, text="Cancel", command=lambda: fin(False)).pack(side="left", padx=6)
            w.protocol("WM_DELETE_WINDOW", lambda: fin(False))

            def poll():
                if not state["open"]:
                    return
                try:
                    pos = scanner.nose.position()
                except Exception as e:
                    now.configure(text=f"nosepiece not readable: {e}")
                    w.after(500, poll)
                    return
                now.configure(text=f"now: {scanner.objective_name(pos)}   (wanted: {name}, position {want})", foreground=OK_FG if pos == want else ERR_FG)
                state["n"] = state["n"] + 1 if pos == want else 0
                if state["n"] >= 3:
                    fin(True)
                    return
                w.after(300, poll)
            poll()
        return self._ask(make)

    # ---- live view
    def pause_window(self, scanner, info):
        """The scan is paused (Scanner.pause_now): a live view to take pictures in between (move the stage and focus as
        you like, the chips stay where they are), Save picture (the raw frame as 16-bit TIFF and the view as JPEG, in
        <local_root>/snapshots/<date>/), the exposure; "resume", "recalibrate" (the coarse knob was used: the focus set again
        by hand at Resume, Scanner._pause_recalibrate) or "stop"."""
        s = scanner.s

        def banner(on):
            b = getattr(self, "_banner", None)
            if b is None:
                return
            try:
                if on:
                    if b.get("timer") is None:                    # (not hidden by its own button meanwhile)
                        b["win"].deiconify()
                        b["win"].lift()
                    self._banner_pause("PAUSE THE SCAN", True)
                else:
                    b["win"].withdraw()                            # (it would cover the pause window)
                    self._banner_pause("Paused (see the Scan paused window)", False)
            except Exception:
                pass
        self._post(lambda: (self.pause_btn.configure(text="Paused (see its window)", state="disabled"), banner(False)))

        def make(done):
            w = self._dialog("Scan paused")
            ttk.Label(w, text="The scan is paused", font=("Segoe UI", 13, "bold")).pack(padx=12, pady=(10, 0), anchor="w")
            ttk.Label(w, text=(f"It stopped at: {info.get('phase')} (stage {f1(info['x'])}, {f1(info['y'])} um, Z {f3(info['z'])} um, "
                               f"objective {info.get('objective')}, exposure {f1(info['exposure'])} ms).\n"
                               "Take your pictures: move the stage (joystick), focus (coarse and fine knobs), change the objective "
                               "and the exposure as you like, but do not touch the chips that are being scanned.\n"
                               "When you are done: \"Return to the scan\".  The scan's objective and camera settings are put back, "
                               "and a check window shows the scan's last picture beside the live view at the same place: you "
                               "focus it again (coarse, then fine knob) and confirm focus and position before the scan goes on."),
                      justify="left", wraplength=900).pack(padx=12, anchor="w")
            top = ttk.Frame(w)
            top.pack(padx=8, pady=8)
            side = ttk.Frame(top)
            tools = ttk.Frame(side)
            lv = LiveView(self, top, scanner, free_z=True, tools=tools, controls=False)
            lv.frame.pack(side="left")
            side.pack(side="left", padx=(12, 0), anchor="n")
            if info.get("hundred"):
                lv.set_objective(True)
            # the objective in use (another one may be used for the pictures): read from the nosepiece, else chosen here;
            # the live view's stage steps and flat field follow it
            orow = ttk.Frame(side)
            orow.pack(anchor="w")
            ttk.Label(orow, text="Objective in:").pack(side="left")
            names = list((s.objective_positions or {}).keys()) or ["5x", "10x", "50x", "100x"]
            if info.get("objective") not in names:
                names.append(info.get("objective"))
            objv = tk.StringVar(value=info.get("objective"))
            ocb = ttk.Combobox(orow, textvariable=objv, values=names, width=7, state="readonly" if scanner.nose is None else "disabled")
            ocb.pack(side="left", padx=4)
            ocb.bind("<<ComboboxSelected>>", lambda e: lv.set_mag(objv.get()))
            onote = ttk.Label(side, text=("(read from the nosepiece)" if scanner.nose is not None else
                                          "(set it when you change the objective: the stage steps follow)"), foreground="#9a96b8")
            onote.pack(anchor="w")

            def watch_objective():
                if not w.winfo_exists():
                    return
                if scanner.nose is not None:
                    try:
                        nm = scanner.objective_name(scanner.nose.position())
                        if nm != objv.get() and nm.endswith("x"):
                            objv.set(nm)
                            lv.set_mag(nm)
                    except Exception:
                        pass
                w.after(500, watch_objective)
            msg = ttk.Label(side, text="", wraplength=250, justify="left")
            snap_dir = Path(s.local_root) / "snapshots" / time.strftime("%Y%m%d")

            def save():
                """Save image: the picture as shown (black and white, contrast, scale bar, measurements) at the camera's
                full resolution, where and as what the Save dialog says (JPG, PNG, TIFF; the TIFF also records the stage
                position, Z, exposure, objective and um per pixel)."""
                fr = getattr(lv, "last_frame", None)
                try:
                    if fr is None:                            # (no live frame shown yet: the camera's newest one)
                        c0 = scanner.core
                        if getattr(lv, "streaming", False) and c0.getRemainingImageCount() > 0:
                            fr = scanner.to_array(c0.getLastImage())
                        elif not getattr(lv, "streaming", False):
                            c0.snapImage()
                            fr = scanner.to_array(c0.getImage())
                except Exception:
                    fr = None
                if fr is None:
                    msg.configure(text="No picture yet: wait for the live view.")
                    return
                c = scanner.core
                meta = {"stage_x_um": c.getXPosition(scanner.xy), "stage_y_um": c.getYPosition(scanner.xy),
                        "z_um": scanner.z_read() if scanner.has_z else "", "exposure_ms": c.getExposure(),
                        "objective": objv.get(), "um_per_px": round(lv.um_per_px(1.0), 5), "taken": "during a pause of the scan"}
                img, _ = to_display(fr, lv.dark, lv.sub, s.display_level, False, 1.0, lv.flat, getattr(scanner, "mono_view", False))
                img = lv.finish(flipped(img, lv.flip()), 1.0)
                try:
                    snap_dir.mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
                from tkinter import filedialog
                path = filedialog.asksaveasfilename(parent=w, title="Save image", initialdir=str(snap_dir),
                                                    initialfile=f"pause_{time.strftime('%Y%m%d_%H%M%S')}", defaultextension=".jpg",
                                                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")])
                if not path:
                    return
                path = Path(path)
                ext = path.suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                    path, ext = path.with_suffix(path.suffix + ".jpg"), ".jpg"
                try:
                    if ext in (".tif", ".tiff"):
                        import tifffile
                        tifffile.imwrite(str(path), img, description="\n".join(f"{k}={v}" for k, v in meta.items()))
                    else:
                        from PIL import Image
                        Image.fromarray(img).save(str(path), **({"quality": 95} if ext in (".jpg", ".jpeg") else {}))
                    msg.configure(text=f"Saved {path.name} in {path.parent}")
                    scanner.say(f"    image saved during the pause: {path}")
                except Exception as e:
                    msg.configure(text=f"Not saved: {type(e).__name__}: {e}")
            ttk.Button(side, text="Save image", style="Accent.TButton", command=save).pack(anchor="w", pady=(10, 0))
            msg.pack(anchor="w", pady=(8, 0))
            tools.pack(anchor="w")                                   # (black and white, histogram, exposure: under these)
            bf = ttk.Frame(w)
            bf.pack(pady=10)

            def fin(v):
                if lv.busy():
                    return
                lv.stop()
                w.destroy()
                done(v)
            ttk.Button(bf, text="Return to the scan", width=20, style="Accent.TButton", command=lambda: fin("resume")).pack(side="left", padx=6)
            # (only Return to the scan, Avram 2026-10-27; closing it would leave the scan waiting unseen)
            w.protocol("WM_DELETE_WINDOW", lambda: msg.configure(text="Press \"Return to the scan\"."))
            lv.start()
            watch_objective()
        r = self._ask(make)
        if r == "stop":
            self._post(lambda: (self.pause_btn.configure(text=PAUSE_TEXT, state="normal" if self.job_running else "disabled"), banner(False)))
        else:
            self._post(lambda: self.pause_btn.configure(text="Returning to the scan (see its window)", state="disabled"))
        return "stop" if r == "stop" else "resume"

    def scale_calibrate(self, scanner, info):
        """The scale bar calibration of one objective (scalecal.run_scale_calibration): a live view with the Measure
        tool; the person draws a line between two points of known distance and types its length.  Returns
        dict(um_per_px, length_um, px), "skip" or "finish"."""
        name = info["objective"]

        def make(done):
            w = self._dialog(f"Scale bar calibration: {name}")
            ttk.Label(w, text=f"Scale bar calibration: the {name} objective", font=("Segoe UI", 13, "bold")).pack(padx=12, pady=(10, 0), anchor="w")
            ttk.Label(w, text=("Put a slide or chip with distances you know under the objective (a stage micrometer, marks of "
                               "known spacing) and focus.  Click Measure (beside the picture), click the two ends of a known "
                               "distance (as far apart as the picture allows: more precise), then type its length."),
                      justify="left", wraplength=1000).pack(padx=12, anchor="w")
            top = ttk.Frame(w)
            top.pack(padx=8, pady=8)
            side = ttk.Frame(top)
            lv = LiveView(self, top, scanner, free_z=True, tools=side, controls=False)
            lv.frame.pack(side="left")
            side.pack(side="left", padx=(12, 0), anchor="n")
            lv.set_mag(name)
            ttk.Separator(side).pack(fill="x", pady=8)
            lr = ttk.Frame(side)
            lr.pack(anchor="w")
            ttk.Label(lr, text="Length of the line:").pack(side="left")
            le = ttk.Entry(lr, width=9)
            le.pack(side="left", padx=4)
            ttk.Label(lr, text="um").pack(side="left")
            res = ttk.Label(side, text="", wraplength=280, justify="left")
            res.pack(anchor="w", pady=(6, 0))
            ttk.Label(side, text=(f"now: {info['current']:.4f} um per pixel" + (" (calibrated before)" if info.get("calibrated") else
                                                                                  " (from the stage calibration and the magnification)")),
                      foreground="#9a96b8", wraplength=280, justify="left").pack(anchor="w")
            bf = ttk.Frame(w)
            bf.pack(pady=10)

            def measured():
                """(pixel length of the last line drawn, um per pixel or None) or None."""
                if not lv.meas:
                    return None
                a, b = lv.meas[-1]
                px = float(np.hypot(b[0] - a[0], b[1] - a[1]))
                try:
                    L = float(le.get())
                except ValueError:
                    return (px, None)
                return (px, L / px if px > 0 and L > 0 else None)

            def show():
                if not w.winfo_exists():
                    return
                m = measured()
                if m is None:
                    res.configure(text="No line yet: Measure, then click its two ends.", foreground="")
                elif m[1] is None:
                    res.configure(text=f"Line: {m[0]:.1f} camera pixels.  Type its length in um.", foreground="")
                else:
                    res.configure(text=f"Line: {m[0]:.1f} camera pixels -> {m[1]:.4f} um per pixel "
                                       f"({100 * (m[1] / info['current'] - 1):+.2f} % against now)", foreground="")
                w.after(300, show)

            def fin(v):
                if lv.busy():
                    return
                if v == "save":
                    m = measured()
                    if m is None or m[1] is None:
                        res.configure(text="Draw a line (Measure, two clicks) and type its length in um first.", foreground="#e06c75")
                        return
                    v = dict(um_per_px=m[1], length_um=float(le.get()), px=m[0])
                lv.stop()
                w.destroy()
                done(v)
            nxt = f"Save, next: {info['next']}" if info.get("next") else "Save and finish"
            ttk.Button(bf, text=nxt, width=22, style="Accent.TButton", command=lambda: fin("save")).pack(side="left", padx=6)
            ttk.Button(bf, text="Skip this objective", width=18, command=lambda: fin("skip")).pack(side="left", padx=6)
            ttk.Button(bf, text="Finish", width=10, command=lambda: fin("finish")).pack(side="left", padx=6)
            w.protocol("WM_DELETE_WINDOW", lambda: fin("finish"))
            lv.start()
            show()
        r = self._ask(make)
        return r if isinstance(r, dict) or r in ("skip", "finish") else "finish"

    def pause_check(self, scanner, info):
        """Returning to the scan after a pause (Scanner.pause_now): the scan's last picture (info["ref"]: frame, what; the
        stage is at its place, the focus drive at the reading that was in focus there) beside the live view, a crosshair in
        the middle of both.  The person focuses (coarse, then fine knob) and ticks "Focus looks good" and "Position looks
        good"; "Continue the scan" only then.  "The position is wrong": a message to whoever runs the scan, the scan stays
        paused.  Returns "ok" or "stop"."""
        s = scanner.s
        ref = info.get("ref")
        who = (info.get("contact") or {}).get("name") or "the person running the scan"

        def make(done):
            w = self._dialog("Return to the scan: check focus and position")
            # (the two pictures side by side, as large as the screen allows: their scale against the camera's)
            sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
            k = max(0.3, min(1.0, (sw - 90) / (2.0 * scanner.W + 40), (sh - 360) / float(scanner.H)))
            ttk.Label(w, text="Check before the scan goes on", font=("Segoe UI", 13, "bold")).pack(padx=12, pady=(10, 0), anchor="w")
            ttk.Label(w, text=(f"The {info.get('objective')} objective is in.  Focus the live view (left; coarse knob, then fine "
                               "knob) until it is as sharp as the picture on the right, check both show the same place under "
                               "the crosses, then tick both boxes below."),
                      justify="left", wraplength=max(600, sw - 120)).pack(padx=12, anchor="w")
            top = ttk.Frame(w)
            top.pack(padx=8, pady=8)
            lf, rf = ttk.Frame(top), ttk.Frame(top)            # (the live view left, the picture to compare with right)
            lf.pack(side="left", padx=(0, 16), anchor="n")
            rf.pack(side="left", anchor="n")
            ttk.Label(lf, text="Live view now", font=("Segoe UI", 13, "bold")).pack()
            lv = LiveView(self, lf, scanner, free_z=True, cross=True, controls=False, scale=k)
            lv.frame.pack()
            if info.get("hundred"):
                lv.set_objective(True)
            ttk.Label(rf, text=("Picture taken when the scan paused" if ref is not None and ref.get("at_pause") else
                                "The scan's last picture" + (f" ({ref['what']})" if ref is not None else "")),
                      font=("Segoe UI", 13, "bold")).pack()
            pic = tk.Label(rf, borderwidth=0)
            pic.pack()

            def show_ref():
                """The picture to compare with, at the live view's scale and flip, a cross in its middle."""
                if ref is None:
                    pic.configure(text="(no picture to compare with: check the place and focus by eye)", fg="#9a96b8",
                                  bg=self.bg, width=60, height=10)
                    return
                try:
                    img, _ = to_display(ref["frame"], lv.dark, lv.sub, s.display_level, False, k, lv.flat,
                                        getattr(scanner, "mono_view", False))
                    img = LiveView.draw_cross(flipped(img, lv.flip()))
                    h_, w_ = img.shape[:2]
                    ph = self.tk.PhotoImage(master=rf, data=f"P6 {w_} {h_} 255 ".encode() + np.ascontiguousarray(img).tobytes(),
                                            format="PPM")
                    pic.configure(image=ph)
                    pic.image = ph
                except Exception as e:
                    pic.configure(text=f"(picture not shown: {e})")
            show_ref()
            lv.on_flip = show_ref                              # (flipping the live view flips the picture too)
            # the two ticks, apart from the live view's own settings: under a white line, centred, larger
            tk.Frame(w, height=2, bg="white").pack(fill="x", padx=12, pady=(6, 10))
            try:
                ttk.Style().configure("Check.TCheckbutton", font=("Segoe UI", 15, "bold"))
            except Exception:
                pass
            cf = ttk.Frame(w)
            cf.pack(anchor="center")
            fok, pok = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
            ttk.Checkbutton(cf, text="Focus looks good (as sharp as the picture)", variable=fok,
                            style="Check.TCheckbutton").pack(anchor="w")
            ttk.Checkbutton(cf, text="Position looks good (the same place under both crosses)", variable=pok,
                            style="Check.TCheckbutton").pack(anchor="w", pady=(6, 0))
            msg = ttk.Label(w, text="", wraplength=1000, justify="center")
            msg.pack(padx=12, anchor="center", pady=(6, 0))
            bf = ttk.Frame(w)
            bf.pack(pady=10)

            def fin(v):
                if lv.busy():
                    return
                lv.stop()
                w.destroy()
                done(v)
            go = ttk.Button(bf, text="Continue the scan", width=20, style="Accent.TButton", command=lambda: fin("ok"))
            go.pack(side="left", padx=6)
            go.state(["disabled"])

            def ticks(*_):
                go.state(["!disabled"] if fok.get() and pok.get() else ["disabled"])
            fok.trace_add("write", ticks)
            pok.trace_add("write", ticks)

            def wrong():
                scanner._pause_notice("the position after a pause looks wrong",
                                      f"{s.name}: someone paused the scan to take pictures; when returning to it, the live view did "
                                      f"not show the place of the scan's last picture ({ref['what'] if ref is not None else 'where it paused'}). "
                                      "The scan stays paused until it is checked at the microscope (the chips may have moved).")
                scanner.say(f"  the position after the pause was reported wrong: {who} told; the scan stays paused")
                msg.configure(text=f"{who} has been told (Slack / email).  The scan stays paused: wait for them, or stop the scan.  "
                                   "If the position is right after all, tick both and continue.", foreground="#e06c75")
            ttk.Button(bf, text=f"The position is wrong: tell {who}", command=wrong).pack(side="left", padx=6)
            ttk.Button(bf, text="Stop the scan", width=14, command=lambda: fin("stop")).pack(side="left", padx=6)
            w.protocol("WM_DELETE_WINDOW", lambda: msg.configure(text="Tick both and press \"Continue the scan\", or \"Stop the scan\"."))
            lv.start()
        r = self._ask(make)
        self._post(lambda: (self.pause_btn.configure(text=PAUSE_TEXT, state="normal" if self.job_running else "disabled"),
                            self._banner_back(r != "stop")))
        return "stop" if r == "stop" else "ok"

    def _banner_back(self, on):
        """After a pause: the Scan in progress window again (unless hidden by its own button), its Pause button."""
        b = getattr(self, "_banner", None)
        if b is None:
            return
        try:
            if on and b.get("timer") is None:
                b["win"].deiconify()
                b["win"].lift()
            self._banner_pause("PAUSE THE SCAN", on)
        except Exception:
            pass

    def ask_focus(self, scanner, title, what, timeout=None, image=None, button="Use this focus", cross=False):
        """Live view to focus by hand.  image: a picture of where the flake should be (the 10x crop the analysis wrote,
        the 100x field boxed), shown beside the live view so the user can check the stage is on the right flake.
        cross: a cross at the centre of the live view (centring a flake)."""
        def make(done):
            w = self._dialog(title)
            top = ttk.Frame(w)
            top.pack(padx=8, pady=8)
            lv = LiveView(self, top, scanner, cross=cross)
            lv.frame.pack(side="left")
            if image and Path(image).is_file():
                try:
                    from PIL import Image, ImageTk
                    im = flipped(Image.open(image), lv.flip())          # (as the live view beside it)
                    im.thumbnail((360, 360))
                    ph = ImageTk.PhotoImage(im)
                    side = ttk.Frame(top)
                    side.pack(side="left", padx=(12, 0), anchor="n")
                    ttk.Label(side, text="Where the flake should be\n(10x scan; box = 100x field)", justify="left").pack(anchor="w", pady=(0, 6))
                    lab = tk.Label(side, image=ph, borderwidth=0)
                    lab.image = ph
                    lab.pack(anchor="w")
                except Exception:
                    pass
            ttk.Label(w, text=what + ("" if cross else "\n\nFocus with the Z buttons (or the focus control), then:"), justify="left",
                          font=("Segoe UI", 10), wraplength=900).pack(padx=12, anchor="w")
            cd = ttk.Label(w, text="")
            cd.pack(padx=12, anchor="w")
            bf = ttk.Frame(w)
            bf.pack(pady=10)

            def fin(v):
                if lv.busy():
                    return
                lv.stop()
                w.destroy()
                done(v)
            ttk.Button(bf, text=button, width=max(16, len(button) + 2), style="Accent.TButton", command=lambda: fin(0)).pack(side="left", padx=6)
            ttk.Button(bf, text="Skip", width=10, command=lambda: fin(1)).pack(side="left", padx=6)
            ttk.Button(bf, text="Stop", width=10, command=lambda: fin(2)).pack(side="left", padx=6)
            w.protocol("WM_DELETE_WINDOW", lambda: fin(1))
            if timeout:
                t_end = time.time() + timeout

                def tick():
                    if not w.winfo_exists():
                        return
                    left = t_end - time.time()
                    if left <= 0:
                        fin(3)
                        return
                    cd.configure(text=f"Without an answer within {int(left)} s the predicted focus is used and the scan goes on.")
                    w.after(500, tick)
                tick()
            lv.start()
        return self._ask(make)

    def ask_mounted(self, scanner, plan):
        """Mounted 1 cm chips: for each block (plan: its number of chips), each chip's centre (material and oxide asked
        at the block's first chip), then 3 corners of the block (chip corners easy to find again on the AFM: its
        reference points).  Returns dict(pieces=[(x, y, z, material, oxide, block), ...], corners={block: [(x, y,
        z), ...]}) or None."""
        def make(done):
            w = self._dialog("Mounted chips: centres and reference corners")
            lv = LiveView(self, w, scanner, free_z=True, cross=True)
            lv.frame.grid(row=0, column=0, rowspan=3, padx=8, pady=8)
            steps = [("chip", m, i) for m in range(1, len(plan) + 1) for i in range(1, plan[m - 1] + 1)]
            steps = [st for m in range(1, len(plan) + 1) for st in [x for x in steps if x[1] == m] + [("corner", m, j) for j in (1, 2, 3)]]
            got, mat = [], {}
            state = {"mat": scanner.s.material, "ox": scanner.s.oxide_nm}
            info = ttk.Label(w, text="", justify="left", anchor="nw", width=46, wraplength=400)
            info.grid(row=0, column=1, padx=8, pady=8, sticky="nw")
            bf = ttk.Frame(w)
            bf.grid(row=1, column=1, padx=8, sticky="nw")
            add = ttk.Button(bf, text="", width=30, style="Accent.TButton")
            add.pack(anchor="w", pady=3)
            undo = ttk.Button(bf, text="Undo the last one", width=30)
            undo.pack(anchor="w", pady=3)

            def cancel():
                if not lv.busy():
                    lv.stop(); w.destroy(); done(None)
            ttk.Button(bf, text="Cancel", width=30, command=cancel).pack(anchor="w", pady=3)

            def refresh():
                if len(got) >= len(steps):
                    lv.stop(); w.destroy()
                    done(dict(pieces=[(g[3][0], g[3][1], g[3][2], g[4][0], g[4][1], g[1]) for g in got if g[0] == "chip"],
                              corners={m: [g[3] for g in got if g[0] == "corner" and g[1] == m] for m in range(1, len(plan) + 1)}))
                    return
                kind, m, i = steps[len(got)]
                if kind == "chip":
                    txt = (f"Block {m} of {len(plan)}, chip {i} of {plan[m - 1]}: move to the approximate centre of the chip "
                           "(10x; within ~1 mm is enough), focus roughly, then press the button.")
                    add.configure(text=f"Add chip {i} of block {m} here")
                else:
                    txt = (f"Block {m}: reference corner {i} of 3. Move to a chip corner of this block that you will "
                           "find again on the AFM (use corners far apart, e.g. on the outer chips), put it in the cross at the "
                           "centre of the view, then press the button.")
                    add.configure(text=f"Add corner {i} of block {m} here")
                done_txt = "\n".join(f"  {'chip' if g[0] == 'chip' else 'corner'} {g[2]} of block {g[1]}: {f1(g[3][0])}, {f1(g[3][1])} um"
                                     for g in got[-8:])
                info.configure(text=txt + (f"\n\nSo far ({len(got)} of {len(steps)}):\n" + done_txt if got else ""))
                undo.configure(state="normal" if got else "disabled")

            def add_here():
                if lv.busy():
                    return
                c = scanner.core
                x, y = c.getXPosition(scanner.xy), c.getYPosition(scanner.xy)
                z = scanner.z_read() if scanner.has_z else 0.0
                kind, m, i = steps[len(got)]
                if kind == "chip" and any(g[0] == "chip" and np.hypot(g[3][0] - x, g[3][1] - y) < 3000 for g in got):
                    from tkinter import messagebox
                    messagebox.showinfo("Mounted chips", "That is within 3 mm of a chip already added: move to the next chip first.", parent=w)
                    return
                if kind == "chip" and m not in mat:     # (the block's material and oxide: asked at its first chip)
                    lv.pause(True)

                    def ans(r):
                        lv.pause(False)
                        if r is None:
                            return
                        state["mat"], state["ox"] = r
                        mat[m] = r
                        got.append((kind, m, i, (x, y, z), r))
                        scanner.say(f"  block {m} chip {i}: centre {f1(x)}, {f1(y)} um, Z {f1(z)}, {r[0]} on {f1(r[1])} nm oxide")
                        refresh()
                    self._piece_info(f"Block {m}: material and oxide", f"The chips of block {m}", state["mat"], state["ox"], ans)
                    return
                got.append((kind, m, i, (x, y, z), mat.get(m)))
                scanner.say(f"  block {m} {kind} {i}: {f1(x)}, {f1(y)} um, Z {f1(z)}")
                refresh()

            def undo_last():
                if got and not lv.busy():
                    g = got.pop()
                    if g[0] == "chip" and not any(x[0] == "chip" and x[1] == g[1] for x in got):
                        mat.pop(g[1], None)
                    refresh()
            add.configure(command=add_here)
            undo.configure(command=undo_last)
            w.protocol("WM_DELETE_WINDOW", cancel)
            refresh()
            lv.start()
        return self._ask(make)

    def ask_pieces(self, scanner):
        """Live viewer for adding the pieces: move to each piece's centre (stage control or the arrow buttons), focus
        roughly, "Add piece here" (+ material and oxide).  Returns [(x, y, z, material, oxide), ...] or None."""
        def make(done):
            w = self._dialog("Multi-piece scan: pieces")
            lv = LiveView(self, w, scanner, free_z=True)
            lv.frame.grid(row=0, column=0, rowspan=3, padx=8, pady=8)
            pieces, state = [], {"mat": scanner.s.material, "ox": scanner.s.oxide_nm}
            info = ttk.Label(w, text="", justify="left", anchor="nw", width=44, wraplength=380)
            info.grid(row=0, column=1, padx=8, pady=8, sticky="nw")
            bf = ttk.Frame(w)
            bf.grid(row=1, column=1, padx=8, sticky="nw")
            add = ttk.Button(bf, text="Add piece 1 here", width=24, style="Accent.TButton")
            add.pack(anchor="w", pady=3)
            fin_b = ttk.Button(bf, text="Done", width=24, state="disabled")
            fin_b.pack(anchor="w", pady=3)
            def cancel():
                if not lv.busy():
                    lv.stop(); w.destroy(); done(None)
            ttk.Button(bf, text="Cancel", width=24, command=cancel).pack(anchor="w", pady=3)

            def refresh():
                txt = ("Move the stage to the approximate centre of piece {} (10x) and focus roughly, then press \"Add piece {} here\"."
                       .format(len(pieces) + 1, len(pieces) + 1))
                if pieces:
                    txt += "\n\nPieces so far:\n" + "\n".join(f"  piece {i + 1}: {f1(p[0])}, {f1(p[1])} um, Z {f1(p[2])}, {p[3]} on {p[4]:g} nm"
                                                              for i, p in enumerate(pieces))
                info.configure(text=txt)
                add.configure(text=f"Add piece {len(pieces) + 1} here")
                fin_b.configure(text=f"Done ({len(pieces)} piece{'s' if len(pieces) != 1 else ''})", state="normal" if pieces else "disabled")

            def add_piece():
                if lv.busy():
                    return
                c = scanner.core
                x, y = c.getXPosition(scanner.xy), c.getYPosition(scanner.xy)
                z = scanner.z_read() if scanner.has_z else 0.0
                if any(np.hypot(p[0] - x, p[1] - y) < 1000 for p in pieces):
                    from tkinter import messagebox
                    messagebox.showinfo("Pieces", "That is within 1 mm of a piece already added: move to the next piece first.", parent=w)
                    return
                lv.pause(True)

                def got(r):
                    lv.pause(False)
                    if r is None:
                        return
                    state["mat"], state["ox"] = r
                    pieces.append((x, y, z, r[0], r[1]))
                    scanner.say(f"  piece {len(pieces)}: centre {f1(x)}, {f1(y)} um, Z {f1(z)}, {r[0]} on {f1(r[1])} nm oxide")
                    refresh()
                self._piece_info(f"Piece {len(pieces) + 1}: material and oxide", f"Piece {len(pieces) + 1} at {f1(x)}, {f1(y)} um",
                                 state["mat"], state["ox"], got)
            add.configure(command=add_piece)
            def finish():
                if not lv.busy():
                    lv.stop(); w.destroy(); done(pieces)
            fin_b.configure(command=finish)
            w.protocol("WM_DELETE_WINDOW", cancel)
            refresh()
            lv.start()
        return self._ask(make)

    def ask_polygons(self, scanner):
        """Scan areas drawn by hand (chip_type "polygon"): a live viewer with a cross; for each area, put the cross on
        each of its corners in turn and "Add point" (as many as wanted, in order round the area: the last joins the
        first), then "Finish polygon" (+ material and oxide); then the next area's first point, or "Done".  A map beside
        the live view shows the areas so far and the field of view.  Returns [dict(points=[(x, y, z), ...], material,
        oxide), ...] (one per area) or None."""
        from tkinter import messagebox

        def make(done):
            w = self._dialog("Scan areas drawn by hand: polygons")
            lv = LiveView(self, w, scanner, free_z=True, cross=True)
            lv.frame.grid(row=0, column=0, rowspan=3, padx=8, pady=8)
            polys, cur = [], []
            state = {"mat": scanner.s.material, "ox": scanner.s.oxide_nm, "open": True}
            side = ttk.Frame(w)
            side.grid(row=0, column=1, padx=8, pady=8, sticky="nw")
            info = ttk.Label(side, text="", justify="left", anchor="nw", width=50, wraplength=420)
            info.pack(anchor="w")
            cw, ch = 420, 300
            cv = tk.Canvas(side, width=cw, height=ch, bg=self.bg, highlightthickness=1, highlightbackground="#55536a")
            cv.pack(anchor="w", pady=(10, 2))
            ttk.Label(side, text="Map, turned as the live view: green = polygons finished, yellow = the one being drawn, pink = "
                      "the field of view now", foreground="#9a96b8", wraplength=420, justify="left").pack(anchor="w")
            # how the 10x focus points are focused (polygon_focus_per_cm2 of each polygon's area, spread over it)
            fm = tk.StringVar(value="manual" if (scanner.s.polygon_focus_mode or scanner.s.focus_mode) == "manual" else "auto")
            ff_ = ttk.Frame(side)
            ff_.pack(anchor="w", pady=(10, 0))
            ttk.Label(ff_, text=f"10x focus points ({scanner.s.polygon_focus_per_cm2:g} per cm², spread over each polygon; "
                      "the 100x uses autofocus):").pack(anchor="w")
            ttk.Radiobutton(ff_, text="Autofocus", value="auto", variable=fm).pack(anchor="w", padx=(16, 0))
            ttk.Radiobutton(ff_, text="Manual: I focus each point (the stage goes to each one)", value="manual",
                            variable=fm).pack(anchor="w", padx=(16, 0))
            ex = lv.exposure_box(side, mode="auto", pady=(10, 0))      # (this view only: the scan's exposure comes back after)
            ttk.Label(ex, text="Only for this view: the scan keeps its own exposure.", foreground="#9a96b8").pack(anchor="w")
            bf = ttk.Frame(w)
            bf.grid(row=1, column=1, padx=8, sticky="nw")
            add = ttk.Button(bf, text="", width=34, style="Accent.TButton")
            add.pack(anchor="w", pady=3)
            fin_b = ttk.Button(bf, text="", width=34)
            fin_b.pack(anchor="w", pady=3)
            undo = ttk.Button(bf, text="Undo the last point", width=34)
            undo.pack(anchor="w", pady=3)
            done_b = ttk.Button(bf, text="", width=34)
            done_b.pack(anchor="w", pady=(12, 3))
            CUR, VIEW = "#e5c07b", "#ff50c8"

            def here():
                try:
                    c = scanner.core
                    return c.getXPosition(scanner.xy), c.getYPosition(scanner.xy)
                except Exception:
                    return None

            def draw():
                cv.delete("all")
                pos = here()
                pts = [q for a in polys for q in a["points"]] + cur + ([pos] if pos else [])
                if not pts:
                    return
                # the map as the live view shows the sample: stage um -> the camera's picture directions (M: pixel ->
                # stage), flipped as the live view, in um
                M = np.asarray(scanner.M, float)
                fh, fv = lv.flip()
                T = np.diag([-1.0 if fh else 1.0, -1.0 if fv else 1.0]) @ np.linalg.inv(M) * np.sqrt(abs(np.linalg.det(M)))
                HX, HY = np.sqrt(abs(np.linalg.det(M))) * scanner.W / 2, np.sqrt(abs(np.linalg.det(M))) * scanner.H / 2
                Q = np.asarray([q[:2] for q in pts], float) @ T.T
                x0, x1, y0, y1 = Q[:, 0].min() - 2 * HX, Q[:, 0].max() + 2 * HX, Q[:, 1].min() - 2 * HY, Q[:, 1].max() + 2 * HY
                k = min((cw - 20) / (x1 - x0), (ch - 20) / (y1 - y0))
                ox, oy = (cw - k * (x1 - x0)) / 2, (ch - k * (y1 - y0)) / 2

                def P(x, y):
                    u, v = T @ (x, y)
                    return ox + (u - x0) * k, oy + (v - y0) * k
                for i, a in enumerate(polys):
                    cv.create_polygon(*[v for q in a["points"] for v in P(q[0], q[1])], outline=OK_FG, fill="", width=2)
                    cv.create_text(*P(*polygon_interior_point([q[:2] for q in a["points"]])), text=str(i + 1), fill=OK_FG,
                                   font=("Segoe UI", 12, "bold"))
                if len(cur) >= 2:
                    cv.create_line(*[v for q in cur for v in P(q[0], q[1])], fill=CUR, width=2)
                if len(cur) >= 3:
                    cv.create_line(*P(cur[-1][0], cur[-1][1]), *P(cur[0][0], cur[0][1]), fill=CUR, dash=(4, 3))
                for j, q in enumerate(cur):
                    u, v = P(q[0], q[1])
                    cv.create_oval(u - 3, v - 3, u + 3, v + 3, fill=CUR, outline="")
                    cv.create_text(u + 8, v - 8, text=str(j + 1), fill=CUR)
                if pos:
                    if cur:
                        cv.create_line(*P(cur[-1][0], cur[-1][1]), *P(*pos), fill=VIEW, dash=(2, 3))
                    u, v = P(*pos)                      # (the field: a rectangle in the picture's directions)
                    cv.create_rectangle(u - HX * k, v - HY * k, u + HX * k, v + HY * k, outline=VIEW)

            def poll():
                if not state["open"] or not w.winfo_exists():
                    return
                if not lv.busy():
                    draw()
                w.after(700, poll)

            def refresh():
                n = len(polys) + 1
                if not cur:
                    if not polys:
                        txt = ("Polygon 1: move the stage so that the cross is on the first corner of the area to scan (10x), "
                               "focus roughly, then press \"Add point 1\".")
                    else:
                        txt = (f"{len(polys)} polygon{'s' if len(polys) > 1 else ''} ready.  Press \"Start the scan\" to scan "
                               f"the inside of {'it' if len(polys) == 1 else 'them'} at 10x.\n\nTo scan another area too: move "
                               f"the cross to the first corner of polygon {n} and press \"Add point 1 of polygon {n}\".")
                    add.configure(text=f"Add point 1 of polygon {n}" if polys else "Add point 1")
                else:
                    txt = (f"Polygon {n}: {len(cur)} point{'s' if len(cur) > 1 else ''} so far.  Move the cross to the next "
                           f"corner and press \"Add point {len(cur) + 1}\" (go round the area in order; the last point joins "
                           f"the first by itself).  When all its corners are in (at least 3), press \"Finish polygon {n}\".")
                    add.configure(text=f"Add point {len(cur) + 1}")
                lines = [f"  polygon {i + 1}: {len(a['points'])} points, {polygon_area_um2(a['points']) / 1e6:.2f} mm2, "
                         f"{a['material']} on {a['oxide']:g} nm" for i, a in enumerate(polys)]
                lines += [f"  polygon {n} point {j + 1}: {f1(q[0])}, {f1(q[1])} um" for j, q in enumerate(cur)][-8:]
                info.configure(text=txt + ("\n\nSo far:\n" + "\n".join(lines) if lines else ""))
                add.configure(style="TButton" if polys and not cur else "Accent.TButton")
                fin_b.configure(text=f"Finish polygon {n}", state="normal" if len(cur) >= 3 else "disabled")
                undo.configure(state="normal" if cur or polys else "disabled")
                done_b.configure(text=f"Start the scan ({len(polys)} polygon{'s' if len(polys) != 1 else ''})",
                                 state="normal" if polys and not cur else "disabled",
                                 style="Accent.TButton" if polys and not cur else "TButton")
                draw()

            def add_point():
                if lv.busy():
                    return
                c = scanner.core
                x, y = c.getXPosition(scanner.xy), c.getYPosition(scanner.xy)
                z = scanner.z_read() if scanner.has_z else 0.0
                if cur and np.hypot(x - cur[-1][0], y - cur[-1][1]) < 5.0:
                    messagebox.showinfo("Polygons", "That is where the last point is: move the cross to the next corner first.", parent=w)
                    return
                cur.append((x, y, z))
                scanner.say(f"  polygon {len(polys) + 1} point {len(cur)}: {f1(x)}, {f1(y)} um, Z {f1(z)}")
                refresh()

            def finish():
                if lv.busy() or len(cur) < 3:
                    return
                n, pts = len(polys) + 1, list(cur)
                if len(pts) > 3 and np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < 20.0:
                    pts.pop()                           # (the first point added again to close it: it closes by itself)
                area = polygon_area_um2([q[:2] for q in pts])
                if area < 100.0:
                    messagebox.showinfo("Polygons", f"The points of polygon {n} are (nearly) on one line: add a corner off that "
                                        "line first.", parent=w)
                    return
                x = polygon_crossing([q[:2] for q in pts])
                if x is not None:
                    e = lambda i: f"{i + 1}-{(i + 1) % len(pts) + 1}"                # noqa: E731
                    messagebox.showinfo("Polygons", f"Polygon {n}: the edges from point {e(x[0])} and {e(x[1])} cross.  The "
                                        "points must go round the area in order: undo the points back to before the crossing "
                                        "and add them again.", parent=w)
                    return
                lv.pause(True)

                def ans(r):
                    lv.pause(False)
                    if r is None:
                        return
                    state["mat"], state["ox"] = r
                    polys.append(dict(points=pts, material=r[0], oxide=r[1]))
                    cur.clear()
                    scanner.say(f"  polygon {n}: {len(pts)} points, {area / 1e6:.2f} mm2, {r[0]} on {f1(r[1])} nm oxide")
                    refresh()
                    next_step(n, len(pts), area)
                self._piece_info(f"Polygon {n}: material and oxide", f"Polygon {n}: {len(pts)} points, {area / 1e6:.2f} mm2",
                                 state["mat"], state["ox"], ans)

            def next_step(n, npts, area):
                """After a polygon is finished: the scan now, or another polygon first."""
                q = self._dialog(f"Polygon {n} finished")
                q.transient(w)
                ttk.Label(q, text=f"Polygon {n} is finished ({npts} points, {area / 1e6:.2f} mm²).\n\nStart the 10x scan of "
                          f"the inside of {'it' if n == 1 else f'the {n} polygons'} now, or add another polygon first?",
                          justify="left", wraplength=420).pack(padx=16, pady=(14, 10), anchor="w")
                qb = ttk.Frame(q)
                qb.pack(pady=(0, 14))

                def pick(scan):
                    q.destroy()
                    if scan:
                        finish_all()
                ttk.Button(qb, text=f"Start the scan ({n} polygon{'s' if n > 1 else ''})", style="Accent.TButton",
                           command=lambda: pick(True)).pack(side="left", padx=6)
                ttk.Button(qb, text="Add another polygon", command=lambda: pick(False)).pack(side="left", padx=6)
                q.protocol("WM_DELETE_WINDOW", lambda: pick(False))
                q.grab_set()

            def undo_last():
                if lv.busy():
                    return
                if cur:
                    cur.pop()
                elif polys:                             # (the last polygon open again: its material is asked again)
                    cur.extend(polys.pop()["points"])
                refresh()

            def close(r):
                state["open"] = False
                lv.stop(); w.destroy(); done(r)

            def finish_all():
                if not lv.busy() and polys and not cur:
                    close([dict(a, focus_mode=fm.get()) for a in polys])

            def cancel():
                if lv.busy():
                    return
                if (polys or cur) and not messagebox.askyesno("Polygons", "Close and forget the points marked so far?", parent=w):
                    return
                close(None)
            ttk.Button(bf, text="Cancel", width=34, command=cancel).pack(anchor="w", pady=3)
            add.configure(command=add_point)
            fin_b.configure(command=finish)
            undo.configure(command=undo_last)
            done_b.configure(command=finish_all)
            w.protocol("WM_DELETE_WINDOW", cancel)
            lv.on_flip = draw                           # (the map turns with the live view)
            refresh()
            lv.start()
            poll()
        return self._ask(make)

    # ---- running the scan
    def run(self, make):
        """Run scans in a worker thread while Tk runs here; returns when the window is closed.  make: a Scanner (one
        job), or a function returning a new Scanner: then after each job (once its analysis has finished) the setup of
        the next job starts; cancelling that setup ends the program."""
        factory = None if hasattr(make, "run") else make

        def work():
            uploads, sc, first = [], None, True
            self.bg_uploads = uploads
            while True:
                sc = make if factory is None else factory()
                if not first:
                    self.log("")
                    self.log("=" * 100)
                    self.log("next job")
                first = False
                ok, t_job = False, time.time()
                self.job_running = True
                self.pause_requested = False
                self._post(lambda: (self.start_btn.configure(text="Start screen", state="normal"),
                                    self.stop_btn.configure(text="Stop the scan", state="normal", command=self._stop),
                                    self.pause_btn.configure(text=PAUSE_TEXT, state="normal")))
                try:
                    sc.run()
                    self.progress(sc, "finished")
                    ok = True
                except Stop as e:
                    self.progress(sc, f"stopped: {e}")
                    sc.say(f"stopped: {e}")
                except Exception as e:
                    import traceback
                    sc.say("ERROR: " + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
                    self.progress(sc, f"failed: {type(e).__name__}: {e}")
                finally:
                    if sc.manifest and sc.stop_all:
                        try:
                            sc.write_manifest(sc.manifest, sc.pieces, "stopped")
                        except Exception:
                            pass
                    if getattr(sc, "upload_thread", None) is not None:
                        uploads.append(sc.upload_thread)
                    try:                                    # (Slack / email: flakeid/scope/notify.py)
                        from .notify import job_end
                        job_end(sc, ok, sc.stop_all or self.stop_requested, (time.time() - t_job) / 60.0)
                    except Exception:
                        pass
                    self.banner_off()
                    self.pause_requested = False
                    self._post(lambda: self.pause_btn.configure(text=PAUSE_TEXT, state="disabled"))
                    try:
                        sc.close_nosepiece()            # (every job, also flat field / capture / stopped: the next opens it again)
                    except Exception:
                        pass
                self.job_running = False
                if factory is None:
                    break
                if self.back_to_start:                     # "Start screen" during the job
                    sc.say("back to the start screen")
                elif not ok or sc.stop_all:
                    # stopped, cancelled or failed: wait for "Start screen" (another job) or "Close"
                    self.idle.clear()
                    self.idle_choice = None
                    self._post(lambda: (self.start_btn.configure(text="Start screen", state="normal"),
                                        self.stop_btn.configure(text="Close", state="normal", command=self._close)))
                    self.progress(sc, getattr(self, "_last_status", "") + "   - \"Start screen\" for another job, or \"Close\"")
                    self.idle.wait()
                    if self.idle_choice != "start":
                        break
                # back to the setup for the next job
                self.stop_requested = self.back_to_start = False
                self.progress(sc, "set up the next job (cancel to quit)")
            if any(t.is_alive() for t in uploads):
                self._post(lambda: (self.start_btn.configure(state="disabled"), self.stop_btn.configure(text="Close", state="disabled")))
                self.progress(sc, "copying the results to the share ...")
                for t in uploads:
                    t.join()
                self.progress(sc, "results copied")
            if self.idle_choice == "close":
                self._post(self.root.destroy)
                return
            self._post(lambda: (self.start_btn.configure(state="disabled"),
                                self.stop_btn.configure(text="Close", state="normal", command=self.root.destroy)))
        threading.Thread(target=work, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        """The window's close button: with a run still finishing in the background (its export, the copy to the share)
        the program would go on invisibly, holding the microscope (a new start would then wait for ever): ask."""
        busy = [t for t in getattr(self, "bg_uploads", []) if t.is_alive()]
        if not busy:
            self.root.destroy()
            return
        names = ", ".join(t.name.replace("finish ", "") for t in busy)

        def ask():
            a = self.confirm("Close the scan program",
                             f"{names}: the 100x export and the copy to the share are still running (the line under the status shows "
                             "how far).\n\n\"Close when done\": the window stays and closes itself when they are done (the microscope "
                             "stays reserved until then).\n\"Close now\": the results stay on this PC and are not copied to the share "
                             "(the export goes on by itself; copy them later).",
                             ("Close when done", "Close now", "Cancel"))
            if a == 0:
                self.progress(None, "closing when the background export and copy are done ...")
                for t in busy:                          # (then the program ends: the microscope is free again)
                    t.join()
                self._post(self.root.destroy)
            elif a == 1:
                import os
                os._exit(0)
        threading.Thread(target=ask, daemon=True).start()


class LiveView:
    """Colour live view with focus and stage controls, inside a Tk window (runs in the Tk thread)."""

    def __init__(self, ui, parent, scanner, free_z=False, cross=False, tools=False, controls=True, scale=None):
        """tools: the picture tools (black and white, scale bar, measuring, histogram with black / white points, auto /
        manual exposure; the pause window, 2026-10-27): True (under the picture) or a frame to put them in (a column
        beside it).  controls=False: no stage / focus buttons (nor arrow keys): the pause window, where the stage and the
        focus are moved by hand.  exposure_box(): the auto / manual exposure box alone (the polygon window).  scale: the
        picture's size against the camera's (default live_scale; the pause check sizes its two pictures to the screen)."""
        tk = ui.tk
        self.ui, self.sc, self.free_z, self.cross = ui, scanner, free_z, cross
        self.frame = ttk.Frame(parent)
        self.img_lbl = tk.Label(self.frame, bg="black", borderwidth=0)
        self.img_lbl.grid(row=0, column=0, columnspan=3)
        self.running = self.paused = False
        # the display's substrate level: the scan camera's reference; for the IR (domain) camera the levels of its own
        # first frame (the scan camera's are far off: the picture opened dark or white until "Balance colour", 2026-10-19)
        self.sub = None if getattr(scanner, "mono_view", False) else scanner.substrate_rgb()
        self.enh = tk.BooleanVar(value=False)
        self.zoom = tk.BooleanVar(value=False)
        self.readout = ttk.Label(self.frame, text="", font=("Consolas", 9), anchor="w", justify="left")
        self.readout.grid(row=1, column=0, columnspan=3, sticky="w")
        opt = ttk.Frame(self.frame)
        opt.grid(row=2, column=0, sticky="w", columnspan=1 if controls else 3)
        ttk.Checkbutton(opt, text="Enhanced contrast", variable=self.enh).pack(side="left")
        ttk.Checkbutton(opt, text="1:1 (centre)", variable=self.zoom).pack(side="left")
        ttk.Button(opt, text="Balance colour", command=self.balance).pack(side="left", padx=4)
        # the picture flipped (display only; the arrows follow): remembered for every live view on this PC
        fh, fv = remembered_flip(scanner.s)
        self.flip_h, self.flip_v = tk.BooleanVar(value=fh), tk.BooleanVar(value=fv)
        self.on_flip = None                                          # (a window's callback: its map follows)
        ff = ttk.Frame(self.frame)
        ff.grid(row=3, column=0, columnspan=3, sticky="w")
        for t, var in (("Flip left-right", self.flip_h), ("Flip up-down", self.flip_v)):
            ttk.Checkbutton(ff, text=t, variable=var, command=self._flip_changed).pack(side="left", padx=(0, 8))
        zf = ttk.Frame(self.frame)
        if controls:
            zf.grid(row=2, column=1, padx=10)
        ttk.Label(zf, text="Focus").pack(side="left")
        self.zstep = tk.StringVar(value="2")
        ttk.Combobox(zf, textvariable=self.zstep, values=("0.5", "2", "10", "50"), width=5, state="readonly").pack(side="left", padx=4)
        ttk.Label(zf, text="um").pack(side="left")
        ttk.Button(zf, text="Up", command=lambda: self.move_z(+1)).pack(side="left", padx=2)
        ttk.Button(zf, text="Down", command=lambda: self.move_z(-1)).pack(side="left", padx=2)
        self.af_btn = ttk.Button(zf, text="Autofocus", command=self.autofocus)
        self.af_btn.pack(side="left", padx=(8, 2))
        self.af_running = False
        jf = ttk.Frame(self.frame)
        if controls:
            jf.grid(row=2, column=2, sticky="e")
        ttk.Label(jf, text="Stage").grid(row=0, column=0, rowspan=2)
        self.jstep = tk.StringVar(value="0.5 field")
        ttk.Combobox(jf, textvariable=self.jstep, values=("0.1 field", "0.5 field", "1 field", "5 fields"), width=9,
                     state="readonly").grid(row=0, column=1, rowspan=2, padx=4)
        for (r, c_, t, d) in ((0, 3, "^", (0, -1)), (1, 2, "<", (-1, 0)), (1, 3, "v", (0, 1)), (1, 4, ">", (1, 0))):
            ttk.Button(jf, text=t, width=2, command=lambda d=d: self.jog(*d)).grid(row=r, column=c_)
        top = parent.winfo_toplevel()
        if controls:
            for key, d in (("<Left>", (-1, 0)), ("<Right>", (1, 0)), ("<Up>", (0, -1)), ("<Down>", (0, 1))):
                top.bind(key, lambda e, d=d: self.jog(*d))
            top.bind("<Prior>", lambda e: self.move_z(+1))
            top.bind("<Next>", lambda e: self.move_z(-1))
        self.flat = self._load_flat(scanner)
        self.flat_half = half_flat(self.flat) if self.flat is not None else None
        self.streaming = False
        self.last_readout = 0.0
        self.n_frames, self.t_frames = 0, time.time()
        self.dark = getattr(scanner, "DARK", None)
        if self.dark is None:
            self.dark = scanner.s.dark * (scanner.full_scale() + 1) / 4096.0
        # the picture tools (display only; the raw frames are not changed): black and white, black / white points on a
        # histogram of the shown picture's brightness, auto / manual exposure
        self.bw = tk.BooleanVar(value=False)
        self.sbar = tk.BooleanVar(value=True)                   # (the scale bar, of the objective in use)
        self.lv_lo, self.lv_hi = 0.0, 255.0
        self.ae = tk.StringVar(value="manual")
        self.t_ae = self.t_hist = 0.0
        self.tools = self.exp_e = None
        self.meas, self.meas_a, self.meas_cur = [], None, None      # (measuring: fixed segments, first point, cursor)
        self.meas_on = tk.BooleanVar(value=False)
        self.scale = float(scanner.s.live_scale if scale is None else scale)
        self._disp = (self.scale, 0.0, 0.0)                          # (shown px -> frame px: frame = shown / scale + offset)
        if tools is not None and tools is not False:
            self._build_tools(tk, None if tools is True else tools)
            self.img_lbl.bind("<Button-1>", self._meas_click)
            self.img_lbl.bind("<Motion>", self._meas_move)

    def exposure_box(self, parent, mode="manual", **pack):
        """The exposure box alone (without the other picture tools) in `parent`, starting in `mode` ("auto" / "manual";
        the polygon window, 2026-10-27)."""
        self.ae.set(mode)
        return self._build_exposure(parent, **pack)

    # ---- picture tools (tools=True)
    HIST_W, HIST_H = 256, 80

    def _build_tools(self, tk, into=None):
        f = self.tools = ttk.Frame(self.frame if into is None else into)
        if into is None:
            f.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        else:
            f.pack(anchor="w", pady=(10, 0))
        side = "left" if into is None else "top"                       # (beside each other under the picture, else stacked)
        left = ttk.Frame(f)
        left.pack(side=side, anchor="nw")
        cb = ttk.Frame(left)
        cb.pack(anchor="w")
        ttk.Checkbutton(cb, text="Black and white", variable=self.bw).pack(side="left")
        ttk.Checkbutton(cb, text="Scale bar", variable=self.sbar).pack(side="left", padx=(12, 0))
        mb = ttk.Frame(left)
        mb.pack(anchor="w", pady=(4, 0))
        self.meas_btn = ttk.Checkbutton(mb, text="Measure", variable=self.meas_on, command=self._meas_toggle, style="Toolbutton")
        self.meas_btn.pack(side="left")
        ttk.Button(mb, text="Clear measurements", command=self.clear_measurements).pack(side="left", padx=(8, 0))
        self.meas_lbl = ttk.Label(left, text="", foreground="#9a96b8")
        self.meas_lbl.pack(anchor="w")
        hb = ttk.Frame(left)
        hb.pack(anchor="w", pady=(4, 0))
        ttk.Label(hb, text="Contrast (drag the lines)").pack(side="left")
        ttk.Button(hb, text="Auto", width=6, command=self.levels_auto).pack(side="left", padx=(8, 2))
        ttk.Button(hb, text="Reset", width=6, command=lambda: self.set_levels(0.0, 255.0)).pack(side="left", padx=2)
        self.hist = tk.Canvas(left, width=self.HIST_W, height=self.HIST_H, bg="#15151a", highlightthickness=1,
                              highlightbackground="#444")
        self.hist.pack(anchor="w", pady=(4, 0))
        self.hist.bind("<Button-1>", self._hist_drag)
        self.hist.bind("<B1-Motion>", self._hist_drag)
        self.hist_lbl = ttk.Label(left, text="black 0, white 255")
        self.hist_lbl.pack(anchor="w")
        self._build_exposure(f, side=side, padx=(18, 0) if into is None else 0, pady=0 if into is None else (12, 0))
        self._draw_hist(None)

    def _build_exposure(self, parent, **pack):
        """The exposure box: Auto / Manual, and in Manual \u25c0 / \u25b6 (10 % less / more) or a typed time."""
        ex = ttk.Frame(parent)
        ex.pack(anchor="nw", **pack)
        ttk.Label(ex, text="Exposure", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        er = ttk.Frame(ex)
        er.pack(anchor="w", pady=(2, 0))
        ttk.Radiobutton(er, text="Auto", value="auto", variable=self.ae, command=self._ae_changed).pack(side="left")
        ttk.Radiobutton(er, text="Manual", value="manual", variable=self.ae, command=self._ae_changed).pack(side="left", padx=(8, 0))
        mr = self.exp_row = ttk.Frame(ex)
        mr.pack(anchor="w", pady=(4, 0))
        self.exp_dn = ttk.Button(mr, text="\u25c0", width=3, command=lambda: self.step_exposure(1 / 1.1))
        self.exp_dn.pack(side="left")
        self.exp_e = ttk.Entry(mr, width=8)
        self.exp_e.pack(side="left", padx=4)
        self.exp_e.bind("<Return>", lambda e: self.type_exposure())
        ttk.Label(mr, text="ms").pack(side="left")
        self.exp_up = ttk.Button(mr, text="\u25b6", width=3, command=lambda: self.step_exposure(1.1))
        self.exp_up.pack(side="left", padx=(4, 0))
        self.exp_set = ttk.Button(mr, text="Set", width=5, command=self.type_exposure)
        self.exp_set.pack(side="left", padx=(6, 0))
        self.exp_msg = ttk.Label(ex, text="\u25c0 / \u25b6: 10 % less / more; or type it and press Set (Enter)",
                                 foreground="#9a96b8", wraplength=240)
        self.exp_msg.pack(anchor="w", pady=(2, 0))
        self._show_exposure()
        self._ae_changed()
        return ex

    def _show_exposure(self):
        if self.exp_e is None:
            return
        try:
            v = float(self.sc.core.getExposure())
        except Exception:
            return
        if self.ui.root.focus_get() is not self.exp_e or self.ae.get() == "auto":
            off = self.exp_e.instate(["disabled"])              # (Auto: greyed out, but showing the exposure now)
            self.exp_e.state(["!disabled"])
            self.exp_e.delete(0, "end")
            self.exp_e.insert(0, f"{v:.3g}")
            if off:
                self.exp_e.state(["disabled"])

    def _ae_changed(self):
        auto = self.ae.get() == "auto"
        for b in (self.exp_dn, self.exp_up, self.exp_set):
            b.state(["disabled"] if auto else ["!disabled"])
        self.exp_e.state(["disabled"] if auto else ["!disabled"])
        self.exp_msg.configure(text="Auto: the exposure follows the picture's brightness" if auto else
                               "\u25c0 / \u25b6: 10 % less / more; or type it and press Set (Enter)")

    EXP_MIN_MS, EXP_MAX_MS = 0.02, 2000.0

    def _set_exp(self, v):
        v = float(min(max(v, self.EXP_MIN_MS), self.EXP_MAX_MS))
        self.set_exposure(v)
        self._show_exposure()

    def step_exposure(self, k):
        self._set_exp(float(self.sc.core.getExposure()) * k)

    def type_exposure(self):
        try:
            v = float(self.exp_e.get())
            assert self.EXP_MIN_MS <= v <= self.EXP_MAX_MS
        except (ValueError, AssertionError):
            self.exp_msg.configure(text=f"Exposure: a number of ms from {self.EXP_MIN_MS:g} to {self.EXP_MAX_MS:g}")
            return
        self._set_exp(v)

    def auto_exposure(self, fr):
        """Auto: the exposure set so the brightest part of the raw frame (99.5th percentile) sits at ~70 % of the camera's
        range, a step every 0.4 s at most (software: the camera's own auto mode stays off for the scan)."""
        now = time.time()
        if now - self.t_ae < 0.4:
            return
        self.t_ae = now
        a = np.asarray(fr)
        a = a[::4, ::4]
        top = float(np.percentile(a, 99.5))
        fs = 255.0 if a.dtype == np.uint8 else float(self.sc.full_scale())
        dk = float(np.mean(self.dark)) if np.ndim(self.dark) else float(self.dark or 0.0)
        lvl = (top - dk) / max(fs - dk, 1.0)
        if 0.6 <= lvl <= 0.8:
            return
        k = 0.5 if lvl >= 0.98 else min(max(0.7 / max(lvl, 1e-3), 0.5), 2.0) ** 0.8
        self._set_exp(float(self.sc.core.getExposure()) * k)

    def set_levels(self, lo, hi):
        lo, hi = float(min(max(lo, 0.0), 254.0)), float(min(max(hi, 1.0), 255.0))
        if hi <= lo + 1:
            hi = lo + 1
        self.lv_lo, self.lv_hi = lo, hi
        if self.tools is not None:
            self.hist_lbl.configure(text=f"black {lo:.0f}, white {hi:.0f}")
            self._draw_hist(getattr(self, "_hist_counts", None))

    def levels_auto(self):
        L = getattr(self, "_lum", None)
        if L is None:
            return
        lo, hi = np.percentile(L, [0.5, 99.5])
        self.set_levels(lo, max(hi, lo + 2))

    def _hist_drag(self, e):
        v = e.x / self.HIST_W * 255.0
        if abs(v - self.lv_lo) <= abs(v - self.lv_hi):
            self.set_levels(v, self.lv_hi)
        else:
            self.set_levels(self.lv_lo, v)

    def _draw_hist(self, counts):
        cv, W, H = self.hist, self.HIST_W, self.HIST_H
        cv.delete("all")
        if counts is not None and counts.max() > 0:
            h = np.log1p(counts.astype(float))
            h = h / max(h.max(), 1e-9) * (H - 4)
            pts = []
            for i, v in enumerate(h):
                x = i / 255.0 * W
                pts += [x, H, x, H - v]
            for i in range(0, len(pts), 4):
                cv.create_line(*pts[i:i + 4], fill="#8a8aa0")
        for v, col in ((self.lv_lo, "#4ea3ff"), (self.lv_hi, "#ffffff")):
            x = v / 255.0 * W
            cv.create_line(x, 0, x, H, fill=col, width=3)

    _RINGS = {}

    @staticmethod
    def draw_cross(img):
        """A cross and a ring at the middle of a picture (centring; the pause check's two pictures)."""
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        cy, cx, a, g = h // 2, w // 2, max(h, w) // 25, max(h, w) // 120
        col = np.array((255, 80, 200), np.uint8)
        t0, t1 = -(max(1, max(h, w) // 450) // 2), max(1, max(h, w) // 450) - max(1, max(h, w) // 450) // 2   # (1 px; 2 on big pictures)
        img[cy + t0:cy + t1, max(cx - a, 0):cx - g] = col
        img[cy + t0:cy + t1, cx + g:cx + a] = col
        img[max(cy - a, 0):cy - g, cx + t0:cx + t1] = col
        img[cy + g:cy + a, cx + t0:cx + t1] = col
        ring = LiveView._RINGS.get((h, w))
        if ring is None:
            yy, xx = np.ogrid[:h, :w]
            ring = LiveView._RINGS[(h, w)] = np.abs(np.hypot(yy - cy, xx - cx) - 2.2 * a) < 0.4 * (t1 - t0) + 0.4
        img[ring] = col
        return img

    def objective_in_view(self):
        """The objective the view is set for ("100x", "10x", "50x", ...: from view_scale, set_mag)."""
        vs = float(getattr(self.sc, "view_scale", 1.0))
        return "100x" if abs(vs * 9.54 - 1) < 1e-3 else f"{10.0 / vs:g}x"

    def um_per_px(self, disp_scale):
        """um per pixel of the shown picture: the scale bar calibration of the objective in use (scalecal: a slide with
        known distances; the objective from view_scale), else the stage calibration (M: um per camera pixel at 10x) x the
        objective (view_scale: 100x 1 / 9.54, else 10 / magnification; set_mag); / the picture's scale (live_scale, or 1)."""
        sc = self.sc
        vs = float(getattr(sc, "view_scale", 1.0))
        name = self.objective_in_view()
        cal = getattr(sc, "scale_cal", None)
        if cal is None:
            try:
                from .scalecal import load_scale_calibration
                cal = load_scale_calibration(sc.s.local_root)
            except Exception:
                cal = {}
            sc.scale_cal = cal
        if name in cal:
            return float(cal[name]) / max(float(disp_scale), 1e-6)
        M = np.asarray(sc.M, float)
        return float(np.sqrt(abs(np.linalg.det(M)))) * vs / max(float(disp_scale), 1e-6)

    # ---- measuring (the pause window): click, the ruler follows the cursor with its length, click again to fix it
    def _meas_toggle(self):
        self.meas_a = self.meas_cur = None
        self.meas_lbl.configure(text="click the first point, then the second" if self.meas_on.get() else "")

    def _to_frame(self, x, y):
        s, ox, oy = self._disp
        return (x / s + ox, y / s + oy)

    def _meas_click(self, e):
        if not self.meas_on.get():
            return
        p = self._to_frame(e.x, e.y)
        if self.meas_a is None:
            self.meas_a, self.meas_cur = p, p
            self.meas_lbl.configure(text="now the second point")
        else:
            self.meas.append((self.meas_a, p))
            L = np.hypot(p[0] - self.meas_a[0], p[1] - self.meas_a[1]) * self.um_per_px(1.0)
            self.meas_a = self.meas_cur = None
            self.meas_lbl.configure(text=f"{len(self.meas)} measured (last {self._len_txt(L)}); click for another")

    def _meas_move(self, e):
        if self.meas_on.get() and self.meas_a is not None:
            self.meas_cur = self._to_frame(e.x, e.y)

    def clear_measurements(self):
        self.meas, self.meas_a, self.meas_cur = [], None, None
        if self.tools is not None:
            self._meas_toggle()

    @staticmethod
    def _len_txt(L):
        return f"{L:.2f} um" if L < 10 else f"{L:.1f} um" if L < 1000 else f"{L / 1000:.3f} mm"

    @staticmethod
    def _text(img, s, org, fs):
        """White text with a thin black outline (readable on any picture, no background box)."""
        import cv2
        th = max(1, int(round(fs * 2.0)))
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

    def draw_measurements(self, img, disp_scale, ox=0.0, oy=0.0):
        """The measured segments (and the one being drawn) on a picture of scale disp_scale whose top-left is frame pixel
        (ox, oy): white lines, their lengths beside them (um, from the objective in use)."""
        segs = list(self.meas) + ([(self.meas_a, self.meas_cur)] if self.meas_a is not None and self.meas_cur is not None else [])
        if not segs:
            return img
        import cv2
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        upp = self.um_per_px(1.0)
        fs = max(0.4, w / 1700.0)
        lw = max(2, int(round(w / 500.0)))
        for a, b in segs:
            pa = (int(round((a[0] - ox) * disp_scale)), int(round((a[1] - oy) * disp_scale)))
            pb = (int(round((b[0] - ox) * disp_scale)), int(round((b[1] - oy) * disp_scale)))
            cv2.line(img, pa, pb, (255, 255, 255), lw, cv2.LINE_AA)
            for q in (pa, pb):
                cv2.circle(img, q, lw + 2, (255, 255, 255), -1, cv2.LINE_AA)
            L = float(np.hypot(b[0] - a[0], b[1] - a[1])) * upp
            lab = self._len_txt(L)
            (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, int(round(fs * 2.0))))
            # (beside the line, not on it: off its middle along its normal, on the upper side, the text centred there)
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            nl = max(float(np.hypot(dx, dy)), 1e-6)
            nx, ny = -dy / nl, dx / nl
            if ny > 0:
                nx, ny = -nx, -ny
            off = 0.5 * float(np.hypot(tw, tht)) + 6 + lw
            cx, cy = (pa[0] + pb[0]) / 2 + nx * off, (pa[1] + pb[1]) / 2 + ny * off
            org = (int(min(max(cx - tw / 2, 2), w - tw - 2)), int(min(max(cy + tht / 2, tht + 2), h - 2)))
            self._text(img, lab, org, fs)
        return img

    SCALE_BAR_UM = {"100x": 10.0}       # fixed scale bar lengths (Avram, 2026-10-27: 10 um at 100x); others: a round length
    _fonts = {}

    @classmethod
    def _font(cls, px):
        """A normal TrueType font (Segoe UI, else Arial) of px pixels, or None (then OpenCV's own)."""
        if px not in cls._fonts:
            f = None
            try:
                from PIL import ImageFont
                for name in ("segoeui.ttf", "arial.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                             "DejaVuSans.ttf"):
                    try:
                        f = ImageFont.truetype(name, px)
                        break
                    except OSError:
                        continue
            except ImportError:
                pass
            cls._fonts[px] = f
        return cls._fonts[px]

    def scale_bar(self, img, disp_scale):
        """A scale bar (bottom left) on the shown picture: 10 um at 100x (SCALE_BAR_UM), else a round length (1, 2, 5 x
        10^n um) about a fifth of its width; its length above it in plain white (a normal font, no outline)."""
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        upp = self.um_per_px(disp_scale)
        if not np.isfinite(upp) or upp <= 0:
            return img
        L = self.SCALE_BAR_UM.get(self.objective_in_view())
        if L is None:
            want = 0.2 * w * upp
            e = 10.0 ** np.floor(np.log10(want))
            L = max(k * e for k in (1, 2, 5) if k * e <= want)
        n = int(round(L / upp))
        th = max(3, h // 110)
        x0, y0 = max(10, w // 40), h - max(12, h // 25)
        img[y0:y0 + th, x0:x0 + n] = 255
        font = self._font(max(12, int(round(w / 55.0))))
        if font is None:                                        # (no TrueType font: OpenCV's, plain white)
            import cv2
            fs = max(0.35, w / 2200.0)
            cv2.putText(img, f"{L:g} um" if L >= 1 else f"{L * 1000:g} nm", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), max(1, int(round(fs * 2.0))), cv2.LINE_AA)
            return img
        from PIL import Image, ImageDraw
        lab = f"{L:g} \u00b5m" if L >= 1 else f"{L * 1000:g} nm"
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        b = d.textbbox((0, 0), lab, font=font)
        d.text((x0 - b[0], y0 - 6 - b[3]), lab, font=font, fill=(255, 255, 255))
        return np.asarray(im).copy()

    def finish(self, img, disp_scale=None, offset=(0.0, 0.0)):
        """The shown picture (8-bit RGB) through the picture tools: black / white points, black and white, the scale bar,
        the measurements (disp_scale: the picture's scale against the camera's, default the live view's; offset: the frame
        pixel at its top-left, the 1:1 centre view)."""
        if self.tools is None:
            return img
        L = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        self._lum = L[::4, ::4]
        now = time.time()
        if now - self.t_hist > 0.2:
            self.t_hist = now
            self._hist_counts = np.bincount(np.clip(L[::2, ::2], 0, 255).astype(np.int64).ravel(), minlength=256)[:256]
            self._draw_hist(self._hist_counts)
        if self.bw.get():
            img = np.repeat(L[..., None], 3, axis=2)
        if self.lv_lo > 0.0 or self.lv_hi < 255.0:
            img = (np.asarray(img, np.float32) - self.lv_lo) * (255.0 / (self.lv_hi - self.lv_lo))
        img = np.clip(img, 0, 255).astype(np.uint8)
        ds = self.scale if disp_scale is None else disp_scale
        if self.sbar.get():
            img = self.scale_bar(img, ds)
        return self.draw_measurements(img, ds, *offset)

    @staticmethod
    def _load_flat(sc):
        """The flat field for the live view: 10x (settings flat10), or at 100x the flat100 entry for the piece's oxide."""
        from ..mm import FlatField
        path = sc.s.flat10
        if getattr(sc, "objective_now", "") in ("50x", "5x"):         # (device photos: no flat field for these)
            return None
        if getattr(sc, "view_scale", 1.0) < 1.0 and sc.s.flat100:
            ox = getattr(sc, "oxide", sc.s.oxide_nm)
            best = min(sc.s.flat100, key=lambda e: abs(float(str(e).split("=", 1)[0]) - ox), default=None)
            path = str(best).split("=", 1)[1] if best else ""
        if not path:
            return None
        try:
            return FlatField.load(path)
        except Exception:
            return None

    def start(self):
        """Live view: the camera runs continuously (as Micro-Manager's live mode) and the newest frame is shown; a snap
        per frame costs ~0.2 s with the XC10.  Falls back to snaps if the camera cannot stream."""
        self.running = True
        c = self.sc.core
        try:
            if c.isSequenceRunning():
                self.sc.stop_sequence()
            c.startContinuousSequenceAcquisition(0)
            self.streaming = True
        except Exception:
            self.streaming = False
        self.tick()

    def stop(self):
        self.running = False
        if self.streaming:
            try:
                self.sc.stream_off()                     # (waits until the camera has stopped: the Raptor takes a moment)
            except Exception:
                pass
            self.streaming = False

    def pause(self, on):
        self.paused = on

    def set_exposure(self, ms):
        """A new exposure while the view runs (the camera's stream is restarted around it)."""
        c = self.sc.core
        self.paused = True
        try:
            if self.streaming:
                self.sc.stop_sequence()
            c.setExposure(float(ms))
            if self.streaming:
                c.startContinuousSequenceAcquisition(0)
        finally:
            self.paused = False

    def set_mag(self, name):
        """Stage steps and flat field for the objective `name` ("5x", "10x", "50x", "100x"; the pause window: whatever
        objective the person uses): the steps in proportion to the magnification (100x: the calibrated 9.54), the flat
        field of 10x / 100x, none for the others."""
        try:
            mag = float(str(name).lower().rstrip("x"))
        except ValueError:
            return
        self.sc.objective_now = name
        self.sc.view_scale = 1.0 / 9.54 if abs(mag - 100) < 1e-6 else 10.0 / mag
        self.flat = self._load_flat(self.sc) if mag in (10.0, 100.0) else None
        self.flat_half = half_flat(self.flat) if self.flat is not None else None
        self.sub = None                                              # (the colour balance again for that objective)

    def set_objective(self, hundred):
        """Stage steps and flat field for the 10x or the 100x objective."""
        self.sc.view_scale = 1.0 / 9.54 if hundred else 1.0
        self.flat = self._load_flat(self.sc)
        self.flat_half = half_flat(self.flat) if self.flat is not None else None

    def balance(self):
        self.sub = None

    def busy(self):
        """True while the autofocus runs (the window's buttons wait: it moves the focus and uses the camera)."""
        if self.af_running:
            self.readout.configure(text="wait for the autofocus to finish ...")
        return self.af_running

    def autofocus(self):
        """Coarse autofocus around the current focus (the scan focuses finely later): the 10x coarse sweep, at 100x the
        wide first-flake sweep, run in a thread while the live view pauses.  If it finds no focus, Z goes back and a
        note says to focus by hand."""
        import threading
        sc = self.sc
        if self.af_running or not sc.has_z:
            return
        self.af_running = True
        self.af_btn.configure(state="disabled", text="Autofocusing ...")
        self.readout.configure(text="autofocus running ...")
        self.paused = True
        was_streaming = self.streaming
        if self.streaming:
            try:
                sc.stop_sequence()
            except Exception:
                pass
            self.streaming = False
        z_start = sc.z_read()
        p = sc.s.af100_first if getattr(sc, "view_scale", 1.0) < 1.0 else sc.s.af10
        box = {}

        def work():
            zref_old = sc.zref
            sc.zref = z_start                                   # the Z limit is taken around where you are now
            try:
                if getattr(sc, "view_scale", 1.0) < 1.0:
                    box["r"] = sc.autofocus(z_start, p, fine=False, limit_check=False)
                else:
                    # wide coarse sweep (a sample in another holder can be far off), then a short sweep around its peak
                    rng, st = sc.s.live_af_range_um, sc.s.live_af_step_um
                    old_dz, old_ext = sc.s.max_dz_um, sc.s.af_extend
                    sc.s.max_dz_um = max(old_dz, rng + sc.s.live_af_refine[0] + sc.s.af_backlash_um)
                    sc.s.af_extend = 0
                    try:
                        r1 = sc.autofocus(z_start, (rng, st, 0, 1, p[4]), fine=False, limit_check=False)
                        box["r"] = r1
                        if r1[2]:
                            r2 = sc.autofocus(r1[0], (sc.s.live_af_refine[0], sc.s.live_af_refine[1], 0, 1, p[4]), fine=False, limit_check=False)
                            box["r"] = r2 if r2[2] else r1
                    finally:
                        sc.s.max_dz_um, sc.s.af_extend = old_dz, old_ext
            except Exception as e:
                box["err"] = f"{type(e).__name__}: {e}"
                try:
                    sc.z_write(z_start)
                    sc.core.waitForDevice(sc.zdev)
                except Exception:
                    pass
            finally:
                sc.zref = zref_old
            self.ui._post(done)

        def done():
            self.af_running = False
            if self.af_btn.winfo_exists():
                self.af_btn.configure(state="normal", text="Autofocus")
            if was_streaming and self.running:
                try:
                    if not sc.core.isSequenceRunning():          # (the autofocus may have left the stream on)
                        sc.core.startContinuousSequenceAcquisition(0)
                    self.streaming = True
                except Exception:
                    pass
            self.paused = False
            r = box.get("r")
            if r is not None and r[2]:
                self.readout.configure(text=f"autofocus: Z {f3(r[0])} um (peak / median sharpness {r[1]:.2f})")
                return
            span = p[0] if getattr(sc, "view_scale", 1.0) < 1.0 else sc.s.live_af_range_um
            why = box.get("err") or (f"no clear focus peak within +-{span:g} um (peak / median {r[1]:.2f}): bare substrate, "
                                     "too little texture in the view, or too far out of focus" if r is not None else "unknown error")
            self.ui.notice(self.frame.winfo_toplevel(), "Autofocus failed",
                           f"Autofocus failed: {why}.\n\nThe focus is back where it was.  Focus manually with the Up / Down "
                           "buttons (or PgUp / PgDn), or move to a spot with flakes and try again.")
        threading.Thread(target=work, daemon=True).start()

    def move_z(self, sign):
        sc = self.sc
        if not sc.has_z or self.af_running:
            return
        z = sc.z_read() + sign * float(self.zstep.get())
        if self.free_z:
            sc.z_write(z); sc.core.waitForDevice(sc.zdev)
        elif not sc.set_z(z):
            self.readout.configure(text=f"Z {f1(z)} is beyond the allowed range (max_dz_um) around the start / prediction")

    def flip(self):
        """(left-right, up-down) as the live view shows the picture now."""
        return bool(self.flip_h.get()), bool(self.flip_v.get())

    def _flip_changed(self):
        remember_flip(self.sc.s, *self.flip())
        if self.on_flip:
            self.on_flip()

    def jog(self, dx, dy):
        """Move the view by a fraction of the field in the directions shown (the picture as displayed, flipped or not;
        pixel -> stage through M)."""
        sc = self.sc
        if self.af_running:
            return
        fh, fv = self.flip()
        dx, dy = (-dx if fh else dx), (-dy if fv else dy)
        f = {"0.1 field": 0.1, "0.5 field": 0.5, "1 field": 1.0, "5 fields": 5.0}[self.jstep.get()]
        M = sc.M * getattr(sc, "view_scale", 1.0)
        d = M @ np.array([dx * sc.W * f, dy * sc.H * f])
        c = sc.core
        x, y = c.getXPosition(sc.xy) + d[0], c.getYPosition(sc.xy) + d[1]
        if not self.free_z and not sc.move_allowed(x, y):
            self.readout.configure(text="that move would go beyond the allowed area (max_travel_um from the start, or the scan's box)")
            return
        c.setXYPosition(sc.xy, x, y)
        c.waitForDevice(sc.xy)

    def tick(self):
        if not self.running:
            return
        if not self.img_lbl.winfo_exists():
            return
        if not self.paused:
            try:
                self.show()
            except Exception as e:
                self.readout.configure(text=f"live view error: {e}")
        self.img_lbl.after(15, self.tick)

    def show(self):
        sc = self.sc
        c = sc.core
        if self.streaming:
            if c.getRemainingImageCount() == 0:
                return                                               # no new frame yet
            fr = sc.to_array(c.getLastImage())
            c.clearCircularBuffer()
        else:
            c.snapImage()
            fr = sc.to_array(c.getImage())
        self.last_frame = fr                                         # (the pause window's Save picture)
        if self.exp_e is not None and self.ae.get() == "auto":
            self.auto_exposure(fr)
        zoom = self.zoom.get()
        if zoom:
            h, w = fr.shape[:2]
            vw, vh = int(w * self.scale) // 2 * 2, int(h * self.scale) // 2 * 2
            y0, x0 = (h - vh) // 2 // 2 * 2, (w - vw) // 2 // 2 * 2
            part = fr[y0:y0 + vh, x0:x0 + vw]
            fl = None
            if self.flat is not None:
                from ..mm import FlatField
                fl = FlatField(flat=self.flat.flat[y0:y0 + vh, x0:x0 + vw], dark=self.flat.dark[y0:y0 + vh, x0:x0 + vw]
                               if np.ndim(self.flat.dark) >= 2 else self.flat.dark)
            img, sub = to_display(part, self.dark, self.sub, sc.s.display_level, self.enh.get(), 1.0, fl, getattr(sc, "mono_view", False))
            if self.sub is None:
                self.sub = sub
        else:
            fl = self.flat_half if self.scale == 0.5 else self.flat
            img, sub = to_display(fr, self.dark, self.sub, sc.s.display_level, self.enh.get(), self.scale, fl, getattr(sc, "mono_view", False))
            if self.sub is None:
                self.sub = sub
        ds, off = (1.0, (float(x0), float(y0))) if zoom else (self.scale, (0.0, 0.0))
        self._disp = (ds, off[0], off[1])
        img = self.finish(flipped(img, self.flip()), ds, off)        # (display only)
        if self.cross:                                               # centring: a cross and a ring at the centre
            img = self.draw_cross(img)
        try:
            from PIL import Image, ImageTk
            im = ImageTk.PhotoImage(Image.fromarray(img))
        except Exception:                                            # (no PIL ImageTk: Tk's own PPM, as the camera panel)
            h_, w_ = img.shape[:2]
            im = self.ui.tk.PhotoImage(master=self.img_lbl, data=f"P6 {w_} {h_} 255 ".encode() + np.ascontiguousarray(img).tobytes(),
                                       format="PPM")
        self.img_lbl.configure(image=im)
        self.img_lbl.image = im
        self.n_frames += 1
        now = time.time()
        if now - self.last_readout > 0.5:                            # (stage / focus readout: serial round trips)
            self.n_frames, self.t_frames, self.last_readout = 0, now, now
            z = sc.z_read() if sc.has_z else 0.0
            obj = ""
            if getattr(sc, "nose", None) is not None:
                try:
                    obj = f"objective {sc.objective_name(sc.nose.position())}   "
                except Exception:
                    obj = ""
            self.readout.configure(text=f"{obj}stage {f1(c.getXPosition(sc.xy))}, {f1(c.getYPosition(sc.xy))} um   Z {f3(z)} um")
