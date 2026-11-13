"""Screenshots of every window of the scan program where the user does something, for the scan guide.

    C:\\FlakeID\\run_flakeid.bat screenshots                    (pictures in C:\\FlakeScans\\dialog_screenshots)
    C:\\FlakeID\\run_flakeid.bat screenshots --out D:\\shots --theme light

It runs the real scan program (the same windows, theme and text) on the simulated microscope (flakeid.scope.mock:
two synthetic chips, a simulated nosepiece, a stand-in for the analysis), and a "director" answers each window the
way a user would: it waits until the window has settled, saves a picture of it, then fills it in and presses the
button.  Nothing touches the microscope, the camera or Micro-Manager, and nothing is written to the real settings,
flat fields or calibration (everything goes to a temporary folder).  Takes about 3-5 minutes; do not use the mouse or
keyboard while it runs (windows must stay on top to be photographed).

The session: start screen -> objective check -> adding two pieces (material and oxide) -> results folder (the Windows
folder picker is skipped) -> run name -> scan plan and main window -> switch to 100x -> 100x imaging window ->
rhombohedral-domains question -> start screen again -> 100x flat field (field-stop focus, Filmetrics value) ->
capture mode -> quit.  The pictures are numbered in that order (the names are used by docs/scan_guide.md); an
index.html shows them all.  Copy the folder to docs/scan_guide_images/ in the repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="flakeid_shots_"))
os.environ["HOME"] = os.environ["USERPROFILE"] = str(TMP / "home")      # (calibrations made by the flat field stay here)
(TMP / "home").mkdir()

import numpy as np  # noqa: E402

OFF100 = np.array([24.0, -18.3])        # the simulated 100x objective's XY offset (as MockCore)


# ---------------------------------------------------------------- the simulated sample
def make_chip(rng, sub, n_flakes, size=4000):
    import cv2
    im = np.empty((size, size, 3), np.uint8)
    im[:] = sub
    cols = [(95, 125, 215), (60, 160, 230), (190, 150, 210), (230, 200, 120), (120, 90, 170)]
    for _ in range(n_flakes):
        c = (int(rng.integers(300, size - 300)), int(rng.integers(300, size - 300)))
        ax = (int(rng.integers(10, 90)), int(rng.integers(6, 50)))
        cv2.ellipse(im, c, ax, float(rng.uniform(0, 180)), 0, 360, cols[int(rng.integers(len(cols)))], -1)
    return cv2.GaussianBlur(im, (0, 0), 0.8)


def make_patch(rng, sub, col, w=1300, h=1000):
    import cv2
    im = np.empty((h, w, 3), np.uint8)
    im[:] = sub
    pts = np.array([[w * 0.5 + r * np.cos(a), h * 0.5 + 0.7 * r * np.sin(a)]
                    for a, r in zip(np.linspace(0, 2 * np.pi, 9)[:-1], rng.uniform(150, 330, 8))], np.int32)
    cv2.fillPoly(im, [pts], col)
    cv2.line(im, tuple(pts[0]), tuple(pts[4]), tuple(int(v * 0.92) for v in col), 6)
    return cv2.GaussianBlur(im, (0, 0), 1.0)


def build_scene():
    from flakeid.scope.mock import MockCore
    rng = np.random.default_rng(5)
    core = MockCore(1.09, x0=20000.0, y0=20000.0)
    chips = [((20000.0, 20000.0), (150, 120, 190), "hbn"), ((27000.0, 20500.0), (125, 105, 175), "graphite")]
    targets = []
    for (cx, cy), sub, mat in chips:
        core.add_chip(make_chip(rng, sub, 140), cx, cy, float(rng.uniform(-4, 4)), 0.0)
        for k in range(3):
            px, py = cx + rng.uniform(-1200, 1200), cy + rng.uniform(-1200, 1200)
            core.add_patch(make_patch(rng, sub, [(90, 130, 220), (170, 140, 205), (60, 170, 225)][k]), px, py)
            targets.append(((cx, cy), (px, py), mat))
    core.patch_um_per_px = 0.114
    core.z = core.true_focus(20000.0, 20000.0) + 3.0
    return core, chips, targets


class DocNose:
    """Simulated coded nosepiece: follows the simulated objective; `force` shows another position (so the first
    objective window appears) until the director clears it."""

    def __init__(self, core, positions):
        self.core, self.positions, self.unit, self.port, self.force = core, positions, "SCU,U-D7RES (simulated)", "mock", None

    def position(self):
        if self.force is not None:
            return self.force
        return self.positions["100x"] if self.core.objective == 100 else self.positions["10x"]

    def close(self):
        pass


def fake_analysis(run_root, targets, stop):
    """Stand-in for the analysis: a piece's target list once its tiles are all on disk."""
    done = set()
    while not stop.is_set():
        for mf in Path(run_root).glob("*_run_*.json"):
            try:
                man = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for p in man.get("pieces", []):
                nm = p["name"]
                pos, tiles = Path(run_root) / f"{nm}_scan.pos", Path(run_root) / p.get("tiles", f"{nm}_tiles")
                if nm in done or not pos.exists() or not tiles.exists():
                    continue
                try:
                    n_plan = len(json.loads(pos.read_text())["map"]["StagePositions"]["array"])
                except Exception:
                    continue
                if len(list(tiles.glob("*/metadata.txt"))) < n_plan:
                    continue
                c = np.array(p["center"][:2])
                rows = [t for t in targets if np.hypot(*(np.array(t[0]) - c)) < 2500]
                tmp = Path(run_root) / f"{nm}_targets.csv.tmp"
                with open(tmp, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["order", "label", "stage_x_um", "stage_y_um", "z10_um", "thickness_nm"])
                    for k, (_, (px, py), mat) in enumerate(rows):
                        w.writerow([k + 1, f"{k + 1:02d}_F{k + 1:03d}", f"{px + OFF100[0]:.2f}", f"{py + OFF100[1]:.2f}", "",
                                    [3.4, 12.0, 25.0][k % 3] if mat == "graphite" else [18.0, 35.0, 60.0][k % 3]])
                tmp.replace(Path(run_root) / f"{nm}_targets.csv")
                done.add(nm)
        time.sleep(1.0)


# ---------------------------------------------------------------- pictures
def grab(w, path):
    """Picture of window w (with its title bar on Windows)."""
    from PIL import ImageGrab
    w.update_idletasks()
    box = None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        try:
            hwnd = int(w.wm_frame(), 16)
            r = wintypes.RECT()
            # the visible frame (Windows 10 adds invisible resize borders to GetWindowRect)
            if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(r), ctypes.sizeof(r)) != 0:
                ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
            box = (r.left, r.top, r.right, r.bottom)
        except Exception:
            box = None
    if box is None:
        x, y = w.winfo_rootx(), w.winfo_rooty()
        box = (x, max(0, y - 28), x + w.winfo_width(), y + w.winfo_height())
    ImageGrab.grab(bbox=box).save(str(path))


def widgets(w):
    out = [w]
    for c in w.winfo_children():
        out += widgets(c)
    return out


def find(w, cls, text=None):
    for c in widgets(w):
        if c.winfo_class() != cls:
            continue
        if text is None:
            return c
        try:
            t = c.cget("text")
        except Exception:
            continue
        if re.search(text, str(t)):
            return c
    return None


def press(w, text):
    b = find(w, "TButton", text) or find(w, "Button", text)
    if b is None:
        raise RuntimeError(f"no button {text!r} in {w.title()!r}")
    b.invoke()


def set_entry(e, v):
    e.delete(0, "end")
    e.insert(0, v)


# ---------------------------------------------------------------- the director
class Director:
    def __init__(self, ui, core, out):
        self.ui, self.core, self.out = ui, core, out
        self.seen, self.counts, self.shots, self.n = set(), {}, [], 0
        self.nose = None
        ui.root.after(500, self.poll)

    def shot(self, w, name, caption):
        self.n += 1
        fn = f"{name}.png"
        try:
            w.lift()
            w.attributes("-topmost", True)
            w.update()
            time.sleep(0.25)
            grab(w, self.out / fn)
            w.attributes("-topmost", False)
            self.shots.append((fn, w.title(), caption))
            print(f"  picture {fn}: {w.title()}")
        except Exception as e:
            print(f"  picture {fn} failed: {e}")

    def poll(self):
        r = self.ui.root
        try:
            if True:
                for w in r.winfo_children():
                    if w.winfo_class() == "Toplevel" and str(w) not in self.seen and w.winfo_ismapped():
                        self.seen.add(str(w))
                        title = w.title()
                        k = self.counts[title] = self.counts.get(title, 0) + 1
                        # live views need a moment for the first frames; other windows only to be drawn
                        delay = 3000 if find(w, "Label", None) is not None and self.has_live(w) else 900
                        r.after(delay, lambda w=w, t=title, k=k: self.act(w, t, k))
        except Exception as e:
            print("director:", e)
        r.after(250, self.poll)

    @staticmethod
    def has_live(w):
        return any(c.winfo_class() == "Label" and str(c.cget("bg")) == "black" for c in widgets(w))

    def fallback(self, w):
        """A step went wrong: answer the window the usual way so the session goes on (and say so)."""
        try:
            if not w.winfo_exists():
                return
            for t in ("^Use this focus$", "^OK$", "^Start$", "^Done", "^No$", "^Keep the current one$", "^Go on anyway$"):
                b = find(w, "TButton", t)
                if b is not None:
                    print(f"  (fallback: pressed '{b.cget('text')}' in {w.title()!r})")
                    b.invoke()
                    return
        except Exception as e:
            print("director fallback:", e)

    def run_steps(self, gen, w=None):
        """An action with pauses: a generator yielding milliseconds to wait."""
        def step():
            try:
                d = next(gen)
            except StopIteration:
                return
            except Exception as e:
                print("director step:", e)
                self.fallback(w)
                return
            self.ui.root.after(int(d), step)
        step()

    def act(self, w, title, k):
        if not w.winfo_exists():
            return
        self.run_steps(self.steps(w, title, k), w)

    def steps(self, w, title, k):
        ui, core = self.ui, self.core
        if title == "FlakeIdentifier":                                    # the start screen
            if k == 1:
                self.shot(w, "01_start_screen", "Start screen: choose what to do (scan chips is selected).")
                press(w, "^Start$")
            elif k == 2:
                find(w, "TRadiobutton", "flat field").invoke()
                yield 300
                cbs = [c for c in widgets(w) if c.winfo_class() == "TCombobox"]
                cbs[0].set("100x")
                ents = [c for c in widgets(w) if c.winfo_class() == "TEntry"]
                set_entry(ents[0], "285")
                set_entry(ents[1], "283.6")
                yield 400
                self.shot(w, "20_start_screen_flat", "Start screen set for a 100x flat field of a 285 nm wafer measured "
                                                    "at 283.6 nm, focused on the field stop.")
                press(w, "^Start$")
            elif k == 3:
                find(w, "TRadiobutton", "Capture").invoke()
                yield 300
                self.shot(w, "30_start_screen_capture", "Start screen set for capturing raw 100x images.")
                press(w, "^Start$")
            else:
                self.shot(w, "40_start_screen_again", "Back at the start screen after a job: start the next one or Quit.")
                press(w, "^Quit$")
        elif title.startswith("Objective: "):
            if "10x" in title:
                self.shot(w, "02_objective_10x", "The program checks the objective: turn to 10x and the window closes by itself.")
                if self.nose is not None:
                    self.nose.force = None
            else:
                self.shot(w, "09_objective_100x", "After the last piece: switch to 100x without touching the focus; the window "
                                                  "closes by itself when the nosepiece reports 100x.")
                core.mockObjective(100)
        elif title == "Multi-piece scan: pieces":
            core.mockFocus()
            yield 2500
            self.shot(w, "03_add_pieces", "Adding pieces: centre piece 1 in the live view, focus roughly, then 'Add piece 1 here'.")
            press(w, "Add piece 1 here")
            while self.counts.get("Piece 1: material and oxide", 0) == 0 or any(
                    c.winfo_class() == "Toplevel" and c.title().startswith("Piece 1") for c in ui.root.winfo_children()):
                yield 300
            core.setXYPosition("XY", 27000.0 + 150, 20500.0 - 100)
            core.mockFocus()
            yield 2500
            press(w, "Add piece 2 here")
            while self.counts.get("Piece 2: material and oxide", 0) == 0 or any(
                    c.winfo_class() == "Toplevel" and c.title().startswith("Piece 2") for c in ui.root.winfo_children()):
                yield 300
            yield 1500
            self.shot(w, "06_pieces_added", "Both pieces added (listed on the right): press 'Done'.")
            press(w, "^Done")
        elif re.match(r"Piece \d: material and oxide", title):
            n = int(title[6])
            cb = find(w, "TCombobox")
            ents = [c for c in widgets(w) if c.winfo_class() == "TEntry"]
            if n == 1:
                cb.set("hBN"); set_entry(ents[0], "285")
            else:
                cb.set("graphite"); set_entry(ents[0], "90")
            yield 300
            self.shot(w, f"0{3 + n}_piece_{n}_material", f"Piece {n}: material and the nominal oxide thickness.")
            press(w, "^OK$")
            return
        elif title == "Multi-piece scan: name":
            set_entry(find(w, "TEntry"), "demo")
            yield 300
            self.shot(w, "07_run_name", "Name of the run (empty = today's date).")
            press(w, "^OK$")
        elif title.endswith("scan plan"):
            if self.counts.get("_plan", 0) == 0:
                self.counts["_plan"] = 1
                self.shot(w, "08_scan_plan", "The plan of each piece: chip outline, tiles and focus points (just for information).")
                yield 500
                self.shot(ui.root, "08b_main_window", "The main window during the scan: progress, the log and 'Stop the scan'.")
            w.destroy()
        elif title.startswith("100x imaging"):
            if self.counts.get("_100x", 0) == 0:
                self.counts["_100x"] = 1
                self.shot(w, "10_100x_imaging", "During the 100x images: skip the rest of a piece, or stop the 100x part.")
        elif title.startswith("Waiting for the analysis"):
            if self.counts.get("_wait", 0) == 0:
                self.counts["_wait"] = 1
                self.shot(w, "10b_waiting_for_analysis", "The 100x part waits for a piece's target list (the analysis is still working on it).")
        elif title.startswith("Rhombohedral domains"):
            self.shot(w, "11_domains_question", "Graphite pieces: image the thick flakes' stacking domains with the IR camera? (No here.)")
            press(w, "^No$")
        elif title.startswith("Flat field (") or title == "Field-stop focus":
            core.mockFocus()
            yield 2500
            self.shot(w, "21_flat_position" if title.startswith("Flat") else "21b_fieldstop_by_hand",
                      "Flat field: a clean bare area, field stop closed until its edge shows, focused on the edge.")
            press(w, "Use this focus")
        elif title == "Field stop":
            self.shot(w, "22_fieldstop", "Field stop: " + ("reopen it to the scanning setting." if "Open" in find(w, "TLabel").cget("text")
                                                           else "close it until the edge shows."))
            press(w, "^OK$")
        elif title.endswith(": dark frames"):
            self.shot(w, "23_dark_frames", "Dark frames: block the light, then OK (Cancel = no dark frames).")
            press(w, "^OK$")
        elif title.startswith("New flat field"):
            yield 500
            self.shot(w, "24_flat_preview", "Preview: the new flat field, and its change from the one in use.")
            w.destroy()
        elif title == "Flat field":
            self.shot(w, "25_flat_result", "The flat field's summary (dust, focus check, calibration): use it or keep the current one.")
            press(w, "Keep the current one" if find(w, "TButton", "Keep") else "^OK$")
        elif title.startswith("Capture: pick"):
            core.mockFocus()
            yield 2500
            self.shot(w, "31_capture_pick", "Capture: move to the flake in the live view, focus, then 'Use this focus'.")
            press(w, "Use this focus")
        elif title == "Capture":
            self.shot(w, "32_capture_another", "Capture another flake? (No ends the capture.)")
            press(w, "^No$")
        elif find(w, "TButton", "Use this focus"):                         # focus by hand (autofocus failed, manual mode)
            core.mockFocus()
            yield 2500
            self.shot(w, "50_focus_by_hand", f"Focus by hand: {title}.")
            press(w, "Use this focus")
        else:                                                              # anything else: picture, then the first button
            self.shot(w, "60_" + re.sub(r"[^A-Za-z0-9]+", "_", title)[:40], f"Other window: {title}.")
            b = find(w, "TButton")
            if b is not None:
                b.invoke()

    def write_index(self):
        rows = "\n".join(f'<figure><img src="{fn}" alt=""><figcaption><b>{fn}</b> &middot; {t}<br>{c}</figcaption></figure>'
                         for fn, t, c in self.shots)
        (self.out / "index.html").write_text(
            "<!doctype html><meta charset=utf-8><title>Scan program windows</title><style>body{font:14px sans-serif;"
            "margin:24px;background:#f4f3f8}figure{margin:0 0 28px}img{max-width:100%;box-shadow:0 1px 6px #0003}</style>"
            f"<h1>Scan program windows</h1><p>{time.strftime('%Y-%m-%d %H:%M')}, flakeid.scope on the simulated microscope.</p>"
            + rows, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="C:/FlakeScans/dialog_screenshots" if sys.platform == "win32" else "dialog_screenshots")
    ap.add_argument("--theme", default="dark", choices=("dark", "light", "native"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from flakeid.scope.scan import Scanner
    from flakeid.scope.settings import load
    from flakeid.scope.ui import TkUI

    core, chips, targets = build_scene()
    positions = {"10x": 2, "100x": 5, "domain": 3}
    sfile = TMP / "scan_settings.json"
    sfile.write_text(json.dumps({"objective_positions": positions}), encoding="utf-8")
    scans = TMP / "scans"
    stop = threading.Event()
    threading.Thread(target=fake_analysis, args=(scans, targets, stop), daemon=True).start()

    class DocUI(TkUI):
        def mock_hook(self, scanner, name, *args):          # the simulated user's hands (field stop, dark, objective)
            fn = getattr(core, name, None)
            if fn:
                fn(*args)

        def ask_dir(self, title, initial):                  # the Windows folder picker is not photographed
            d = TMP / "results"
            d.mkdir(exist_ok=True)
            return str(d)

    ui = DocUI("FlakeIdentifier scan", theme=a.theme)
    director = Director(ui, core, out)
    first = {"v": True}

    def factory():
        s = load(str(sfile), ask_mode=True, settings_file=str(sfile), local_root=str(scans), flats_dir=str(TMP / "flats"),
                 local_analysis=False, z_settle_ms=0.0, verbose=False, target_exposure_ms=8.0, flat_frames=9,
                 flat_dark_frames=4, domain_imaging="ask", targets_wait_s=600.0, analysis_wait_h=0.01)
        sc = Scanner(core, s, ui)
        sc.nose = director.nose = DocNose(core, positions)
        if first["v"]:
            sc.nose.force = positions["100x"]               # the first job starts on the wrong objective: the window appears
            first["v"] = False
        return sc

    def finish():                                           # closes the program once the last job has ended
        if str(ui.stop_btn.cget("text")) == "Close":
            director.write_index()
            stop.set()
            print(f"\n{len(director.shots)} pictures in {out}  (index.html shows them all)")
            ui.root.destroy()
            return
        ui.root.after(1000, finish)
    ui.root.after(5000, finish)
    print(f"taking the pictures (temporary files in {TMP}); do not use the mouse or keyboard ...")
    ui.run(factory)
    shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
