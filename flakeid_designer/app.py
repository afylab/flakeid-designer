"""vdW Stack Designer as a desktop app: the designer's local server and page in a native window (pywebview: the
system's web engine, WebKit on macOS, WebView2 on Windows 10/11), opened and closed from the Dock / taskbar like any
app.  The page, the libraries (scan-results folders), designs and decisions are the same as in the browser version.

    python -m flakeid_designer.app            (or the packaged app: tools/designer_app/)
"""
from __future__ import annotations

import os
import socket
import re
import sys
import threading
import time
from pathlib import Path

try:
    from . import __main__ as D
except ImportError:                                  # (run as a script / from the packaged app)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import flakeid_designer.__main__ as D            # type: ignore

TITLE = "vdW Stack Designer"
APP_PORT = 8766                                      # (the browser version uses 8765)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Api:
    """Called from the page (window.pywebview.api): what a browser does but an app window does not."""

    # (only methods here: pywebview walks every public attribute of this object to build the page's bridge, and on
    # Windows walking into the window's native .NET object froze the app; private names, "_...", are not walked)
    def __init__(self):
        self._window = None

    def save_file(self, name, text):
        """A native Save dialog for the page's exports (catalog CSV, design JSON); returns the path or ''."""
        import webview
        r = self._window.create_file_dialog(webview.SAVE_DIALOG, save_filename=name)
        if not r:
            return ""
        path = r if isinstance(r, str) else r[0]
        Path(path).write_text(text, encoding="utf-8")
        return path

    def open_external(self, url):
        import webbrowser
        if str(url).startswith(("http://", "https://")):
            webbrowser.open(url)


def versions():
    """The versions that matter for the window and the builds, for the log (a PC's problems differ by version: 2026-10-20,
    Python 3.14 on the stacker PC)."""
    out = [f"Python {sys.version.split()[0]}"]
    for name in ("webview", "clr_loader", "pythonnet", "numpy", "cv2", "scipy", "skimage"):
        try:
            import importlib.metadata as md
            dist = {"webview": "pywebview", "cv2": "opencv-python-headless", "skimage": "scikit-image"}.get(name, name)
            out.append(f"{dist} {md.version(dist)}")
        except Exception:
            pass
    return ", ".join(out)


def watchdog(port, win, every=15.0, slow=5.0, quiet=120.0):
    """Every `every` s: does the designer's server answer, and does the window run a line of script, within `slow` s?
    If not, what every thread is doing goes to the log (at most every `quiet` s): the window froze on the stacker PC
    while it built runs (2026-10-20) and nothing in the log said why."""
    import traceback
    import urllib.request
    probe = {"busy": False}
    last = [0.0]

    def ui_probe(done):
        try:
            win.evaluate_js("1")
        except Exception:
            pass
        done.set()
        probe["busy"] = False

    while True:
        time.sleep(every)
        problems = []
        t0 = time.time()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=slow * 4) as r:
                r.read()
            dt = time.time() - t0
            if dt > slow:
                problems.append(f"the designer's server took {dt:.1f} s to answer")
        except Exception as e:
            problems.append(f"the designer's server did not answer in {time.time() - t0:.0f} s ({type(e).__name__})")
        if not probe["busy"]:                         # (one window probe at a time: a hung window keeps it)
            probe["busy"] = True
            done = threading.Event()
            threading.Thread(target=ui_probe, args=(done,), daemon=True, name="window probe").start()
            if not done.wait(slow):
                problems.append(f"the window did not run a line of script within {slow:g} s")
        else:
            problems.append("the window has not run the previous line of script yet")
        if problems and time.time() - last[0] > quiet:
            last[0] = time.time()
            names = {t.ident: t.name for t in threading.enumerate()}
            lines = [f"{time.strftime('%H:%M:%S')} WATCHDOG: " + "; ".join(problems)]
            try:
                lines.append(f"  builds: {D.BUILDER.status()}; free memory {D.free_memory_gb()} GB")
            except Exception:
                pass
            for tid, frame in sys._current_frames().items():
                lines.append(f"  --- thread {names.get(tid, tid)}")
                lines += ["    " + ln.rstrip() for ln in traceback.format_stack(frame)[-8:]]
            sys.stderr.write("\n".join(lines) + "\n")


def running_designer(port):
    """Is a designer app already serving on this port?  -> its version, or None."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=3) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d.get("version") or "?" if d.get("app") else None
    except Exception:
        return None


def storage_folder(base):
    """The window's browser storage folder (WebView2 / WebKit).  On Windows a folder still held by an earlier
    session's WebView2 processes (an app ended in Task Manager leaves them running) made the next window hang with a
    blank, frozen page (2026-10-20, the stacker PC): a locked one is not used; this session gets a folder of its own
    (its page's remembered state, e.g. the runs ticked, starts empty) and the log says so."""
    base = Path(base)
    if sys.platform != "win32":
        return base
    lock = base / "EBWebView" / "lockfile"
    if not lock.exists():
        return base
    try:
        lock.unlink()                                # (a lock from a session that has ended: removed, folder reused)
        return base
    except OSError:
        pass
    alt = base.with_name(base.name + f"_{os.getpid()}")
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} the window's storage {base} is still in use (an earlier session's "
                     f"WebView2 processes: end them in Task Manager, or restart the PC); this session uses {alt}\n")
    return alt


WEBVIEW2_FEATURES_OFF = ["CalculateNativeWinOcclusion"]
WEBVIEW2_ARGS = ["--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
                 "--disable-background-timer-throttling"]


def merge_browser_args(base, features_off, args):
    """Chromium takes only the last --disable-features=: one merged list (pywebview sets ElasticOverscroll)."""
    toks, feats = [], []
    for t in (base or "").split():
        if t.startswith("--disable-features="):
            feats += [f for f in t.split("=", 1)[1].split(",") if f]
        else:
            toks.append(t)
    for f in features_off:
        if f not in feats:
            feats.append(f)
    for t in args:
        if t.startswith("--disable-features="):
            feats += [f for f in t.split("=", 1)[1].split(",") if f and f not in feats]
        elif t not in toks:
            toks.append(t)
    return " ".join(([f"--disable-features={','.join(feats)}"] if feats else []) + toks)


def webview2_flags(cfg):
    """Our browser arguments added to the ones pywebview gives WebView2 (it sets them itself: the
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS variable is ignored), just after it makes the creation properties and
    before the browser starts."""
    extra = list(WEBVIEW2_ARGS) + str(cfg.get("app_browser_args") or "").split()
    if cfg.get("app_gpu") is False:
        extra.append("--disable-gpu")
    try:
        import webview.platforms.edgechromium as E
    except Exception as e:
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} window: browser arguments not added ({type(e).__name__}: {e})\n")
        return
    if getattr(E.EdgeChrome, "_flakeid_args", False):
        return
    orig = E.EdgeChrome.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        try:
            p = self.webview.CreationProperties
            p.AdditionalBrowserArguments = merge_browser_args(str(p.AdditionalBrowserArguments or ""), WEBVIEW2_FEATURES_OFF, extra)
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} window: WebView2 arguments {p.AdditionalBrowserArguments}\n")
        except Exception as e:
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} window: browser arguments not added ({type(e).__name__}: {e})\n")
    E.EdgeChrome.__init__ = init
    E.EdgeChrome._flakeid_args = True


def main():
    import webview
    cfg = D.load_config()
    roots, data, library = D.start_library(cfg)
    store = D.Store(roots, data, library, added=cfg.get("added_runs"))
    threading.Thread(target=store.scan, daemon=True).start()
    # a fixed port: the page keeps a few things in its browser storage (runs ticked, the open stack), which belong to
    # the address; another port only when this one is taken (then those start empty)
    port = APP_PORT
    try:
        t = socket.socket(); t.bind(("127.0.0.1", port)); t.close()
    except OSError:
        other = running_designer(port)
        if other:
            # the designer is already open (perhaps hidden behind other windows, or its window stuck): no second window
            # (two windows on one storage folder hang on Windows); the running one is opened in the browser instead
            import webbrowser
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} the designer ({other}) is already running on port {port}: "
                             "opened in the browser instead of a second window\n")
            webbrowser.open(f"http://127.0.0.1:{port}/")
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.user32.MessageBoxW(None, f"{TITLE} is already running, so it was opened in your web browser.\n\n"
                                                           "If its own window is stuck, close it in Task Manager and start the "
                                                           "designer again.", TITLE, 0x40)
                except Exception:
                    pass
            return 0
        port = free_port()
    srv = D.make_server(port, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    api = Api()
    win = webview.create_window(f"{TITLE}", f"http://127.0.0.1:{port}/", js_api=api, width=1440, height=900,
                                min_size=(900, 600), background_color="#f4f6f5", text_select=True)
    api._window = win

    def native_pick(kind, prompt, start, exts):
        # pywebview takes "<letters and spaces> (*.a;*.b)" only: "(GDS)" in the prompt made every GDS pick fail
        label = " ".join(re.sub(r"[^\w ]+", " ", prompt).split()) or "Files"
        types = (f"{label} ({';'.join('*.' + e for e in exts)})", "All files (*.*)") if exts else ()
        try:
            r = win.create_file_dialog(webview.FOLDER_DIALOG if kind == "folder" else webview.OPEN_DIALOG,
                                       directory=start or "", allow_multiple=False, file_types=types)
        except ValueError:                               # (a filter this pywebview does not take: then without one)
            r = win.create_file_dialog(webview.FOLDER_DIALOG if kind == "folder" else webview.OPEN_DIALOG,
                                       directory=start or "", allow_multiple=False)
        if not r:
            return None
        return str(r if isinstance(r, str) else r[0])
    D.NATIVE_PICK = native_pick
    # debugging the window (Windows: it froze on the stacker PC when Plan / Stack was opened, 2026-10-20, and the page's
    # stack could not be seen): FLAKEID_DESIGNER_DEBUG=1 or setting "app_debug": the developer tools in the window (F12,
    # right-click > Inspect) and port 9222 for edge://inspect.  "app_gpu": false: WebView2 without the graphics card.
    # (pywebview sets WebView2's browser arguments itself, so WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS is ignored; its
    # remote-debugging setting is the one way to pass more)
    debug = bool(os.environ.get("FLAKEID_DESIGNER_DEBUG")) or bool(cfg.get("app_debug"))
    if sys.platform == "win32":
        # the page froze on the stacker PC (Plan / Stack opened; not with the developer tools attached, which keep a
        # page awake, 2026-10-20): WebView2 (Chromium) thought the window hidden and froze the page (no script, no
        # drawing; scrolling still worked).  Its window-occlusion check and the backgrounding of hidden pages off;
        # setting "app_browser_args": more flags (e.g. to try one without a rebuild)
        webview2_flags(cfg)
    if debug:
        webview.settings["REMOTE_DEBUGGING_PORT"] = 9222
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} window: developer tools on (port 9222)\n")
    threading.Thread(target=watchdog, args=(port, win), daemon=True, name="watchdog").start()
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    webview.settings["ALLOW_DOWNLOADS"] = True
    # window icon: Windows takes only .ico files (a .png stops the window from opening) and already has the .exe's
    # own icon; macOS / Linux get the png
    icon = None if sys.platform == "win32" else D.HERE / "static" / "designer_256.png"
    webview.start(icon=str(icon) if icon is not None and icon.is_file() else None, private_mode=False, debug=debug,
                  storage_path=str(storage_folder(Path.home() / ".flakeid_designer_app")))
    srv.shutdown()                                   # the window closed: the app quits
    return 0


def run():
    """main(), with what goes wrong recorded: a packaged app has no console (on Windows sys.stdout / sys.stderr are
    None), so output goes to the designer's log file, and a failure to start shows a message box (Windows)."""
    log = getattr(D, "LOG", Path.home() / ".flakeid_designer_log.txt")
    if sys.stderr is None or sys.stdout is None or not getattr(sys.stderr, "isatty", lambda: False)():
        try:
            f = open(log, "a", encoding="utf-8", buffering=1)
            f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {TITLE} (app) {getattr(D, 'VERSION', '')}, Python {sys.version.split()[0]}, {sys.platform}\n")
            sys.stdout = sys.stderr = f
            try:                                          # (a crash inside a library: where it was, in the log)
                import faulthandler
                faulthandler.enable(file=f, all_threads=True)
            except Exception:
                pass
        except OSError:
            pass
    try:
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} versions: {versions()}; {os.cpu_count()} cores, "
                         f"{D.free_memory_gb() or 0:.1f} GB memory free\n")
    except Exception:
        pass
    try:
        return main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        try:
            sys.stderr.write(tb)
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, f"{TITLE} could not start:\n\n{tb.strip().splitlines()[-1]}\n\n"
                                                       f"Details are in {log}", TITLE, 0x10)
            except Exception:
                pass
        return 1


def run_helper(name, args):
    """--helper NAME ARGS: a job the designer runs in a separate process, without a window (the packaged app has no
    python to run the scripts with).  marks: the KLayout export (marks_to_klayout.py)."""
    if sys.platform == "win32" and (sys.stdout is None or sys.stderr is None):
        # started from a command window (not by the designer, which reads its output): print there (the app has no
        # console of its own), so a build run by hand shows its progress bar instead of looking frozen
        try:
            import ctypes
            if ctypes.windll.kernel32.AttachConsole(-1):
                sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
                sys.stderr = sys.stdout
                sys.stdout.write("\n")
        except Exception:
            pass
    try:
        sys.stdout.reconfigure(line_buffering=True)   # (the designer shows the export's progress as it prints it)
    except Exception:
        pass
    if name == "marks":
        try:
            from flakeid_designer import marks_to_klayout as MK
        except ImportError:
            import marks_to_klayout as MK              # type: ignore
        sys.argv = ["marks_to_klayout"] + list(args)
        r = MK.main()
        return r if isinstance(r, int) else 0
    if name == "bundle":                              # (a run's designer files, as the designer's builder makes them: tests)
        D.builder_main()
        mdb = __import__("make_designer_bundle")
        r = mdb.main(list(args))
        return r if isinstance(r, int) else 0
    if name == "crop":                                # (offline stacks: flake pictures cut out of the run's sheets)
        try:
            from flakeid_designer import offline as OFL
        except ImportError:
            import offline as OFL                      # type: ignore
        return OFL.crop_main(list(args))
    print(f"unknown helper {name!r}")
    return 2


def check_builder():
    """--check-builder: can this copy build runs' designer files (the packaged builder and its modules load)?"""
    try:
        D.builder_main()
        for m in ("make_catalog_100x", "terraces100", "flakeid.screen100", "flakeid.branch", "flakeid.afm"):
            __import__(m)
        import numpy as _np
        from flakeid.branch import chip_threshold
        chip_threshold(_np.r_[_np.linspace(-3, -1, 40), _np.linspace(2, 4, 40)])   # (hBN thin / thick: numpy, no scikit-learn)
        # worker processes (the build's hBN terraces run in them): started as the build starts them
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        mdb = __import__("make_designer_bundle")
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as ex:
            got = list(ex.map(mdb._ping, [1, 2, 3]))
        if got != [2, 4, 6]:
            raise RuntimeError(f"worker processes answered {got}")
        # imported flakes: the thickness estimate (optical model, camera profile) and the auto outline, on a made-up picture
        import numpy as np
        try:
            from . import manual as MN
        except ImportError:
            import manual as MN                  # type: ignore
        im = np.full((200, 300, 3), (181, 139, 202), np.uint8)
        im[60:140, 90:210] = (240, 233, 162)
        o = MN.auto_outline(im, [150, 100])["outline"]
        e = MN.estimate(im, o, "hbn", 285.0)
        if not e["candidates"]:
            raise RuntimeError("the thickness estimate gave no candidates")
        from flakeid.afm import analyse_auto           # (AFM lists: the images' step heights)
        z = np.random.default_rng(0).normal(0, 0.3, (128, 128))
        z[:, 64:] += 20.0
        st = [L.get("step_nm") for L in analyse_auto(z, 100.0)["levels"] if "step_nm" in L]
        if not st or abs(max(st) - 20.0) > 1.0:
            raise RuntimeError(f"the AFM step analysis gave {st} for a 20 nm step")
        print("builder OK:", D.builder_root(), "(worker processes OK; imported-flake outline and thickness OK; AFM steps OK)")
        return 0
    except Exception as e:
        print("builder NOT usable:", type(e).__name__, e)
        return 1


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()             # (the app's own worker processes: the build's hBN terraces)
    if len(sys.argv) > 2 and sys.argv[1] == "--helper":
        sys.exit(run_helper(sys.argv[2], sys.argv[3:]))
    if "--check-builder" in sys.argv:
        sys.exit(check_builder())
    sys.exit(run())
