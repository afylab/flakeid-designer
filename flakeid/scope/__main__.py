"""Run the chip scan from Python on the microscope PC (Micro-Manager GUI closed).

    python -m flakeid.scope --mm "C:\\Program Files\\Micro-Manager-2.0" --cfg "C:\\Program Files\\Micro-Manager-2.0\\autoscan_withir.cfg" \\
        [--settings scan_settings.json] [--mode multi|chip|targets|flat] [--save-to Z:\\...]

Settings (exposures, focus, analysis, ...) are those of tools/mm_chip_scan.bsh (flakeid/scope/settings.py); a JSON
file overrides them, the options here override the file.  Multi mode (default): the pieces are added in a live viewer,
scanned to a local folder (local_root, C:/FlakeScans), analysed on this PC (python -m flakeid.live in the background),
imaged at 100x, and the results (viewers, target lists, logs, 100x and domain images; not the raw tiles) copied to the
folder chosen at the start (or --save-to).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def launched_directly():
    """Started from the desktop / taskbar shortcut (pythonw.exe, no console, not through run_flakeid.bat): set up what
    run_flakeid.bat does (Anaconda off the PATH, the original PATH kept for the camera's libraries), and send the
    output that would go to the console to a log file.  Returns the log file, or None."""
    import os
    if sys.platform != "win32" or os.environ.get("FLAKEID_ORIG_PATH"):
        return None
    exe = Path(sys.executable).parent
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    os.environ["FLAKEID_ORIG_PATH"] = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join([str(exe), str(exe / "Scripts"), os.path.join(sysroot, "system32"), sysroot])
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT)
    logp = None
    if sys.stdout is None or sys.stderr is None:         # pythonw: no console
        for d in (Path("C:/FlakeScans"), Path(os.environ.get("TEMP", "."))):
            try:
                d.mkdir(parents=True, exist_ok=True)
                logp = d / "flakeid_console_log.txt"
                f = open(logp, "a", encoding="utf-8", buffering=1)
                import time
                f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} started from the shortcut\n")
                sys.stdout = sys.stderr = f
                break
            except OSError:
                continue
    return logp


def enable_crash_log():
    """A crash of Python itself (an access violation in a device driver or in Tk: the program vanishes without an error
    message) writes the stack of every thread to <C:/FlakeScans>/crash_log.txt (faulthandler; 2026-10-27, Python closed
    when Pause was pressed).  Returns the file, or None."""
    import faulthandler
    import os
    import time
    if sys.platform != "win32":
        return None
    for d in (Path("C:/FlakeScans"), Path(os.environ.get("TEMP", "."))):
        try:
            d.mkdir(parents=True, exist_ok=True)
            f = open(d / "crash_log.txt", "a", encoding="utf-8", buffering=1)
            f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} started (pid {os.getpid()}); a crash's stacks follow\n")
            faulthandler.enable(file=f, all_threads=True)
            enable_crash_log.file = f                    # (kept open for the program's life)
            return d / "crash_log.txt"
        except OSError:
            continue
    return None


def load_core(mm, cfg, log=print, corelog=None):
    """Micro-Manager with the configuration cfg loaded (every device opened and started).  corelog: Micro-Manager's own
    log goes to this file (started afresh), with its debug messages while the devices start (which one it waits for)."""
    import glob
    import os
    try:
        from pymmcore_plus import CMMCorePlus
        core = CMMCorePlus()
    except ImportError:
        import pymmcore
        core = pymmcore.CMMCore()
    if not glob.glob(os.path.join(mm, "mmgr_dal_*.dll")) and not glob.glob(os.path.join(mm, "libmmgr_dal_*")):
        alt = os.path.dirname(os.path.abspath(cfg))
        if glob.glob(os.path.join(alt, "mmgr_dal_*.dll")):
            log(f"(no device drivers in {mm}; using {alt})")
            mm = alt
    # Windows must find the libraries the drivers load (e.g. the XIMEA API): the Micro-Manager folder, C:\XIMEA, the
    # original PATH without Anaconda (see tools/win7/mm_python_test.py)
    sys.path.insert(0, str(ROOT / "tools" / "win7"))
    try:
        from mm_python_test import prepare_dll_search
        cwd = os.getcwd()
        prepare_dll_search(mm)
        os.chdir(cwd)
    except Exception as e:
        log(f"(library search path not prepared: {e})")
    core.setDeviceAdapterSearchPaths([mm])
    if corelog:
        try:
            core.setPrimaryLogFile(str(corelog), True)
            core.enableDebugLog(True)
        except Exception as e:
            log(f"(Micro-Manager log not redirected: {e})")
    core.loadSystemConfiguration(cfg)
    if corelog:
        try:
            core.enableDebugLog(False)                        # (the scan itself: Micro-Manager's normal messages only)
        except Exception:
            pass
    return core


def cfg_devices(cfg):
    """[(label, library, adapter)] of the devices in a Micro-Manager configuration file (its Device lines)."""
    out = []
    try:
        for ln in Path(cfg).read_text(encoding="utf-8", errors="replace").splitlines():
            q = [x.strip() for x in ln.split(",")]
            if len(q) >= 4 and q[0] == "Device":
                out.append((q[1], q[2], q[3]))
    except OSError:
        pass
    return out


CONNECT_HINT_S = 45.0          # the start-up window explains what to do when the microscope has not answered by then


def connect_window(load, corelog, cfg, hint_s=CONNECT_HINT_S):
    """Run load() (opening the microscope: Micro-Manager starts every device) in a thread behind a window that shows it
    is happening (2026-10-27: after a restart of the PC the program seemed to do nothing for a long time, a second start
    then said it was already running): the time so far, the devices of the configuration and Micro-Manager's latest
    messages (which device it is starting).  After hint_s s it says what usually helps (Stream Motion once after a PC
    restart, then start again) and offers Quit (the stuck device call cannot be stopped from here: the program ends, which
    frees the microscope).  Returns load()'s result, or raises its error."""
    import os
    import threading
    import time
    import tkinter as tk
    box = {}

    def work():
        try:
            box["r"] = load()
        except BaseException as e:                            # (shown by main as the start error)
            box["e"] = e
    th = threading.Thread(target=work, daemon=True, name="microscope start")
    t0 = time.time()
    th.start()
    th.join(1.5)                                              # (a quick start: no window at all)
    if not th.is_alive():
        if "e" in box:
            raise box["e"]
        return box["r"]
    root = tk.Tk()
    root.title("FlakeIdentifier: connecting to the microscope")
    root.configure(padx=16, pady=12)
    tk.Label(root, text="Connecting to the microscope ...", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    devs = cfg_devices(cfg)
    tk.Label(root, text="Micro-Manager is starting the devices of " + str(cfg) + (":\n" + ", ".join(f"{d[0]} ({d[1]})" for d in devs) if devs else ""),
             justify="left", wraplength=760).pack(anchor="w", pady=(4, 0))
    el = tk.Label(root, text="", font=("Segoe UI", 11))
    el.pack(anchor="w", pady=(8, 0))
    tk.Label(root, text="Micro-Manager's latest messages:", fg="#666").pack(anchor="w", pady=(8, 0))
    txt = tk.Text(root, width=110, height=8, font=("Consolas", 9), wrap="none")
    txt.pack(anchor="w")
    hint = tk.Frame(root)
    tk.Label(hint, text=("The microscope has not finished starting.  This often happens after a restart of the PC: the camera "
                         "or the stage is not ready yet (the device in the messages above is the one it waits for).\n\n"
                         "What usually helps: press Quit, start Stream Motion, wait until it shows the live picture, close it, "
                         "then start the scan program again.\n\nIf it keeps happening, send " + str(corelog) + " (it says "
                         "which device does not answer)."), justify="left", wraplength=760, fg="#b00020").pack(anchor="w")
    bf = tk.Frame(hint)
    bf.pack(anchor="w", pady=(8, 0))
    state = {"hint_at": hint_s, "shown": False}

    def keep():
        hint.pack_forget()
        state["shown"], state["hint_at"] = False, time.time() - t0 + 60.0

    def quit_():
        print(f"start-up abandoned after {time.time() - t0:.0f} s (Quit): the microscope did not finish starting; see {corelog}", flush=True)
        os._exit(2)
    tk.Button(bf, text="Keep waiting", command=keep, padx=12).pack(side="left")
    tk.Button(bf, text="Quit", command=quit_, padx=12).pack(side="left", padx=(8, 0))
    root.protocol("WM_DELETE_WINDOW", quit_)
    last = {"n": -1}

    def tail():
        try:
            with open(corelog, "rb") as fh:
                fh.seek(0, 2)
                n = fh.tell()
                if n == last["n"]:
                    return
                last["n"] = n
                fh.seek(max(0, n - 6000))
                lines = [ln for ln in fh.read().decode("utf-8", "replace").splitlines() if ln.strip()][-8:]
        except (OSError, TypeError):                              # (no log file)
            return
        txt.delete("1.0", "end")
        txt.insert("end", "\n".join(ln[-160:] for ln in lines))

    def poll():
        if not th.is_alive():
            root.destroy()
            return
        s_ = time.time() - t0
        el.configure(text=f"{s_:.0f} s so far")
        tail()
        if s_ >= state["hint_at"] and not state["shown"]:
            hint.pack(anchor="w", pady=(10, 0))
            state["shown"] = True
            print(f"start-up: the microscope has not finished starting after {s_:.0f} s (see {corelog})", flush=True)
        root.after(500, poll)
    poll()
    root.mainloop()
    th.join(0.5)
    print(f"microscope started in {time.time() - t0:.0f} s", flush=True)
    if "e" in box:
        raise box["e"]
    return box["r"]


# one scan program at a time: a second one would wait for ever loading the microscope the first one holds (e.g. one
# whose window was closed while a run's export finished in the background, 2026-10-19)
def _lock_file(local_root):
    return Path(local_root or "C:/FlakeScans") / "flakeid_running.json"


def _alive(pid):
    try:
        pid = int(pid)
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x1000, False, pid)                  # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(h, ctypes.byref(code))
            k.CloseHandle(h)
            return bool(ok) and code.value == 259                  # STILL_ACTIVE
        import os
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def running_instance(local_root):
    """The record of another scan program still running ({pid, started, state}), or None."""
    import os
    try:
        d = json.loads(_lock_file(local_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if int(d.get("pid") or 0) == os.getpid() or not _alive(d.get("pid")):
        return None
    return d


def wait_for_exit(pid, timeout_s=120.0):
    """A restart after an update (FLAKEID_WAIT_PID): wait until the program that started this one has ended (it holds
    the microscope until then)."""
    import time
    t_end = time.time() + timeout_s
    while _alive(pid) and time.time() < t_end:
        time.sleep(0.5)


def claim_instance(local_root, state=""):
    """Record this program as the one running (state: what it does, for a second start's message); removed at exit."""
    import atexit
    import os
    import time
    f = _lock_file(local_root)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(dict(pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M"), state=state)), encoding="utf-8")
    except OSError:
        return

    def release():
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("pid") == os.getpid():
                f.unlink()
        except (OSError, ValueError):
            pass
    atexit.register(release)


def main(argv=None):
    logp = launched_directly()
    enable_crash_log()
    try:
        return _main(argv)
    except SystemExit:
        raise
    except BaseException as e:
        import traceback
        traceback.print_exc()
        if logp is not None:                             # (no console to show it: a message box)
            try:
                import tkinter
                from tkinter import messagebox
                r = tkinter.Tk()
                r.withdraw()
                messagebox.showerror("FlakeIdentifier", f"The scan program could not start:\n\n{type(e).__name__}: {e}\n\n"
                                                        f"Details: {logp}")
                r.destroy()
            except Exception:
                pass
        return 1


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flakeid.scope", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm", help="Micro-Manager install folder (device drivers)")
    ap.add_argument("--cfg", help="hardware configuration (.cfg)")
    ap.add_argument("--settings", help="JSON file overriding settings (flakeid/scope/settings.py)")
    ap.add_argument("--mode", choices=["multi", "chip", "targets", "flat", "capture", "retarget", "domaintest", "device", "offsets",
                                       "scalecal"])
    ap.add_argument("--run-dir", help="local folder for the images and run files (default: local_root, C:/FlakeScans)")
    ap.add_argument("--name", help="run / piece name")
    ap.add_argument("--results", help="results folder of the analysis on this PC (results_dir)")
    ap.add_argument("--save-to", "--share", dest="share", help="where the results are copied at the end (default: asked)")
    ap.add_argument("--targets", help="targets mode: the targets CSV")
    ap.add_argument("--focus", choices=["auto", "manual"])
    ap.add_argument("--no-analysis", action="store_true", help="do not start the analysis on this PC (it runs elsewhere)")
    ap.add_argument("--headless", action="store_true", help="no windows (simulator / tests)")
    ap.add_argument("--print-settings", action="store_true", help="print all settings as JSON and stop")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override one setting (flakeid/scope/settings.py), e.g. --set hundred_save_frames=true; the value is "
                         "read as JSON when it can be (numbers, true/false, lists), else as text")
    a = ap.parse_args(argv)
    from .settings import dump, load
    ov = dict(mode=a.mode, run_dir=a.run_dir, results_dir=a.results, copy_results_to=a.share, targets_csv=a.targets, focus_mode=a.focus)
    if a.name:
        ov["run_name" if (a.mode or "multi") == "multi" else "name"] = a.name
    if a.no_analysis:
        ov["local_analysis"] = False
    for kv in a.set:
        if "=" not in kv:
            ap.error(f"--set {kv}: use NAME=VALUE")
        k_, v_ = kv.split("=", 1)
        try:
            ov[k_.strip()] = json.loads(v_)
        except ValueError:
            ov[k_.strip()] = v_
    ov["settings_file"] = a.settings or ""
    if not a.mode and not a.headless:
        ov["ask_mode"] = True               # (the desktop shortcut: a start screen chooses scan / flat field / capture)
    print(f"flakeid.scope: python {sys.version.split()[0]}, settings {a.settings}", flush=True)
    s = load(a.settings, **ov)
    if a.print_settings:
        print(json.dumps(dump(s), indent=1))
        return 0
    if not (a.mm and a.cfg):
        ap.error("--mm and --cfg are needed")
    from .scan import Scanner
    import os
    wait = os.environ.pop("FLAKEID_WAIT_PID", "")      # (restarted after an update: the old program ends first)
    if wait.isdigit():
        print(f"waiting for the previous program (process {wait}) to end ...", flush=True)
        wait_for_exit(int(wait))
    if not a.headless:
        other = running_instance(s.local_root)
        if other:                                       # (it holds the microscope: loading it again would hang)
            raise RuntimeError(f"the scan program is already running (process {other.get('pid')}, since {other.get('started', '?')}"
                               + (f": {other['state']}" if other.get("state") else "") + ").  Wait until it is done (its window "
                               "says how far; it closes itself after a background export), or end pythonw.exe in the Task "
                               "Manager.")
        claim_instance(s.local_root)
    print(f"loading the microscope ({a.cfg}) ...", flush=True)
    corelog = Path(s.local_root or "C:/FlakeScans") / "mm_startup_log.txt"   # (Micro-Manager's log: which device it waits for)
    try:
        corelog.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        corelog = None
    if a.headless:
        core = load_core(a.mm, a.cfg, corelog=corelog)
    else:                                               # (a window while the devices start: it can take long, or hang)
        core = connect_window(lambda: load_core(a.mm, a.cfg, corelog=corelog), corelog, a.cfg)
    print("microscope ready", flush=True)
    if a.headless:
        from .ui import AutoUI
        ui = AutoUI(core, analysis=s.local_analysis)
        Scanner(core, s, ui).run()
    else:
        from .ui import TkUI
        ui = TkUI(theme=s.ui_theme)
        # one job after the other: each with fresh settings (a run changes its copy: names, folders, ...)
        # (settings read again for every job: a new flat field adopted in one job is used by the next)
        ui.run(lambda: Scanner(core, load(a.settings, **ov), ui))
    return 0


if __name__ == "__main__":
    sys.exit(main())
