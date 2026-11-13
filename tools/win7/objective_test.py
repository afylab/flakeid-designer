"""Does the objective window close by itself when the objective is changed?  A minimal test of the coded nosepiece
reading and the scan program's "Objective: ..." window, without the camera, stage or Micro-Manager.

    C:\\FlakeID\\run_flakeid.bat objtest                 (settings from C:\\FlakeID\\scan_settings.json)
    C:\\FlakeID\\run_flakeid.bat objtest --port COM1 --watch 20

Close the scan program, Olympus Stream and Micro-Manager first (only one program can have the port open).
Steps (each prints PASS / FAIL and what it means):
  1. settings: nosepiece_port and objective_positions are set (without them the scan shows a plain OK window that
     cannot advance by itself);
  2. the port opens and the U-CBS answers;
  3. a second open while the first is still open (what a previous job that did not close the port causes): expected
     to FAIL on Windows; it shows the message to look for in the scan log;
  4. watch: turn the nosepiece through its positions for --watch seconds; every reading is printed with its time
     (None = between positions or no answer);
  5. the scan program's own window (the same code, from a worker thread as in a scan): it asks for the other
     objective; turn to it, and the window must close by itself about 1 s after it is in.
Everything also goes to objective_test_log.txt in the current folder: send it.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LOG = open("objective_test_log.txt", "w", encoding="utf-8")


def say(msg=""):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default="C:/FlakeID/scan_settings.json")
    ap.add_argument("--port", help="override nosepiece_port")
    ap.add_argument("--watch", type=float, default=20.0, help="seconds to watch while you turn the nosepiece")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds the objective window waits")
    a = ap.parse_args(argv)
    say(f"objective test, {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. settings
    st = {}
    try:
        st = json.loads(Path(a.settings).read_text(encoding="utf-8"))
        say(f"settings: {a.settings}")
    except Exception as e:
        say(f"settings: {a.settings} not read ({e})")
    port = a.port or st.get("nosepiece_port") or ""
    positions = st.get("objective_positions") or {}
    say(f"  nosepiece_port = {port!r}, objective_positions = {positions}")
    if not port or not positions.get("10x") or not positions.get("100x"):
        say("1 FAIL: without nosepiece_port and the 10x / 100x positions the scan cannot read the objective: it shows a window "
            "with OK that you must press (no auto-advance). Add them to scan_settings.json, e.g.\n"
            '   "nosepiece_port": "COM1", "objective_positions": {"10x": 2, "100x": 5, "domain": 3}')
        if not port:
            return 1
        positions = {"10x": 2, "100x": 5, **positions}
    else:
        say("1 PASS: settings")

    # 2. open the port
    from flakeid.scope.nosepiece import Nosepiece
    try:
        t0 = time.time()
        nose = Nosepiece(port)
        p0 = nose.position()
        say(f"2 PASS: {nose.unit} on {port}, position {p0} ({time.time() - t0:.2f} s)")
    except Exception as e:
        say(f"2 FAIL: {type(e).__name__}: {e}")
        say("   'Access is denied' / 'PermissionError' / 'could not open port': another program has the port (the scan program, "
            "Olympus Stream, Micro-Manager with a configuration using it). No answer: wrong port, or the box is off.")
        return 1
    names = {v: k for k, v in positions.items()}

    # 3. a second open while the first is open
    try:
        n2 = Nosepiece(port)
        n2.close()
        say("3 note: a second open worked here (on Windows it normally fails)")
    except Exception as e:
        say(f"3 expected failure of a second open: {type(e).__name__}: {e}")
        say("   If the scan log of a job shows 'WARNING: nosepiece not read (...)' with this message, the port was still held by an "
            "earlier job (fixed 2026-10-04: every job now closes it).")

    # 4. watch
    say(f"4 watch: turn the nosepiece through its positions now ({a.watch:g} s) ...")
    t0, last, n, n_none, slow = time.time(), "start", 0, 0, 0
    while time.time() - t0 < a.watch:
        tq = time.time()
        p = nose.position()
        dt = time.time() - tq
        n += 1
        n_none += p is None
        slow += dt > 0.5
        if p != last:
            say(f"   {time.time() - t0:6.2f} s: position {p} ({names.get(p, 'between / no answer' if p is None else 'not in objective_positions')}), "
                f"query {dt * 1000:.0f} ms")
            last = p
        time.sleep(0.3)
    say(f"   {n} readings, {n_none} None, {slow} slower than 0.5 s")
    say("4 " + ("PASS" if n_none < n else "FAIL: no position read at all") + ": readings")

    # 5. the scan program's window, opened from a worker thread as in a scan
    from flakeid.scope.ui import TkUI
    want_name = "100x" if names.get(nose.position()) != "100x" else "10x"
    want = positions[want_name]
    polls = []

    class Logged:                               # the nosepiece as the window sees it (every reading recorded)
        def position(self):
            p = nose.position()
            polls.append((time.time(), p, threading.current_thread().name))
            return p

    class FakeScanner:
        nose = Logged()
        s = type("S", (), {"objective_positions": positions})()

        def objective_name(self, pos):
            return names.get(pos, "unknown" if pos is None else f"position {pos}")

    ui = TkUI("objective test")
    ui.log(f"Turn the nosepiece to {want_name} (position {want}): the window should close by itself ~1 s after it is in.")
    res = {}

    def work():
        t0 = time.time()
        res["ok"] = ui.wait_objective(FakeScanner(), want_name, want, f"Turn to the {want_name} objective (position {want}). This window "
                                      "should close by itself about 1 s after it is in.")
        res["t"] = time.time() - t0
        ui._post(ui.root.destroy)

    threading.Thread(target=work, daemon=True, name="worker").start()
    ui.root.after(int(a.timeout * 1000), ui.root.destroy)
    ui.root.mainloop()
    t_in = next((t for t, p, _ in polls if p == want), None)
    if "ok" not in res:
        say(f"5 FAIL: the window did not close within {a.timeout:g} s")
    elif res["ok"] and t_in is not None:
        say(f"5 PASS: the window closed by itself {polls[-1][0] - t_in:.1f} s after the objective was in ({res['t']:.1f} s in all)")
    else:
        say(f"5 window closed by a button (result {res['ok']}), not by itself")
    t0p = polls[0][0] if polls else 0
    say(f"   {len(polls)} readings by the window (thread {polls[0][2] if polls else '-'}); changes:")
    last = "start"
    for t, p, _ in polls:
        if p != last:
            say(f"   {t - t0p:6.2f} s: {p}")
            last = p
    if "ok" not in res and t_in is not None:
        say("   the position was read but the window did not close: a problem in the window's polling (send this log)")
    elif t_in is None:
        say(f"   position {want} was never read while the window was open")
    nose.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
