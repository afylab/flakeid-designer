"""Simulated device photos (start screen "Device photos"): 100x, 50x, 10x, 5x and one extra 100x picture, then the
photos, terraces and page (flakeid.devicephoto), and where they go:

1. no stack folder (older settings): <device>_<date>/ copied to device_photos_to (as before);
2. a stack folder (out/device/stacks/DQD8 with DQD8.stack and 01_stacking/), step "contacts": 02_contacts/ with the
   file names and step.json of docs/stack_workflow.md ("Stack folders"), the raw TIFFs deleted;
3. the same folder, step "SETs": 03_SETs/;
4. a stack folder that cannot be reached (share down): all kept locally, then
   python -m flakeid.scope.devicephotos --resend <working folder> <stack folder>  ->  04_tgetch/;
5. the start screen (Tk): its Device photos box filled (a new folder, made on Start) returns the stack folder, step,
   description and the device name (skip with --no-tk).

    python tools/mm_mock/run_mock_device.py [--image a_100x.jpg] [--no-tk]
"""
from __future__ import annotations

import csv
import glob
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.scope.mock import MockCore  # noqa: E402
from flakeid.scope.scan import Scanner  # noqa: E402
from flakeid.scope.settings import load  # noqa: E402
from flakeid.scope.ui import AutoUI  # noqa: E402


def run_once(out, demo, step=None, stack=None, legacy=False):
    """One simulated device-photos job; returns (scanner, previews shown, the working folder)."""
    core = MockCore(1.09, x0=50000.0, y0=20000.0)
    if demo:
        core.add_patch(cv2.cvtColor(cv2.imread(demo[2]), cv2.COLOR_BGR2RGB), 50000.0, 20000.0)
        core.patch_um_per_px = 1.0894 / 9.54
        core.setXYPosition("XY", 50000.0 + core.off100[0], 20000.0 + core.off100[1])   # (the 100x view on the patch)
    (out / "settings.json").write_text(json.dumps({"objective_positions": {"10x": 2, "100x": 5}}))
    kw = dict(device_photos_to=str(out / "share")) if legacy else dict(
        device_stack_dir=str(stack), device_step=step, device_step_description=f"{step}: what was done\nsecond line")
    s = load(None, mode="device", device_name="mockdev A", local_root=str(out / "local"), settings_file=str(out / "settings.json"),
             z_settle_ms=0.0, verbose=False, device_frames=4, device_shift_frames=2,
             flat10=str(ROOT / "flats/flat_10x_3ms.npz"), flat100=("285=" + str(ROOT / "flats/flat_100x_285nm.npz"),), **kw)
    ui = AutoUI(core, analysis=False)
    asked = {"n": 0}
    base_confirm = ui.confirm

    def confirm(title, msg, buttons=("OK", "Cancel")):          # one extra 100x picture, then Done
        if title == "Device photos" and buttons and buttons[0] == "Done":
            asked["n"] += 1
            return 1 if asked["n"] == 1 else 0
        return base_confirm(title, msg, buttons)
    ui.confirm = confirm
    previews = []

    def ask_preview(scanner, title, msg, path, buttons=("Keep", "Retake", "Stop")):   # the first photo retaken once
        previews.append((title, Path(path).name, msg))
        return 1 if len(previews) == 1 else 0
    ui.ask_preview = ask_preview
    # the simulator's objective offsets (as the objective offsets tool writes them): the stage follows objective changes
    s.objective_offsets = {f"{m}x": dict(dx=float(o[0] - core.off100[0]), dy=float(o[1] - core.off100[1]),
                                         dz=float(z - core.zoff100))
                           for m, o, z in ((100, core.off100, core.zoff100), (50, core.off_obj[50], core.zoff_obj[50]),
                                           (10, np.zeros(2), 0.0), (5, core.off_obj[5], core.zoff_obj[5]))}
    before = set((out / "local" / "device_photos").glob("*")) if (out / "local" / "device_photos").is_dir() else set()
    sc = Scanner(core, s, ui)
    sc.run()
    if sc.logf:
        sc.logf.close()
        sc.logf = None
    new = sorted(set((out / "local" / "device_photos").glob("*")) - before)
    return sc, previews, (new[-1] if new else None)


def check_step(stack, nn, step, d, label):
    """The step folder NN_<step> of the stack folder: names, step.json, page links; the raw TIFFs gone from d."""
    dev = "DQD8"
    sd = stack / f"{nn:02d}_{step}"
    errs = []
    if not sd.is_dir():
        print(f"{label}: no {sd.name} in {stack} ({sorted(p.name for p in stack.iterdir())})")
        return False
    names = sorted(p.name for p in sd.iterdir())
    b = f"{dev}_{step}"
    want = [f"{b}_100x.jpg", f"{b}_50x.jpg", f"{b}_10x.jpg", f"{b}_5x.jpg", f"{b}_100x_2.jpg", f"{b}.html", "step.json"]
    errs += [f"missing {n}" for n in want if n not in names]
    rec = json.loads((sd / "step.json").read_text(encoding="utf-8")) if (sd / "step.json").is_file() else {}
    keys = {"format", "name", "description", "at", "by", "device", "oxide_nm", "source", "photos"}
    if set(rec) != keys:
        errs.append(f"step.json keys {sorted(rec)}")
    if rec.get("format") != "vdw-fabstep/1" or rec.get("name") != step or rec.get("device") != dev or \
            rec.get("source") != "scanner device photos" or not rec.get("description", "").startswith(step + ": what was done\nsecond") or \
            not isinstance(rec.get("at"), int) or rec["at"] < 1.7e12 or not rec.get("by") or rec.get("oxide_nm") != 285.0:
        errs.append("step.json fields " + json.dumps({k: rec.get(k) for k in keys - {"photos"}}))
    pk = {"objective", "label", "file", "um_per_px", "w", "h", "terrace_map", "terraces"}
    listed = set()
    for ph in rec.get("photos", []):
        if set(ph) != pk:
            errs.append(f"photo keys {sorted(ph)}")
        listed.add(ph["file"])
        if ph["terrace_map"]:
            listed.add(ph["terrace_map"])
        for t in ph["terraces"]:
            if set(t) != {"n", "area_um2", "contrast", "hc"}:
                errs.append(f"terrace keys {sorted(t)}")
            listed.add(t["hc"])
    objs = [(p["objective"], p["file"]) for p in rec.get("photos", [])]
    if [o for o, _ in objs] != ["100x", "50x", "10x", "5x", "100x"]:
        errs.append(f"objectives {objs}")
    extra = set(names) - listed - {f"{b}.html", "step.json"}
    if extra or listed - set(names):
        errs.append(f"files not in step.json {sorted(extra)}, listed but missing {sorted(listed - set(names))}")
    t100 = [n for n in names if "_T" in n]
    if not any(n.startswith(f"{b}_100x_T") for n in t100) or not any(n.startswith(f"{b}_100x_2_T") for n in t100) \
            or f"{b}_100x_terraces.jpg" not in names or f"{b}_100x_2_terraces.jpg" not in names:
        errs.append(f"terrace files {t100}")
    if any((sd / n).stat().st_size == 0 for n in names):
        errs.append("empty files")
    import re
    page = (sd / f"{b}.html").read_text(encoding="utf-8") if (sd / f"{b}.html").is_file() else ""
    links = set(re.findall(r'(?:src|href)="([^"]+)"', page))
    if not links or any("/" in l or "\\" in l or not (sd / l).is_file() for l in links):
        errs.append(f"page links {sorted(l for l in links if '/' in l or not (sd / l).is_file())[:5]}")
    if "what was done" not in page:
        errs.append("description not on the page")
    tifs = list(d.glob("*.tif")) + list(d.glob("*/*.tif"))
    kept = sorted(p.name for p in d.iterdir())
    if tifs or (d / "photos").exists() or not any(n.endswith("_log.txt") for n in kept):
        errs.append(f"working folder after: {kept}")
    meta = json.loads(next(d.glob("*_meta.json")).read_text())
    if meta.get("sent_to") != str(sd):
        errs.append(f"meta sent_to {meta.get('sent_to')}")
    print(f"{label}: {sd.name}: {len(names)} files ({', '.join(n for n in names if '_T' not in n)}, {len(t100)} terrace frames); "
          f"step.json by {rec.get('by')!r}; working folder kept {kept}" + (f"\n   ERRORS: {errs}" if errs else ""))
    return not errs


def tk_start_screen(out):
    """The start screen's Device photos box, filled and Started by a callback (worker thread asks, as the scan does)."""
    import threading
    try:
        from flakeid.scope.ui import TkUI
        ui = TkUI("device photos start screen test")
    except Exception as e:
        print(f"start screen: no Tk ({type(e).__name__}: {e}): not tested")
        return True
    from tkinter import messagebox, ttk
    import tkinter as tk
    asked = []
    messagebox.askyesno = lambda *a, **k: asked.append(a) or True     # ("make the folder?": yes)
    local = out / "tk_local"
    local.mkdir(parents=True)
    newdir = out / "stacks" / "NewDev 3"
    s = load(None, local_root=str(local), oxide_nm=90.0)
    sc = type("FakeScanner", (), {"s": s})()
    res, seen = {}, {}

    def walk(wd):
        yield wd
        for c in wd.winfo_children():
            yield from walk(c)

    def fill():
        win = ui._start_win
        if win is None:
            ui.root.after(100, fill)
            return
        ws = list(walk(win))
        next(x for x in ws if isinstance(x, ttk.Radiobutton) and x.cget("text") == "Device photos").invoke()
        dv = next(x for x in ws if isinstance(x, ttk.LabelFrame) and "Device photos" in x.cget("text"))
        dws = list(walk(dv))
        ents = [x for x in dws if isinstance(x, ttk.Entry)]
        seen["browse"] = any(isinstance(x, ttk.Button) and x.cget("text").startswith("Browse") for x in dws)
        ents[0].delete(0, "end")
        ents[0].insert(0, str(newdir))
        ents[1].delete(0, "end")
        ents[1].insert(0, "top gate")
        next(x for x in dws if isinstance(x, tk.Text)).insert("1.0", "Ti/Au top gate, 5/45 nm\nlift-off in acetone")
        start = next(x for x in ws if isinstance(x, ttk.Button) and x.cget("text") == "Start")
        ui.root.after(700, lambda: (seen.update(label=[x.cget("text") for x in dws if isinstance(x, ttk.Label)
                                                       and x.cget("text").startswith(("device:", "(folder"))]), start.invoke()))

    def work():
        res["r"] = ui.ask_mode(sc)
        ui._post(ui.root.destroy)
    ui.root.after(300, fill)
    threading.Thread(target=work, daemon=True).start()
    ui.root.after(20000, ui.root.destroy)
    ui.root.mainloop()
    r = res.get("r") or {}
    mem = json.loads((local / "device_photos.json").read_text()) if (local / "device_photos.json").is_file() else {}
    ok = (r.get("mode") == "device" and r.get("stack_dir") == str(newdir) and r.get("step") == "top gate" and newdir.is_dir()
          and r.get("description") == "Ti/Au top gate, 5/45 nm\nlift-off in acetone" and r.get("device_name") == "NewDev 3"
          and r.get("oxide") == 90.0 and len(asked) == 1 and mem.get("stack_dir") == str(newdir) and seen.get("browse"))
    print(f"start screen: returned {r}; folder made {newdir.is_dir()} (asked {len(asked)}x); remembered {mem}; "
          f"device label {seen.get('label')}; Browse button {seen.get('browse')}: {'ok' if ok else 'WRONG'}")
    return ok


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="a 100x picture (JPEG) to put on the simulated chip (default: a demo 100x image if present)")
    ap.add_argument("--no-tk", action="store_true", help="leave out the start-screen (Tk) test")
    a = ap.parse_args(argv)
    out = ROOT / "tools" / "mm_mock" / "out" / "device"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    demo = [x for x in sorted(glob.glob(str(ROOT / "example images/demo_high_magnification/*/*_100x.jpg"))) if " X" not in x]
    if a.image:
        demo = [a.image] * 3
    # 1. no stack folder: the old way (a folder copied to device_photos_to)
    sc, previews, d = run_once(out, demo, legacy=True)
    ok = d is not None
    if ok:
        rows = list(csv.DictReader(open(next(d.glob("*_index.csv")))))
        objs = [r["objective"] for r in rows]
        shifts = [len([f for f in r["shift_files"].split(";") if f]) for r in rows]
        page = next(d.glob("*.html"), None)
        rec = json.loads(next(d.glob("*_photos.json")).read_text()) if page else {}
        copied = (out / "share" / d.name / (page.name if page else "x")).is_file()
        print(f"objectives {objs}, shifted pictures {shifts}, page {page.name if page else None}, copied to the share {copied}")
        for p in rec.get("photos", []):
            print(f"  {p['objective']:>4} {p['label']:<7} {p['w']}x{p['h']} {p['um_per_px']:.4f} um/px, shifts combined {p['shifts_combined']}, "
                  f"terraces {len(p['terraces'])}, substrate found {p['substrate_found']}")
        comb = [p["shifts_combined"] for p in rec.get("photos", []) if p["objective"] == "100x"]
        tifs = sorted(f.name for f in d.glob("*.tif"))
        listed = sorted([r["file"] for r in rows] + [f for r in rows for f in r["shift_files"].split(";") if f])
        print(f"previews shown {len(previews)} (first retaken); TIFFs on disk = listed: {tifs == listed}")
        ok = len(previews) == 6 and tifs == listed and objs == ["100x", "50x", "10x", "5x", "100x"] and shifts == [4, 4, 0, 0, 4] and page is not None and copied and all(comb)
    print("1. no stack folder:", "ok" if ok else "WRONG")
    # 2, 3. a stack folder: steps 02_contacts, 03_SETs
    stack = out / "stacks" / "DQD8"
    (stack / "01_stacking").mkdir(parents=True)
    (stack / "DQD8.stack").write_text(json.dumps({"format": "vdw-stack/1", "name": "DQD8"}))
    for nn, step in ((2, "contacts"), (3, "SETs")):
        sc, previews, d = run_once(out, demo, step=step, stack=stack)
        ok &= d is not None and d.name.startswith(f"DQD8_{step}_") and check_step(stack, nn, step, d, f"{nn}. step {step}")
    # 4. the stack folder unreachable (share down): kept locally; then --resend
    down = out / "share_down" / "DQD8"
    sc, previews, d = run_once(out, demo, step="tgetch", stack=down)
    kept = d is not None and len(list(d.glob("*.tif"))) == 17 and (d / "photos").is_dir() and not down.exists()
    logt = next(d.glob("*_log.txt")).read_text(encoding="utf-8") if d else ""
    said = "--resend" in logt and "could not be written" in logt
    print(f"4. stack folder down: everything kept locally {kept}, the log says how to resend {said}")
    from flakeid.scope.devicephotos import main as dp_main
    rc = dp_main(["--resend", str(d), str(stack)])
    ok &= kept and said and rc == 0 and check_step(stack, 4, "tgetch", d, "4. after --resend") and "resend" in \
        next(d.glob("*_log.txt")).read_text(encoding="utf-8")
    try:
        dp_main(["--resend", str(d), str(stack)])
        again = False
    except SystemExit:
        again = True                                            # (sent already: refused)
    ok &= again and sorted(p.name for p in stack.iterdir()) == ["01_stacking", "02_contacts", "03_SETs", "04_tgetch", "DQD8.stack"]
    print(f"   a second --resend refused {again}; stack folder now {sorted(p.name for p in stack.iterdir())}")
    if not a.no_tk:
        ok &= tk_start_screen(out)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
