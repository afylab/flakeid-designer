"""Raw 100x images of flakes, for developing the 100x processing: each flake's averaged frame, every single frame and the
shifted frames (as a run takes them), and frames around it for the optics pattern.  Nothing is deleted afterwards.

    C:\\FlakeID\\run_flakeid.bat capture                (where the stage is: pick each flake in the live view)
    C:\\FlakeID\\run_flakeid.bat capture --name hbn_test
    C:\\FlakeID\\run_flakeid.bat capture C:\\FlakeScans\\pleasework_1_targets.csv --flake 11   (a scanned flake, chip as scanned)
    (then copy C:\\FlakeScans\\capture_<name>\\ and the 100x flat field, e.g. C:\\FlakeID\\FlakeIdentifier\\flats\\flat_100x_285nm.npz)

--flake is the flake number (F011 = flake 11, as the viewer and the designer number them); --label picks a target by its
label instead.  The chip must be on the stage as it was scanned (same holder position): the target list's stage
coordinates are used.  The scan program opens in targets mode: put in the 100x objective when asked; it focuses on
the flake (the 100x - 10x focus offset is taken from the run's log when it is there), takes the images, then
--pattern frames (default 24) at random points up to 250 um around the flake.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

DEST = Path(r"C:\FlakeID")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="?", help="a piece's target list (<piece>_targets.csv); without it: capture where the stage is")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--flake", type=int, help="flake number (F011 = 11)")
    g.add_argument("--label", help="target label, e.g. 04_F011_60nm")
    ap.add_argument("--pattern", type=int, default=24, help="frames around the flake for the optics pattern (default 24; 0 = none)")
    ap.add_argument("--name", help="capture name (default capture_<piece>_F<flake>)")
    ap.add_argument("--mm", default=r"C:\Program Files\Micro-Manager-2.0")
    ap.add_argument("--cfg", default=r"C:\Program Files\Micro-Manager-2.0\autoscan_withir.cfg")
    ap.add_argument("--settings", default=str(DEST / "scan_settings.json"))
    a = ap.parse_args(argv)
    if not a.targets:
        # in place: the scan program's capture mode (live view: move to each flake, focus, "Use this focus")
        cmd = [sys.executable, "-X", "utf8", "-m", "flakeid.scope", "--mm", a.mm, "--cfg", a.cfg, "--settings", a.settings,
               "--mode", "capture", "--set", f"pattern_frames={a.pattern}"] + (["--name", a.name] if a.name else [])
        print("starting the scan program in capture mode (images go to C:\\FlakeScans\\<name>\\) ...")
        return subprocess.call(cmd, cwd=str(DEST / "FlakeIdentifier"))
    if a.flake is None and a.label is None:
        ap.error("with a target list, give --flake or --label")
    tl = Path(a.targets)
    rows = list(csv.DictReader(open(tl, newline="")))
    if a.flake is not None:
        hit = [r for r in rows if str(r.get("flake_id", "")).strip() == str(a.flake)
               or re.search(rf"_F0*{a.flake}_", r.get("label", ""))]
    else:
        hit = [r for r in rows if r.get("label") == a.label]
    if not hit:
        print(f"no target {'flake %d' % a.flake if a.flake is not None else a.label} in {tl}")
        return 1
    t = hit[0]
    piece = re.sub(r"_targets\.csv$", "", tl.name)
    name = a.name or f"capture_{piece}_" + (f"F{a.flake:03d}" if a.flake is not None else re.sub(r"\W", "_", a.label))
    out = tl.parent / name
    out.mkdir(parents=True, exist_ok=True)
    one = out / f"{name}_targets.csv"
    with open(one, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(t))
        w.writeheader()
        w.writerow(t)
    print(f"target: {t['label']} (flake {t.get('flake_id')}, {t.get('thickness_nm')} nm) at {t['stage_x_um']}, {t['stage_y_um']} um")
    # the 100x - 10x focus offset measured in that run (its log: "Next time: focus_offset_um = ..."), else measured here
    off = "NaN"
    for lg in sorted(tl.parent.glob("*_run_*_log.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
        m = re.findall(r"focus_offset_um = (-?\d+\.\d+)", lg.read_text(errors="replace"))
        if m:
            off = m[0]
            print(f"focus offset 100x - 10x from {lg.name}: {off} um")
            break
    cmd = [sys.executable, "-X", "utf8", "-m", "flakeid.scope", "--mm", a.mm, "--cfg", a.cfg, "--settings", a.settings,
           "--mode", "targets", "--targets", str(one), "--name", name,
           "--set", f"out_dir={json.dumps(str(out))}", "--set", "hundred_save_frames=true",
           "--set", f"pattern_frames={a.pattern}", "--set", f"focus_offset_um={off}"]
    chip = tl.parent / f"{piece}_chip.json"
    if chip.is_file():
        cmd += ["--set", f"chip_json={json.dumps(str(chip))}"]
    print("starting the scan program in targets mode ...")
    r = subprocess.call(cmd, cwd=str(DEST / "FlakeIdentifier"))
    print(f"\nimages: {out}\\{name}_100x  (send that folder, and the 100x flat field from {DEST}\\FlakeIdentifier\\flats)")
    return r


if __name__ == "__main__":
    sys.exit(main())
