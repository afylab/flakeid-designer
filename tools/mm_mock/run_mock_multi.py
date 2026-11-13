"""Run tools/mm_chip_scan.bsh in MODE = "multi" against the simulated microscope.

Three real chip scans (bn24/25/26) lie on the stage at their own positions, rotations and heights,
on a tilted and bowed surface; demo 100x images are placed on each chip as the "flakes".  A fake
analysis (this harness) writes <piece>_targets.csv once a piece's tiles are all on disk (the last
one late, so the script has to wait).  The simulated user switches to 100x when asked: finer pixels,
the objective's XY offset (24, -18.3 um) and focus offset (-41 um), more defocus blur.

With --graphite the pieces are graphite: the target lists get a thickness_nm column (some flakes thinner than
3 layers), and after the 100x images the simulated user puts in the stacking-domain configuration (MockCore:
camera "DomainCam", XY offset (12.5, -7.0) um and focus offset +6.3 um from 100x); the flakes of 3+ layers
must be imaged again, in focus.  --domain-offset-known gives the script the offsets (as mm_domain_test.bsh
would); otherwise the focus offset is found at the first flake.

    python tools/mm_mock/run_mock_multi.py --java .../bin/java --jars DIR [--graphite [--domain-offset-known]]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.background import detect_chip  # noqa: E402

M10 = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])
UM = 1.09                    # um per px of the stitched chip images
UM100 = 1.0894 / 9.54
OFF100 = np.array([24.0, -18.3])
ZOFF100 = -41.0
OFFD, ZOFFD = np.array([36.5, -25.3]), -34.7          # MockCore defaults: the domain configuration


def img_to_stage(u, shape, pose):
    cx, cy, rot = pose[:3]
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    d = (np.atleast_2d(u) - np.array([shape[1] / 2.0, shape[0] / 2.0])) * UM
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1) + np.array([cx, cy])


def stage_to_img(T, shape, pose):
    cx, cy, rot = pose[:3]
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    d = np.atleast_2d(T) - np.array([cx, cy])
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1) / UM + np.array([shape[1] / 2.0, shape[0] / 2.0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--java", required=True)
    ap.add_argument("--jars", required=True)
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/multi"))
    ap.add_argument("--tilt", nargs=2, type=float, default=(2.0, -1.5), help="um/mm")
    ap.add_argument("--bow", type=float, default=1.5, help="um of bow 5 mm from the bow centre")
    ap.add_argument("--patches", type=int, default=3, help="100x 'flakes' per chip")
    ap.add_argument("--late", type=float, default=25.0, help="the last target list appears this many s after its scan")
    ap.add_argument("--chips", default="bn24,bn25,bn26")
    ap.add_argument("--graphite", action="store_true", help="graphite pieces: stacking-domain imaging after 100x")
    ap.add_argument("--domain-offset-known", action="store_true", help="the script gets the domain offsets")
    ap.add_argument("--glitch", type=int, default=-1, help="snap number that returns a garbage frame (camera glitch)")
    ap.add_argument("--materials", help="per piece material:oxide, e.g. graphite:90,hbn:285 (PIECE_MATERIALS; "
                                        "graphite pieces get stacking-domain images)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    run_dir = out / "run"
    run_dir.mkdir(parents=True)
    today = datetime.date.today().strftime("%Y%m%d")
    (run_dir / f"{today}_1_tiles").mkdir()            # an earlier scan: the new pieces must start at index 2

    names = a.chips.split(",")
    poses = [(50000.0, 20000.0, 2.0, 0.0), (66000.0, 21000.0, -3.0, 25.0), (58000.0, 35000.0, 4.0, -15.0)][: len(names)]
    rng = np.random.default_rng(7)
    demo = [f for f in sorted(glob.glob(str(ROOT / "example images/demo_high_magnification/*/*_100x.jpg"))) if " X" not in f]
    chips, centres, patches, masks = [], [], [], []
    k_demo = 0
    for nm, pose in zip(names, poses):
        path = ROOT / "example images" / f"{nm}.jpg"
        im = cv2.imread(str(path))
        sc = 0.125
        small = cv2.cvtColor(cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        m = detect_chip(small.astype(np.float32), 0)
        masks.append((m, sc, im.shape))
        ys, xs = np.nonzero(m)
        cen = img_to_stage(np.array([xs.mean(), ys.mean()]) / sc, im.shape, pose)[0]
        centres.append(cen)
        chips.append((path, pose))
        er = cv2.erode(m.astype(np.uint8), np.ones((90, 90), np.uint8))
        yy, xx = np.nonzero(er)
        for _ in range(a.patches):
            j = rng.integers(len(yy))
            pc = img_to_stage(np.array([xx[j], yy[j]]) / sc, im.shape, pose)[0]
            patches.append((nm, pc, demo[k_demo % len(demo)])); k_demo += 1
    tx, ty = a.tilt[0] / 1000, a.tilt[1] / 1000
    bowc = np.mean(centres, 0)
    bowK = a.bow / 5000.0 ** 2
    z0 = -90.0 - tx * centres[0][0] - ty * centres[0][1]

    def chip_dz(x, y):
        for (path, pose), (m, sc, shape) in zip(chips, masks):
            u = stage_to_img(np.array([x, y]), shape, pose)[0]
            if 0 <= u[0] < shape[1] and 0 <= u[1] < shape[0]:
                return pose[3]
        return 0.0

    def true_focus(x, y, obj=10):
        sx, sy = (x - OFF100[0], y - OFF100[1]) if obj == 100 else (x - OFFD[0], y - OFFD[1]) if obj == "domain" else (x, y)
        z = z0 + tx * x + ty * y + bowK * ((x - bowc[0]) ** 2 + (y - bowc[1]) ** 2) + chip_dz(sx, sy)
        return z + (ZOFF100 if obj == 100 else ZOFFD if obj == "domain" else 0.0)
    thick = {}                    # label -> thickness_nm of the flake (graphite)

    # pieces: chip centres + an offset, focused roughly by the simulated user (+-3 um)
    pc_txt = ", ".join("{ %.1f, %.1f, %.2f }" % (c[0] + o[0], c[1] + o[1], true_focus(c[0] + o[0], c[1] + o[1]) + rng.normal(0, 3))
                       for c, o in zip(centres, rng.normal(0, 300, (len(centres), 2))))

    # fake analysis: target lists once a piece's tiles are complete
    stop = threading.Event()
    written = {}

    def analysis():
        while not stop.is_set():
            for mf in run_dir.glob("*_run_*.json"):
                try:
                    man = json.loads(mf.read_text())
                except Exception:
                    continue
                for i, p in enumerate(man["pieces"]):
                    nm = p["name"]
                    if nm in written:
                        continue
                    pos = run_dir / f"{nm}_scan.pos"
                    tiles = run_dir / p["tiles"]
                    if not pos.exists() or not tiles.exists():
                        continue
                    n_plan = len(json.loads(pos.read_text())["map"]["StagePositions"]["array"])
                    if len(list(tiles.glob("*/metadata.txt"))) < n_plan:
                        continue
                    last = i == len(man["pieces"]) - 1
                    if last and nm not in written and a.late > 0:
                        written.setdefault("_late_" + nm, time.time())
                        if time.time() - written["_late_" + nm] < a.late:
                            continue
                    c = np.array(p["center"])
                    chip = names[int(np.argmin([np.hypot(*(c - q)) for q in centres]))]
                    rows = [(f"{k + 1:02d}_F{k + 1:03d}_{chip}", pc + OFF100 + rng.normal(0, 3, 2))
                            for k, (n_, pc, f) in enumerate([q for q in patches if q[0] == chip])]
                    tmp = run_dir / f"{nm}_targets.csv.tmp"
                    with open(tmp, "w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["order", "label", "stage_x_um", "stage_y_um", "z10_um", "thickness_nm"])
                        for k, (lab, xy) in enumerate(rows):
                            th = [0.67, 3.4, 12.0][k % 3]          # 2 layers (no domain image), 10 layers, ~36 layers
                            thick[lab] = th
                            w.writerow([k + 1, lab, f"{xy[0]:.2f}", f"{xy[1]:.2f}", "", th])
                    tmp.rename(run_dir / f"{nm}_targets.csv")
                    written[nm] = time.time()
                    print(f"HARNESS wrote {nm}_targets.csv ({len(rows)} targets)", flush=True)
            time.sleep(2)

    th = threading.Thread(target=analysis, daemon=True)
    th.start()

    script = (ROOT / "tools/mm_chip_scan.bsh").read_text()
    pm = [(m.split(":")[0], float(m.split(":")[1])) for m in a.materials.split(",")] if a.materials else None
    if pm:
        a.graphite = any(m == "graphite" for m, _ in pm)
    subs = ((("MODE", '"multi"'), ("OUT_DIR", f'"{out.as_posix()}"'), ("RUN_DIR", f'"{run_dir.as_posix()}"'),
                 ("RUN_NAME", '""'), ("PIECE_CENTERS", f"new double[][] {{ {pc_txt} }}"), ("AF_MANUAL_FALLBACK", '"never"'),
                 ("TARGETS_WAIT_S", "900.0"), ("FOCUS_MODE", '"auto"'),
                 ("MATERIAL", '"graphite"' if a.graphite else '"hbn"'), ("DOMAIN_CAMERA", '"DomainCam"'), ("DOMAIN_CAMERA_PROPS", "new String[][] { }"),
                 ("DOMAIN_EXPOSURE_MS", "4.0"),
                 ("PIECE_MATERIALS", "new Object[][] { " + ", ".join('{ "%s", %.1f }' % q for q in pm) + " }" if pm else "null"))
                 + ((("DOMAIN_XY_OFFSET_UM", "new double[] { %.3f, %.3f }" % tuple(OFFD - OFF100)),
                     ("DOMAIN_FOCUS_OFFSET_UM", "%.3f" % (ZOFFD - ZOFF100))) if a.domain_offset_known else
                    (("DOMAIN_XY_OFFSET_UM", "new double[] { %.3f, %.3f }" % tuple(OFFD - OFF100)),)))
    for k, v in subs:
        script, n = re.subn(rf'^{k}\s*=.*$', f'{k} = {v};', script, flags=re.M)
        assert n == 1, k
    first = centres[0]
    prelude = (f'mmc = new MockCore("{chips[0][0].as_posix()}", {UM}, -1e9, -1e9, 0.0, '
               f'{M10[0, 0]}, {M10[0, 1]}, {M10[1, 0]}, {M10[1, 1]}, {first[0]}, {first[1]}, 1L);\n'
               + "".join(f'mmc.addChip("{p.as_posix()}", {pose[0]}, {pose[1]}, {pose[2]}, {pose[3]});\n' for p, pose in chips)
               + "".join(f'mmc.addPatch("{Path(f).as_posix()}", {pc[0]}, {pc[1]});\n' for _, pc, f in patches)
               + f'mmc.patchUmPerPx = {UM100}; mmc.zOff100 = {ZOFF100}; mmc.offX100 = {OFF100[0]}; mmc.offY100 = {OFF100[1]};\n'
               f'mmc.tiltX = {tx}; mmc.tiltY = {ty}; mmc.z0 = {z0}; mmc.bowK = {bowK}; mmc.bowX = {bowc[0]}; mmc.bowY = {bowc[1]};\n'
               f'mmc.z = {true_focus(*first) + 2.0}; mm = new MockStudio();\nmmc.glitchAt = {a.glitch};\nt0 = System.currentTimeMillis();\n')
    post = '\nprint("MOCK snaps " + mmc.getSnaps() + " seconds " + (System.currentTimeMillis() - t0) / 1000.0);\n'
    run = out / "run.bsh"
    run.write_text(prelude + script + post)
    jars = Path(a.jars)
    cp = ":".join(str(p) for p in (jars / "bsh.jar", jars / "ij.jar", jars / "classes"))
    t_start = time.time()
    proc = subprocess.Popen([a.java, "-Xmx8g", "-Djava.awt.headless=true", "-cp", cp, "bsh.Interpreter", str(run)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = []
    for line in proc.stdout:
        log.append(line)
        if any(k in line for k in ("====", "HARNESS", "domain", "MOCK camera", "MOCK", "ERROR", "Exception", "FAILED", "focus surface", "pieces:",
                                   "rejected", "accepted", "wide search", "scan finished", "images (", "focus offset 100x", "gave up", "Error")):
            print(line.rstrip(), flush=True)
    proc.wait()
    stop.set()
    (out / "stdout.txt").write_text("".join(log))
    print(f"run time {time.time() - t_start:.0f} s, exit {proc.returncode}")

    # ---- score
    man = json.loads(next(run_dir.glob("*_run_*.json")).read_text())
    print("manifest state:", man["state"], [(p["name"], p["state"]) for p in man["pieces"]])
    ok = True
    for i, p in enumerate(man["pieces"]):
        nm = p["name"]
        c = np.array(p["center"])
        ci = int(np.argmin([np.hypot(*(c - q)) for q in centres]))
        m, sc, shape = masks[ci]
        pose = chips[ci][1]
        pos = json.loads((run_dir / f"{nm}_scan.pos").read_text())["map"]["StagePositions"]["array"]
        T = np.array([q["DevicePositions"]["array"][0]["Position_um"]["array"] for q in pos])
        chipj = json.loads((run_dir / f"{nm}_chip.json").read_text())
        fov = np.array(chipj["field_um"])
        cover = np.zeros(m.shape, np.uint8)
        for t in T:
            cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov / 2 + t
            cv2.fillPoly(cover, [np.round(stage_to_img(cs, shape, pose) * sc).astype(np.int32)], 1)
        covered = float((cover[m] > 0).mean())
        # tile Z (from the tiles' metadata, real um) against the true focus
        zt = []
        for md in (run_dir / f"{nm}_tiles").glob("*/metadata.txt"):
            d = json.loads(md.read_text())
            fm = next(v for k, v in d.items() if k.startswith("Metadata"))
            u = stage_to_img(np.array([fm["XPositionUm"], fm["YPositionUm"]]), shape, pose)[0] * sc
            if not (0 <= u[0] < m.shape[1] and 0 <= u[1] < m.shape[0] and m[int(u[1]), int(u[0])]):
                continue           # tile centred off the chip (the mock's piece height ends with its image)
            zt.append(fm["ZPositionUm"] - true_focus(fm["XPositionUm"], fm["YPositionUm"]))
        zt = np.array(zt)
        idx = run_dir / f"{nm}_100x" / f"{nm}_100x_index.csv"
        rows = list(csv.DictReader(open(idx))) if idx.exists() else []
        e100 = np.array([float(r["z_um"]) - true_focus(float(r["stage_x_um"]), float(r["stage_y_um"]), 100) for r in rows])
        how = [r["focus"] for r in rows]
        n_t = len([q for q in patches if q[0] == names[ci]])
        print(f"{nm} ({names[ci]}): {len(T)} tiles, chip covered {covered:.3f}, surface {chipj.get('focus_surface', {}).get('type', 'plane')}, "
              f"tile focus error max {np.abs(zt).max():.2f} um (rms {np.sqrt((zt ** 2).mean()):.2f}); "
              f"100x {len(rows)}/{n_t} imaged {how}, focus error " + " ".join(f"{e:+.2f}" for e in e100))
        ok &= covered > 0.98 and len(rows) == n_t and (len(e100) == 0 or np.abs(e100).max() < 0.6) and np.abs(zt).max() < 4.0
    if a.graphite:
        nd, ed, n_want = 0, [], 0
        for p in man["pieces"]:
            nm = p["name"]
            r100 = list(csv.DictReader(open(run_dir / f"{nm}_100x" / f"{nm}_100x_index.csv"))) if (run_dir / f"{nm}_100x" / f"{nm}_100x_index.csv").exists() else []
            if p.get("material", "graphite") == "graphite":
                n_want += sum(1 for r in r100 if thick.get(r["label"], 0) >= 0.84)
            di = run_dir / f"{nm}_domains" / f"{nm}_domains_index.csv"
            rows = list(csv.DictReader(open(di))) if di.exists() else []
            nd += len(rows)
            for r in rows:
                e = float(r["z_um"]) - true_focus(float(r["stage_x_um"]), float(r["stage_y_um"]), "domain")
                ed.append(e)
                print(f"  domain {nm} {r['label']} {r['thickness_nm']} nm: focus {r['focus']}, error {e:+.2f} um")
        ed = np.array(ed)
        print(f"domain images {nd} of {n_want} flakes of 3+ layers; focus error max {np.abs(ed).max() if len(ed) else float('nan'):.2f} um")
        ok &= nd == n_want and n_want > 0 and np.abs(ed).max() < 0.6
    if pm:
        for p, (m, ox) in zip(man["pieces"], pm):
            sp = json.loads((run_dir / p["tiles"] / "scan_params.json").read_text())
            cj = json.loads((run_dir / f"{p['name']}_chip.json").read_text())
            got = [(p.get("material"), p.get("oxide_nm")), (sp["material"], sp["oxide_nm"]), (cj["material"], cj["oxide_nm"])]
            good = all(g == (m, ox) for g in got)
            print(f"{p['name']}: material/oxide in run file, scan_params.json, chip.json: {got} -> {'OK' if good else 'WRONG'}")
            ok &= good
    names_ok = [p["name"] for p in man["pieces"]] == [f"{today}_{i + 2}" for i in range(len(names))]
    print("piece names", [p["name"] for p in man["pieces"]], "OK" if names_ok else "WRONG")
    print("RESULT", "PASS" if ok and names_ok and man["state"] == "done" else "FAIL")


if __name__ == "__main__":
    main()
