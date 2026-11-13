"""Re-imaging the 100x targets of a piece that moved (flakeid.scope mode "retarget") in the Python simulator.

    python tools/mm_mock/run_mock_retarget.py [--reloaded [--start-off UM]] [--shift DX DY] [--rot DEG] [--auto]

A synthetic chip with a few hundred flakes is "scanned" (its flake list, target list and chip file written as the
analysis would), then moved on the stage (a small shift in place, or with --reloaded a larger shift and a rotation, as
when it is taken off and put back), and the program finds it again and images the targets: by hand (the default; the
simulated user centres the flake nearest the stage, within 1.5 um), or with --auto from its flakes.  PASS:
every target imaged within 10 um of where it really is now.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

UM = 1.09
OFF100 = np.array([24.0, -18.3])


def chip_image(rng, size=5000, n=420):
    im = np.empty((size, size, 3), np.uint8)
    im[:] = (150, 120, 190)
    cols = [(95, 125, 215), (60, 160, 230), (190, 150, 210), (230, 200, 120), (120, 90, 170), (200, 180, 230)]
    fl = []
    for _ in range(n):
        u, v = rng.uniform(150, size - 150, 2)
        a, b = rng.uniform(6, 40), rng.uniform(5, 25)
        cv2.ellipse(im, (int(u), int(v)), (int(a), int(b)), float(rng.uniform(0, 180)), 0, 360, cols[int(rng.integers(len(cols)))], -1)
        fl.append((u, v, math.pi * a * b * UM * UM))
    return cv2.GaussianBlur(im, (0, 0), 0.8), fl


def img_to_stage(u, v, size, cx, cy, rot_deg):
    """Stage position that centres chip-image pixel (u, v) (MockCore's chip layer, 10x)."""
    a = math.radians(rot_deg)
    Rinv = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) * UM
    return Rinv @ (np.array([u, v]) - size / 2.0) + np.array([cx, cy])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "tools/mm_mock/out/retarget"))
    ap.add_argument("--reloaded", action="store_true")
    ap.add_argument("--shift", nargs=2, type=float, default=None)
    ap.add_argument("--rot", type=float, default=None)
    ap.add_argument("--start-off", type=float, default=None, help="put back: start this far (um) from the chip centre (default within 300)")
    ap.add_argument("--auto", action="store_true", help="found automatically (no flakes centred by hand)")
    ap.add_argument("--slip", type=float, default=0.0, help="the focus slips by this much (um) after the 4th 100x image")
    ap.add_argument("--viewer", action="store_true", help="give the piece a (minimal) viewer and a results folder: the re-image "
                    "then updates both; PASS also needs every image matched to its flake in the viewer and the copy updated")
    ap.add_argument("--share-only", action="store_true", help="with --viewer: the viewer only in the results folder's copy "
                    "(none on this PC): the re-image updates the copy directly")
    ap.add_argument("--patch100", default="", help="folder of 100x demo images (example images/demo_high_magnification): one at "
                    "each target, so the 100x autofocus has something to focus on")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = np.random.default_rng(3)
    img, fl = chip_image(rng)
    size = img.shape[0]
    cx0, cy0, rot0 = 30000.0, 20000.0, 1.0
    shift = np.array(a.shift) if a.shift else (np.array([740.0, -460.0]) if a.reloaded else np.array([55.0, -38.0]))
    drot = a.rot if a.rot is not None else (4.0 if a.reloaded else 0.25)
    cx1, cy1, rot1 = cx0 + shift[0], cy0 + shift[1], rot0 + drot

    # the first scan's files
    piece = "demo_1"
    rdir = out / "results" / "demo_run_1" / piece
    rdir.mkdir(parents=True)
    old = [img_to_stage(u, v, size, cx0, cy0, rot0) for u, v, _ in fl]
    new = [img_to_stage(u, v, size, cx1, cy1, rot1) for u, v, _ in fl]
    with open(rdir / f"{piece}_tiles_live_flakes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flake_id", "score", "area_um2", "stage_x_um", "stage_y_um"])
        for i, ((u, v, ar), p) in enumerate(zip(fl, old)):
            w.writerow([i + 1, 0.9, f"{ar:.1f}", f"{p[0]:.2f}", f"{p[1]:.2f}"])
    big = [i for i, f in enumerate(fl) if f[2] > 1500][:10]
    with open(out / f"{piece}_targets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "label", "flake_id", "score", "thickness_nm", "stage_x_um", "stage_y_um", "z_um", "z10_um"])
        for k, i in enumerate(big):
            p = old[i] + OFF100
            w.writerow([k + 1, f"{k + 1:02d}_F{i + 1:03d}", i + 1, 0.9, 30.0, f"{p[0]:.2f}", f"{p[1]:.2f}", "", ""])
    if a.viewer:                                      # the first run's viewer (flakes at their first-scan positions) and its copy
        vd = (out / "share" / "demo_run_1" / piece) if a.share_only else rdir / f"{piece}_tiles_viewer"
        (vd / "100x").mkdir(parents=True)
        D = dict(name=f"{piece}_tiles", flakes=[dict(id=i + 1, sx=round(float(p[0]), 2), sy=round(float(p[1]), 2)) for i, p in enumerate(old)])
        (vd / "index.html").write_text("<script>\nconst D=" + json.dumps(D) + ";\nconst H100={};\n</script>\n")
        (out / "share" / "demo_run_1" / piece).mkdir(parents=True, exist_ok=True)
    corners = [img_to_stage(u, v, size, cx0, cy0, rot0).tolist() for u, v in ((size, size), (0, size), (0, 0), (size, 0))]   # the scan's order: +X+Y, -X+Y, -X-Y, +X-Y
    (out / f"{piece}_chip.json").write_text(json.dumps(dict(material="hbn", oxide_nm=285.0, corners=corners)))

    # the chip, moved
    core = MockCore(UM, x0=cx0, y0=cy0)
    core.add_chip(img, cx1, cy1, rot1, 0.0)
    if a.patch100:
        import glob
        demo = [f for f in sorted(glob.glob(str(Path(a.patch100) / "*" / "*_100x.jpg"))) if " X" not in f]
        for k, i in enumerate(big):
            core.add_patch(cv2.cvtColor(cv2.imread(demo[k % len(demo)]), cv2.COLOR_BGR2RGB), new[i][0], new[i][1])
        core.patch_um_per_px = 1.0894 / 9.54
    start = (cx1 + rng.uniform(-300, 300), cy1 + rng.uniform(-300, 300)) if a.reloaded else (cx0 + 500, cy0 - 400)
    if a.reloaded and a.start_off:
        start = (cx1 + a.start_off * 0.8, cy1 - a.start_off * 0.6)
    truth = {f"{k + 1:02d}_F{i + 1:03d}": new[i] + OFF100 for k, i in enumerate(big)}
    tp = np.array(list(truth.values()))

    def centre():                                     # the user: the flake nearest the stage, centred within ~1.5 um
        d = np.hypot(tp[:, 0] - core.x, tp[:, 1] - core.y)
        j = int(np.argmin(d))
        if d[j] < 600:
            core.x, core.y = (tp[j] + rng.normal(0, 1.0, 2)).tolist()
    core.mockCentre = centre
    core.setXYPosition("XY", *start)
    core.z = core.true_focus(*start)
    s = load(None, mode="retarget", local_root=str(out), retarget_piece=piece, retarget_reloaded=a.reloaded, z_settle_ms=0.0,
             verbose=False, target_exposure_ms=8.0, focus_offset_um=float(core.zoff100), af_manual_fallback="never",
             retarget_manual_points=0 if a.auto else 3, out_dir=str(out / "share"))
    sc = Scanner(core, s, AutoUI(core, analysis=False))
    focus_err, n_img = {}, [0]
    orig_image_flake = sc.image_flake

    def image_flake(P, t, tag, **kw):                 # the focus error of each image (a retake's replaces it); the slip
        r = orig_image_flake(P, t, tag, **kw)
        focus_err[t[4]] = core.z - core.true_focus(core.x, core.y)
        n_img[0] += 1
        if n_img[0] == 4 and a.slip:
            core.zoff100 += a.slip
            print(f"  (simulated focus slip: {a.slip:+.1f} um)")
        return r
    sc.image_flake = image_flake
    sc.run()

    # score: where each target was imaged against where its flake is now
    idx = next(out.glob("*_retake1_100x/*_100x_index.csv"))
    errs = []
    with open(idx, newline="") as fh:
        for r in csv.DictReader(fh):
            e = math.hypot(float(r["stage_x_um"]) - truth[r["label"]][0], float(r["stage_y_um"]) - truth[r["label"]][1])
            errs.append(e)
    print(f"chip moved {shift[0]:+.0f}, {shift[1]:+.0f} um, rotated {drot:+.2f} deg ({'put back' if a.reloaded else 'in place'})")
    print(f"{len(errs)} of {len(big)} targets imaged; position error median {np.median(errs):.1f} um, max {max(errs):.1f} um")
    fe = np.abs(list(focus_err.values()))
    print(f"focus error median {np.median(fe):.2f} um, max {fe.max():.2f} um; focus jumps logged: {len(sc.focus_jumps)}")
    ok = len(errs) == len(big) and max(errs) < 10.0 and fe.max() < 1.0
    if a.viewer:
        import re
        pages = (("viewer", rdir / f"{piece}_tiles_viewer" / "index.html"),) if not a.share_only else ()
        for where, page in pages + (("copy", out / "share" / "demo_run_1" / piece / "index.html"),):
            H = json.loads(re.search(r"const H100=(.*?);\n", page.read_text(), re.S).group(1)) if page.is_file() else {}
            right = sum(1 for k, v in H.items() for e in v if re.search(r"_F(\d+)", e["lab"]) and int(re.search(r"_F(\d+)", e["lab"]).group(1)) == int(k))
            print(f"{where}: 100x images for {len(H)} flakes, {right} on the right flake")
            ok = ok and len(H) == len(big) and right == len(big)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
