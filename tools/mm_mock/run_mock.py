"""Run tools/mm_chip_scan.bsh against a simulated microscope and score its scan plan.

The mock (MockCore.java) serves 10x RAW16 Bayer frames, vignetted, noisy and
flipped like the XC10, cut from a stitched chip scan. The real script runs
unchanged in BeanShell, apart from its OUT_DIR/NAME settings. The resulting
tiles are compared with the chip outline from flakeid.background.detect_chip.

    python tools/mm_mock/run_mock.py "example images/bn24.jpg" --rot 4 --offset 800 -500
Needs a Java 8 runtime (--java), bsh.jar and ij.jar (--jars DIR).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flakeid.background import detect_chip  # noqa: E402

M = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])
O = np.array([50000.0, 20000.0])       # stage coordinate of the image centre
UM_PER_PX = 1.09


def stage_to_img(T, shape, rot):
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    d = (np.atleast_2d(T) - O)
    u = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1) / UM_PER_PX
    return u + np.array([shape[1] / 2.0, shape[0] / 2.0])


def img_to_stage(u, shape, rot):
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    d = (np.atleast_2d(u) - np.array([shape[1] / 2.0, shape[0] / 2.0])) * UM_PER_PX
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1) + O


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help='stitched scan, or "synthetic:rect|arc|corner" (1-inch wafer pieces)')
    ap.add_argument("--rot", type=float, default=0.0, help="chip rotation relative to the stage (deg)")
    ap.add_argument("--offset", nargs=2, type=float, default=(0.0, 0.0), help="start offset from the chip centre (um)")
    ap.add_argument("--edge-points", type=int, default=None, help="EDGE_POINTS in the script (default: the script's)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tilt", nargs=2, type=float, default=(0.0, 0.0), metavar=("DZDX", "DZDY"),
                    help="true chip tilt in um/mm (the mock user focuses on it)")
    ap.add_argument("--focus-points", type=int, default=None)
    ap.add_argument("--neighbour", type=float, default=None, metavar="GAP_UM",
                    help="place a copy of the chip this far beyond its -X edge (a neighbouring chip)")
    ap.add_argument("--fault-x", type=float, default=None, help="X moves go to this factor x the command")
    ap.add_argument("--lock-gamma", action="store_true", help="camera ignores gamma changes (tests the read-back)")
    ap.add_argument("--focus-mode", default=None, help='"auto" or "manual" (manual = simulated user)')
    ap.add_argument("--z-error", type=float, default=0.0, help="true focus minus the start Z (um)")
    ap.add_argument("--exposure", type=float, default=None, help="EXPOSURE_MS in the script")
    ap.add_argument("--auto-exposure", action="store_true")
    ap.add_argument("--residue", type=float, default=None, metavar="COVER",
                    help="paint tape residue and flakes over this fraction of a 2.5 mm area around the start")
    ap.add_argument("--z-scale", type=float, default=12.0, help="drive units per um (mock) = Z_SCALE in the script")
    ap.add_argument("--run", action="store_true", help="RUN_SCAN = true (exercises the acquisition fallback)")
    ap.add_argument("--java", required=True)
    ap.add_argument("--jars", required=True, help="folder with bsh.jar, ij.jar and the compiled mock classes in classes/")
    a = ap.parse_args(argv)

    global UM_PER_PX
    synthetic = a.image.startswith("synthetic:")
    if synthetic:
        # 1-inch piece of a 4-inch wafer: rectangle (25.4 x 24 mm) intersected with the wafer disc (R 50.8 mm)
        kind = a.image.split(":", 1)[1]
        UM_PER_PX = 3.0
        W_, H_ = 12000, 11600                                   # 36 x 34.8 mm field
        yy, xx = np.mgrid[0:H_, 0:W_].astype(np.float32)
        X = (xx - W_ / 2) * UM_PER_PX / 1000.0; Y = (yy - H_ / 2) * UM_PER_PX / 1000.0   # mm, image +y down
        chip = (np.abs(X) <= 12.7) & (np.abs(Y) <= 12.0)
        R = 50.8
        if kind == "arc":        # the -Y (image top) side is the wafer rim, apex at y = -12
            chip &= (X ** 2 + (Y - (-12.0 + R)) ** 2) <= R * R
            chip &= Y >= -13.7                                  # (rectangle top pushed out so the rim is the edge)
        elif kind == "corner":   # the rim cuts off the (+X, -Y) corner
            cxw, cyw = 12.7 - 40.0, -12.0 + 36.0
            chip &= ((X - cxw) ** 2 + (Y - cyw) ** 2) <= R * R
        del X, Y, xx, yy
        rng = np.random.default_rng(11)
        img = np.full((H_, W_, 3), 22, np.uint8)
        img[chip] = (200, 130, 150)                             # BGR: purple-ish SiO2 like the Stream scans
        for _ in range(4000):                                   # flakes
            cxp, cyp = int(rng.integers(0, W_)), int(rng.integers(0, H_))
            if not chip[cyp, cxp]:
                continue
            col = [(120, 220, 240), (150, 230, 180), (190, 170, 240)][int(rng.integers(0, 3))]
            cv2.ellipse(img, (cxp, cyp), (int(rng.integers(2, 12)), int(rng.integers(2, 12))), float(rng.uniform(0, 180)), 0, 360, col, -1)
        img[~chip] = 22
        del chip
        out = Path(a.out or ROOT / "tools/mm_mock/out" / kind)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "synthetic.png"), img)
        a.image = str(out / f"synthetic_{kind}.png")
        cv2.imwrite(a.image, img)
    else:
        img = cv2.imread(a.image)
    sc = 0.125
    out = Path(a.out or ROOT / "tools/mm_mock/out" / Path(a.image).stem)
    out.mkdir(parents=True, exist_ok=True)
    image_path = Path(a.image)
    if a.neighbour is not None:
        # [copy][gap][chip] on the dark surround; the script starts on the right-hand chip
        sm = cv2.cvtColor(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        m = detect_chip(sm.astype(np.float32), 0)
        ys_, xs_ = np.nonzero(m)
        x0, x1 = int(xs_.min() / sc), int(xs_.max() / sc)
        y0, y1 = int(ys_.min() / sc), int(ys_.max() / sc)
        chipc = img[y0:y1, x0:x1]
        gap = int(a.neighbour / UM_PER_PX)
        bg = np.median(img[:50, :50].reshape(-1, 3), axis=0).astype(np.uint8)
        pad = 400
        comp = np.empty((chipc.shape[0] + 2 * pad, 2 * chipc.shape[1] + gap + 2 * pad, 3), np.uint8)
        comp[:] = bg
        comp[pad:pad + chipc.shape[0], pad:pad + chipc.shape[1]] = chipc
        comp[pad:pad + chipc.shape[0], pad + chipc.shape[1] + gap:pad + 2 * chipc.shape[1] + gap] = chipc
        image_path = out / "composite.jpg"
        cv2.imwrite(str(image_path), comp, [cv2.IMWRITE_JPEG_QUALITY, 92])
        img = comp
    small = cv2.cvtColor(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    gt = detect_chip(small.astype(np.float32), 0)
    if a.neighbour is not None:            # ground truth: the right-hand chip only
        n, lab = cv2.connectedComponents(gt.astype(np.uint8))
        rights = [(np.nonzero(lab == k)[1].mean(), k) for k in range(1, n) if (lab == k).sum() > 1000]
        gt = lab == max(rights)[1]
    ys, xs = np.nonzero(gt)
    centre_u = np.array([xs.mean(), ys.mean()]) / sc
    start = img_to_stage(centre_u, img.shape, a.rot)[0] + np.array(a.offset)

    if a.residue:
        # tape residue (hazy grey-blue, streaky) and flakes over COVER of the area around the start
        rimg = img.copy()
        c0 = stage_to_img(start, img.shape, a.rot)[0]
        R = int(1250 / UM_PER_PX)
        rng = np.random.default_rng(5)
        yy, xx = np.mgrid[-R:R, -R:R]
        mask = np.zeros((2 * R, 2 * R), np.uint8)
        while (mask[(xx ** 2 + yy ** 2) < R * R] > 0).mean() < a.residue:
            cx, cy = rng.integers(0, 2 * R, 2)
            ax_, ay_ = rng.integers(20, 160, 2)
            cv2.ellipse(mask, (int(cx), int(cy)), (int(ax_), int(ay_)), float(rng.uniform(0, 180)), 0, 360, int(rng.integers(1, 4)), -1)
        x0_, y0_ = int(c0[0]) - R, int(c0[1]) - R
        sub = rimg[y0_:y0_ + 2 * R, x0_:x0_ + 2 * R].astype(np.float32)
        cols = {1: (185, 180, 170), 2: (120, 200, 230), 3: (60, 160, 90)}          # BGR: haze, yellow flake, green flake
        for k_, col in cols.items():
            m_ = mask == k_
            sub[m_] = 0.45 * sub[m_] + 0.55 * np.array(col, np.float32) + rng.normal(0, 6, (m_.sum(), 3))
        rimg[y0_:y0_ + 2 * R, x0_:x0_ + 2 * R] = np.clip(sub, 0, 255).astype(np.uint8)
        image_path = out / "residue.jpg"
        cv2.imwrite(str(image_path), rimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / "residue_start.jpg"), rimg[y0_:y0_ + 2 * R, x0_:x0_ + 2 * R])
    script = (ROOT / "tools/mm_chip_scan.bsh").read_text()
    script = re.sub(r'^OUT_DIR\s*=.*$', f'OUT_DIR = "{out.as_posix()}";', script, flags=re.M)
    script = re.sub(r'^NAME\s*=.*$', f'NAME = "{Path(a.image).stem}";', script, flags=re.M)
    if a.focus_points is not None:
        script = re.sub(r'^FOCUS_POINTS\s*=.*$', f'FOCUS_POINTS = {a.focus_points};', script, flags=re.M)
    if a.focus_mode:
        script = re.sub(r'^FOCUS_MODE\s*=.*$', f'FOCUS_MODE = "{a.focus_mode}";', script, flags=re.M)
    if a.exposure is not None:
        script = re.sub(r'^EXPOSURE_MS\s*=.*$', f'EXPOSURE_MS = {a.exposure};', script, flags=re.M)
    if a.auto_exposure:
        script = re.sub(r'^AUTO_EXPOSURE\s*=.*$', 'AUTO_EXPOSURE = true;', script, flags=re.M)
    script = re.sub(r'^Z_SCALE\s*=.*$', f'Z_SCALE = {a.z_scale};', script, flags=re.M)
    if a.run:
        script = re.sub(r'^RUN_SCAN\s*=.*$', 'RUN_SCAN = true;', script, flags=re.M)
    if a.edge_points is not None:
        script = re.sub(r'^EDGE_POINTS\s*=.*$', f'EDGE_POINTS = {a.edge_points};', script, flags=re.M)
    prelude = (f'mmc = new MockCore("{Path(image_path).resolve().as_posix()}", {UM_PER_PX}, {O[0]}, {O[1]}, {a.rot}, '
               f'{M[0, 0]}, {M[0, 1]}, {M[1, 0]}, {M[1, 1]}, {start[0]}, {start[1]}, 1L);\nmm = new MockStudio();\n'
               + ('mmc.lockGamma = true;\n' if a.lock_gamma else '')
               + (f'mmc.faultScaleX = {1 / a.fault_x};\n' if a.fault_x else '') +
               f'mmc.tiltX = {a.tilt[0] / 1000}; mmc.tiltY = {a.tilt[1] / 1000}; mmc.z = -90.0; mmc.z0 = -90.0 + {a.z_error} - mmc.tiltX * {start[0]} - mmc.tiltY * {start[1]};\n'
               't0 = System.currentTimeMillis();\n')
    post = ('\nprint("MOCK snaps " + mmc.getSnaps() + " moves " + mmc.getMoves() + " travel_mm " + (mmc.getTravel() / 1000.0)'
            ' + " end " + mmc.getXPosition("XY") + " " + mmc.getYPosition("XY") + " loaded " + mm.list.getNumberOfPositions()'
            ' + " seconds " + (System.currentTimeMillis() - t0) / 1000.0);\n')
    run = out / "run.bsh"
    run.write_text(prelude + script + post)
    jars = Path(a.jars)
    cp = ":".join(str(p) for p in (jars / "bsh.jar", jars / "ij.jar", jars / "classes"))
    r = subprocess.run([a.java, "-Xmx4g", "-Djava.awt.headless=true", "-cp", cp, "bsh.Interpreter", str(run)],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode or "Error" in r.stderr or "Exception" in r.stdout:
        print(r.stderr[-3000:])
    name = Path(a.image).stem
    chip = json.loads((out / f"{name}_chip.json").read_text())
    pos = json.loads((out / f"{name}_scan.pos").read_text())["map"]["StagePositions"]["array"]
    tiles = np.array([p["DevicePositions"]["array"][0]["Position_um"]["array"] for p in pos])
    # .pos Z is in drive units; the script sets the mock drive's step size (rounded to 4 decimals by the mock)
    ts = 0.025 / 12
    zs = (round(ts * 1e4) / 1e4) / ts if "Z_STEPSIZE_UM = 0;" not in script else a.z_scale
    tz = np.array([p["DevicePositions"]["array"][1]["Position_um"]["array"][0] for p in pos]) / zs
    z0 = -90.0 + a.z_error - a.tilt[0] / 1000 * start[0] - a.tilt[1] / 1000 * start[1]
    zerr = tz - (z0 + a.tilt[0] / 1000 * tiles[:, 0] + a.tilt[1] / 1000 * tiles[:, 1])
    fov = np.array(chip["field_um"])

    # score on the reduced ground-truth grid
    cover = np.zeros(gt.shape, np.uint8)
    for t in tiles:
        cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov / 2 + t
        u = stage_to_img(cs, img.shape, a.rot) * sc
        cv2.fillPoly(cover, [np.round(u).astype(np.int32)], 1)
    # detected chip region from its sides (lines, arcs, bounding edges), on the ground-truth grid
    gy, gx = np.mgrid[0:gt.shape[0], 0:gt.shape[1]]
    T = img_to_stage(np.stack([gx.ravel(), gy.ravel()], 1) / sc, img.shape, a.rot)
    inside = np.ones(len(T), bool)
    for sd in chip.get("sides", []):
        pnt, nrm = np.array(sd["point"]), np.array(sd["normal"])
        inside &= (T - pnt) @ nrm <= 0
        if sd["type"] == "arc":
            c_ = np.array(sd["arc"][:2]); inside &= np.hypot(*(T - c_).T) <= sd["arc"][2]
    poly = inside.reshape(gt.shape).astype(np.uint8)
    u = stage_to_img(np.array(chip["corners"]), img.shape, a.rot) * sc
    covered = (cover[gt] > 0).mean()
    iou = (poly.astype(bool) & gt).sum() / (poly.astype(bool) | gt).sum()
    px_um = UM_PER_PX / sc
    # edge error: GT boundary pixels -> distance to the detected polygon outline
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    b = max(cnts, key=len).reshape(-1, 2)
    pcs, _ = cv2.findContours(poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pc = max(pcs, key=len).astype(np.float32)
    d = np.array([abs(cv2.pointPolygonTest(pc, (float(x), float(y)), True)) for x, y in b[::5]]) * px_um
    tile_area = len(tiles) * fov.prod()
    chip_area = gt.sum() * px_um ** 2
    res = dict(image=name, rot=a.rot, offset=list(a.offset), tiles=len(tiles), chip_covered=round(float(covered), 4),
               polygon_iou=round(float(iou), 4), edge_err_um_median=round(float(np.median(d)), 1),
               edge_err_um_p95=round(float(np.percentile(d, 95)), 1),
               tile_area_over_chip=round(float(tile_area / chip_area), 2), fitted_rotation=round(chip["rotation_deg"], 2),
               size_mm=[round(v / 1000, 3) for v in chip["size_um"]],
               tile_z_err_um_max=round(float(np.abs(zerr).max()), 2), tile_z_range_um=round(float(np.ptp(tz)), 1),
               focus_points=len(chip.get("focus_points", [])))
    # overlay for inspection
    vis = cv2.resize(img, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
    for t in tiles:
        cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * fov / 2 + t
        cv2.polylines(vis, [np.round(stage_to_img(cs, img.shape, a.rot) * sc).astype(np.int32)], True, (200, 200, 200), 1)
    cv2.polylines(vis, [pc.astype(np.int32)], True, (0, 0, 255), 2)
    for p in chip["edge_points"]:
        q = stage_to_img(np.array(p), img.shape, a.rot)[0] * sc
        cv2.circle(vis, tuple(int(v) for v in q), 5, (255, 0, 0), -1)
    q = stage_to_img(start, img.shape, a.rot)[0] * sc
    cv2.circle(vis, tuple(int(v) for v in q), 7, (0, 200, 0), -1)
    cv2.imwrite(str(out / f"{name}_check.jpg"), vis)
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
