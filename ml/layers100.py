"""Graphite layer count check: the terrace's contrast measured in its own 100x picture (independent of the 10x call
that names the shot) against the thin-film model.  python layers100.py -> results/layers100.json + summary.

Contrast per channel = mean colour inside terrace a (the scope's _regions.json polygon, eroded 1 um) / the bare
substrate (the commonest colour of the frame outside the terrace, dilated 5 um), - 1; from the saved 100x JPEG
(display gamma: ratios are taken to the camera gamma of the 10x profile to compare with the model)."""
import json, os, re, collections, sys
import numpy as np, cv2
from paths import DATA, OUT

SHARE = os.environ.get("FLAKEID_SHARE", "Z:/scanresults")
RUNS = ("graphite_20261019_run_1", "graphite_20261019_run_5")


def measure(jpg, reg):
    im = cv2.imdecode(np.fromfile(jpg, np.uint8), cv2.IMREAD_COLOR)
    if im is None or not reg.get("terrace"):
        return None
    px = reg["px_um"]
    T = np.zeros(im.shape[:2], np.uint8)
    for p in reg["terrace"]:
        cv2.fillPoly(T, [np.array(p, np.float64).reshape(-1, 2).round().astype(np.int32)], 1)
    inner = cv2.erode(T, np.ones((max(3, int(1 / px)),) * 2, np.uint8)) > 0
    outer = cv2.dilate(T, np.ones((int(5 / px),) * 2, np.uint8)) == 0
    if inner.sum() < 500 or outer.sum() < 5000:
        return None
    rgb = im[..., ::-1].astype(np.float64)
    o = rgb[outer]
    q = (o // 4).astype(int)                           # the commonest colour (4-level bins) outside the flake
    key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    mode = np.bincount(key).argmax()
    sub = np.median(o[key == mode], axis=0)
    near = np.abs(o / sub - 1).max(1) < 0.03           # bare substrate: within 3 % of that colour
    sub = np.median(o[near], axis=0)
    ter = np.median(rgb[inner], axis=0)
    return dict(contrast=(ter / sub - 1).round(4).tolist(), terrace_um2=reg.get("terrace_um2"), sub_frac=float(near.mean()))


rows = []
for run in RUNS:
    for s in json.load(open(DATA / "share100" / "index.json", encoding="utf-8")):
        if s["run"] != run:
            continue
        m = re.search(r"_F\d+_(\d+)(L|nm)_", s["file"])
        if not m:
            continue
        reg = json.load(open(os.path.join(SHARE, s["file"][:-4] + "_regions.json"), encoding="utf-8"))
        r = measure(os.path.join(SHARE, s["file"]), reg)
        if r:
            r.update(key=s["key"], run=run, file=s["file"], call=m.group(1) + m.group(2))
            rows.append(r)
json.dump(rows, open(OUT / "layers100.json", "w"))
for call in sorted({r["call"] for r in rows}, key=lambda c: (c[-1] != "L", int(re.sub(r"\D", "", c)))):
    c = np.array([r["contrast"] for r in rows if r["call"] == call])
    if len(c) >= 3:
        print(f"{call:6s} n {len(c):3d}  median contrast R {100 * np.median(c[:, 0]):+6.1f} %  G {100 * np.median(c[:, 1]):+6.1f} %  "
              f"B {100 * np.median(c[:, 2]):+6.1f} %   G spread (IQR) {100 * (np.percentile(c[:, 1], 75) - np.percentile(c[:, 1], 25)):.1f} %")
