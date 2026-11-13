"""Every 100x shot on the share with its rule outputs, for setup C's pretraining: python prep/share100.py
-> <bundle>/share100/: X.npy (pictures 384 x 512, uint8 BGR), M.npy (masks 96 x 128: terrace, debris, steps, fine
terraces; uint8 0/1), index.json (key run:piece-fNNNN, run, material, file, screen features of _regions.json).
Reads the share once (slow: ~1 MB per shot); material from the run's designer bundle, else the run's name."""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA

SHARE = os.environ.get("FLAKEID_SHARE", "Z:/scanresults")
OUT = DATA / "share100"; OUT.mkdir(exist_ok=True)
H, W, MH, MW = 384, 512, 96, 128
SUF = re.compile(r"_(avg|c8|sg|debris|dt|avg_c8|avg_sg|mosaic)\.jpg$")
FID = re.compile(r"_F(\d+)_")

shots = []
for run in sorted(os.listdir(SHARE)):
    rd = os.path.join(SHARE, run)
    if run.startswith("_") or not os.path.isdir(rd):
        continue
    mat = {}
    bj = os.path.join(rd, "designer", "bundle.json")
    if os.path.isfile(bj):
        mat = {f["id"]: f.get("material") for f in json.load(open(bj, encoding="utf-8"))["flakes"]}
    for piece in sorted(os.listdir(rd)):
        h = os.path.join(rd, piece, "100x")
        if not os.path.isdir(h):
            continue
        fs = set(os.listdir(h))
        for f in sorted(fs):
            reg = f[:-4] + "_regions.json"
            if not f.endswith(".jpg") or SUF.search(f) or reg not in fs:
                continue
            m = FID.search(f)
            key = f"{run}:{piece}-f{int(m.group(1)):04d}" if m else f"{run}:{piece}-{f[:-4]}"
            mt = mat.get(key) or ("graphite" if "graphite" in run else "hbn" if run.startswith("bn_") else None)
            shots.append(dict(key=key, run=run, piece=piece, material=mt, file=f"{run}/{piece}/100x/{f}", reg=os.path.join(h, reg)))
print(len(shots), "shots;", {m: sum(s["material"] == m for s in shots) for m in {s["material"] for s in shots}}, flush=True)

X = np.lib.format.open_memmap(str(OUT / "X.npy"), "w+", np.uint8, (len(shots), H, W, 3))
M = np.lib.format.open_memmap(str(OUT / "M.npy"), "w+", np.uint8, (len(shots), 4, MH, MW))
index, t0 = [], time.time()
for i, s in enumerate(shots):
    r = json.load(open(s.pop("reg"), encoding="utf-8"))
    im = cv2.imdecode(np.fromfile(os.path.join(SHARE, s["file"]), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        continue
    fh, fw = im.shape[:2]
    X[i] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    sx, sy = MW / fw, MH / fh
    for c, polys in enumerate((r.get("terrace") or [], r.get("debris") or [], r.get("steps") or [],
                               [p["poly"][0] if isinstance(p.get("poly"), list) and p["poly"] and isinstance(p["poly"][0], list)
                                else p.get("poly") or [] for p in r.get("fine") or []])):
        hi = np.zeros((MH * 4, MW * 4), np.uint8)       # drawn at 4x, then area-averaged: thin steps keep a trace
        for p in polys:
            if len(p) >= 6:
                pts = (np.array(p, np.float64).reshape(-1, 2) * [sx * 4, sy * 4]).round().astype(np.int32)
                cv2.fillPoly(hi, [pts], 1)
        M[i, c] = cv2.resize(hi, (MW, MH), interpolation=cv2.INTER_AREA) > 0
    s["screen"] = (r.get("screen") or {}).get("features") or {}
    s["i"] = i
    index.append(s)
    if i % 100 == 0:
        print(i, len(shots), f"{time.time() - t0:.0f} s", flush=True)
X.flush(); M.flush()
json.dump(index, open(OUT / "index.json", "w", encoding="utf-8"), indent=0)
print("done", len(index), f"{time.time() - t0:.0f} s")
