"""10x crops (images/c150, as prep/crops10.py cuts them: 144 px at level max, 1.09 um/px) for the 100x shots on the share
that set10.json has none for (imaged flakes without an outcome label), from the live flake tables (live10/) and the
viewer tile pyramids on the share.  python prep/crops10_share.py"""
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA

SHARE = Path(os.environ.get("FLAKEID_SHARE", "Z:/scanresults"))
OUT = DATA / "images" / "c150"


class Pyr:
    def __init__(self, run, pc):
        self.d = SHARE / run / pc / "tiles" / "image"
        self.top = max(int(p.name) for p in self.d.iterdir() if p.name.isdigit())
        self.cache = {}

    def tile(self, L, c, r):
        k = (L, c, r)
        if k not in self.cache:
            p = self.d / str(L) / f"{c}_{r}.jpg"
            self.cache[k] = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) if p.is_file() else None
        return self.cache[k]

    def crop(self, L, x, y, n):
        """n x n px centred on (x, y) in level-L pixels (black outside)."""
        out = np.zeros((n, n, 3), np.uint8)
        x0, y0 = int(round(x - n / 2)), int(round(y - n / 2))
        for r in range(max(0, y0 // 256), (y0 + n - 1) // 256 + 1):
            for c in range(max(0, x0 // 256), (x0 + n - 1) // 256 + 1):
                t = self.tile(L, c, r)
                if t is None:
                    continue
                tx, ty = c * 256, r * 256
                ax0, ay0 = max(x0, tx), max(y0, ty)
                ax1, ay1 = min(x0 + n, tx + t.shape[1]), min(y0 + n, ty + t.shape[0])
                if ax1 > ax0 and ay1 > ay0:
                    out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = t[ay0 - ty:ay1 - ty, ax0 - tx:ax1 - tx]
        return out


have = {s["key"] for s in json.load(open(DATA / "sets" / "set10.json"))}
todo = {}
for s in json.load(open(DATA / "share100" / "index.json", encoding="utf-8")):
    if s["key"] not in have and "-f" in s["key"]:
        todo.setdefault((s["run"], s["piece"]), set()).add(int(s["key"].rsplit("-f", 1)[1]))
n = miss = 0
for (run, pc), fids in sorted(todo.items()):
    fl = glob.glob(str(DATA / "live10" / run / pc / "*_tiles_live_flakes.csv"))
    if not fl:
        print(run, pc, "no live table:", len(fids), "skipped"); miss += len(fids)
        continue
    pos = {int(r["flake_id"]): (float(r["x_px"]), float(r["y_px"])) for r in csv.DictReader(open(fl[0], encoding="utf-8"))}
    P = Pyr(run, pc)
    for f in sorted(fids):
        if f not in pos:
            miss += 1
            continue
        p = OUT / f"{run}__{pc}-f{f:04d}.png"
        if not p.is_file():
            cv2.imwrite(str(p), P.crop(P.top, *pos[f], 144))
        n += 1
    print(run, pc, len(fids), flush=True)
print("crops", n, "missing", miss)
