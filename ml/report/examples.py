"""Representative flakes for the summary page -> examples.json (with base64 JPEG thumbnails).
100x: graphite flakes where the network changed the call (kept flakes the hand model ranked low that the combination
ranks high, discards it ranked high that the combination ranks low) and ones both get wrong; hBN: kept flakes the network
alone ranks lowest vs the hand model (what it misses).  10x: highest / lowest network-scored crops with their outcome."""
import base64, csv, json
from pathlib import Path
import numpy as np, cv2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import SETS, OUT, dpath

HERE = Path(__file__).parent


def thumb(p, w=360):
    im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    s = w / im.shape[1]
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()


def pct(z):                     # rank among all flakes of the set, 0..100 (100 = most "keep")
    return 100.0 * (np.argsort(np.argsort(z)) + 0.5) / len(z)


out = {}
# graphite 100x
G = {s["key"]: s for s in json.load(open(SETS / "graphite100.json")) if s["ok"]}
gk = sorted(G)
Z = np.load(OUT / "z_graphite.npz")
y, h, b = Z["y"], pct(Z["hand_s"]), pct(Z["resnet50_32_both"])
n = pct(Z["resnet50_32"])
def pick(mask, order, k):
    idx = np.where(mask)[0]
    return [int(i) for i in idx[np.argsort(order[idx])][:k]]
rows = []
for kind, ids in [("rescued", pick((y == 1) & (h < 40), -(b - h), 4)), ("caught", pick((y == 0) & (h > 60), b - h, 4)),
                  ("missed", pick((y == 1), np.maximum(h, b), 2)), ("fooled", pick((y == 0), -np.minimum(h, b), 2))]:
    for i in ids:
        rows.append(dict(kind=kind, key=gk[i], y=int(y[i]), hand=round(h[i]), net=round(n[i]), both=round(b[i]),
                         img=thumb(dpath(G[gk[i]]["img"]))))
out["graphite100"] = rows
# hBN 100x
R = {r["key"]: r for r in csv.DictReader(open(SETS / "hbn100.csv", encoding="utf-8"))}
hk = sorted(R)
Z = np.load(OUT / "z_hbn.npz")
y, h, n, b = Z["y"], pct(Z["hand_s"]), pct(Z["dinov2s_128"]), pct(Z["dinov2s_128_both"])
rows = []
for kind, ids in [("net_misses", pick((y == 1) & (h > 50), n - h, 4)), ("net_sees", pick((y == 0) & (h > 60), n - h, 4))]:
    for i in ids:
        rows.append(dict(kind=kind, key=hk[i], y=int(y[i]), hand=round(h[i]), net=round(n[i]), both=round(b[i]),
                         img=thumb(dpath(R[hk[i]]["img"]))))
out["hbn100"] = rows
# 10x crops: highest network-scored kept and wasted targets, lowest-scored kept
S = {s["key"]: s for s in json.load(open(SETS / "set10.json"))}
for mat, var in (("hbn", "resnet50_both"), ("graphite", "dinov2s_X150")):
    Z = np.load(OUT / f"z10_{mat}.npz"); keys = Z["keys"]; y = Z["y"]
    n, h = pct(Z[var]), pct(Z["hand_s"])
    rows = []
    for kind, mask, order in (("net_high_kept", y == 1, -n), ("net_high_wasted", y == 0, -n), ("net_low_kept", y == 1, n)):
        idx = np.where(mask)[0]; idx = idx[np.argsort(order[idx])][:3]
        for i in idx:
            s = S[keys[i]]
            rows.append(dict(kind=kind, key=str(keys[i]), y=int(y[i]), outcome=s["outcome"], net=round(n[i]), hand=round(h[i]),
                             img=thumb(dpath(s["c400"]), 240), img150=thumb(dpath(s["c150"]), 240)))
    out[f"{mat}10"] = rows
json.dump(out, open(OUT / "examples.json", "w"))
for k, v in out.items():
    print(k, [(r["kind"], r["key"], r["y"], r["hand"], r["net"], r.get("both")) for r in v])
