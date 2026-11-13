"""Every graphite flake of the built designer bundles on the share scored by the network (results/graphite_net.onnx through
OpenCV, as the build would) and tiered with results/graphite_combo.json, next to its current tier (pick model v3).
python score_bundles.py -> results/graphite_bundle_scores.json + a table per run.  Reads only; writes nothing to the share."""
import json, os, collections, time
import numpy as np, cv2
from paths import SETS, OUT

SHARE = os.environ.get("FLAKEID_SHARE", "Z:/scanresults")
combo = json.load(open(OUT / "graphite_combo.json"))
a, b = combo["coef"][1], combo["intercept"]            # (the pick logit's weight is 0)
net = cv2.dnn.readNetFromONNX(str(OUT / "graphite_net.onnx"))
dec = {s["key"]: s["y"] for s in json.load(open(SETS / "graphite100.json")) if s["ok"]}


def tier(p):
    return "out" if p < combo["reject_below"] else ("keep" if p >= combo["keep_above"] else "review")


rows, t0 = [], time.time()
for run in sorted(os.listdir(SHARE)):
    d = os.path.join(SHARE, run, "designer")
    if run.startswith("_") or not os.path.isfile(os.path.join(d, "bundle.json")):
        continue
    B = json.load(open(os.path.join(d, "bundle.json"), encoding="utf-8"))
    atl = {}
    for f in B["flakes"]:
        if f.get("material") != "graphite":
            continue
        pic = f.get("img")
        if isinstance(pic, dict):
            if pic["s"] not in atl:
                atl[pic["s"]] = cv2.imdecode(np.fromfile(os.path.join(d, pic["s"]), np.uint8), cv2.IMREAD_COLOR)
            im = atl[pic["s"]][pic["y"]:pic["y"] + pic["h"], pic["x"]:pic["x"] + pic["w"]]
        elif pic:
            im = cv2.imdecode(np.fromfile(os.path.join(d, pic), np.uint8), cv2.IMREAD_COLOR)
        else:
            continue
        if im is None:
            continue
        x = cv2.resize(im, (512, 384), interpolation=cv2.INTER_AREA).astype(np.float32).transpose(2, 0, 1)[None]
        net.setInput(x)
        z = float(net.forward()[0, 0])
        p = 1 / (1 + np.exp(-(a * z + b)))
        t = (f.get("terraces") or [{}])[0]
        old = t.get("tier") or ""
        rows.append(dict(key=f["id"], run=run, net_logit=round(z, 3), p_new=round(float(p), 4), tier_new=tier(p),
                         p_old=t.get("p_keep"), tier_old={"reject": "out"}.get(old, old), y=dec.get(f["id"])))
    print(run, sum(r["run"] == run for r in rows), f"{time.time() - t0:.0f} s", flush=True)
json.dump(rows, open(OUT / "graphite_bundle_scores.json", "w"), indent=0)

print("\ncurrent tier -> new tier (all graphite flakes in the bundles; decided: kept / discarded)")
for run in sorted({r["run"] for r in rows}) + ["all"]:
    R = [r for r in rows if run in ("all", r["run"])]
    c = collections.Counter((r["tier_old"], r["tier_new"]) for r in R)
    old, new = collections.Counter(r["tier_old"] for r in R), collections.Counter(r["tier_new"] for r in R)
    print(f"\n{run}: {len(R)} flakes ({sum(r['y'] == 1 for r in R)} kept, {sum(r['y'] == 0 for r in R)} discarded, "
          f"{sum(r['y'] is None for r in R)} undecided)")
    print(f"  now: {dict(old)}   new: {dict(new)}")
    if run == "all":
        for (o, n), k in sorted(c.items()):
            ks = sum(1 for r in R if (r["tier_old"], r["tier_new"]) == (o, n) and r["y"] == 1)
            ds = sum(1 for r in R if (r["tier_old"], r["tier_new"]) == (o, n) and r["y"] == 0)
            print(f"    {o:>7s} -> {n:7s} {k:4d}   (kept {ks}, discarded {ds})")
