"""Shot-quality check of the 100x pictures (experiment E): which shots do not show the targeted flake well, and how
much of the screening and targeting error they account for.  python shotcheck.py -> results/shotcheck.json + report

Flags, from each shot's _regions.json (the scope's own 100x analysis; read from the share):
  whole_frame   terrace a covers >= 95 % of the frame (a flake larger than the field, or substrate taken as terrace)
  not_found     no terrace of the 10x flake's colour near the expected place (seed.found false)
  no_fit        the terrace found does not match the 10x terrace's size / colour (seed.fits false)
  off_centre    the terrace was found more than 25 um from where the stage put the 10x flake (seed.offset_um)
  defocused     the strongest edges in the picture are soft for its contrast range: 99.5th percentile of the
                gradient over the 0.5 - 99.5 percentile range of brightness < 0.25 (share100 pictures, 384 x 512;
                sharp shots 1 - 2.5).  The screen's own focus number (Laplacian variance around terrace a) is no use
                here: large clean flakes with few edges in view score as low as blurred ones
  faint         almost no contrast in the picture (brightness range < 20 grey levels): nothing visible, or a
                1 - 2 layer graphite flake (these are kept more often than the rest), so not a fault and not in
                'any'.  Kept graphite flakes flagged whole_frame are mostly such thin flakes the terrace finder lost.
Screening error: held-out combined 100x scores (z_<set>_bundle.npz); a 'confident error' is a discard in its run's top
10 % or a keep in its bottom 10 %."""
import csv, json, os, sys, collections
import numpy as np, cv2
from sklearn.metrics import roc_auc_score
from paths import DATA, OUT, SETS

SHARE = os.environ.get("FLAKEID_SHARE", "Z:/scanresults")
FLAGS = ["whole_frame", "not_found", "no_fit", "off_centre", "defocused", "faint"]
FAULTS = FLAGS[:5]


def read():
    rows = []
    for s in json.load(open(DATA / "share100" / "index.json", encoding="utf-8")):
        p = os.path.join(SHARE, s["file"][:-4] + "_regions.json")
        try:
            r = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sd = r.get("seed") or {}
        fw, fh = r.get("frame_wh") or [0, 0]
        px = r.get("px_um") or 0
        sc = (r.get("screen") or {}).get("features") or {}
        rows.append(dict(key=s["key"], run=s["run"], material=s["material"], file=s["file"],
                         terrace_um2=r.get("terrace_um2") or 0.0, frame_um2=fw * fh * px * px,
                         found=sd.get("found"), fits=sd.get("fits"), moved=sd.get("moved"),
                         offset=float(np.hypot(*(sd.get("offset_um") or [0, 0]))), focus=sc.get("focus"),
                         t_nm=sc.get("t_nm") or r.get("t_nm"), tier=(r.get("screen") or {}).get("tier")))
    return rows


def flag(rows):
    mat = {}                                           # shots outside a designer bundle: their piece's material
    for r in rows:
        if r["material"]:
            mat.setdefault(r["file"].split("/")[1], r["material"])
    for r in rows:
        r["material"] = r["material"] or mat.get(r["file"].split("/")[1]) or ("graphite" if (r["t_nm"] or 99) < 8 else "hbn")
        r["whole_frame"] = r["frame_um2"] > 0 and r["terrace_um2"] >= 0.95 * r["frame_um2"]
        r["not_found"] = r["found"] is False
        r["no_fit"] = r["fits"] is False
        r["off_centre"] = r["offset"] > 25
    I = {s["file"]: s["i"] for s in json.load(open(DATA / "share100" / "index.json", encoding="utf-8"))}
    X = np.load(str(DATA / "share100" / "X.npy"), mmap_mode="r")
    for r in rows:
        L = cv2.GaussianBlur(cv2.cvtColor(np.asarray(X[I[r["file"]]]), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.7)
        g = cv2.magnitude(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))
        lo, hi = np.percentile(L, [0.5, 99.5])
        r["contrast"], r["sharp"] = float(hi - lo), float(np.percentile(g, 99.5) / max(hi - lo, 1.0))
        r["faint"] = r["contrast"] < 20
        r["defocused"] = not r["faint"] and r["sharp"] < 0.25
    for r in rows:
        r["any"] = any(r[f] for f in FAULTS)
    return rows


def std(z, runs):
    o = z.astype(float).copy()
    for r in np.unique(runs):
        s = runs == r; o[s] = (z[s] - z[s].mean()) / (z[s].std() + 1e-9)
    return o


def main():
    rows = flag(read())
    json.dump(rows, open(OUT / "shotcheck.json", "w", encoding="utf-8"))
    print(f"{len(rows)} shots read\n\nshare of shots flagged, per run:")
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["run"], r["material"])].append(r)
    print(f"  {'run':28s} {'mat':8s} {'shots':>5s} " + " ".join(f"{f:>11s}" for f in FLAGS + ["any"]))
    for (run, m), R in sorted(by.items()):
        print(f"  {run:28s} {m:8s} {len(R):5d} " + " ".join(f"{100 * np.mean([r[f] for r in R]):10.0f}%" for f in FLAGS + ["any"]))
    print(f"  {'all':28s} {'':8s} {len(rows):5d} " + " ".join(f"{100 * np.mean([r[f] for r in rows]):10.0f}%" for f in FLAGS + ["any"]))
    shot = {}
    for r in rows:
        shot.setdefault(r["key"], r)
    out = {}
    for mat in ("graphite", "hbn"):                    # 100x screening on the decided flakes
        Z = np.load(OUT / f"z_{mat}_bundle.npz")
        keys = sorted(json.load(open(OUT / "cache_c_keys.json")))
        G = {s["key"] for s in json.load(open(SETS / "graphite100.json")) if s["ok"]}
        H = {r["key"] for r in csv.DictReader(open(SETS / "hbn100.csv", encoding="utf-8"))}
        keys = np.array(sorted(G if mat == "graphite" else H))
        y, runs, z = Z["y"], Z["runs"], Z["ft_bundle_both"]
        has = np.array([k in shot for k in keys])
        F = {f: np.array([bool(shot[k][f]) if k in shot else False for k in keys]) for f in FLAGS + ["any"]}
        pct = np.zeros(len(y))
        for r in np.unique(runs):
            s = np.where(runs == r)[0]; pct[s] = (np.argsort(np.argsort(z[s])) + 0.5) / len(s)
        err = ((y == 0) & (pct > 0.9)) | ((y == 1) & (pct < 0.1))
        print(f"\n{mat}: {len(y)} decided flakes, {has.sum()} with a shot on the share; {err.sum()} confident errors")
        for f in FLAGS + ["any"]:
            s = F[f] & has; u = ~F[f] & has
            if s.sum() == 0:
                continue
            a_s = roc_auc_score(y[s], z[s]) if 0 < y[s].sum() < s.sum() else float("nan")
            print(f"  {f:11s} {s.sum():4d} flakes ({100 * s.mean():4.1f} %), kept {100 * y[s].mean():4.1f} % (rest {100 * y[u].mean():4.1f} %), "
                  f"confident errors {err[s].sum():3d} = {100 * err[s].mean():4.1f} % of them (rest {100 * err[u].mean():4.1f} %), "
                  f"AUC flagged {a_s:.3f} / rest {roc_auc_score(y[u], z[u]):.3f}")
        s = F["any"] & has
        out[mat] = dict(flagged=int(s.sum()), n=int(has.sum()), err_flagged=int(err[s].sum()), err=int(err[has].sum()),
                        auc_all=round(roc_auc_score(y[has], z[has]), 3), auc_clean=round(roc_auc_score(y[has & ~s], z[has & ~s]), 3),
                        examples=[str(k) for k in keys[s & err]][:12])
    L = list(csv.DictReader(open(SETS / "target_labels_10x.csv", encoding="utf-8")))
    print("\n10x targets with a shot: outcome by flag (share of shots)")
    for f in FLAGS + ["any"]:
        c = collections.Counter(r["outcome"] for r in L if r["key"] in shot and shot[r["key"]][f])
        u = collections.Counter(r["outcome"] for r in L if r["key"] in shot and not shot[r["key"]][f])
        n, m = sum(c.values()), sum(u.values())
        if n:
            print(f"  {f:11s} {n:4d}: keep {100 * c['keep'] / n:4.1f} %, dropped {100 * c['dropped'] / n:4.1f} %, discard {100 * c['discard'] / n:4.1f} %"
                  f"   (rest: keep {100 * u['keep'] / m:4.1f} %, dropped {100 * u['dropped'] / m:4.1f} %)")
    json.dump(out, open(OUT / "shotcheck_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
