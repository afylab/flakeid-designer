"""Choose AFM thickness-calibration targets on a Micro-Manager scan and prepare the AFM page.

    python tools/make_afm_set.py "testing images/bigbntest_2_tiles" "testing images/bigbntest_2_results" \
        --flat "example images/micromanager_test/flat_6.3x.npz" --img100 "testing images/bigbntest_2_100x" -o afm_set

Selection (for fitting the RAW16 camera profile): terraces spread over the predicted thickness range
(bins below), each large, flat, isolated and with a straight stretch of edge on clean bare substrate,
where the AFM scan goes (an edge, not the centre: the step height is the measurement).  Flakes with
several terraces in different bins, terraces whose colour also fits another interference order, and
flakes imaged at 100x are preferred.  The list starts with three reference flakes far apart (to fit the
microscope -> AFM stage transform) and then follows a short stage path.

Writes <out>/data.json and images (10x close-up and context crops, 100x frame, chip overview) for
tools/afm_page.html.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from flakeid.mm import StageMap, load_dataset  # noqa: E402
from flakeid.targets import order_path  # noqa: E402

# (lo, hi, number of flakes, priority) in predicted nm
BINS = [(2, 6, 2, 1), (6, 12, 2, 1), (12, 20, 2, 1), (20, 30, 2, 1), (30, 45, 2, 1), (45, 60, 2, 1),
        (60, 80, 2, 1), (80, 100, 2, 1), (100, 135, 1, 2), (135, 175, 2, 1), (175, 260, 1, 2)]
DEMO_SUB = np.array([220.0, 151.0, 192.0])
SCAN_UM = 25.0


def viewer_dir(results: Path) -> Path:
    """The chip viewer's folder of a piece's results: <piece>/<name>_tiles_viewer/ (the analysis's own results folder)
    or the piece folder itself (the copy on the share: viewer on top, tables in data/)."""
    results = Path(results)
    if (results / "index.html").is_file():
        return results
    return next(results.glob("*_viewer"))


def results_file(results: Path, pattern: str) -> Path:
    """A results table (e.g. *_live_terraces.csv) in either layout (next to the viewer folder, or in data/)."""
    results = Path(results)
    for d in (results, results / "data"):
        hit = next(d.glob(pattern), None) if d.is_dir() else None
        if hit is not None:
            return hit
    raise FileNotFoundError(f"no {pattern} in {results} or {results / 'data'}")


def load_viewer_data(results: Path):
    v = viewer_dir(results) / "index.html"
    h = v.read_text()
    i = h.index("const D=") + 8
    return json.loads(h[i:h.index(";\n", i)])


def knee(d, K=200.0):
    v = d.copy()
    m = v > K
    v[m] = K + (255 - K) * (1 - np.exp(-(v[m] - K) / (2.5 * (255 - K))))
    return np.clip(v, 0, 255).astype(np.uint8)


def to_display(rgb16, sub):
    """mosaic crop (linear) -> 8-bit with the substrate at the designer_demo hBN substrate colour."""
    return knee(rgb16.astype(np.float32) / sub * DEMO_SUB)


def poly_mask(polys, x0, y0, shape):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        a = (np.array(p, np.float32).reshape(-1, 2) - [x0, y0])
        cv2.fillPoly(m, [np.round(a).astype(np.int32)], 1)
    return m.astype(bool)


def edge_site(T, other_ok, px_um, scan_um=SCAN_UM):
    """Best scan centre on the boundary of terrace mask T: outside it bare substrate (or other_ok), inside the
    terrace, and the scan box clean.  Returns (x, y, quality 0..1) in crop px, or None."""
    T8 = T.astype(np.uint8)
    din = cv2.distanceTransform(np.pad(T8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * px_um
    dout = cv2.distanceTransform(np.pad(1 - T8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * px_um
    # the outline is blurred over ~2-3 um at 10x and bright flakes have a glare rim: judge the outside 4-12 um out
    inb = (din >= 2) & (din <= 8)
    outb = (dout >= 4) & (dout <= 12)
    k = max(3, int(round(scan_um / px_um)) | 1)
    box = lambda m: cv2.boxFilter(m.astype(np.float32), -1, (k, k), normalize=False)   # noqa: E731
    good_out = box(outb & other_ok) / np.maximum(box(outb), 1)
    good_in = box(inb & T) / np.maximum(box(inb), 1)
    purity = box(T | other_ok) / (k * k)
    s = good_out * good_in * purity
    bnd = T & ~cv2.erode(T8, np.ones((3, 3), np.uint8)).astype(bool)
    # some terrace and some outside in the box (the edge runs through it), and at least half a box from the crop border
    half = k // 2
    bnd[:half, :] = bnd[-half:, :] = False
    bnd[:, :half] = bnd[:, -half:] = False
    if not bnd.any():
        return None
    ys, xs = np.nonzero(bnd)
    j = int(np.argmax(s[ys, xs]))
    return float(xs[j]), float(ys[j]), float(s[ys[j], xs[j]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles")
    ap.add_argument("results")
    ap.add_argument("--flat")
    ap.add_argument("--img100", help="100x targets folder (images of some of these flakes)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--edge-margin", type=float, default=400.0, help="skip flakes this close to the chip edge (um)")
    ap.add_argument("--cache", help="keep the mosaic in this .npy (re-runs skip the stitching)")
    ap.add_argument("--pre", type=int, default=8, help="candidates per bin examined in the image")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    D = load_viewer_data(Path(a.results))
    fj = json.loads(next(Path(a.results).glob("*_flakes.json")).read_text())
    smap = StageMap.from_dict(fj["metadata"]["stage_map"])
    T = {int(k): v for k, v in D["terraces"].items()}
    F = {f["id"]: f for f in D["flakes"]}
    px_um = D["pixel_um"]

    # 100x frames by flake id
    im100 = {}
    if a.img100:
        for r in csv.DictReader(open(next(Path(a.img100).glob("*_100x_index.csv")))):
            fid = int(r["label"].split("_F")[1].split("_")[0])
            im100[fid] = (Path(a.img100) / r["file"], r.get("focus", ""))

    # ---- pre-selection from the table
    def base_score(t):
        f = F[t["fl"]]
        q = set(t["q"].split(";"))
        if q & {"saturated", "chip_edge", "irregular_shape", "poor_colour_match"}:
            return -1
        s = min(t["d"], 45.0) / 45.0
        s *= 1.0 if "low_confidence" not in q else 0.6
        s *= 1.0 / (1.0 + 10 * (f["crowd"] or 0))
        s *= 1.0 + 0.15 * (f["n"] - 1 if f["n"] > 1 else 0)
        s *= 1.25 if t["fl"] in im100 else 1.0
        s *= 1.2 if "ambiguous_thickness" in q else 1.0
        return s

    print("loading the mosaic ...")

    cache = Path(a.cache) if a.cache else None
    if cache is not None and cache.exists():
        rgb = np.load(cache, mmap_mode="r")
    else:
        img, _ = load_dataset(a.tiles, flat=a.flat, verbose=False)
        rgb = img.rgb
        if cache is not None:
            np.save(cache, rgb)
    sub_ref = np.array(fj["params"]["substrate_rgb"], np.float32)
    H, W = rgb.shape[:2]
    # distance to the chip edge (dark surround or no data): the residue band along the edge is often
    # mistaken for thin flakes, and edges are awkward on the AFM
    es = 0.05
    small = cv2.resize(np.asarray(rgb[::8, ::8]), None, fx=es * 8, fy=es * 8, interpolation=cv2.INTER_AREA).astype(np.float32)
    chip = (small.mean(2) > 0.5 * float(sub_ref.mean())).astype(np.uint8)
    chip = cv2.morphologyEx(chip, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edge_dist = cv2.distanceTransform(np.pad(chip, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] / es * px_um     # um

    def far_from_edge(f, um=a.edge_margin):
        # the whole flake (its box) away from the edge; and no giant "flakes" (residue bands, clusters)
        if f["n"] > 8 or max(f["bw"], f["bh"]) * px_um > 800:
            return False
        for x, y in ((f["bx"], f["by"]), (f["bx"] + f["bw"], f["by"]), (f["bx"], f["by"] + f["bh"]),
                     (f["bx"] + f["bw"], f["by"] + f["bh"]), (f["tx"], f["ty"])):
            xi, yi = int(x * es), int(y * es)
            if not (0 <= yi < edge_dist.shape[0] and 0 <= xi < edge_dist.shape[1]) or edge_dist[yi, xi] < um:
                return False
        return True

    pre = []
    for lo, hi, n, pr in BINS:
        c = sorted((t for t in T.values() if lo <= t["t"] < hi and t["d"] >= 12), key=lambda t: -base_score(t))
        c = [t for t in c if base_score(t) > 0 and far_from_edge(F[t["fl"]])][: a.pre]
        pre += [(lo, hi, t) for t in c]
    print(f"{len(pre)} pre-candidates")

    def crop(x0, y0, x1, y1):
        x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), W), min(int(y1), H)
        return rgb[y0:y1, x0:x1], x0, y0

    def local_sub(c):
        v = c.reshape(-1, 3).astype(np.float32)
        r = v / sub_ref
        near = np.abs(r - 1).max(1) < 0.08
        return np.median(v[near], 0) if near.sum() > 200 else sub_ref

    # ---- edge sites for the pre-candidates
    rated = []
    n_edge = 0
    for lo, hi, t in pre:
        f = F[t["fl"]]
        if not far_from_edge(f):
            n_edge += 1
            continue
        m = 80
        c, x0, y0 = crop(f["bx"] - m, f["by"] - m, f["bx"] + f["bw"] + m, f["by"] + f["bh"] + m)
        if c.size == 0:
            continue
        sub = local_sub(c)
        cs = cv2.GaussianBlur(c.astype(np.float32), (0, 0), 1.0) / sub - 1
        substrate = np.abs(cs).max(2) < 0.06
        Tm = poly_mask(t["poly"], x0, y0, c.shape[:2])
        if Tm.sum() < 50:
            continue
        site = edge_site(Tm, substrate, px_um)
        kind = "substrate"
        if site is None or site[2] < 0.6:
            # step to another terrace of the same flake
            sib = np.zeros_like(Tm)
            for tid in f["terr"]:
                if tid != t["id"]:
                    sib |= poly_mask(T[tid]["poly"], x0, y0, c.shape[:2])
            s2 = edge_site(Tm, sib | substrate, px_um) if sib.any() else None
            if s2 is not None and (site is None or s2[2] > site[2] + 0.1):
                site, kind = s2, "step"
        if site is None or site[2] < 0.6:
            continue
        rated.append(dict(bin=(lo, hi), t=t, site=(site[0] + x0, site[1] + y0), sq=site[2], kind=kind,
                          score=base_score(t) * (0.5 + site[2])))
    print(f"{len(rated)} with a usable edge site ({n_edge} skipped: within {a.edge_margin:.0f} um of the chip edge)")
    for lo, hi, n, pr in BINS:
        rs = [r for r in rated if r["bin"] == (lo, hi)]
        print(f"   {lo:3d}-{hi:<3d} nm: {len(rs):2d} usable" + (f", best edge {max(r['sq'] for r in rs):.2f}" if rs else ""))

    # ---- final pick: per bin, distinct flakes; a flake chosen for one bin may also serve another
    chosen, used_fl = [], set()
    for lo, hi, n, pr in BINS:
        cand = sorted((r for r in rated if r["bin"] == (lo, hi) and r["t"]["fl"] not in used_fl), key=lambda r: -r["score"])
        for r in cand[:n]:
            r["priority"] = pr
            chosen.append(r)
            used_fl.add(r["t"]["fl"])
    # group by flake: every chosen terrace of a flake, plus its other terraces worth measuring in the same scan area
    flakes = {}
    for r in chosen:
        flakes.setdefault(r["t"]["fl"], []).append(r)
    print(f"{len(chosen)} terraces on {len(flakes)} flakes")

    # ---- order: 3 reference flakes far apart, then a short path from the last one
    ids = list(flakes)
    P = np.array([[F[i]["sx"], F[i]["sy"]] for i in ids])
    best, refs = -1, None
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            for k in range(j + 1, len(ids)):
                area = abs(np.cross(P[j] - P[i], P[k] - P[i])) / 2
                big = min(F[ids[q]]["bd"] for q in (i, j, k))
                sc = area * min(big, 40)
                if sc > best:
                    best, refs = sc, [i, j, k]
    # visit the references in a short loop, then the rest from the last reference
    refs = [refs[q] for q in order_path(P[refs])]
    rest = [q for q in range(len(ids)) if q not in refs]
    start = P[refs[-1]]
    sub_P = np.vstack([start[None], P[rest]])
    path = order_path(sub_P)[1:] if len(rest) else []
    order = refs + [rest[q - 1] for q in path]

    # ---- images and records
    ov_s = 3000.0 / max(W, H)
    ov = cv2.resize(rgb, None, fx=ov_s, fy=ov_s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "img" / "overview.jpg"), cv2.cvtColor(to_display(ov, sub_ref), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    from rank100x import load_contrast  # noqa: E402
    flat100 = ROOT / "testing images/bigbntest_100x_score/flat100x.npy"
    flat100 = np.load(flat100) if flat100.exists() else None
    recs = []
    for n_, q in enumerate(order, 1):
        fid = ids[q]
        f = F[fid]
        rs = flakes[fid]
        # close-up: flake + margin, at full resolution (max 900 px)
        m = max(60, int(0.4 * max(f["bw"], f["bh"])))
        c, x0, y0 = crop(f["bx"] - m, f["by"] - m, f["bx"] + f["bw"] + m, f["by"] + f["bh"] + m)
        s = min(1.0, 900.0 / max(c.shape[:2]))
        disp = to_display(c, local_sub(c))
        disp = cv2.resize(disp, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else disp
        cv2.imwrite(str(out / "img" / f"f{fid}_close.jpg"), cv2.cvtColor(disp, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        close = dict(x0=x0, y0=y0, s=s, w=disp.shape[1], h=disp.shape[0])
        # context: 1.6 mm around, at ~3.6 um/px
        cx, cy = f["tx"], f["ty"]
        R = int(800 / px_um)
        c2, x2, y2 = crop(cx - R, cy - R, cx + R, cy + R)
        s2 = 440.0 / max(c2.shape[:2])
        d2 = cv2.resize(to_display(c2, local_sub(c2)), None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "img" / f"f{fid}_ctx.jpg"), cv2.cvtColor(d2, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        ctx = dict(x0=x2, y0=y2, s=s2, w=d2.shape[1], h=d2.shape[0])
        has100 = False
        if fid in im100 and flat100 is not None:
            C, *_ = load_contrast(im100[fid][0], flat100, [705.0, 905.0, 588.0])
            d3 = knee((C + 1) * DEMO_SUB.astype(np.float32))
            d3 = cv2.resize(d3, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "img" / f"f{fid}_100x.jpg"), cv2.cvtColor(d3, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            has100 = True
        terr = []
        chosen_ids = {r["t"]["id"]: r for r in rs}
        for tid in f["terr"]:
            t = T[tid]
            r = chosen_ids.get(tid)
            site = None
            if r is not None:
                sx_, sy_ = smap.to_stage(*r["site"])
                site = dict(px=[round(r["site"][0], 1), round(r["site"][1], 1)], stage=[round(sx_, 1), round(sy_, 1)],
                            quality=round(r["sq"], 2), kind=r["kind"])
            terr.append(dict(id=tid, t=t["t"], tlo=t["tlo"], thi=t["thi"], alt=t["alt"], d=t["d"], q=t["q"], c=t["c"],
                             poly=t["poly"], target=bool(r), priority=r["priority"] if r else None,
                             bin=list(r["bin"]) if r else None, site=site))
        recs.append(dict(n=n_, ref=q in refs, flake=fid, score=f["score"], best_t=f["t"], area=f["area"], d=f["bd"],
                         stage=[f["sx"], f["sy"]], target_px=[f["tx"], f["ty"]], bbox=[f["bx"], f["by"], f["bw"], f["bh"]],
                         poly=f["poly"], terraces=terr, close=close, ctx=ctx, has100=has100,
                         focus100=im100.get(fid, (None, ""))[1]))
    travel = sum(np.hypot(*(np.array(recs[i]["stage"]) - recs[i - 1]["stage"])) for i in range(1, len(recs)))
    meta = dict(scan=Path(a.tiles).name, material=D["material"], oxide_fit=D["oxide"], nominal_oxide=285,
                pixel_um=px_um, mosaic_wh=[W, H], overview=dict(file="img/overview.jpg", s=ov_s),
                stage_map=fj["metadata"]["stage_map"], scan_um=SCAN_UM, travel_mm=round(travel / 1000, 1),
                bins=[dict(lo=lo, hi=hi, n=n, priority=pr) for lo, hi, n, pr in BINS])
    (out / "data.json").write_text(json.dumps(dict(meta=meta, flakes=recs), separators=(",", ":")))
    print(f"{len(recs)} flakes, {sum(1 for r in recs for t in r['terraces'] if t['target'])} target terraces, "
          f"path {travel / 1000:.1f} mm; refs {[recs[i]['flake'] for i in range(3)]}")


if __name__ == "__main__":
    main()
