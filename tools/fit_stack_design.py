#!/usr/bin/env python3
"""As-built fit of a designed stack: where each layer of a designer plan ended up in pictures of the finished stack.

    python tools/fit_stack_design.py --design <design.json> --runs <scanresults> --image100 <natural 100x.jpg>
        [--wide <Olympus Stream 50x.jpg>] [--wide-rot180] -o <out>

The flakes are the plan's (their 100x outlines in the scan runs' designer bundles, 0.229 um/px).  The pictures: a
100x frame from the scan program's camera orientation (e.g. a capture exported to JPEG; 0.1143 um/px) and optionally
a 50x picture of the whole stack (0.2288 um/px; Olympus Stream pictures are rotated 180 deg against the scan camera:
--wide-rot180).  The 100x frame is placed into the 50x picture (feature match); edges come from the 100x frame where
it covers the stack and from the 50x elsewhere.

1. The whole plan (every layer's outline at its planned pose) is matched against the edges: every rotation, both
   mirror states (a flip-over transfer), every position.
2. Each layer is then refined near the pose step 1 predicts for it (--turn, default +-40 deg; --shift, +-25 um): its outline against the
   edges, with the penalty for covering bare substrate.
DQD5 (2026-10-05, against the hand alignment): the hBN layers and the bottom graphite within 0.4-1.9 um and 1.2 deg;
the 2L graphene 1.3 um but 9 deg (1.2 um / 1.4 deg with --edge-mode both, which costs the 4.3 nm graphite); the 6.6 nm
top graphite wrong in every mode (12-20 um: put on a thick hBN edge).
Writes <out>/overlay.jpg, <out>/fit.json, and <out>/design_asbuilt.json: a copy of the plan with the fitted poses
(each layer keeps its planned pose in `plan`) and the 100x picture placed under them, for the designer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_stack_layers import TRUNC, fit_one, outline_points, substrate_mask  # noqa: E402

UM100 = 0.1143                     # um per pixel, 100x, scan camera (full resolution)
UM = 2 * UM100                     # working scale: the half-resolution 100x = full-resolution 50x = the flake outlines


def rot(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def edges(img, mask_box=None):
    a = cv2.GaussianBlur(img, (0, 0), 1.2)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    e = np.zeros(img.shape[:2], np.uint8)
    for ch in cv2.split(lab):
        e |= cv2.Canny(ch, 12, 30)
    if mask_box is not None:
        x0, y0, x1, y1 = mask_box
        e[y0:y1, x0:x1] = 0
    return e


def log_edges(img, K=400.0, lo=18, hi=40):
    """Edges of the log brightness (each channel): a step of a few % (2L graphene, thin graphite under hBN) is as strong
    here in a dark region as in a bright one, and the noise of dark regions does not grow into edges."""
    a = cv2.GaussianBlur(img.astype(np.float32) + 1.0, (0, 0), 1.2)
    e = np.zeros(img.shape[:2], np.uint8)
    for ch in cv2.split(np.log(a)):
        u8 = np.clip((ch - np.median(ch)) * K + 128, 0, 255).astype(np.uint8)
        e |= cv2.Canny(u8, lo, hi)
    return e


def similarity(src, dst):
    """src -> dst similarity (2x3) from ORB features (contrast-equalised grey)."""
    orb = cv2.ORB_create(5000)

    def kd(im):
        g = cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        return orb.detectAndCompute(g, None)
    k1, d1 = kd(src)
    k2, d2 = kd(dst)
    mt = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.float32([k1[m.queryIdx].pt for m in mt])
    p2 = np.float32([k2[m.trainIdx].pt for m in mt])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    return A, int(inl.sum()), len(mt)


def load_layers(design, runs):
    bundles, out = {}, []
    for L in design["layers"]:
        run, fid = L["tid"].split(":", 1)
        if run not in bundles:
            bundles[run] = json.loads((Path(runs) / run / "designer" / "bundle.json").read_text())
        b = bundles[run]
        f = next(f for f in b["flakes"] if L["tid"].startswith(f["id"] + "."))
        t = next(t for t in f["terraces"] if t["id"] == L["tid"])
        f = dict(f, crop_px=[f["frame"]["x0"], f["frame"]["y0"]])
        out.append(dict(L=L, f=f, t=t, name=f["id"].split(":")[1], c=np.array(t["c"], float)))
    return out


def world_pts(e, P):
    """Cut-out px -> the plan's world um (the designer's convention)."""
    L = e["L"]
    q = (P - e["c"]) * e["f"]["px_um"] * ([-1, 1] if L.get("mirror") else [1, 1])
    return q @ rot(L["rot"]).T + [L["x"], L["y"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--runs", required=True, help="the scan results folder holding the plan's runs (<run>/designer/bundle.json)")
    ap.add_argument("--image100", required=True, help="100x picture in the scan camera's orientation (0.1143 um/px)")
    ap.add_argument("--wide", help="50x picture of the whole stack (0.2288 um/px)")
    ap.add_argument("--wide-rot180", action="store_true", help="the 50x picture is rotated 180 deg (Olympus Stream)")
    ap.add_argument("--mirror", choices=("auto", "no", "yes"), default="auto", help="the stack mirrored against the plan")
    ap.add_argument("--turn", type=float, default=40.0, help="each layer searched within this many deg of its planned rotation")
    ap.add_argument("--shift", type=float, default=25.0, help="... and this many um of its planned position")
    ap.add_argument("--explain", action="store_true", help="take the edges of layers already fitted (thick first) out for the later "
                    "ones (DQD5: worse; graphite edges often coincide with hBN edges)")
    ap.add_argument("--support", type=float, default=0.62, help="combo: a layer with at least this fraction of its outline within "
                    "0.7 um of a natural-picture edge is fixed; the others are fitted again on the high-contrast edges")
    ap.add_argument("--edge-mode", choices=("combo", "natural", "both", "log"), default="combo",
                    help="edges of the 100x picture: combo (default): every layer on the natural picture's edges, then the "
                         "weakly supported ones (--support) again on the natural + log-brightness edges with the fixed layers' "
                         "edges taken out; natural; both (natural + log: finds thin graphene, DQD5's 2L 1.2 um / 1.4 deg, but "
                         "the 4.3 nm graphite then 3 um / 14 deg off); log only")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    design = json.loads(Path(a.design).read_text())
    lay = load_layers(design, a.runs)
    full100 = cv2.imread(a.image100, cv2.IMREAD_COLOR)
    h100 = cv2.resize(full100, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    e100n, e100l = edges(h100), log_edges(h100)
    e100 = {"log": e100l, "natural": e100n, "both": e100n | e100l, "combo": e100n}[a.edge_mode]
    s100 = substrate_mask(h100)
    if a.wide:
        wide = cv2.imread(a.wide, cv2.IMREAD_COLOR)
        H, W = wide.shape[:2]
        sb = (int(0.88 * W), int(0.93 * H), W, H)                   # Stream scale bar (bottom right)
        ew = edges(wide, sb)
        if a.wide_rot180:
            wide, ew = cv2.rotate(wide, cv2.ROTATE_180), cv2.rotate(ew, cv2.ROTATE_180)
        canvas = wide
        A, ni, nm = similarity(h100, wide)
        sA = math.hypot(A[0, 0], A[1, 0])
        print(f"100x frame in the 50x picture: scale {sA:.4f}, rotation {math.degrees(math.atan2(A[1, 0], A[0, 0])):.2f} deg, "
              f"shift {A[:, 2].round(1).tolist()} ({ni} of {nm} feature matches)")
        foot = cv2.warpAffine(np.ones(h100.shape[:2], np.uint8), A, (W, H), flags=cv2.INTER_NEAREST)
        foot = cv2.erode(foot, np.ones((7, 7), np.uint8))
        e = np.where(foot > 0, cv2.warpAffine(e100, A, (W, H), flags=cv2.INTER_NEAREST), ew)
        e_hc = np.where(foot > 0, cv2.warpAffine(e100n | e100l, A, (W, H), flags=cv2.INTER_NEAREST), ew)
        subm = np.where(foot > 0, cv2.warpAffine(s100, A, (W, H), flags=cv2.INTER_NEAREST), substrate_mask(wide))
    else:
        canvas, e, subm = h100, e100, s100
        e_hc = e100n | e100l
        A = np.float32([[1, 0, 0], [0, 1, 0]])
    dt = np.minimum(cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5), TRUNC).astype(np.float32)
    inside = np.ones_like(dt)
    cv2.imwrite(str(out / "edges.png"), e)
    mirrors = {"auto": (False, True), "no": (False,), "yes": (True,)}[a.mirror]

    # 1. the whole plan
    P = []
    for x in lay:
        pts = outline_points(x["f"], (x["f"]["frame"]["w"], x["f"]["frame"]["h"]))
        x["pts"] = pts
        P.append(world_pts(x, pts) / UM)
    P = np.vstack(P)
    t0 = time.time()
    g = fit_one(P, np.zeros(2), dt, inside, np.arange(0, 360, 2.0), mirrors)
    g = fit_one(P, np.zeros(2), dt, inside, np.arange(g[0][1] - 2, g[0][1] + 2.01, 0.25), (g[0][2],))[0] if g else None
    if g is None:
        raise SystemExit("the plan does not fit in the pictures")
    gsc, gang, gm, gt = g
    print(f"whole plan: rotation {gang:.2f} deg{' mirrored' if gm else ''}, score {gsc:.2f} px ({time.time() - t0:.0f} s)")

    def plan_to_canvas(w):                                          # plan world um -> canvas px
        return (np.asarray(w, float) / UM * ([-1, 1] if gm else [1, 1])) @ rot(gang).T + gt

    # 2. each layer near its predicted pose
    def dtmap(em, taken=None):
        if taken is not None:
            em = np.where(taken > 0, 0, em).astype(np.uint8)
        return np.minimum(cv2.distanceTransform((em == 0).astype(np.uint8), cv2.DIST_L2, 5), TRUNC).astype(np.float32)

    def outline_canvas(x, ang, m, t):
        Pq = np.array(x["f"]["outline"], float).reshape(-1, 2)
        return ((Pq - x["c"]) * ([-1, 1] if m else [1, 1])) @ rot(ang).T + t

    def support(x, ang, m, t, dtm):
        """Fraction of the outline within 3 px (~0.7 um) of an edge."""
        q = ((x["pts"] - x["c"]) * ([-1, 1] if m else [1, 1])) @ rot(ang).T + t
        qi = np.round(q).astype(int)
        ok = (qi[:, 0] >= 0) & (qi[:, 1] >= 0) & (qi[:, 0] < dtm.shape[1]) & (qi[:, 1] < dtm.shape[0])
        return float((dtm[qi[ok, 1], qi[ok, 0]] <= 3).mean()) if ok.any() else 0.0

    def fit_layer(x, dtm):
        L, f = x["L"], x["f"]
        hh, ww_ = f["rect"][3], f["rect"][2]
        fmask = np.zeros((hh, ww_), np.uint8)
        cv2.fillPoly(fmask, [np.round(np.array(f["outline"], float).reshape(-1, 2)).astype(np.int32)], 1)
        ang0 = (gang - L["rot"]) if gm else (gang + L["rot"])
        m0 = bool(L.get("mirror")) != gm
        tp = plan_to_canvas([L["x"], L["y"]])
        r1 = fit_one(x["pts"], x["c"], dtm, inside, np.arange(ang0 - a.turn, ang0 + a.turn + 0.01, 1.0), (m0,), fmask, subm, t0=tp, rad=a.shift / UM)
        if not r1:
            return None
        sc, ang, m, t = fit_one(x["pts"], x["c"], dtm, inside, np.arange(r1[0][1] - 1, r1[0][1] + 1.01, 0.25), (m0,), fmask, subm,
                                t0=tp, rad=a.shift / UM)[0]
        return dict(x=x, score=sc, ang=ang, m=m, t=t, d_um=float(np.hypot(*(t - tp))) * UM, dang=((ang - ang0 + 180) % 360) - 180,
                    ang0=ang0, tp=tp)

    def say(r, note=""):
        x = r["x"]
        print(f"  {x['name']} ({x['f']['material']} {float(x['t']['t']):g} nm): {r['d_um']:.1f} um and {r['dang']:+.1f} deg from the plan's pose, "
              f"score {r['score']:.2f} px" + note)

    res = []
    used = np.zeros_like(e)
    if a.edge_mode == "combo":
        dtn = dtmap(e)
        for x in lay:
            r = fit_layer(x, dtn)
            if r is None:
                print(f"  {x['name']}: no pose near the plan's")
                continue
            r["support"] = support(x, r["ang"], r["m"], r["t"], dtn)
            r["stage"] = "natural"
            res.append(r)
        fixed = [r for r in res if r["support"] >= a.support]
        weak = [r for r in res if r["support"] < a.support]
        for r in fixed:
            cv2.polylines(used, [np.round(outline_canvas(r["x"], r["ang"], r["m"], r["t"])).astype(np.int32)], True, 1, 5)
            say(r, f", edge support {r['support']:.2f}: fixed")
        dth = dtmap(e_hc, used)                                     # high contrast, the fixed layers' edges taken out
        for r in weak:
            r2 = fit_layer(r["x"], dth)
            if r2 is None:
                say(r, f", edge support {r['support']:.2f}: kept (no high-contrast pose)")
                continue
            r2["support"], r2["support_natural"], r2["stage"] = support(r2["x"], r2["ang"], r2["m"], r2["t"], dth), r["support"], "high contrast"
            res[res.index(r)] = r2
            say(r2, f", edge support {r['support']:.2f} on the natural edges: fitted again on the high-contrast ones (support there "
                    f"{r2['support']:.2f})")
    else:
        # the clearest first (hBN, thickest first, then graphite, thickest first); with --explain the edges a fitted layer
        # explains are taken out for the ones after it
        dt0 = dtmap(e)
        for x in sorted(lay, key=lambda x: (x["f"]["material"] != "hbn", -float(x["t"]["t"]))):
            r = fit_layer(x, dtmap(e, used) if a.explain and used.any() else dt0)
            if r is None:
                print(f"  {x['name']}: no pose near the plan's")
                continue
            r["support"], r["stage"] = support(x, r["ang"], r["m"], r["t"], dt0), a.edge_mode
            res.append(r)
            cv2.polylines(used, [np.round(outline_canvas(x, r["ang"], r["m"], r["t"])).astype(np.int32)], True, 1, 5)
            say(r)

    # overlay on the canvas
    ov = canvas.copy()
    cols = [(255, 160, 60), (60, 220, 255), (90, 230, 120), (230, 120, 230), (80, 120, 255), (200, 200, 70)]
    res.sort(key=lambda r: lay.index(r["x"]))
    for i, r in enumerate(res):
        x = r["x"]
        Pq = np.array(x["f"]["outline"], float).reshape(-1, 2)
        q = ((Pq - x["c"]) * ([-1, 1] if r["m"] else [1, 1])) @ rot(r["ang"]).T + r["t"]
        cv2.polylines(ov, [np.round(q).astype(np.int32)], True, cols[i % len(cols)], 2, cv2.LINE_AA)
        cv2.putText(ov, f"{len(lay) - lay.index(x)}. {x['name']}", tuple(int(v) for v in r["t"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    cols[i % len(cols)], 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # each layer on the 100x picture (full resolution): fitted outline solid, the pose the whole-plan fit predicted dashed
    Ainv = cv2.invertAffineTransform(A)
    for i, r in enumerate(res):
        x = r["x"]
        Pq = np.array(x["f"]["outline"], float).reshape(-1, 2)
        im = full100.copy()
        for ang, m, t, dashed in ((r["ang0"], r["m"], r["tp"], True), (r["ang"], r["m"], r["t"], False)):
            q = ((Pq - x["c"]) * ([-1, 1] if m else [1, 1])) @ rot(ang).T + t
            u = 2.0 * (q @ Ainv[:, :2].T + Ainv[:, 2])
            u = np.round(u).astype(np.int32)
            if dashed:
                for k in range(0, len(u), 2):
                    cv2.line(im, tuple(u[k]), tuple(u[(k + 1) % len(u)]), (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.polylines(im, [u], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, f"{x['name']} ({x['f']['material']}, {x['t']['t']:g} nm): {r['d_um']:.1f} um, {r['dang']:+.1f} deg from the plan",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"layer_{len(lay) - lay.index(x)}_{x['name']}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # back into the plan's frame: canvas = R(gang) M (world / UM) + gt  ->  world = UM M R(-gang) (canvas - gt)
    Mg = np.diag([-1.0, 1.0]) if gm else np.eye(2)

    def canvas_to_plan(p):
        return UM * (Mg @ rot(-gang) @ (np.asarray(p, float) - gt))
    layers = []
    for x in lay:
        L = dict(x["L"])
        r = next((r for r in res if r["x"] is x), None)
        L["plan"] = dict(x=L["x"], y=L["y"], rot=L["rot"], mirror=bool(L.get("mirror")))
        if r is not None:
            w = canvas_to_plan(r["t"])
            L["x"], L["y"] = round(float(w[0]), 2), round(float(w[1]), 2)
            L["rot"] = round(((gang - r["ang"]) if gm else (r["ang"] - gang)) % 360, 3)
            L["mirror"] = bool(r["m"] != gm)
        layers.append(L)
    # the 100x picture in the plan's frame: full-res px u -> canvas A (u / 2) -> plan
    a_ang = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    sA = math.hypot(A[0, 0], A[1, 0])
    cimg = canvas_to_plan(A[:, :2] @ (np.array([full100.shape[1], full100.shape[0]], float) / 4.0) + A[:, 2])
    simg = dict(url="", w=int(full100.shape[1]), h=int(full100.shape[0]), f=1.0, s=round(UM100 * sA, 6),
                rot=round(((gang - a_ang) if gm else (a_ang - gang)) % 360, 3), mirror=bool(gm), tx=round(float(cimg[0]), 2),
                ty=round(float(cimg[1]), 2), opacity=0.9, name=Path(a.image100).name)
    asb = dict(design, name=f"{design.get('name', 'stack')} · as built (automatic fit)", layers=layers, stack_image=simg,
               as_built=dict(of=None, of_name=design.get("name"), made=int(time.time() * 1000), method="automatic fit (tools/fit_stack_design.py)"))
    asb.pop("updated", None)
    (out / "design_asbuilt.json").write_text(json.dumps(asb, indent=1))
    fit = dict(plan=dict(rotation_deg=gang, mirrored=gm, score_px=gsc, translation_px=gt.tolist()), um_px=UM,
               picture100_in_canvas=A.tolist(), layers=[dict(id=r["x"]["L"]["tid"], name=r["x"]["name"], score_px=round(r["score"], 2),
                                                             from_plan_um=round(r["d_um"], 1), from_plan_deg=round(r["dang"], 2),
                                                             canvas_angle_deg=r["ang"], canvas_mirror=bool(r["m"]), canvas_t_px=r["t"].tolist(),
                                                             edges=r.get("stage"), edge_support=round(r.get("support", float("nan")), 3))
                                                        for r in res])
    (out / "fit.json").write_text(json.dumps(fit, indent=1))
    print(f"wrote {out / 'overlay.jpg'}, {out / 'fit.json'}, {out / 'design_asbuilt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
