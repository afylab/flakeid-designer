#!/usr/bin/env python3
"""Prototype of the automatic as-built fit: where each flake of a stack ended up in the final stack image.

    python tools/fit_stack_layers.py <stack folder> [--image DQD3_stack_100x.jpg] [-o out]

Needs the stack's designer run (tools/make_stack_bundle.py: flake outlines at 0.229 um/px) and a stack image at the
same scale (100x, BX51 + XC10).  For each flake the outline (without the stretches along the image frame or the text
overlay, which are not flake edges) is matched against the edges of the stack image: every rotation (2 deg steps,
then 0.25), both mirror states and every position, scored by the mean distance of the outline to the nearest edge
(truncated at 3 um).  Writes <out>/fit.json (poses, scores, the runner-up), <out>/overlay.png and a design file the
designer can open (layers at the fitted poses, the stack image under them).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import fftconvolve

TRUNC = 13.0          # px at 0.229 um/px: ~3 um
OUTSIDE = 0.6         # cost of an outline point outside the picture, as a fraction of TRUNC


def edge_map(img):
    a = cv2.GaussianBlur(img, (0, 0), 1.2)
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
    e = np.zeros(img.shape[:2], np.uint8)
    for ch in cv2.split(lab):
        e |= cv2.Canny(ch, 12, 30)
    h, w = e.shape
    e[int(0.89 * h):, :int(0.35 * w)] = 0                          # Olympus Stream text overlay
    return e


def outline_points(f, full_wh, step=1.5):
    """Points along the flake outline (cut-out px), leaving out stretches on the original image's frame or overlay."""
    P = np.array(f["outline"], float).reshape(-1, 2)
    k = 0.5
    W, H = full_wh[0] * k, full_wh[1] * k
    ox, oy = f["crop_px"][0] * k, f["crop_px"][1] * k
    pts = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(1, int(np.hypot(*(b - a)) / step))
        for t in np.arange(n) / n:
            q = a + t * (b - a)
            gx, gy = q[0] + ox, q[1] + oy
            if gx < 3 or gy < 3 or gx > W - 4 or gy > H - 4:
                continue
            if gy > 0.89 * H - 4 and gx < 0.35 * W + 4:
                continue
            pts.append(q)
    return np.array(pts)


def bubble_map(img, ring_px=4):
    """Bubbles trapped between layers collect along flake edges (the easiest cue to a thin layer's outline): small
    round spots against a smooth background (LoG blobs in any Lab channel, round = the Hessian's two curvatures alike,
    so the responses along straight edges drop out).  Returns (centres, mask of the discs within ring_px of them)."""
    from skimage.feature import blob_log, hessian_matrix, hessian_matrix_eigvals
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = img.shape[:2]
    out = []
    for ch in range(3):
        c = lab[..., ch]
        r = np.abs(c - cv2.medianBlur(np.clip(c, 0, 255).astype(np.uint8), 15).astype(np.float32))
        r[int(0.89 * h):, :int(0.35 * w)] = 0
        r /= max(float(r.max()), 1.0)
        for y, x, sg in blob_log(r, min_sigma=1.2, max_sigma=6, num_sigma=8, threshold=0.08):
            H = hessian_matrix(cv2.GaussianBlur(r, (0, 0), float(sg)), sigma=1.0, order="rc")
            e = hessian_matrix_eigvals(H)
            l1, l2 = abs(e[0][int(y), int(x)]), abs(e[1][int(y), int(x)])
            if max(l1, l2) > 0 and min(l1, l2) / max(l1, l2) > 0.35:
                out.append((x, y))
    m = np.zeros((h, w), np.uint8)
    for x, y in out:
        cv2.circle(m, (int(x), int(y)), ring_px, 1, -1)
    return np.array(out), m.astype(np.float32)


def substrate_mask(img):
    """Bare substrate in the stack image: the commonest colour (the chip around the stack), within a few %."""
    a = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
    q = (a // 6).astype(np.int32)
    key = q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]
    vals, cnt = np.unique(key[::3, ::3], return_counts=True)
    sub = np.median(a[key == vals[cnt.argmax()]], axis=0)
    return (np.abs(a / np.maximum(sub, 1) - 1).max(axis=2) < 0.05).astype(np.float32)


def fit_one(pts, c, dt, inside, angles, mirrors, fmask=None, subm=None, lam=TRUNC, bub=None, beta=0.0, t0=None, rad=None):
    """Best (score, angle, mirror, t) over angles x mirrors x translations: pose maps cut-out p to R M (p - c) + t.
    score = mean edge distance of the outline + lam x the fraction of the flake's area that would lie on bare
    substrate (a layer cannot sit where the stack image shows plain chip).  t0, rad: only translations within rad px
    of t0 (a layer near where the design puts it)."""
    H, W = dt.shape
    best = []
    for m in mirrors:
        for ang in angles:
            a = math.radians(ang)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            q = (pts - c) * ([-1, 1] if m else [1, 1])
            q = q @ R.T
            lo = np.floor(q.min(0)).astype(int)
            qi = np.round(q - lo).astype(int)
            hh, ww = qi[:, 1].max() + 1, qi[:, 0].max() + 1
            imp = np.zeros((hh, ww), np.float32)
            np.add.at(imp, (qi[:, 1], qi[:, 0]), 1.0)
            # sum over points of dt[p + t]: correlate dt with the point image; the stack image is padded so that part of
            # the outline may lie outside it (a layer larger than the picture), scored on the points inside (>= 60 %)
            dtp = np.pad(dt, ((hh, hh), (ww, ww)))
            inp = np.pad(inside, ((hh, hh), (ww, ww)))
            S = fftconvolve(dtp, imp[::-1, ::-1], mode="valid")
            C = fftconvolve(inp, imp[::-1, ::-1], mode="valid")
            n = imp.sum()
            ok = C >= 0.6 * n
            if not ok.any():
                continue
            # points outside the picture count as 60 % of the worst distance: a pose hanging off the picture must not
            # win on the few points it leaves inside (the stack picture is taken centred on the stack)
            sc = np.where(ok, (S + OUTSIDE * TRUNC * (n - C)) / n, 1e9)
            if bub is not None and beta > 0:
                # reward: the fraction of the outline that runs past bubbles (within a few px)
                Bc = fftconvolve(np.pad(bub, ((hh, hh), (ww, ww))), imp[::-1, ::-1], mode="valid")
                sc = np.where(ok, sc - beta * TRUNC * Bc / n, 1e9)
            if fmask is not None and subm is not None:
                # the flake's area rotated the same way, anchored so that its centre c lands on the same translation
                Mf = np.float32([[R[0, 0] * (-1 if m else 1), R[0, 1], 0], [R[1, 0] * (-1 if m else 1), R[1, 1], 0]])
                hm, wm = fmask.shape
                corners = (np.array([[0, 0], [wm, 0], [0, hm], [wm, hm]], float) - c) * ([-1, 1] if m else [1, 1]) @ R.T
                lo2 = np.floor(corners.min(0)).astype(int)
                Mf[:, 2] = -(Mf[:, :2] @ c) - lo2
                sz = np.ceil(corners.max(0) - lo2).astype(int) + 1
                fm = cv2.warpAffine(fmask.astype(np.float32), Mf, (int(sz[0]), int(sz[1])), flags=cv2.INTER_NEAREST)
                # align its anchor with the outline's: translation index k of the outline <-> t = idx - lo (+pad); here t = idx2 - lo2
                sp = np.pad(subm, ((hh, hh), (ww, ww)))
                ip = np.pad(inside, ((hh, hh), (ww, ww)))
                Sf = fftconvolve(np.pad(sp, ((0, fm.shape[0]), (0, fm.shape[1]))), fm[::-1, ::-1], mode="valid")
                Cf = fftconvolve(np.pad(ip, ((0, fm.shape[0]), (0, fm.shape[1]))), fm[::-1, ::-1], mode="valid")
                # outline translation index (yy, xx) means t = (xx - ww, yy - hh) - lo; mask index for the same t: t + lo2 + pad
                yy0, xx0 = np.mgrid[0:sc.shape[0], 0:sc.shape[1]]
                ty, tx = yy0 - hh - lo[1], xx0 - ww - lo[0]
                my, mx = ty + lo2[1] + hh, tx + lo2[0] + ww
                okm = (my >= 0) & (mx >= 0) & (my < Sf.shape[0]) & (mx < Sf.shape[1])
                frac = np.ones_like(sc)
                frac[okm] = Sf[my[okm], mx[okm]] / np.maximum(Cf[my[okm], mx[okm]], 1.0)
                sc = np.where(ok, sc + lam * frac, 1e9)
            if t0 is not None and rad is not None:
                gy, gx = np.mgrid[0:sc.shape[0], 0:sc.shape[1]]
                far = (gx - ww - lo[0] - t0[0]) ** 2 + (gy - hh - lo[1] - t0[1]) ** 2 > rad * rad
                sc = np.where(far, 1e9, sc)
            k = int(np.argmin(sc))
            yy, xx = np.unravel_index(k, sc.shape)
            if sc[yy, xx] >= 1e9:
                continue
            t = np.array([xx - ww, yy - hh], float) - lo             # translation of the centre c
            best.append((float(sc[yy, xx]), ang, m, t))
    best.sort(key=lambda b: b[0])
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--image", help="stack image file in the folder (default: *_stack_100x.jpg)")
    ap.add_argument("--bundle", help="the stack's designer run folder (default: <folder>/designer)")
    ap.add_argument("--mirror", choices=("auto", "no", "yes"), default="auto",
                    help="whether the stack is mirrored against the flake images (the same for every layer; auto: fitted per layer)")
    ap.add_argument("--no-bw", action="store_true", help="do not add the edges of the high-contrast (*_bw) stack image")
    ap.add_argument("--bubbles", type=float, default=0.5, help="weight of the bubble cue (0 = off)")
    ap.add_argument("--only", help="fit only the flakes whose name contains this")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    mirrors = {"auto": (False, True), "no": (False,), "yes": (True,)}[a.mirror]
    folder = Path(a.folder)
    bdir = Path(a.bundle) if a.bundle else folder / "designer"
    out = Path(a.out) if a.out else folder / "fit"
    out.mkdir(parents=True, exist_ok=True)
    b = json.loads((bdir / "bundle.json").read_text())
    simg_path = folder / a.image if a.image else sorted(p for p in folder.glob("*_stack_100x*.jpg") if "_bw" not in p.name)[0]
    full = cv2.imread(str(simg_path), cv2.IMREAD_COLOR)
    img = cv2.resize(full, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    e = edge_map(img)
    bw = sorted(p for p in folder.glob("*_stack_100x*bw*.jpg"))
    if bw and not a.no_bw:                                         # the high-contrast version shows buried edges better
        bwi = cv2.resize(cv2.imread(str(bw[0]), cv2.IMREAD_COLOR), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        e |= edge_map(bwi)
    dt = np.minimum(cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5), TRUNC).astype(np.float32)
    inside = np.ones_like(dt)
    subm = substrate_mask(img)
    bpts, bub = bubble_map(img)
    print(f"  {len(bpts)} bubbles")
    cv2.imwrite(str(out / "substrate.png"), (subm * 255).astype(np.uint8))
    res = []
    overlay = img.copy()
    cols = [(255, 160, 60), (60, 220, 255), (90, 230, 120), (230, 120, 230), (80, 120, 255), (200, 200, 70)]
    for i, f in enumerate(b["flakes"]):
        if a.only and a.only not in f["id"]:
            continue
        pts = outline_points(f, (f["frame"]["w"], f["frame"]["h"]))
        c = np.array(f["terraces"][0]["c"], float)
        x, y, w, h = f["rect"]
        fmask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(fmask, [np.round(np.array(f["outline"], float).reshape(-1, 2)).astype(np.int32)], 1)
        coarse = fit_one(pts, c, dt, inside, np.arange(0, 360, 2.0), mirrors, fmask, subm, bub=bub, beta=a.bubbles)
        if not coarse:
            print(f"  {f['id']}: does not fit in the stack image")
            continue
        s0, a0, m0, _ = coarse[0]
        fine = fit_one(pts, c, dt, inside, np.arange(a0 - 2, a0 + 2.01, 0.25), (m0,), fmask, subm, bub=bub, beta=a.bubbles)
        sc, ang, m, t = fine[0]
        # runner-up: the best pose more than 10 deg or 15 px away from the winner
        alt = next((x for x in coarse if abs(((x[1] - a0 + 180) % 360) - 180) > 10 or x[2] != m0 or np.hypot(*(x[3] - coarse[0][3])) > 15), None)
        res.append(dict(id=f["id"], score_px=round(sc, 2), angle_deg=round(ang, 2), mirror=bool(m), t_px=[round(v, 1) for v in t],
                        runner_up=dict(score_px=round(alt[0], 2), angle_deg=alt[1], mirror=bool(alt[2])) if alt else None, n_points=int(len(pts))))
        print(f"  {f['id']}: angle {ang:.2f} deg{' mirrored' if m else ''}, score {sc:.2f} (edge distance + substrate penalty, px)"
              + (f"; runner-up {alt[0]:.2f} px at {alt[1]:.0f} deg{' mirrored' if alt[2] else ''}" if alt else ""))
        # draw the fitted outline
        R = np.array([[math.cos(math.radians(ang)), -math.sin(math.radians(ang))], [math.sin(math.radians(ang)), math.cos(math.radians(ang))]])
        P = np.array(f["outline"], float).reshape(-1, 2)
        q = ((P - c) * ([-1, 1] if m else [1, 1])) @ R.T + t
        cv2.polylines(overlay, [np.round(q).astype(np.int32)], True, cols[i % len(cols)], 2)
        cv2.putText(overlay, f["id"].split(":")[1], tuple(int(v) for v in t), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cols[i % len(cols)], 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "overlay.png"), overlay)
    cv2.imwrite(str(out / "edges.png"), e)
    (out / "fit.json").write_text(json.dumps(dict(image=simg_path.name, um_px=0.229, results=res), indent=1))
    # a design for the designer: the stack image at full resolution (0.1143 um/px) centred on the origin, the layers
    # at the fitted poses (world um = (half-res px - image centre) * 0.229)
    h2, w2 = img.shape[:2]
    layers = []
    for r in res[::-1]:                                            # designer order: bottom first
        tid = r["id"] + ".a"
        x, y = (r["t_px"][0] - w2 / 2) * 0.2286, (r["t_px"][1] - h2 / 2) * 0.2286
        layers.append(dict(uid=r["id"].split(":")[1], tid=tid, x=round(x, 2), y=round(y, 2), rot=r["angle_deg"] % 360, mirror=r["mirror"],
                           opacity=0.5, showImg=False, showOutline=True, inOverlap=True, locked=False))
    design = dict(name=f"{folder.name} · as built (automatic fit)", layers=layers[::-1][::-1], stack_image=dict(
        url="", w=int(full.shape[1]), h=int(full.shape[0]), f=1.0, s=0.1143, rot=0, mirror=False, tx=0.0, ty=0.0, opacity=0.95, name=simg_path.name),
        as_built=dict(of=None, of_name=folder.name, made=None, method="automatic fit prototype"))
    (out / "design.json").write_text(json.dumps(design, indent=1))
    print(f"wrote {out / 'overlay.png'}, {out / 'fit.json'}, {out / 'design.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
