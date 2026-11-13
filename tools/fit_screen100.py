#!/usr/bin/env python3
"""Fit the 100x screening models (profiles/screen100_<material>.json) on the user's 100x labels.

    python tools/fit_screen100.py extract RUN_DIR LABELS_DIR --material hbn --oxide 303 -o feats_bn2.csv [--overlays DIR]
    python tools/fit_screen100.py fit feats_bn1.csv feats_bn2.csv --material hbn [--recall 0.9]
    python tools/fit_screen100.py fit-decisions RUN_DIR ... --data <scanresults>/_designer_data --material graphite \
        [--keep 0.95] [--clip none,2,3,4] [-C 0.03,0.1,0.3,1,3] [--group run] [--dry-run] [-o scores.csv]
    python tools/fit_screen100.py fit-defects screen_features.csv [--material hbn] [--keep 0.95] [--centre run] [-o scores.csv]

extract: every labelled frame of a 100x run (raw TIFFs, as written by mm_chip_scan.bsh) is read like the viewer export
(dark subtracted, the run's static optics pattern divided out) and flakeid.screen100.features() computed; the label
(scoring-page document: verdict, tags) is joined.  LABELS_DIR holds <image stem>.json documents.
fit: logistic regression of keep = (useful or maybe) on screen100.FEATURES, leave-one-out predictions to choose the
threshold that keeps --recall of the flakes the user kept, and the resulting share of 'not' flakes removed.
fit-decisions: the "pick" model (profile key "pick": the score the designer shows, and its keep / review tiers) from
the keep / discard decisions made in the designer (<data>/review/*.jsonl; "picked up" counts as keep), on the
screening features saved with each flake's 100x image in the runs' viewers (RUN_DIR: the run folder, or a local copy
of its chip viewers: <run>/<any>/<chip>/index.html + 100x/*_regions.json).  numpy only.  Each |z| clip and C of the
grid is evaluated leaving one run out; the profile gets the best pooled held-out AUC; the nested estimate (clip and C
chosen on the training runs alone) is printed and stored.  The pick model's reject_below keeps --keep of the keep +
used flakes (pooled): below it a flake is tier "reject", shown as screened out in the designer (2026-10-22: 6 runs,
486 decisions, clip 2, C 0.3, held-out AUC 0.783, nested 0.757; the 2026-10-14 model scored 0.72 / 0.76 on the two
2026-10-19 runs it never saw, the refit 0.81 / 0.835 held out).  The label model above stays the gate that drops flakes before the designer (reject_below).  Why: for graphite
the label model did not predict the decisions (AUC 0.52 on 128 decisions, 2026-10-14); the decisions prefer smooth,
residue-free and not too large flakes.
fit-defects: the hBN "pick" model from the designer decisions on the label model's score (p_quality, as logit) and
two defect features of terrace a in the 100x picture (flakeid.defects100: clean_d, edge_um_per_1000; --centre run /
chip: relative to the run's / chip's median, for comparison).  The table: one row per decided flake with run, scan,
decision (keep / used / discard), p_quality (or p_keep), clean_d, edge_um_per_1000 (ok = 1; --features: other defect
columns), v (the defects version, flakeid.defects100; 1 when missing: the model records it as "defects"), the defects
measured on the pictures as the bundles store them (2026-10-22: flakedata/screen_features_view.csv, version 1;
screen_features_view_v2.csv, version 2).  Evaluated leaving one
run out; reject_below is set so that --keep of the keep + used flakes stay (pooled over runs), keep_above as the
graphite pick model (0.5: the classes are weighted equally).  numpy only (no scikit-learn).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flakeid import screen100 as S  # noqa: E402
from flakeid.hundred import static_pattern  # noqa: E402
from flakeid.mm import BAYER_CODE  # noqa: E402


def linear(p, dark=128.0):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), BAYER_CODE).astype(np.float32) - dark


def extract(a):
    run, lab_dir = Path(a.run), Path(a.labels)
    labels = {}
    for f in lab_dir.glob("*.json"):
        d = json.loads(f.read_text())
        d = d.get("data", d)
        if d.get("verdict"):
            labels[f.stem] = d
    files = sorted(p for p in run.glob("*.tif") if p.stem in labels)
    print(f"{run.name}: {len(files)} labelled frames")
    pat_path = Path(a.out).with_suffix(".pattern.npy")
    if pat_path.is_file():
        pat = np.load(pat_path)
    else:
        pat = static_pattern(linear(p) for p in sorted(run.glob("*.tif")))
        if pat is not None:
            np.save(pat_path, pat)
    if a.overlays:
        Path(a.overlays).mkdir(parents=True, exist_ok=True)
    rows = []
    for k, p in enumerate(files):
        m = re.search(r"_(\d+)nm", p.stem)
        t = float(m.group(1)) if m else 30.0
        lin = linear(p)
        if pat is not None:
            lin = lin / pat
        f, reg, extra = S.features(lin, a.material, a.oxide, t)
        d = labels[p.stem]
        rows.append(dict(id=p.stem, run=run.name, material=a.material, verdict=d["verdict"], tags=";".join(d.get("tags") or []),
                         note=d.get("note") or "", **{kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in f.items()}))
        if a.overlays and "dt" in extra:
            from flakeid.hundred import analyse
            c8, _, _ = analyse(lin)
            o = cv2.cvtColor(c8, cv2.COLOR_RGB2BGR)
            for r in extra["fine"]:
                for q in r["poly"]:
                    cv2.polylines(o, [np.round(np.array(q).reshape(-1, 2)).astype(np.int32)], True,
                                  (255, 255, 0) if r["dt_nm"] > 0 else (0, 255, 255), 2, cv2.LINE_AA)
            for q in extra["wrinkles"]:
                cv2.polylines(o, [np.round(np.array(q).reshape(-1, 2)).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
            dt = np.clip(128 + extra["dt"] / 1.5 * 127, 0, 255).astype(np.uint8)
            dtc = cv2.applyColorMap(dt, cv2.COLORMAP_TWILIGHT_SHIFTED)
            cv2.imwrite(str(Path(a.overlays) / f"{p.stem}.jpg"),
                        np.hstack([cv2.resize(o, (688, 519)), cv2.resize(dtc, (688, 519))]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(files)}", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")


def fit(a):
    rows = [r for f in a.csv for r in csv.DictReader(open(f)) if r["material"] == a.material]
    X = np.array([S.vector({k: float(v) for k, v in r.items() if k in S.FEATURES or k in
                            ("terrace_um2", "clean_um2", "focus") and v not in ("", None)}) for r in rows])
    y = np.array([r["verdict"] in ("useful", "maybe") for r in rows], int)
    fill = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, fill)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    C = a.C
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    p = cross_val_predict(model, Z, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p)
    order = np.sort(p[y == 1])
    thr = float(order[int(np.floor((1 - a.recall) * len(order)))]) if len(order) else 0.5
    kept = p >= thr
    print(f"{a.material}: {len(y)} labelled ({y.sum()} useful/maybe), LOO AUC {auc:.3f}")
    print(f"  threshold {thr:.3f}: keeps {kept[y == 1].mean():.0%} of useful/maybe, removes {(~kept[y == 0]).mean():.0%} of 'not' "
          f"({(~kept).sum()} of {len(y)} flakes removed)")
    for v in ("useful", "maybe"):
        lost = [r["id"] for r, k in zip(rows, kept) if r["verdict"] == v and not k]
        if lost:
            print(f"  {v} removed: {lost}")
    model.fit(Z, y)
    coef = model.coef_[0]
    for k, c in sorted(zip(S.FEATURES, coef), key=lambda t: -abs(t[1])):
        print(f"    {k:16s} {c:+.2f}")
    out = dict(material=a.material, features=S.FEATURES, mean=mu.tolist(), std=sd.tolist(), fill=fill.tolist(),
               coef=coef.tolist(), intercept=float(model.intercept_[0]), threshold=round(thr, 4),
               evaluation=dict(n=int(len(y)), n_keep=int(y.sum()), loo_auc=round(float(auc), 3),
                               recall_target=a.recall, kept_useful_maybe=round(float(kept[y == 1].mean()), 3),
                               removed_not=round(float((~kept[y == 0]).mean()), 3)),
               source=[str(Path(f).name) for f in a.csv])
    prof = S.PROFILE_DIR / f"screen100_{a.material}.json"
    if prof.is_file():                                  # (the pick model and the tiers are kept)
        old = json.loads(prof.read_text())
        out.update({k: old[k] for k in ("pick", "reject_below", "keep_above") if k in old})
    prof.write_text(json.dumps(out, indent=1))
    print(f"wrote profiles/screen100_{a.material}.json")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "run", "verdict", "tags", "p_keep", "kept"])
            for r, pp, k in zip(rows, p, kept):
                w.writerow([r["id"], r["run"], r["verdict"], r["tags"], round(float(pp), 3), int(k)])


def viewer_pieces(run):
    """The pieces of a run folder (tools/make_designer_bundle.piece_dirs), or of a local copy holding only the chip
    viewers (<run>/<any>/<chip>/index.html with the 100x list, and their 100x/*_regions.json)."""
    from make_designer_bundle import piece_dirs
    out = piece_dirs(run)
    if not out:
        out = [p.parent for p in sorted(run.glob("*/*/index.html")) if "const H100=" in p.read_text(encoding="utf-8")]
    return out


def decision_rows(a, feats):
    """(feature rows, keep 0/1, piece, run) of the flakes decided in the designer that have a saved 100x screening with
    all of feats (images analysed before a feature existed are left out: their gaps would be filled with medians)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from make_afm_set import load_viewer_data, viewer_dir
    from make_catalog_100x import h100, noise
    extra = json.loads(Path(a.extra).read_text(encoding="utf-8")) if getattr(a, "extra", None) else {}
    ex_keys = {k for v in extra.values() for k in v}     # (missing for some flakes: filled with the median)
    last = {}
    for f in sorted((Path(a.data) / "review").glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("id") and (d["id"] not in last or d.get("updated", 0) > last[d["id"]].get("updated", 0)):
                last[d["id"]] = d
    want = {}
    for d in last.values():
        b = d.get("body") or {}
        m = re.match(r"([^:]+):(.+)-f(\d+)\.a$", d["id"])
        if m and b.get("material") == a.material and b.get("decision") in ("keep", "used", "discard"):
            want[(m.group(1), m.group(2), int(m.group(3)))] = int(b["decision"] != "discard")
    rows, ys, groups, runs, keys, n_old, n_first = [], [], [], [], [], 0, 0
    for run in map(Path, a.runs):
        for piece in viewer_pieces(run):
            D = load_viewer_data(piece)
            stem = D["name"].replace("_tiles", "")
            if D["material"] != a.material:
                continue
            vd = viewer_dir(piece)
            for fid_s, imgs in h100(vd).items():
                y = want.get((run.name, stem, int(fid_s)))
                imgs = [e for e in imgs if not e.get("domain")]
                if y is None or not imgs:
                    continue
                # (the catalog's choice among repeated shots, also its half-resolution noise: else another picture; a
                # local copy without the pictures: the first)
                if len(imgs) > 1 and not all((vd / e["f"]).is_file() for e in imgs):
                    imgs, n_first = imgs[:1], n_first + 1
                e = imgs[0] if len(imgs) == 1 else min(imgs, key=lambda e: (e.get("focus") not in ("auto", "manual", None),
                                                                            noise(vd / e["f"], cv2.IMREAD_REDUCED_GRAYSCALE_2)))
                rj = vd / re.sub(r"\.jpg$", "_regions.json", e["f"])
                feat = (json.loads(rj.read_text()).get("screen") or {}).get("features") if rj.is_file() else None
                key = f"{run.name}:{stem}-f{int(fid_s):04d}"
                if feat and key in extra:                   # (features measured afterwards, --extra)
                    feat = dict(feat, **extra[key])
                if feat and not all(k in feat or k in ("log_terrace", "log_clean", "focus_log") or k in ex_keys
                                    for k in feats):
                    n_old += 1
                elif feat:
                    rows.append(feat)
                    ys.append(y)
                    groups.append(stem)
                    runs.append(run.name)
                    keys.append(key)
    print(f"{len(want)} {a.material} decisions; {len(rows)} with a saved 100x screening ({sum(ys)} kept)"
          + (f"; {n_old} left out (analysed before some features existed)" if n_old else "")
          + (f"; {n_first} with repeated shots took the first (pictures not in the copy)" if n_first else ""))
    return rows, np.array(ys, int), np.array(groups), np.array(runs), keys


def fit_decisions(a):
    """The pick model on the saved 100x features, numpy only.  Each (clip, C) of the grid is evaluated leaving one run
    (--group run) or piece out; the profile gets the one with the best pooled held-out AUC.  The honest estimate: the
    held-out run's scores from a model whose (clip, C) was chosen on the other runs alone (nested).  Tiers: reject
    below reject_below (--keep of the keep + used flakes above it, pooled; the designer shows them as screened out),
    keep from keep_above, else review; the label model's gate (profile reject_below) still drops flakes first."""
    prof = S.PROFILE_DIR / f"screen100_{a.material}.json"
    d = json.loads(prof.read_text(encoding="utf-8"))
    drop = [k.strip() for k in a.drop.split(",") if k.strip()]
    feats = [k for k in (d.get("features") or S.FEATURES) if k not in drop]
    feats += [k.strip() for k in a.add.split(",") if k.strip() and k.strip() not in feats]
    rows, y, piece, run, keys = decision_rows(a, feats)
    X = np.array([S.vector(f, feats) for f in rows])
    omit = {}
    for o in filter(None, (v.strip() for v in a.omit_thick.split(","))):  # (feature:t_nm, NaN above: filled)
        k, t = o.split(":")
        if k in feats:
            omit[k] = float(t)
            tt = np.array([float(f.get("t_terrace_nm", np.nan)) for f in rows])     # (from --extra: the bundles' t)
            X[tt > float(t), feats.index(k)] = np.nan
            print(f"{k} left out of {int((tt > float(t)).sum())} flakes thicker than {t} nm")
    yb = y.astype(bool)
    grp = run if a.group == "run" else piece
    sg = SIGNS.get(a.material, {}) if a.signs == "default" else {}
    signs = np.array([sg.get(k, 0) for k in feats], float)
    j_cap = feats.index(a.cap_feature) if a.cap_feature in feats else None
    caps = [None if v.strip().lower() in ("none", "0") else float(v) for v in a.caps.split(",")] if j_cap is not None else [None]

    def opt(v):
        return None if v.strip().lower() in ("none", "0") else float(v)
    grid = [(opt(c), float(C), cap) for c in a.clip.split(",") for C in a.C.split(",") for cap in caps]

    def capped(Xs, cap):
        if cap is None:
            return Xs
        Xs = Xs.copy()
        Xs[:, j_cap] = np.minimum(Xs[:, j_cap], cap)  # (NaN stays NaN: filled later)
        return Xs

    def fit(Xtr, ytr, clip, C, cap=None):
        Xtr = capped(Xtr, cap)
        fill = np.nanmedian(Xtr, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        Xf = np.where(np.isfinite(Xtr), Xtr, fill)
        mu, sd = Xf.mean(0), Xf.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Z = (Xf - mu) / sd
        if clip:
            Z = np.clip(Z, -clip, clip)
        w, b = logreg_signed(Z, ytr, C, signs) if signs.any() else logreg(Z, ytr, C)
        m = dict(features=feats, mean=mu.tolist(), std=sd.tolist(), fill=fill.tolist(), coef=w.tolist(), intercept=b)
        if clip:
            m["clip"] = clip
        if cap is not None:
            m["cap"] = {feats[j_cap]: cap}
        return m

    def z(m, Xs):
        for k, v in (m.get("cap") or {}).items():
            Xs = capped(Xs, v)
        Xf = np.where(np.isfinite(Xs), Xs, np.array(m["fill"]))
        Z = (Xf - np.array(m["mean"])) / np.array(m["std"])
        if m.get("clip"):
            Z = np.clip(Z, -m["clip"], m["clip"])
        return Z @ np.array(m["coef"]) + m["intercept"]

    def held_out(idx, clip, C, cap=None):
        """scores of the rows idx, each group left out in turn (fitted on the other groups of idx)"""
        s = np.full(len(y), np.nan)
        for g in np.unique(grp[idx]):
            te = idx & (grp == g)
            tr = idx & ~te
            s[te] = z(fit(X[tr], y[tr], clip, C, cap), X[te]) if 0 < y[tr].sum() < tr.sum() else 0.0
        return s

    def cut(s, yy, keep):                               # (the score keeping the share keep of the kept flakes)
        return float(np.sort(s[yy])[int(np.floor((1 - keep) * yy.sum()))]) if keep < 1 else -np.inf

    def name(cc):
        return f"clip {cc[0] or 'none'}, C {cc[1]:g}" + (f", {a.cap_feature} <= {cc[2]:g}" if cc[2] is not None else "")

    def two(m):
        return 0 < yb[m].sum() < m.sum()
    allr = np.ones(len(y), bool)
    groups = sorted(set(grp))
    print(f"{len(y)} decided {a.material} flakes ({yb.sum()} keep / used, {(~yb).sum()} discard), {len(groups)} "
          f"{a.group}s; held out by {a.group}:")
    loro = {}
    for cc in grid:
        s = held_out(allr, *cc)
        loro[cc] = s
        per = [_auc(s[grp == g], yb[grp == g]) for g in groups if two(grp == g)]
        print(f"  {name(cc):34s} AUC {_auc(s, yb):.3f} (mean per {a.group} {np.mean(per):.3f})")
    best = max(grid, key=lambda cc: _auc(loro[cc], yb))
    # nested: (clip, C) chosen on the training groups alone; the cutoff from the training groups' scores
    nest, chosen, lost, removed = np.full(len(y), np.nan), {}, 0, 0
    for g in groups:
        te = grp == g
        tr = ~te
        cc = max(grid, key=lambda c: _auc(held_out(tr, *c)[tr], yb[tr]))
        chosen[g] = name(cc)
        m = fit(X[tr], y[tr], *cc)
        nest[te] = z(m, X[te])
        t = cut(z(m, X[tr]), yb[tr], a.keep)
        lost += int((nest[te][yb[te]] < t).sum())
        removed += int((nest[te][~yb[te]] < t).sum())
    old = d.get("pick")
    p_old = np.array([S._logistic(old, f) for f in rows]) if old else np.full(len(y), np.nan)
    print(f"chosen on all {a.group}s: {name(best)} (held-out AUC {_auc(loro[best], yb):.3f}); nested held-out AUC "
          f"{_auc(nest, yb):.3f}; deployed pick model {_auc(p_old, yb) if old else float('nan'):.3f} (partly in-sample)")
    per_run = {}
    for r in sorted(set(run)):
        m = run == r
        if two(m):
            per_run[r] = dict(n=int(m.sum()), n_keep=int(yb[m].sum()), old=round(_auc(p_old[m], yb[m]), 3) if old else None,
                              held_out=round(_auc(loro[best][m], yb[m]), 3), nested=round(_auc(nest[m], yb[m]), 3))
            print(f"  {r:26s} n {m.sum():3d} ({yb[m].sum():3d} kept)  AUC deployed {per_run[r]['old']}  held out "
                  f"{per_run[r]['held_out']}  nested {per_run[r]['nested']}" + (f"  (chose {chosen[r]})" if r in chosen else ""))
    print(f"  cutoff keeping {a.keep:.0%} from the training {a.group}s, on the held-out ones (nested): {removed} of "
          f"{(~yb).sum()} discards removed, {lost} of {yb.sum()} keeps lost")
    pick = fit(X, y, *best)
    p = 1 / (1 + np.exp(-z(pick, X)))
    rej = float(np.floor(cut(p, yb, a.keep) * 1e4) / 1e4) if a.keep < 1 else 0.0   # (rounded down: the keep at the cutoff stays)
    print(f"  all {a.group}s: reject_below {rej:.4f} screens out {int((p[yb] < rej).sum())} of {yb.sum()} keep / used and "
          f"{int((p[~yb] < rej).sum())} of {(~yb).sum()} discards; keep tier (p >= {a.keep_above:g}): "
          f"{int((p[yb] >= a.keep_above).sum())} keep / used, {int((p[~yb] >= a.keep_above).sum())} discards")
    for k, c in sorted(zip(feats, pick["coef"]), key=lambda t: -abs(t[1]))[:10]:
        print(f"    {k:22s} {c:+.2f}")
    import time
    if omit:
        pick["omit_above_t_nm"] = omit
    pick.update(C=best[1], signs={k: int(v) for k, v in zip(feats, signs) if v}, dropped=drop, reject_below=rej, keep_above=a.keep_above, keep_share=a.keep, fitted=time.strftime("%Y-%m-%d"),
                evaluation=dict(n=int(len(y)), n_keep=int(yb.sum()), group=a.group,
                                held_out_auc=round(_auc(loro[best], yb), 3), nested_auc=round(_auc(nest, yb), 3),
                                grid={name(cc): round(_auc(s, yb), 3) for cc, s in loro.items()},
                                per_run=per_run, nested_training_cutoff=dict(removed=removed, keeps_lost=lost),
                                deployed_before_auc=round(_auc(p_old, yb), 3) if old else None),
                source=f"designer decisions ({Path(a.data).name}/review), runs: {', '.join(sorted(set(run)))}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "run", "piece", "decision", "p_old", "p_new", "held_out_z", "nested_z", "p_quality"])
            for k, r, pc, yy, po, pn, s0, s1, f in zip(keys, run, piece, yb, p_old, p, loro[best], nest, rows):
                w.writerow([k, r, pc, "keep" if yy else "discard", round(float(po), 4), round(float(pn), 4),
                            round(float(s0), 3), round(float(s1), 3), round(S._logistic(d, f), 4)])
    if a.dry_run:
        print(f"(dry run: profiles/screen100_{a.material}.json not written)")
        return
    d["pick"] = pick
    if a.reject_below is not None:
        d["reject_below"] = a.reject_below
    prof.write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(f"wrote profiles/screen100_{a.material}.json (pick model: {name(best)}, reject below {rej}, keep from "
          f"{a.keep_above:g}; gate: label model below {d.get('reject_below', d['threshold'])})")


# (coefficient signs the graphite pick model may take, 2026-10-22: size and cleanness up, residue / roughness / wrinkles
# / fine terraces / ridges down; unconstrained fits gave clean_frac and solidity negative weights, collinear with size,
# so clean compact flakes the user kept lost score)
SIGNS = {"graphite": dict(log_terrace=1, log_clean=1, clean_d_um=1, clean_frac=1, solidity=1, wrinkle_free_ratio=1,
                          debris_per_1000=-1, debris_frac=-1, steps_frac=-1, roughness=-1, ridge_density=-1, dt_iqr=-1,
                          dt_p95=-1, orth_p90=-1, fine_n=-1, fine_frac=-1, bubble_per_1000=-1, bubble_frac=-1,
                          wrinkle_n=-1, wrinkle_len_density=-1)}


def logreg_signed(Z, y, C, signs):
    """logreg with sign constraints: coef[j] >= 0 where signs[j] > 0, <= 0 where < 0, free where 0 (L-BFGS-B)."""
    from scipy.optimize import minimize
    y = np.asarray(y, float)
    sw = np.where(y > 0, len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max(len(y) - y.sum(), 1)))

    def f(wb):
        w, b = wb[:-1], wb[-1]
        t = Z @ w + b
        p = 1 / (1 + np.exp(-t))
        g = sw * (p - y)
        return float(np.sum(sw * (np.logaddexp(0, t) - y * t)) + 0.5 / C * w @ w), np.r_[Z.T @ g + w / C, g.sum()]
    bounds = [(0, None) if s > 0 else (None, 0) if s < 0 else (None, None) for s in signs] + [(None, None)]
    r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=2000, gtol=1e-8))
    return r.x[:-1], float(r.x[-1])


def logreg(Z, y, C=1.0, it=60):
    """L2-penalised logistic regression (penalty 1 / C, not on the intercept), the two classes weighted to the same
    total (scikit-learn's class_weight="balanced"): (coef, intercept)."""
    y = np.asarray(y, float)
    sw = np.where(y > 0, len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max(len(y) - y.sum(), 1)))
    A = np.column_stack([Z, np.ones(len(Z))])
    R = np.eye(A.shape[1]) / C
    R[-1, -1] = 0.0
    wb = np.zeros(A.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(A @ wb)))
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + R
        step = np.linalg.solve(H, A.T @ (sw * (p - y)) + R @ wb)
        wb -= step
        if np.abs(step).max() < 1e-9:
            break
    return wb[:-1], float(wb[-1])


def _auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_defects(a):
    from flakeid import defects100 as D
    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8")) if r.get("ok", "1") == "1"
            and r.get("decision") in ("keep", "used", "discard")]
    y = np.array([r["decision"] != "discard" for r in rows])
    vers = {int(r.get("v") or 1) for r in rows}                     # (the defects version the table was measured with)
    if len(vers) != 1:
        sys.exit(f"the table mixes defects versions {sorted(vers)}: measure all flakes the same way")
    run = np.array([r["run"] for r in rows])
    scan = np.array([r["scan"] for r in rows])
    grp = run if a.centre == "run" else scan
    cen = a.centre != "none"
    pq = np.array([float(r.get("p_quality") or r["p_keep"]) for r in rows])
    feats = [k.strip() for k in a.features.split(",") if k.strip()]
    V = np.array([[float(r[k]) for k in feats] for r in rows])
    L = np.array([D._logit(v) for v in pq])

    def centred(Vx, g, ref=None, n0=0.0):
        """features minus their group's median (shrunk to ref by n0 flakes)"""
        out = Vx.copy()
        if not cen:
            return out
        for q in set(g):
            m = g == q
            med = np.median(Vx[m], axis=0)
            out[m] = Vx[m] - (med if ref is None else (m.sum() * med + n0 * ref) / (m.sum() + n0))
        return out

    def asbuilt(Vx, r, ref):
        """run r centred as its bundle grows chip by chip (chips in name order; medians over the chips so far)"""
        out = Vx.copy()
        chips = sorted(set(scan[run == r]))
        for k, c in enumerate(chips):
            sofar = np.isin(scan, chips[:k + 1]) & (run == r)
            n, med = sofar.sum(), np.median(Vx[sofar], axis=0)
            out[scan == c] = Vx[scan == c] - (n * med + a.n0 * ref) / (n + a.n0)
        return out

    def fit(X, yy, inputs):
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        w, b = logreg((X - mu) / sd, yy, a.C)
        return dict(inputs=inputs, mean=mu.tolist(), std=sd.tolist(), coef=w.tolist(), intercept=b)

    def z(m, X):
        return ((X - np.array(m["mean"])) / np.array(m["std"])) @ np.array(m["coef"]) + m["intercept"]

    def cut(s, yy, keep):
        return float(np.sort(s[yy])[int(np.floor((1 - keep) * yy.sum()))])
    inputs = ["p_quality_logit"] + feats
    # leave one run out: the label model's score alone, the model with the run's final centre, and as a bundle sees it
    SC = {"p_quality alone": np.zeros(len(y)), "model": np.zeros(len(y)), "model, as built chip by chip": np.zeros(len(y))}
    fixed = {}
    for r in sorted(set(run)):
        tr, te = run != r, run == r
        SC["p_quality alone"][te] = z(fit(L[tr, None], y[tr], inputs[:1]), L[te, None])
        Xtr = np.column_stack([L, centred(V, grp)])[tr]
        m = fit(Xtr, y[tr], inputs)
        ref = np.median(V[tr], axis=0)
        SC["model"][te] = z(m, np.column_stack([L, centred(V, grp, ref, a.n0)])[te])
        Vb = asbuilt(V, r, ref) if a.centre == "run" else centred(V, grp, ref, a.n0) if cen else V
        SC["model, as built chip by chip"][te] = z(m, np.column_stack([L, Vb])[te])
        st = z(m, Xtr)
        for keep in (0.95, 0.90):                       # (the cutoff from the training runs, on the held-out run)
            t = cut(st, y[tr], keep)
            q = fixed.setdefault(keep, [0, 0])
            q[0] += int((SC["model"][te][~y[te]] < t).sum())
            q[1] += int((SC["model"][te][y[te]] < t).sum())
    print(f"{len(y)} decided flakes ({y.sum()} keep / used, {(~y).sum()} discard), {len(set(run))} runs; centring: {a.centre}"
          + (f" (median shrunk to the training runs' by n0 = {a.n0:g} flakes)" if cen else ""))
    ev = {}
    for name, s in SC.items():
        per = [_auc(s[run == r], y[run == r]) for r in sorted(set(run))]
        rem = {k: int((s[~y] < cut(s, y, k)).sum()) for k in (0.95, 0.90)}
        print(f"  {name:30s} leave-one-run-out AUC {_auc(s, y):.3f} (mean per run {np.mean(per):.3f}); discards removed at "
              f"the cutoff keeping 95 %: {rem[0.95]}, 90 %: {rem[0.90]} (of {(~y).sum()})")
        ev[name] = dict(auc=round(_auc(s, y), 3), auc_per_run=round(float(np.mean(per)), 3), removed_95=rem[0.95], removed_90=rem[0.90])
    print("  model, cutoff from the training runs: " + ", ".join(f"{k:.0%}: {v[0]} discards removed, {v[1]} keeps lost"
                                                               for k, v in fixed.items()))
    # the model on all runs; the cutoffs on its scores with the centring a bundle uses (shrunk run medians)
    m = fit(np.column_stack([L, centred(V, grp)]), y, inputs)
    m["defects"] = vers.pop()                   # (flakeid.defects100 measures the flakes this way for this model)
    if cen:
        m["centre"] = dict(by=a.centre, features=feats, ref=[round(float(v), 3) for v in np.median(V, axis=0)], n0=a.n0)
    cg = {g: D.centres(m, [dict(zip(feats, v)) for v in V[grp == g]]) for g in set(grp)}
    sc = np.array([D.pick_p(m, p, dict(zip(feats, v)), cg[g]) for p, v, g in zip(pq, V, grp)])
    rej = {}
    for keep in sorted({0.95, 0.90, a.keep}, reverse=True):
        t = float(np.floor(cut(sc, y, keep) * 1e4) / 1e4)   # (rounded down: the keep at the cutoff stays)
        rej[keep] = t
        print(f"  all runs, keeping {keep:.0%} of keep + used: reject_below {t:.4f}: rejects {int((sc[y] < t).sum())} of {y.sum()} "
              f"keep / used and {int((sc[~y] < t).sum())} of {(~y).sum()} discards ({int((sc < t).sum())} of {len(y)} flakes)")
    print(f"  keep tier (p >= {a.keep_above:g}): {int((sc >= a.keep_above).sum())} flakes ({int((sc[y] >= a.keep_above).sum())} keep / "
          f"used, {int((sc[~y] >= a.keep_above).sum())} discards)")
    print("  weights (standardised): " + ", ".join(f"{k} {c:+.2f}" for k, c in zip(m["inputs"], m["coef"]))
          + f"; intercept {m['intercept']:+.2f}")
    import time
    m.update(reject_below=rej[a.keep], keep_above=a.keep_above, keep_share=a.keep, C=a.C, fitted=time.strftime("%Y-%m-%d"),
             evaluation=dict(n=int(len(y)), n_keep=int(y.sum()), runs=len(set(run)), loro=ev,
                             loro_training_cutoff={f"{k:.2f}": dict(removed=v[0], keeps_lost=v[1]) for k, v in fixed.items()},
                             reject_below_95=rej[0.95], reject_below_90=rej[0.90]),
             source=f"designer decisions on hBN ({Path(a.csv).name}), runs: {', '.join(sorted(set(run)))}")
    prof = S.PROFILE_DIR / f"screen100_{a.material}.json"
    if a.pick_out:
        Path(a.pick_out).write_text(json.dumps(m, indent=1), encoding="utf-8")
        print(f"wrote the pick model to {a.pick_out}")
    if a.dry_run:
        print(f"(dry run: profiles/screen100_{a.material}.json not written)")
    else:
        d = json.loads(prof.read_text())
        d["pick"] = m
        prof.write_text(json.dumps(d, indent=1))
        print(f"wrote profiles/screen100_{a.material}.json (pick model on the defects: reject below {m['reject_below']}, keep "
              f"from {a.keep_above:g})")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "run", "scan", "decision", "p_quality"] + feats + ["centre_" + k for k in feats]
                       + ["p", "loro_z", "rejected"])
            for r, p, v, g, s0, s1 in zip(rows, pq, V, grp, sc, SC["model"]):
                c = cg[g]
                w.writerow([r.get("key", ""), r["run"], r["scan"], r["decision"], p] + list(v)
                           + [round(c.get(k, 0.0), 3) for k in feats] + [round(float(s0), 4), round(float(s1), 3),
                                                                        int(s0 < m["reject_below"])])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("run")
    e.add_argument("labels")
    e.add_argument("--material", required=True)
    e.add_argument("--oxide", type=float, required=True)
    e.add_argument("--overlays")
    e.add_argument("-o", "--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("csv", nargs="+")
    f.add_argument("--material", required=True)
    f.add_argument("--recall", type=float, default=0.9)
    f.add_argument("-C", type=float, default=0.3)
    f.add_argument("-o", "--out", help="per-flake LOO predictions (.csv)")
    fd = sub.add_parser("fit-decisions")
    fd.add_argument("runs", nargs="+", help="run folders on the share (their pieces' viewers hold the 100x screening)")
    fd.add_argument("--data", required=True, help="the designer's shared data folder (_designer_data)")
    fd.add_argument("--material", required=True)
    fd.add_argument("-C", default="0.03,0.1,0.3,1,3", help="the regularisation(s) to choose from (comma-separated)")
    fd.add_argument("--clip", default="none,2,3,4", help="|z| clips of the standardised features to choose from")
    fd.add_argument("--drop", default="", help="features left out (comma-separated), e.g. focus_log")
    fd.add_argument("--add", default="", help="features added to the profile's list (comma-separated), e.g. from --extra")
    fd.add_argument("--extra", help="JSON {flake key run:scan-fNNNN: {feature: value}}: features measured afterwards "
                                    "(e.g. from the natural pictures), merged into the saved ones; missing ones are filled with the median")
    fd.add_argument("--signs", choices=("default", "none"), default="none", help="default: the coefficient signs of "
                    "SIGNS (sign-constrained fit)")
    fd.add_argument("--omit-thick", default="", help="feature:t_nm, ... left out (filled with the median) of flakes "
                    "whose terrace a is thicker than t_nm (t_terrace_nm, give it with --extra), also when scoring, e.g. focus_log:5")
    fd.add_argument("--cap-feature", default="roughness", help="the feature --caps limits (raw value, before scaling)")
    fd.add_argument("--caps", default="none", help="upper limits of --cap-feature to choose from (none: no cap)")
    fd.add_argument("--group", choices=("run", "piece"), default="run", help="held out together in the evaluation")
    fd.add_argument("--keep", type=float, default=0.95, help="share of the keep + used flakes at or above the pick "
                    "model's reject_below (below: screened out in the designer); 1: no screening out")
    fd.add_argument("--keep-above", type=float, default=0.5, help="pick score from which a flake is 'keep' (else 'review')")
    fd.add_argument("--reject-below", type=float, help="new gate: the label model's score below which flakes are dropped")
    fd.add_argument("--dry-run", action="store_true", help="evaluate and print only: the profile is not written")
    fd.add_argument("-o", "--out", help="per-flake scores (.csv)")
    fx = sub.add_parser("fit-defects")
    fx.add_argument("csv", help="one row per decided flake: run, scan, decision, p_quality, clean_d, edge_um_per_1000")
    fx.add_argument("--material", default="hbn")
    fx.add_argument("-C", type=float, default=1.0)
    fx.add_argument("--centre", choices=("run", "chip", "none"), default="none", help="features relative to the run's / "
                    "chip's median (default none: 2026-10-22, 1009 flakes of 7 runs, run-centred AUC 0.839 against 0.836, but as a "
                    "bundle grows by chips 0.836, and its cutoff lost 11 instead of 6 of 124 keeps on held-out runs)")
    fx.add_argument("--n0", type=float, default=50.0, help="the run median is shrunk to the training runs' by this many "
                    "flakes (a bundle with one chip so far: about half-way)")
    fx.add_argument("--keep", type=float, default=0.95, help="share of the keep + used flakes at or above reject_below")
    fx.add_argument("--keep-above", type=float, default=0.5)
    fx.add_argument("--features", default="clean_d,edge_um_per_1000", help="the defect features (table columns) next to "
                    "p_quality, comma-separated")
    fx.add_argument("--dry-run", action="store_true", help="evaluate and print only: the profile is not written")
    fx.add_argument("--pick-out", help="also write the fitted pick model (JSON) here")
    fx.add_argument("-o", "--out", help="per-flake scores (.csv)")
    a = ap.parse_args(argv)
    {"extract": extract, "fit": fit, "fit-decisions": fit_decisions, "fit-defects": fit_defects}[a.cmd](a)


if __name__ == "__main__":
    main()
