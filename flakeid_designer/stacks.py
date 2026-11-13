"""Stack files (one per stack: plan, stacking steps, final state, exports) and the move of the old designs to them.

    python -m flakeid_designer.stacks backup  <_designer_data>            a checked copy, next to it
    python -m flakeid_designer.stacks migrate <_designer_data> [--dry-run]   backup first (checked), then stacks/

The format and the plan are in docs/stack_workflow.md.  The migration never changes or deletes the old files: every
design file is copied verbatim into its stack file (legacy.sources) and stays in designs/.  It refuses to run
unless a backup of the whole data folder was made and checked file by file (size and SHA-256) in the same call.
"""
from __future__ import annotations

import argparse
import re
import datetime
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

FORMAT = "vdw-stack/1"
POSE_KEYS = ("x", "y", "rot", "mirror")


# ---------------------------------------------------------------- backup
def _sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _files(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file())


def backup(data_dir, log=print):
    """Copy the whole data folder to <data_dir>_backup_<date-time> next to it and check every file against the
    original (size and SHA-256).  Returns the backup folder; raises RuntimeError if anything differs."""
    src = Path(data_dir).resolve()
    if not src.is_dir():
        raise RuntimeError(f"{src} is not a folder")
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dst, n = src.with_name(f"{src.name}_backup_{stamp}"), 2
    while dst.exists():                              # (never into an existing backup)
        dst, n = src.with_name(f"{src.name}_backup_{stamp}_{n}"), n + 1
    log(f"backup: {src} -> {dst}")
    shutil.copytree(src, dst, copy_function=shutil.copy2)
    want = _files(src)
    bad = []
    total = 0
    for p in want:
        q = dst / p.relative_to(src)
        try:
            same = q.is_file() and q.stat().st_size == p.stat().st_size and _sha256(q) == _sha256(p)
        except OSError as e:
            same = False
            bad.append(f"{p.relative_to(src)} ({e})")
        if not same:
            bad.append(str(p.relative_to(src)))
        total += p.stat().st_size
    manifest = dict(source=str(src), made=datetime.datetime.now().isoformat(timespec="seconds"), files=len(want),
                    bytes=total, checked="size and sha256", problems=bad)
    (dst / "_backup_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if bad:
        raise RuntimeError(f"backup {dst}: {len(bad)} file(s) differ from the original: {', '.join(bad[:5])}")
    log(f"backup checked: {len(want)} files, {total / 1e6:.1f} MB, all identical to the original")
    return dst


# ---------------------------------------------------------------- the stack file
def new_stack(sid, name, by=None):
    now = int(time.time() * 1000)
    return dict(format=FORMAT, id=sid, name=name or "Untitled stack", created=now, updated=now, by=by,
                status="planning", plan=dict(layers=[], overlap_um2=0, overlap_d_um=0, revisions=[]),
                steps=[], final=dict(images=[], poses={}, shapes={}, notes=""), exports=[], legacy=dict(sources={}))


def _pose(L, planned=False):
    P = L.get("plan") if planned and isinstance(L.get("plan"), dict) else L
    return {k: P.get(k, L.get(k)) for k in POSE_KEYS}


def _plan_layer(L):
    """A design layer as planned: its planned position if it was moved as built, without the stacking records."""
    out = {k: v for k, v in L.items() if k not in ("plan", "picked", "photos")}
    out.update(_pose(L, planned=True))
    return out


def _picture(si, **extra):
    keep = ("url", "w", "h", "f", "s", "rot", "mirror", "tx", "ty", "opacity", "name")
    return dict({k: si[k] for k in keep if k in si}, **extra)


# ---------------------------------------------------------------- migration
def _norm_name(n):
    """A design's name without the as-built / copy endings (the old Save made new copies of the same stack)."""
    n = re.sub(r"\s*·?\s*as[ -]built\b.*$", "", str(n or "").strip(), flags=re.I)
    n = re.sub(r"\s*\((copy|\d+)\)\s*$|\s+copy(\s*\d+)?\s*$", "", n, flags=re.I)
    return n.strip().lower()


def _is_asbuilt(d):
    """An as-built record: made by "Save as-built copy" (as_built.of; missing when the plan design was never saved
    under that id) or named so ("... · as built ...")."""
    return bool(d.get("as_built")) or bool(re.search(r"\bas[ -]built\b", str(d.get("name") or ""), flags=re.I))


def _flakes(d):
    """The flakes of a design's layers (terrace id without its terrace letter)."""
    return {str(L.get("tid") or "").rsplit(".", 1)[0] for L in d.get("layers") or [] if L.get("tid")}


def _same_stack(a, b):
    """Two designs are copies of one stack: the same name (without the endings) and mostly the same flakes."""
    if _norm_name(a.get("name")) != _norm_name(b.get("name")):
        return False
    fa, fb = _flakes(a), _flakes(b)
    if not fa or not fb:
        return not fa and not fb
    return fa <= fb or fb <= fa or len(fa & fb) / len(fa | fb) >= 0.5


def group_designs(designs):
    """[[design ids of one stack], ...]: copies of the same stack together (as-built copies with the design they were
    made from, or with copies of it by name and flakes when that design is gone)."""
    ids = sorted(designs, key=lambda i: designs[i].get("updated") or 0)
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for k, a in enumerate(ids):
        of = (designs[a].get("as_built") or {}).get("of")
        if of in parent:
            parent[find(a)] = find(of)
        for b in ids[:k]:
            if _same_stack(designs[a], designs[b]):
                parent[find(a)] = find(b)
    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def stacks_from_designs(designs):
    """{design_id: design} -> {stack_id: stack}: each stack's design files (copies made by the old Save, and its
    as-built copies) in one stack file.  The newest plan design is the plan (older ones: plan revisions); the newest
    as-built copy is the final state (older ones: final revisions); picked-up layers: pick-up steps.  A group of only
    as-built copies (their design deleted): the plan from the layers' planned positions."""
    out = {}
    for g in group_designs(designs):
        g = sorted(g, key=lambda i: designs[i].get("updated") or 0)          # oldest first
        plans = [i for i in g if not _is_asbuilt(designs[i])]
        built = [i for i in g if i not in plans]
        main = plans[-1] if plans else built[-1]
        d = designs[main]
        first = designs[(plans or built)[0]].get("name") or "Untitled stack"   # (the original name, not "... copy")
        st = new_stack(main, re.sub(r"\s*·?\s*as[ -]built\b.*$", "", first, flags=re.I).strip() or first, d.get("by"))
        st["created"] = min(designs[i].get("updated") or st["created"] for i in g)
        st["updated"] = max(designs[i].get("updated") or 0 for i in g) or st["updated"]
        for i in g:
            st["legacy"]["sources"][i] = designs[i]
        layers = d.get("layers") or []
        st["plan"]["layers"] = [_plan_layer(L) for L in layers]
        st["plan"]["overlap_um2"], st["plan"]["overlap_d_um"] = d.get("overlap_um2", 0), d.get("overlap_d_um", 0)
        for i in plans[:-1]:                                                # older copies of the plan
            o = designs[i]
            st["plan"]["revisions"].append(dict(at=o.get("updated"), by=o.get("by"), name=o.get("name"), from_design=i,
                                                note="an older copy of this stack (from the old Save)",
                                                layers=[_plan_layer(L) for L in o.get("layers") or []]))
        # picked-up layers (the stacking flow edited the design itself): from the newest design that has any
        src = next((designs[i] for i in reversed(g) if any(L.get("picked") for L in designs[i].get("layers") or [])), None)
        if src:
            sl = src.get("layers") or []
            picked = [L for L in reversed(sl) if L.get("picked")]            # pick-up order: top layer first
            for k, L in enumerate(picked):
                st["steps"].append(dict(id=f"m{k + 1}", kind="pickup", layer_uid=L.get("uid"),
                                        at=(L.get("picked") or {}).get("at"), by=src.get("by"),
                                        photos=[_picture(p, at=p.get("at")) for p in (L.get("photos") or [])],
                                        attachments=[], poses={M.get("uid"): _pose(M) for M in sl if M.get("picked")},
                                        flags={}, notes="(from the old design file)"))
            st["status"] = "stacking"
        # stack pictures on plan designs (the finished stack on a picture): final images
        for i in plans:
            if isinstance(designs[i].get("stack_image"), dict):
                st["final"]["images"].append(_picture(designs[i]["stack_image"], legacy=True, from_design=i))
                st["final"]["poses"] = {L.get("uid"): _pose(L) for L in designs[i].get("layers") or []}
        # as-built copies: the newest is the final state, older ones final revisions
        for n, i in enumerate(built):
            c = designs[i]
            poses = {L.get("uid"): _pose(L) for L in c.get("layers") or []}
            if isinstance(c.get("stack_image"), dict):
                st["final"]["images"].append(_picture(c["stack_image"], legacy=True, from_design=i))
            if n < len(built) - 1:
                st["final"].setdefault("revisions", []).append(dict(at=c.get("updated"), by=c.get("by"),
                                                                    name=c.get("name"), from_design=i, poses=poses))
            else:
                st["final"]["poses"] = poses
            st["status"] = "finalized"
        out[main] = st
    return out


def migrate(data_dir, dry_run=False, log=print):
    """Backup (checked), then one stack file per stack in <data>/stacks/ (existing stack files are left alone).
    Returns (backup folder or None, {stack_id: stack}, written ids)."""
    data = Path(data_dir).resolve()
    ddir = data / "designs"
    designs = {}
    for p in sorted(ddir.glob("*.json")) if ddir.is_dir() else []:
        try:
            designs[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RuntimeError(f"{p} cannot be read ({e}): nothing migrated")     # (never migrate half)
    stacks = stacks_from_designs(designs)
    n_copies = sum(1 for d in designs.values() if _is_asbuilt(d))
    log(f"{len(designs)} design files ({n_copies} as-built copies) -> {len(stacks)} stacks")
    if dry_run:
        return None, stacks, []
    bk = backup(data, log=log)
    sdir = data / "stacks"
    sdir.mkdir(exist_ok=True)
    written = []
    for sid, st in stacks.items():
        f = sdir / f"{sid}.json"
        if f.exists():
            log(f"  {sid}: a stack file exists already: left as it is")
            continue
        tmp = sdir / f".{sid}.tmp"
        tmp.write_text(json.dumps(st), encoding="utf-8")
        os.replace(tmp, f)
        written.append(sid)
    (sdir / "_migration.json").write_text(json.dumps(dict(
        at=datetime.datetime.now().isoformat(timespec="seconds"), backup=str(bk), designs=sorted(designs),
        stacks={sid: sorted(st["legacy"]["sources"]) for sid, st in stacks.items()}, written=written), indent=1),
        encoding="utf-8")
    log(f"{len(written)} stack files written to {sdir} (the design files are unchanged; backup: {bk})")
    return bk, stacks, written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("backup", "migrate"))
    ap.add_argument("data_dir", help="the library's _designer_data folder")
    ap.add_argument("--dry-run", action="store_true", help="migrate: only say what it would do (no backup, no files)")
    a = ap.parse_args(argv)
    try:
        if a.what == "backup":
            backup(a.data_dir)
        else:
            migrate(a.data_dir, dry_run=a.dry_run)
    except RuntimeError as e:
        print("ERROR:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
