"""Device photos (start screen "Device photos"): a finished device or heterostructure photographed at 100x, 50x, 10x
and 5x, then any extra pictures at any of them (when a field of view leaves out something).

For each picture you centre the device and focus in the live view, then "Take photo".  At 100x and 50x the picture is
taken as the scan's 100x pictures of flakes: device_frames frames averaged, then the same (device_shift_frames each) at
device_shifts_um around it (doubled at 50x, so the shift is the same in pixels), which flakeid.devicephoto aligns on the
device and combines: dust in the light path and camera specks are averaged away.  At 10x and 5x: device_frames_low
frames averaged at one position.  Each objective keeps its own exposure (set for the device's substrate level).

With a stack folder (device_stack_dir; the start screen asks for it, a step name and what was done): the photos are a
fabrication step of that device (docs/stack_workflow.md, "Stack folders").  Raw pictures -> the local working folder
<local_root>/device_photos/<device>_<step>_<date>/ (16-bit Bayer TIFFs, <name>_index.csv, <name>_meta.json: step,
description, who, when), made into photos and terrace frames there (flakeid.devicephoto), then written to
<stack folder>/NN_<step>/ (NN = one more than the highest NN_ folder; 01_stacking counts as 01):

    <device>_<step>_<mag>.jpg            main photo of each objective (100x, 50x, 10x, 5x)
    <device>_<step>_<mag>_2.jpg, _3 ...   further photos at the same objective (extra pictures), in the order taken
    <photo>_terraces.jpg, <photo>_T1.jpg  a 100x photo's terrace map and its terraces' colour +-8 % frames, <photo>
                                          being that photo's name without .jpg: <device>_<step>_100x_terraces.jpg,
                                          <device>_<step>_100x_T1.jpg; the second 100x photo's <device>_<step>_100x_2_T1.jpg
    <device>_<step>.html                  the step's page (links into the same folder)
    step.json                             format vdw-fabstep/1 (written last: the designer finds steps by it)

(<device> and <step> with anything but letters, digits, _ . - replaced by _.)  Once every file is there with its size,
the raw TIFFs and the working folder's photos are deleted (its log, index and _photos.json are kept).  If the stack
folder cannot be reached (share down), everything stays in the working folder; send it later with

    python -m flakeid.scope.devicephotos --resend <local working folder> <stack folder>

Without a stack folder (older settings, no start screen): <local_root>/device_photos/<device>_<date>/ as above, its
photos and <name>.html, and the folder copied to device_photos_to (default: device_photos next to the results folder).

The 50x and 5x objectives need not be in objective_positions: you are asked to put them in, and the nosepiece position
read then is remembered in the settings file.
"""
from __future__ import annotations

import datetime
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from ..devicephoto import MAG, ORDER, process

DEFAULT_EXPOSURE_SCALE = {"100x": 1.0, "50x": 0.3, "10x": 1.0, "5x": 0.3}


def _safe(s, default="device"):
    """A name usable in file names (Windows too): anything but letters, digits, _ . - becomes _."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (s or "").strip()).strip("._") or default


def remember_objective(sc, name, pos):
    """Record nosepiece position `pos` as objective `name` in the settings (and the settings file, a backup kept)."""
    s = sc.s
    op = dict(s.objective_positions or {})
    if pos is None or name in op or pos in op.values():
        return
    op[name] = pos
    s.objective_positions = op
    sc.say(f"  nosepiece position {pos} is the {name} objective (remembered)")
    from .objoffsets import update_settings_file
    update_settings_file(sc, {"objective_positions": op})


def _objective(sc, obj):
    """Get objective `obj` in: True, or "skip" / "stop"."""
    r = sc.wait_objective(obj, f"Put in the {obj} objective for the device photo (the window closes by itself when it is in).")
    if r is False:
        return "stop"
    if r is None:                                       # (no nosepiece reader, or a position not known yet)
        a = sc.ui.confirm(f"Objective: {obj}", f"Put in the {obj} objective, then press Continue.", ("Continue", "Skip this one", "Stop"))
        if a == 1:
            return "skip"
        if a not in (0,):
            return "stop"
        sc.ui.mock_hook(sc, "mockObjective", int(MAG[obj]))
        if sc.nose is not None:
            try:
                remember_objective(sc, obj, sc.nose.position())
            except Exception:
                pass
    return True


def _average(sc, n, what):
    fr = [sc.snap_good(f"{what} frame {j + 1}", after_move=j == 0) for j in range(n)]   # (one move, then a burst)
    return np.round(np.mean(np.asarray(fr, np.float32), axis=0)).astype(fr[0].dtype), fr


def _follow_offsets(sc, D, obj):
    """The stage and focus moved by the objective offsets (objective offsets tool) from the last objective to obj, so the
    device stays centred and nearly in focus."""
    from .objoffsets import offset_between
    prev = D.get("last_obj")
    d = offset_between(sc.s, prev, obj) if prev else None
    if d is None:
        return
    c = sc.core
    x, y = c.getXPosition(sc.xy) + d[0], c.getYPosition(sc.xy) + d[1]
    sc.approach_to(x, y)
    sc.set_z(sc.z_read() + d[2])
    sc.say(f"    moved by the {prev} -> {obj} offset: {d[0]:+.2f}, {d[1]:+.2f} um, focus {d[2]:+.2f} um")


def shoot(sc, D, obj, label):
    """One device photo at objective obj, shown for a check and retaken on request: None (kept or skipped) or "stop"."""
    s, c = sc.s, sc.core
    sc.ui.check_stop(sc)
    r = _objective(sc, obj)
    if r != True:                                       # noqa: E712  ("skip" / "stop")
        sc.say(f"  {obj}: {r}")
        return "stop" if r == "stop" else None
    sc.objective_now = obj
    sc.view_scale = 1.0 / 9.54 if obj == "100x" else 10.0 / MAG[obj]
    _follow_offsets(sc, D, obj)
    D["last_obj"] = obj
    sc.stream_off()
    ex0 = D["exp"].get(obj) or (s.target_exposure_ms if obj in ("100x", "50x") else s.exposure_ms) * DEFAULT_EXPOSURE_SCALE[obj]
    c.setExposure(ex0)
    c.waitForDevice(c.getCameraDevice())
    retake = 0
    while True:
        a = sc.ask_focus(f"Device photo: {obj}", f"{D['name']}, {obj}{'' if label == 'main' else ' (extra picture)'}"
                         f"{f' (retake {retake})' if retake else ''}: centre what the photo should show and focus on it, then press "
                         "\"Take photo\".", button="Take photo")
        if a == 2 or a is None:
            return "stop"
        if a != 0:
            sc.say(f"  {obj}: skipped")
            return None
        row, files = _take(sc, D, obj, label)
        v = _review(sc, D, obj, row, files)
        if v == "keep":
            D["idx"].write(row)
            D["idx"].flush()
            sc.ui.progress(sc, f"device photos: {D['n']} taken")
            return None
        for f_ in files:                                # (retake / stop: the pictures of this photo go)
            try:
                (D["dir"] / f_).unlink()
            except OSError:
                pass
        D["n"] -= 1
        if v == "stop":
            sc.say(f"  {obj} {label}: dropped; stopped")
            return "stop"
        retake += 1
        sc.say(f"  {obj} {label}: retake {retake}")


def _review(sc, D, obj, row, files):
    """The photo just taken shown with what was checked: "keep", "retake" or "stop"."""
    from ..devicephoto import load_flat, preview
    import csv
    import io
    cols = ["n", "label", "objective", "file", "shift_files", "shift_um", "frames", "shift_frames", "stage_x_um", "stage_y_um", "z_um",
            "exposure_ms", "dark", "saturated_frac", "um_per_px"]
    r = dict(zip(cols, next(csv.reader(io.StringIO(row.strip())))))
    s = sc.s
    ffp = s.flat10 if obj == "10x" else (sc._flat100_for(s.oxide_nm) if obj == "100x" else None)
    pv = D["dir"] / "preview" / (Path(files[0]).stem + ".jpg")
    pv.parent.mkdir(exist_ok=True)
    try:
        notes, bad = preview(D["dir"], r, load_flat(ffp) if ffp else None, s.oxide_nm, pv)
    except Exception as e:
        notes, bad = [f"(no preview: {type(e).__name__}: {e})"], True
        pv = None
    sc.say("    check: " + "; ".join(notes))
    msg = f"{D['name']}, {obj} ({r['label']}):\n  - " + "\n  - ".join(notes)
    if bad:
        msg += "\n\nSomething may be wrong: look at the picture and retake if needed."
    if pv is None or not pv.is_file():
        a = sc.ui.confirm(f"Device photo: {obj}", msg, ("Keep", "Retake", "Stop"))
    else:
        a = sc.ui.ask_preview(sc, f"Device photo: {obj}", msg, str(pv), ("Keep", "Retake", "Stop"))
    return {0: "keep", 1: "retake"}.get(a, "stop") if a is not None else "keep"


def _take(sc, D, obj, label):
    """Take the pictures of one photo: (its index row, [files])."""
    s, c = sc.s, sc.core
    try:
        sc.check_exposure(True)
    except Exception as e:                              # (over / under exposed: taken anyway, noted in the log)
        sc.say(f"    exposure check: {e}")
    D["exp"][obj] = c.getExposure()
    shifted = obj in ("100x", "50x")
    nfr = max(1, int(s.device_frames if shifted else s.device_frames_low))
    D["n"] += 1
    stem = f"{D['n']:02d}_{obj}_{_safe(label)}"
    x0, y0, z = c.getXPosition(sc.xy), c.getYPosition(sc.xy), sc.z_read()
    sc.set_phase("device photos")
    im, frames = _average(sc, nfr, f"{obj} {label}")
    fs = sc.full_scale()
    sat = max(float((f_ >= fs - 1).mean()) for f_ in frames) if frames[0].ndim == 2 else 0.0
    opos, oname = sc.objective()
    info = {"device": D["name"], "label": label, "objective": obj, "nosepiece_position": "" if opos is None else opos,
            "stage_x_um": x0, "stage_y_um": y0, "z_um": z, "exposure_ms": c.getExposure(), "frames": nfr, "bit_depth": fs,
            "camera": c.getCameraDevice(), "bayer": "RGGB (R at top-left)"}
    sc.save_image(D["dir"] / f"{stem}.tif", im, info)
    files, shf, shu = [f"{stem}.tif"], [], []
    nsh = max(1, int(s.device_shift_frames))
    if shifted:
        k = 100.0 / MAG[obj]
        for j, (dx, dy) in enumerate(s.device_shifts_um or ()):
            sc.ui.check_stop(sc)
            sc.approach_to(x0 + dx * k, y0 + dy * k)
            ims, _ = _average(sc, nsh, f"{obj} {label} shift {j + 1}")
            fn = f"{stem}_s{j + 1}.tif"
            xs, ys = c.getXPosition(sc.xy), c.getYPosition(sc.xy)
            sc.save_image(D["dir"] / fn, ims, dict(info, shift=j + 1, stage_x_um=xs, stage_y_um=ys, frames=nsh))
            shf.append(fn)
            shu.append(f"{xs - x0:.3f}:{ys - y0:.3f}")
        sc.approach_to(x0, y0)                          # (back where the device was centred)
    files += shf
    from .objoffsets import pixel_um
    px = pixel_um(s, obj)
    row = (f"{D['n']},{label},{obj},{stem}.tif,{';'.join(shf)},{';'.join(shu)},{nfr},{nsh if shf else ''},{x0},{y0},{z},"
           f"{c.getExposure()},{sc.DARK},{sat},{'' if px is None else round(px, 6)}\r\n")
    sc.say(f"  {obj} {label}: {stem}.tif ({nfr} frames" + (f", + {len(shf)} shifted x {nsh}" if shf else "") + f"), exposure "
           f"{c.getExposure():.3f} ms, Z {z:.3f}" + (f"   WARNING: {sat * 100:.3f} % of pixels saturated" if sat > 0.001 else ""))
    return row, files


def dest_root(s):
    if s.device_photos_to:
        return Path(s.device_photos_to)
    if s.copy_results_to:
        return Path(s.copy_results_to).parent / "device_photos"
    return None


# ---- stack folders (docs/stack_workflow.md, "Stack folders")

STEP_FORMAT = "vdw-fabstep/1"


def stack_device_name(folder):
    """The device name of a stack folder: its .stack file's name without .stack (the newest, if several), else the
    folder's name."""
    folder = Path(folder)
    try:
        st = sorted(folder.glob("*.stack"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        st = []
    return st[0].stem if st else folder.name


def next_step_number(folder):
    """One more than the highest NN_ folder of a stack folder (01_stacking counts as 01: the first step is 02)."""
    hi = 1
    for p in Path(folder).iterdir():
        m = re.match(r"^(\d{2,})_", p.name)
        if m and p.is_dir():
            hi = max(hi, int(m.group(1)))
    return hi + 1


def remembered_stack_dir(s):
    """The stack folder given last on the start screen (<local_root>/device_photos.json), else device_stack_dir."""
    try:
        f = Path(s.local_root) / "device_photos.json"
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8")).get("stack_dir") or s.device_stack_dir or ""
    except (OSError, ValueError):
        pass
    return s.device_stack_dir or ""


def remember_stack_dir(s, folder):
    try:
        Path(s.local_root).mkdir(parents=True, exist_ok=True)
        (Path(s.local_root) / "device_photos.json").write_text(json.dumps({"stack_dir": str(folder)}), encoding="utf-8")
    except OSError:
        pass


def _who(s):
    """step.json "by": the start screen's "Messages to" name (a Slack name), else the Windows user."""
    try:
        from .ui import remembered_notify
        n = remembered_notify(s)[0].strip()
        if n and "@" not in n:
            return n
    except Exception:
        pass
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return ""


def _meta_path(d):
    return next(iter(sorted(Path(d).glob("*_meta.json"))), None)


def build_step(d, device, log=print):
    """The step's files, named for the stack folder, in <working folder>/step_out/ (rebuilt each time): (that folder,
    [file names], step record).  From the processed photos (<name>_photos.json, photos/) and <name>_meta.json."""
    from ..devicephoto import make_page
    d = Path(d)
    meta = json.loads(_meta_path(d).read_text(encoding="utf-8"))
    rec = json.loads(next(d.glob("*_photos.json")).read_text())
    dev, step = _safe(device), _safe(meta["step"], "step")
    out = d / "step_out"
    if out.exists():
        shutil.rmtree(str(out))
    out.mkdir()
    files, photos, page_photos, seen = [], [], [], {}

    def put(src, name):
        shutil.copy2(str(d / src), str(out / name))
        files.append(name)
        return name
    for ph in rec["photos"]:                        # (in the order taken: the main photo of an objective first)
        obj = ph["objective"]
        seen[obj] = seen.get(obj, 0) + 1
        base = f"{dev}_{step}_{obj}" + ("" if seen[obj] == 1 else f"_{seen[obj]}")
        pp = dict(ph, jpg=put(ph["jpg"], base + ".jpg"), terraces=[])
        tm = put(ph["terrace_map"], base + "_terraces.jpg") if ph.get("terrace_map") else None
        if tm:
            pp["terrace_map"] = tm
        ts = []
        for t_ in ph.get("terraces") or []:
            hc = put(t_["hc"], f"{base}_T{t_['n']}.jpg")
            pp["terraces"].append(dict(t_, hc=hc))
            ts.append(dict(n=t_["n"], area_um2=t_["area_um2"], contrast=t_["contrast"], hc=hc))
        page_photos.append(pp)
        photos.append(dict(objective=obj, label=ph.get("label", ""), file=pp["jpg"], um_per_px=ph["um_per_px"], w=ph["w"],
                           h=ph["h"], terrace_map=tm, terraces=ts))
    srec = {"format": STEP_FORMAT, "name": meta["step"], "description": meta.get("description", ""), "at": meta["at"],
            "by": meta.get("by", ""), "device": device, "oxide_nm": meta.get("oxide_nm", rec.get("oxide_nm")),
            "source": "scanner device photos", "photos": photos}
    page = f"{dev}_{step}.html"
    (out / page).write_text(make_page(dict(rec, device=device, photos=page_photos, step=meta["step"],
                                           description=meta.get("description", ""), at=meta["at"], by=meta.get("by", ""))),
                            encoding="utf-8")
    files.append(page)
    (out / "step.json").write_text(json.dumps(srec, indent=1), encoding="utf-8")
    files.append("step.json")                      # (last: written last to the stack folder)
    return out, files, srec


def send_step(d, stack_dir, log=print):
    """Write the working folder d's step to <stack_dir>/NN_<step>/, check it (every file there, same size), then delete
    the raw TIFFs and d's photos.  Returns the step folder; raises OSError (nothing deleted) if it could not be written."""
    d, root = Path(d), Path(stack_dir)
    if not root.is_dir():
        raise OSError(f"the stack folder {root} cannot be reached (or does not exist)")
    device = stack_device_name(root)
    out, files, srec = build_step(d, device, log)
    step = _safe(srec["name"], "step")
    nn = next_step_number(root)
    while True:                                     # (a folder of that number made meanwhile: the next one)
        dst = root / f"{nn:02d}_{step}"
        try:
            dst.mkdir()
            break
        except FileExistsError:
            nn += 1
    try:
        for f_ in files:                            # (step.json last: the designer finds a step by it)
            shutil.copyfile(str(out / f_), str(dst / f_))
        bad = [f_ for f_ in files if not (dst / f_).is_file() or (dst / f_).stat().st_size != (out / f_).stat().st_size
               or (dst / f_).stat().st_size == 0]
        if bad:
            raise OSError(f"{len(bad)} file(s) not written completely to {dst}: {', '.join(bad[:5])}")
    except Exception:
        shutil.rmtree(str(dst), ignore_errors=True)   # (no half step in the stack folder; all is still in d)
        raise
    nb = sum((dst / f_).stat().st_size for f_ in files)
    log(f"device photos: step {dst.name} of {device} written ({len(files)} files, {nb / 1e6:.1f} MB): {dst}")
    # the working folder: raw TIFFs, previews, photos and the local page go; the log, index, _photos.json and meta stay
    freed = 0
    for p in list(d.glob("*.tif")) + list(d.glob("*.html")):
        freed += p.stat().st_size
        p.unlink()
    for sub in ("preview", "photos", "step_out"):
        if (d / sub).is_dir():
            freed += sum(p.stat().st_size for p in (d / sub).rglob("*") if p.is_file())
            shutil.rmtree(str(d / sub), ignore_errors=True)
    mp = _meta_path(d)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta.update(sent_to=str(dst), sent_at=int(time.time() * 1000), files=files)
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"  raw pictures and working copies deleted from {d} ({freed / 1e6:.0f} MB); its log and lists kept")
    return dst


def resend_command(d, stack_dir):
    return f'python -m flakeid.scope.devicephotos --resend "{d}" "{stack_dir}"'


def run_device_photos(sc):
    s = sc.s
    stack = s.device_stack_dir.strip() if s.device_stack_dir else ""
    if stack:                                       # (a fabrication step of a stack folder)
        try:
            s.device_name = stack_device_name(stack)
        except Exception:
            s.device_name = s.device_name or Path(stack).name
        name = f"{_safe(s.device_name)}_{_safe(s.device_step, 'step')}" + datetime.datetime.now().strftime("_%Y%m%d_%H%M")
    else:
        name = _safe(s.device_name or "device") + datetime.datetime.now().strftime("_%Y%m%d_%H%M")
    d = Path(s.local_root) / "device_photos" / name
    k = 1
    while d.exists():                               # (two jobs in one minute)
        k += 1
        d = Path(s.local_root) / "device_photos" / f"{name}_{k}"
    name = d.name
    d.mkdir(parents=True)
    sc.open_log(d / f"{name}_log.txt")
    sc.say(f"device photos of {s.device_name} (oxide {s.oxide_nm:g} nm) -> {d}")
    if stack:
        meta = dict(format="flakeid-devicephotos-meta/1", device=s.device_name, step=s.device_step.strip(),
                    description=(s.device_step_description or "").strip(), at=int(time.time() * 1000), by=_who(s),
                    oxide_nm=s.oxide_nm, stack_dir=stack)
        (d / f"{name}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        sc.say(f"  fabrication step \"{meta['step']}\" of the stack folder {stack}")
    if s.set_camera:
        sc.apply_camera(s.target_exposure_ms)
    sc.DARK = s.dark * (sc.full_scale() + 1) / 4096.0
    sc.open_nosepiece()
    sc.move_box = None
    D = dict(name=s.device_name or name, dir=d, n=0, exp={}, idx=open(d / f"{name}_index.csv", "w", newline=""))
    D["idx"].write("n,label,objective,file,shift_files,shift_um,frames,shift_frames,stage_x_um,stage_y_um,z_um,exposure_ms,dark,"
                   "saturated_frac,um_per_px\r\n")
    stopped = False
    try:
        for obj in ORDER:
            if shoot(sc, D, obj, "main") == "stop":
                stopped = True
                break
        n_extra = 0
        while not stopped:
            a = sc.ui.confirm("Device photos", f"{D['n']} photo(s) of {D['name']} taken.\n\nAnother picture (when a field of view leaves "
                              "out something)?  Choose its objective:", ("Done", "100x", "50x", "10x", "5x"))
            if not a or a not in (1, 2, 3, 4):
                break
            n_extra += 1
            if shoot(sc, D, ORDER[a - 1], f"extra{n_extra}") == "stop":
                break
    finally:
        D["idx"].close()
        sc.stream_off()
        sc.objective_now = ""
    if not D["n"]:
        sc.say("device photos: none taken")
        return
    sc.set_phase("device photos: processing")
    page = None
    try:
        page = process(d, flat10=s.flat10 or None, flat100=sc._flat100_for(s.oxide_nm), oxide_nm=s.oxide_nm, log=sc.say)
    except Exception as e:
        sc.say(f"device photos: processing failed ({type(e).__name__}: {e}); the raw pictures are in {d}. "
               f"Again with: python -m flakeid.devicephoto \"{d}\"")
    where = str(d)
    if stack:
        if page is None:
            sc.say(f"device photos: not written to the stack folder (processing failed); afterwards: {resend_command(d, stack)}")
        else:
            try:
                dst = send_step(d, stack, log=sc.say)
                where, page = str(dst), dst / f"{_safe(s.device_name)}_{_safe(s.device_step, 'step')}.html"
            except Exception as e:
                msg = (f"The stack folder could not be written ({type(e).__name__}: {e}).\n\nEverything is kept in\n{d}\n\n"
                       f"Send it there later (on this PC) with\n{resend_command(d, stack)}")
                sc.say("device photos: " + msg.replace("\n\n", " ").replace("\n", " "))
                if sc.ui.can_ask:
                    sc.ui.message("Device photos", msg)
    else:
        root = dest_root(s)
        if root is not None:
            try:
                dst = root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(d), str(dst), dirs_exist_ok=True)
                where = str(dst)
                sc.say(f"device photos copied to {dst}")
            except Exception as e:
                sc.say(f"device photos: not copied to {root} ({type(e).__name__}: {e}); they are in {d}")
    sc.report_times()
    if page is not None and sc.ui.can_ask:
        if sc.ui.confirm("Device photos", f"{D['n']} photo(s) of {D['name']} in\n{where}", ("Open the page", "Close")) == 0:
            pg = Path(where) / page.name
            try:
                import os
                os.startfile(str(pg if pg.is_file() else page))     # (Windows)
            except Exception:
                pass


def resend(d, stack_dir, flat10=None, flat100=None, step=None, description=None, log=print):
    """A working folder not written to its stack folder (share down, or processing failed): processed if needed, then
    written as a step (send_step).  Its log gets what happened."""
    d = Path(d)
    lf = next(iter(sorted(d.glob("*_log.txt"))), None)

    def say(m):
        log(m)
        if lf is not None:
            with open(lf, "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%H:%M:%S ") + m + "\n")
    mp = _meta_path(d)
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp else None
    if meta is None:                                # (an older folder, without a stack folder: its step given here)
        if not step:
            raise SystemExit(f"{d}: no *_meta.json; give the step's name with --step")
        idx = next(iter(sorted(d.glob("*_index.csv"))), None)
        if idx is None:
            raise SystemExit(f"{d}: no *_index.csv (not a device-photos folder)")
        mp = d / (idx.name[:-len("_index.csv")] + "_meta.json")
        meta = dict(format="flakeid-devicephotos-meta/1", device="", step=step, description=description or "",
                    at=int(idx.stat().st_mtime * 1000), by="", oxide_nm=None)
    if meta.get("sent_to"):
        raise SystemExit(f"{d} was written to {meta['sent_to']} already (its raw pictures are deleted)")
    if step:
        meta["step"] = step
    if description is not None:
        meta["description"] = description
    meta["stack_dir"] = str(stack_dir)
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    say(f"resend {d} -> stack folder {stack_dir}")
    if not any(d.glob("*_photos.json")) or not (d / "photos").is_dir():
        say("  processing the raw pictures first")
        process(d, flat10=flat10, flat100=flat100, oxide_nm=float(meta.get("oxide_nm") or 285.0), log=say)
    if not meta.get("oxide_nm"):
        meta["oxide_nm"] = json.loads(next(d.glob("*_photos.json")).read_text()).get("oxide_nm")
        mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return send_step(d, stack_dir, log=say)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m flakeid.scope.devicephotos", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--resend", nargs=2, metavar=("WORKING_FOLDER", "STACK_FOLDER"), required=True,
                    help="write a local working folder's photos to the stack folder as a fabrication step")
    ap.add_argument("--flat10", help="10x flat field .npz (only if the photos were not processed yet)")
    ap.add_argument("--flat100", help="100x flat field .npz of the chip's oxide (only if not processed yet)")
    ap.add_argument("--step", help="the step's name (default: the one given on the start screen)")
    ap.add_argument("--description", help="what was done (default: the one given on the start screen)")
    a = ap.parse_args(argv)
    try:
        resend(a.resend[0], a.resend[1], a.flat10, a.flat100, a.step, a.description)
    except OSError as e:
        print(f"not written: {e}; nothing deleted")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
