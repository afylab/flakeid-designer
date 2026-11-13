"""Domain camera test (start screen "Domain camera test", or --mode domaintest): sets up the stacking-domain (IR)
configuration the scan uses for graphite.  It measures the IR camera's XY and focus offsets from the 100x objective, its
pixel size and orientation, and tests its autofocus.  A port of tools/mm_domain_test.bsh (Micro-Manager Script Panel),
with the IR camera handled as the scan's domain imaging handles it (woken up, domain_camera_props, Frame Average 1,
frames averaged here).

  0  at 100x with the scan camera you centre a small, sharp feature on the cross (a flake corner, a graphite edge, dust;
     with edges in X and Y) and focus
  1  100x autofocus and a reference frame
  2  you change to the IR objective, camera and filter (the stage and the focus untouched)
  3  the IR camera set up as for the domain images
  4  its focus: a wide search around the 100x focus (by hand if that fails)
  5  its pixel size, rotation and mirroring, from small stage moves (2 um, then 20 % of the field in X and in Y)
  6  the XY offset (IR - 100x stage position of the same feature): its frame matched against the 100x reference on a
     common stage-aligned grid (edges, which ignore brightness and filter differences), checked at the offset, then
     confirmed or corrected by hand in the live view (the cross)
  7  the focus offset (IR - 100x), a sharpness sweep (+-8 um), the sharpness noise, and the autofocus from three starts
     with domain_af and domain_af_first
  8  the results in <local_root>/domaintest_<date>_<time>/ (log, domaintest_domain.json, the sweep as csv / png, frames),
     and domain_xy_offset_um / domain_focus_offset_um written into the settings file if you agree
The stage ends at the feature with the IR configuration in place; the scan camera is selected again.
"""
from __future__ import annotations

import datetime
import json
import math
import time
from pathlib import Path

import numpy as np

from .scan import f1, f3, isnan

SWEEP_RANGE_UM = 8.0          # diagnostic sweep +- around the IR focus
SWEEP_STEP_UM = 0.2
REPEATS = 8                   # frames at one Z for the noise of the sharpness
TEST_OFFSETS_UM = (0.0, 3.0, -3.0)    # autofocus starts relative to the sharpest Z
CAL_FRACTION = 0.20           # pixel calibration moves: this fraction of the field
MIN_MATCH = 1.5               # an image match counts if its peak is this many times the next best
FRAMES = 4                    # frames averaged for each picture that is matched (the IR camera is noisy at High Gain Off)


# ---------------------------------------------------------------------------------------------- image matching
def gray(img, bayer):
    """(float32 brightness image, binning): a RAW16 colour mosaic (the scan camera) is binned 2x2 (one pixel per cell)."""
    a = np.asarray(img)
    if a.ndim == 3:
        return a[..., :3].astype(np.float32).mean(axis=2), 1
    if bayer:
        h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        return a[:h, :w].astype(np.float32).reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)), 2
    return a.astype(np.float32), 1


def pow2(n):
    p = 64
    while p * 2 <= n:
        p *= 2
    return p


def prep(g, n):
    """The n x n centre: edge strength (Sobel), mean removed, Hann window (the match then depends on where the edges are,
    not on brightness, colour or filter)."""
    import cv2
    h, w = g.shape
    c = np.ascontiguousarray(g[(h - n) // 2:(h - n) // 2 + n, (w - n) // 2:(w - n) // 2 + n], np.float32)
    gx, gy = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    e -= e.mean()
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    return e * win[:, None] * win[None, :]


def xcorr(a, b):
    """The shift s (px) with b(p) = a(p - s), by cross-correlation: (sx, sy, peak / next best peak)."""
    n = a.shape[0]
    C = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))).real)
    inner = C[1:n - 1, 1:n - 1]
    iy, ix = np.unravel_index(int(np.argmax(inner)), inner.shape)
    by, bx = iy + 1, ix + 1
    bv = float(C[by, bx])
    ex = max(4, n // 64)
    m = np.ones_like(C, bool)
    m[max(0, by - ex):by + ex + 1, max(0, bx - ex):bx + ex + 1] = False
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    second = max(float(C[m].max()) if m.any() else 0.0, 0.0)
    l, r, u, d = float(C[by, bx - 1]), float(C[by, bx + 1]), float(C[by - 1, bx]), float(C[by + 1, bx])
    dx = 0.5 * (l - r) / (l - 2 * bv + r) if (l - 2 * bv + r) != 0 else 0.0
    dy = 0.5 * (u - d) / (u - 2 * bv + d) if (u - 2 * bv + d) != 0 else 0.0
    return (bx + max(-0.5, min(0.5, dx)) - n / 2.0, by + max(-0.5, min(0.5, dy)) - n / 2.0, bv / max(second, 1e-12))


def to_grid(img, m, n, g):
    """The image resampled onto an n x n grid aligned with the stage axes, g um per grid pixel, centred on the frame's
    centre.  m: the image's pixel -> stage matrix (um per image pixel).  Outside the image: its mean."""
    import cv2
    m = np.asarray(m, float)
    mi = np.linalg.inv(m)
    sc = g / math.sqrt(abs(np.linalg.det(m)))                   # image px per grid px
    src = np.ascontiguousarray(img, np.float32)
    if sc > 1.5:                                                 # (smoothed first: no aliasing)
        src = cv2.GaussianBlur(src, (0, 0), 0.5 * sc)
    h, w = src.shape
    v = (np.arange(n) - n // 2) * g
    vx, vy = np.meshgrid(v, v)
    px = (w / 2.0 + mi[0, 0] * vx + mi[0, 1] * vy).astype(np.float32)
    py = (h / 2.0 + mi[1, 0] * vx + mi[1, 1] * vy).astype(np.float32)
    return cv2.remap(src, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(src.mean()))


# ---------------------------------------------------------------------------------------------- the test
def run_domain_test(sc):
    s, c = sc.s, sc.core
    out = Path(s.local_root) / ("domaintest_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    sc.open_log(out / "domaintest_log.txt")
    say = sc.say
    say(f"domain camera test -> {out}")
    M100 = np.asarray(sc.M, float) / 9.54                       # 100x pixel -> stage (um per px), the scan's convention
    cam100 = c.getCameraDevice()
    M_scan, vs_scan = sc.M.copy(), getattr(sc, "view_scale", 1.0)
    cams = None

    def avg(n=FRAMES):
        fr = [sc.frame() for _ in range(n)]
        return fr[0] if n == 1 else np.mean(np.asarray(fr, np.float32), axis=0)

    def picture(path, a, bayer):
        """A JPEG to show beside the live view (the 100x reference)."""
        try:
            import cv2
            g, _ = gray(a, bayer)
            lo, hi = np.percentile(g, (0.5, 99.5))
            v = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            hh, ww = v.shape[:2]
            cv2.drawMarker(v, (ww // 2, hh // 2), (200, 80, 255), cv2.MARKER_CROSS, max(hh, ww) // 12, 2)
            cv2.imwrite(str(path), v, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return str(path)
        except Exception:
            return None

    try:
        # ---- 0. the feature, at 100x
        a = sc.ask_focus("Domain camera test: the feature (100x)",
                         "Centre a small, sharp feature on the cross (a flake corner, a graphite edge, dust: something with "
                         "edges in X and Y), with the stage (joystick, arrow buttons or arrow keys), and focus.  Then press "
                         "\"Use this position\".\n\nAfterwards the program asks you to change to the IR objective, camera and "
                         "filter; it measures where the same feature is with them.", None, button="Use this position", cross=True)
        if a == 2:
            raise sc.fail("cancelled")
        X0, Y0 = c.getXPosition(sc.xy), c.getYPosition(sc.xy)
        sc.X0, sc.Y0, sc.Z0 = X0, Y0, sc.z_read()
        say(f"  0. feature at {f1(X0)}, {f1(Y0)} um, Z {f3(sc.Z0)}; 100x camera {cam100}")

        # ---- 1. 100x reference
        r = sc.autofocus(sc.z_read(), s.af100_first)
        if not r[2]:
            a = sc.ask_focus("Domain camera test: 100x focus", "The 100x autofocus found no focus here: focus by hand, then press "
                             "\"Use this focus\".", None)
            if a != 0:
                raise sc.fail("no 100x focus")
        z100 = sc.z_read()
        raw = avg()
        bayer100 = np.asarray(raw).ndim == 2
        sc.save_image(out / "domaintest_100x.tif", np.asarray(raw).astype(np.float32),
                      {"stage_x_um": X0, "stage_y_um": Y0, "z_um": z100, "camera": cam100, "frames": FRAMES})
        ref, b100 = gray(raw, bayer100)
        mRef = M100 * b100
        ref_pic = picture(out / "domaintest_100x.jpg", raw, bayer100)
        say(f"  1. 100x focus Z {f3(z100)} um ({'autofocus' if r[2] else 'by hand'}); reference frame saved (domaintest_100x.tif)")

        # ---- 2. the IR configuration
        ir_pos = (s.objective_positions or {}).get("domain")
        if sc.ui.confirm("Domain camera test: change to the IR configuration",
                         "Change the setup to the IR camera:\n   - the IR objective" + (f" (nosepiece position {ir_pos})" if ir_pos else "")
                         + "\n   - the IR camera" + (f" ({s.domain_camera}, selected by the program)" if s.domain_camera else "")
                         + "\n   - the IR filter\n\nDo not move the stage or the focus.  Press OK when everything is in place.",
                         ("OK", "Cancel")) != 0:
            raise sc.fail("cancelled")
        sc.ui.mock_hook(sc, "mockDomain")
        if sc.wait_objective("domain", "Put in the IR objective (the window closes by itself when it is in).") is False:
            raise sc.fail("cancelled: not on the IR objective")

        # ---- 3. the IR camera
        cams = sc.setup_domain_camera()
        cam_scan, exp_scan, wh_scan, camD, fsD = cams
        sc.view_scale = 1.0 / 9.54                               # (the live view's stage steps until the IR pixel is known)
        g0 = avg()
        bayerD = False
        for _ in range(8):                                       # saturated: a shorter exposure (cameras whose exposure works)
            sat = float((np.asarray(g0) >= fsD - 1).mean())
            if sat <= 0.01:
                break
            e0 = c.getExposure()
            sc.stream_off()
            c.setExposure(e0 / 2.0)
            say(f"    {f1(sat * 100)} % saturated: exposure {f3(e0)} -> {f3(c.getExposure())} ms")
            g0 = avg()
        lo, med, hi = np.percentile(np.asarray(g0, np.float32), (1, 50, 99))
        say(f"  3. IR camera {camD}: {sc.W} x {sc.H}, full scale {int(fsD)}; median {med:.0f}, 1-99 % {lo:.0f} .. {hi:.0f}")
        if hi - lo < 0.002 * fsD:
            raise sc.fail("the IR picture has no contrast (no light, wrong filter, the camera not delivering): see "
                          "run_flakeid.bat irtest")

        # ---- 4. the IR focus (wide search)
        known = not isnan(s.domain_focus_offset_um)
        zs0 = z100 + (s.domain_focus_offset_um if known else 0.0)
        say(f"  4. IR focus: search around Z {f3(zs0)}" + (f" (100x + domain_focus_offset_um {f3(s.domain_focus_offset_um)})" if known else
                                                            " (the 100x focus)") + f", +-{f1(s.domain_search[0])} um")
        r = sc.autofocus(zs0, s.domain_search)
        how = "auto"
        if not r[2]:
            a = sc.ask_focus("Domain camera test: IR focus", "No focus found with the IR camera: focus by hand (Z buttons, or the "
                             "focus knob of the motorised drive), then press \"Use this focus\".", None)
            if a != 0:
                raise sc.fail("no IR focus")
            how = "by hand"
        zD = sc.z_read()
        say(f"    IR focus Z {f3(zD)} um ({how}); IR - 100x = {f3(zD - z100)} um")

        # ---- 5. pixel calibration: stage moves -> image shifts (b(p) = a(p - s) with s = -Md^-1 d)
        g0, gb = gray(avg(), bayerD)
        nC = min(512, pow2(min(g0.shape)))
        a0 = prep(g0, nC)
        sc.move_to(X0 + 2.0, Y0)
        s1 = xcorr(a0, prep(gray(avg(), bayerD)[0], nC))
        px1 = math.hypot(s1[0], s1[1])
        if s1[2] < MIN_MATCH:
            raise sc.fail(f"the IR picture after a 2 um move does not match the one before (match {f1(s1[2])}): is it in focus, with "
                          "structure (a flake edge)?")
        if px1 < 0.5:
            raise sc.fail(f"a 2 um stage move did not shift the IR picture ({f3(px1)} px): is it in focus, with structure?")
        umpx = 2.0 / px1
        D = CAL_FRACTION * nC * umpx
        say(f"  5. pixel calibration: 2 um move -> {f1(px1)} px (about {f3(umpx / gb)} um per camera px); calibration moves {f1(D)} um")
        sc.move_to(X0 + D, Y0)
        sx = xcorr(a0, prep(gray(avg(), bayerD)[0], nC))
        sc.move_to(X0, Y0 + D)
        sy = xcorr(a0, prep(gray(avg(), bayerD)[0], nC))
        sc.move_to(X0, Y0)
        s0 = xcorr(a0, prep(gray(avg(), bayerD)[0], nC))
        say(f"    +X: shift {f3(sx[0])}, {f3(sx[1])} px (match {f1(sx[2])});  +Y: {f3(sy[0])}, {f3(sy[1])} px (match {f1(sy[2])});  "
            f"back at the start: {f3(s0[0])}, {f3(s0[1])} px (stage repeatability)")
        if sx[2] < MIN_MATCH or sy[2] < MIN_MATCH:
            say("    WARNING: weak image match during the calibration: use a feature with more structure")
        mi = np.array([[-sx[0] / D, -sy[0] / D], [-sx[1] / D, -sy[1] / D]])
        if min(sx[2], sy[2]) < 1.1 or abs(np.linalg.det(mi)) < 1e-12:
            raise sc.fail("the pixel calibration failed (the pictures after the moves did not match): use a feature with more structure")
        mDb = np.linalg.inv(mi)                                  # um per (binned) px
        mD = mDb / gb                                            # um per camera px
        pxD = math.sqrt(abs(np.linalg.det(mD)))
        mirror = np.linalg.det(mD) * np.linalg.det(M_scan) < 0
        rot = math.degrees(math.atan2(mD[1, 0], mD[0, 0]) - math.atan2(M_scan[1, 0], M_scan[0, 0]))
        rot = (rot + 180.0) % 360.0 - 180.0
        skew = abs(math.hypot(mD[0, 0], mD[1, 0]) / math.hypot(mD[0, 1], mD[1, 1]) - 1)
        fovX = abs(mD[0, 0]) * sc.W + abs(mD[0, 1]) * sc.H
        fovY = abs(mD[1, 0]) * sc.W + abs(mD[1, 1]) * sc.H
        say(f"    IR pixel -> stage: [{f3(mD[0, 0])}, {f3(mD[0, 1])}; {f3(mD[1, 0])}, {f3(mD[1, 1])}] um per px: {f3(pxD)} um per px, field "
            f"{f1(fovX)} x {f1(fovY)} um; rotated {f1(rot)} deg against the scan camera" + (", MIRRORED" if mirror else "")
            + (f"; X and Y scales differ by {f1(skew * 100)} % (check the calibration)" if skew > 0.03 else ""))
        say(f"    IR pixels are {f3(math.sqrt(abs(np.linalg.det(M100))) / pxD)} x finer than the 100x pixels")
        sc.M, sc.view_scale = mD, 1.0                            # (the live view's stage steps, in the IR picture's directions)

        # ---- 6. XY offset: the IR frame against the 100x reference on a common stage-aligned grid
        ext100 = min(ref.shape[1] * abs(mRef[0, 0]) + ref.shape[0] * abs(mRef[0, 1]), ref.shape[1] * abs(mRef[1, 0]) + ref.shape[0] * abs(mRef[1, 1]))
        extD = min(fovX, fovY)
        nG = 512
        gG = max(max(math.sqrt(abs(np.linalg.det(mRef))), pxD * gb), min(ext100, extD) / nG)
        say(f"  6. XY offset: matching on a {nG} x {nG} grid, {f3(gG)} um per grid px ({f1(nG * gG)} um across; offsets up to "
            f"+-{f1(nG * gG / 2)} um are found)")
        gRef = prep(to_grid(ref, mRef, nG, gG), nG)
        m = xcorr(gRef, prep(to_grid(gray(avg(), bayerD)[0], mDb, nG, gG), nG))
        offX, offY = m[0] * gG, m[1] * gG
        say(f"    match: IR - 100x = {f3(offX)}, {f3(offY)} um (match {f1(m[2])})")
        if m[2] < MIN_MATCH:
            say(f"    WARNING: weak match ({f1(m[2])} < {MIN_MATCH}): the IR picture may look too different; check by hand below")
        xy_how = "image match"
        sc.move_to(X0 + offX, Y0 + offY)
        ck = xcorr(gRef, prep(to_grid(gray(avg(), bayerD)[0], mDb, nG, gG), nG))
        say(f"    check at {f1(X0 + offX)}, {f1(Y0 + offY)}: residual {f3(ck[0] * gG)}, {f3(ck[1] * gG)} um (match {f1(ck[2])})")
        if math.hypot(ck[0] * gG, ck[1] * gG) > 2.0 and ck[2] >= MIN_MATCH:
            offX, offY = offX + ck[0] * gG, offY + ck[1] * gG
            sc.move_to(X0 + offX, Y0 + offY)
            say(f"    corrected once: offset {f3(offX)}, {f3(offY)} um")
        sc.save_image(out / "domaintest_ir_at_offset.tif", np.asarray(avg(), np.float32),
                      {"stage_x_um": X0 + offX, "stage_y_um": Y0 + offY, "z_um": zD, "camera": camD, "frames": FRAMES})
        a = sc.ask_focus("Domain camera test: XY offset", "The feature that was on the cross at 100x (the picture on the right) should "
                         "now be on the cross of the IR picture.  If it is not, centre it with the stage.\n\n\"Use this position\": the "
                         "position now.  \"Skip\": keep the automatic result.", None, image=ref_pic, button="Use this position", cross=True)
        if a == 2:
            raise sc.fail("cancelled")
        if a == 0:
            hx, hy = c.getXPosition(sc.xy) - X0, c.getYPosition(sc.xy) - Y0
            if math.hypot(hx - offX, hy - offY) > 0.05:
                say(f"    centred by hand: {f3(hx)}, {f3(hy)} um ({f3(hx - offX)}, {f3(hy - offY)} from the automatic result)")
                xy_how = "by hand"
            offX, offY = hx, hy

        # ---- 7. focus offset, sweep, noise, autofocus runs (at the feature, IR configuration)
        r = sc.autofocus(zD, s.domain_af)
        if r[2]:
            zD = r[0]
        say(f"  7. focus offset: IR - 100x = {f3(zD - z100)} um (autofocus at the offset position)")
        roi = s.domain_af[4]
        n = int(round(2 * SWEEP_RANGE_UM / SWEEP_STEP_UM)) + 1
        zs = zD - SWEEP_RANGE_UM + np.arange(n) * SWEEP_STEP_UM
        t0, n0 = time.time(), sc.n_snaps
        sc.zref = zD
        bi, ms = sc.sweep(zs, roi)
        dt = time.time() - t0
        valid = ms[ms >= 0]
        med_ = float(np.median(valid)) if len(valid) else 0.0
        half = med_ + 0.5 * (ms[bi] - med_)
        lo_, hi_ = bi, bi
        while lo_ > 0 and ms[lo_ - 1] >= half:
            lo_ -= 1
        while hi_ < n - 1 and ms[hi_ + 1] >= half:
            hi_ += 1
        fwhm = (hi_ - lo_ + 1) * SWEEP_STEP_UM
        ratio = float(ms[bi] / max(med_, 1e-12))
        with open(out / "domaintest_sweep.csv", "w", newline="") as fh:
            fh.write("z_um,sharpness\r\n")
            for z, v in zip(zs, ms):
                fh.write(f"{z},{v}\r\n")
        say(f"    sweep +-{f1(SWEEP_RANGE_UM)} um in {SWEEP_STEP_UM} um steps: sharpest at {f3(zs[bi] - zD)} um from the focus, peak/median "
            f"{f3(ratio)}, peak width (FWHM) {f1(fwhm)} um; {f3(dt / max(sc.n_snaps - n0, 1))} s per frame")
        try:
            import cv2
            pw, ph = 700, 380
            im = np.full((ph, pw, 3), 255, np.uint8)
            v = ms.copy()
            ok = v >= 0
            mn, mx = float(v[ok].min()), float(v[ok].max())
            pts = [(50 + int(i * (pw - 70.0) / (n - 1)), ph - 50 - int((v[i] - mn) / max(mx - mn, 1e-12) * (ph - 80))) for i in range(n) if ok[i]]
            cv2.rectangle(im, (50, 30), (pw - 20, ph - 50), (200, 200, 200), 1)
            cv2.polylines(im, [np.array(pts, np.int32)], False, (200, 60, 0), 2, cv2.LINE_AA)
            cv2.putText(im, f"IR camera: sharpness vs Z   peak/median {ratio:.3f}, FWHM {fwhm:.1f} um", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(im, f"{zs[0]:.1f}", (40, ph - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(im, f"{zs[-1]:.1f}", (pw - 70, ph - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(im, "Z (um)", (pw // 2 - 20, ph - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(out / "domaintest_sweep.png"), im)
        except Exception as e:
            say(f"    (plot not drawn: {e})")
        zBest = float(zs[bi])
        sc.set_z(zBest - s.af_backlash_um); sc.set_z(zBest)
        sc.save_image(out / "domaintest_ir_best.tif", np.asarray(avg(), np.float32), {"z_um": zBest, "camera": camD, "frames": FRAMES})
        rep = np.array([sc.sharpness(roi) for _ in range(REPEATS)])
        cv = float(rep.std() / max(rep.mean(), 1e-12))
        say(f"    sharpness noise at the peak: {f3(cv * 100)} % ({REPEATS} frames)")
        res = []
        for off in TEST_OFFSETS_UM:
            for nm, pp in (("domain_af", s.domain_af), ("domain_af_first", s.domain_af_first)):
                zst = zBest + off
                sc.set_z(zst)
                n0, t1 = sc.n_snaps, time.time()
                r = sc.autofocus(zst, pp)
                say(f"    autofocus ({nm}) from {off:+.1f} um: " + (f"{f3(r[0] - zBest)} um from the sharpest" if r[2] else "FAILED")
                    + f"; {sc.n_snaps - n0} frames, {time.time() - t1:.1f} s")
                res.append((r[0] - zBest, bool(r[2])))
        sc.set_z(zBest - s.af_backlash_um); sc.set_z(zBest)

        # ---- 8. results
        say("  --- result")
        nf = sum(1 for e in res if not e[1])
        worst = max([abs(e[0]) for e in res if e[1]] or [0.0])
        say(f"  * autofocus: {len(res) - nf} of {len(res)} runs passed, worst {f3(worst)} um from the sharpest Z"
            + ("  <-- larger than a quarter of the peak width: a smaller fine step" if worst > max(0.3, fwhm / 4) else ""))
        if ratio < 1.3:
            say(f"  * the sharpness peak is weak (peak/median {f3(ratio)}): more light, or a feature with more edges")
        if cv > 0.05:
            say(f"  * the sharpness is noisy ({f3(cv * 100)} %): more light, or average frames")
        if fwhm > 0 and s.domain_af[1] > fwhm:
            say(f"  * domain_af's coarse step ({s.domain_af[1]} um) is wider than the peak ({f1(fwhm)} um): a smaller coarse step")
        if fwhm < 2.5 * s.domain_af[3]:
            say(f"  * the peak is narrow compared with the fine step ({s.domain_af[3]} um): a smaller fine step")
        zoff = zD - z100
        say(f"  domain_xy_offset_um = [{f3(offX)}, {f3(offY)}]   ({xy_how})")
        say(f"  domain_focus_offset_um = {f3(zoff)}")
        say(f"  IR camera pixel -> stage (um per px, for the analysis): [[{f3(mD[0, 0])}, {f3(mD[0, 1])}], [{f3(mD[1, 0])}, {f3(mD[1, 1])}]]")
        doc = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), camera=camD, width=sc.W, height=sc.H, um_per_px_matrix=mD.ravel().tolist(),
                   um_per_px=pxD, mirrored=bool(mirror), rotation_vs_scan_camera_deg=rot, xy_offset_um=[offX, offY], xy_offset_method=xy_how,
                   focus_offset_um=zoff, z100_um=z100, z_domain_um=zD, stage_100x_um=[X0, Y0], sweep_fwhm_um=fwhm, sweep_peak_ratio=ratio,
                   sharpness_noise=cv, af_runs_passed=len(res) - nf, af_runs=len(res), af_worst_um=worst)
        (out / "domaintest_domain.json").write_text(json.dumps(doc, indent=1))
        save_offsets(sc, offX, offY, zoff, out, mD)
        say(f"done ({sc.n_snaps} frames): {out}")
        say(f"  the stage is at the feature with the IR configuration in place (Z {f3(zBest)}); change back to the scan configuration "
            "before the next run")
    finally:
        sc.M, sc.view_scale, sc.zref = M_scan, vs_scan, float("nan")
        if cams is not None:
            sc.restore_scan_camera(cams[0], cams[1], cams[2], cams[3])


def save_offsets(sc, offX, offY, zoff, out, mD=None):
    """domain_xy_offset_um and domain_focus_offset_um (and domain_px_um_matrix, the IR pixel -> stage matrix the IR
    mosaics plan with) into the settings file, if the user agrees (a dated backup is kept)."""
    s = sc.s
    sf = Path(s.settings_file) if s.settings_file else None
    if sf is None or not sf.is_file():
        sc.say("  no settings file to update: put these values into your scan settings by hand")
        return
    if sc.ui.confirm("Domain camera test: save", f"Save the IR camera's offsets into {sf.name}?\n\n   domain_xy_offset_um = [{offX:.3f}, {offY:.3f}]\n"
                     f"   domain_focus_offset_um = {zoff:.3f}\n" + ("   domain_px_um_matrix (IR pixel -> stage)\n" if mD is not None else "") + "\n(The scan's domain imaging uses them; a copy of the old file is kept.)",
                     ("Save", "Do not save")) != 0:
        sc.say("  not saved into the settings (the values are in the log and domaintest_domain.json)")
        return
    try:
        cfg = json.loads(sf.read_text(encoding="utf-8"))
        bak = sf.with_name(sf.stem + datetime.datetime.now().strftime("_before_domaintest_%Y%m%d_%H%M") + sf.suffix)
        bak.write_text(sf.read_text(encoding="utf-8"), encoding="utf-8")
        cfg["domain_xy_offset_um"] = [round(offX, 3), round(offY, 3)]
        cfg["domain_focus_offset_um"] = round(zoff, 3)
        if mD is not None:
            cfg["domain_px_um_matrix"] = [[round(float(v), 5) for v in row] for row in np.asarray(mD, float)]
        tmp = sf.with_name(sf.name + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        tmp.replace(sf)
        sc.say(f"  saved into {sf} (old version: {bak.name})")
    except Exception as e:
        sc.say(f"  WARNING: the settings were not saved ({type(e).__name__}: {e}); put the values in by hand")
