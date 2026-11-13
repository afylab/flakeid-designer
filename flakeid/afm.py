"""AFM step heights: read Bruker NanoScope .spm files, mask bad data, level, and measure terrace steps.

    from flakeid.afm import read_spm, analyse_height
    img = read_spm("bigbn2.0_00002.spm"); res = analyse_height(img["Height Sensor"], img.pixel_nm)

Pipeline (analyse_height):
  1. bad data: clipped values; spikes and tip-loss streaks = pixels that differ from the same columns in the
     rows just above and below by much more than the noise (real edges are continuous from line to line);
     whole lines that are mostly bad; grown by a margin.
  2. levelling, repeated:
       - height levels = peaks of the histogram of the current flattened image (flat terraces and the substrate);
         each pixel within the level's band belongs to it; eroded so edges and slopes are left out;
       - one fit to all level pixels: z = plane (tilt, optionally curvature) + one offset per level, so the
         tilt comes from every flat area, not only the substrate;
       - line offsets (the scanner's line-to-line jumps) from the level pixels of each line: the median of
         z - fit along the line, so a line crossing a step is not distorted.
  3. steps: each level's offset above the reference level (the lowest one, normally SiO2).  Checked line by
     line: on every scan line with enough pixels of both levels, the difference of their medians; the step is
     the median over lines and its uncertainty 1.25 x MAD / sqrt(lines).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# NanoScope reader
# ---------------------------------------------------------------------------

@dataclass
class SPM:
    channels: dict                    # name -> 2D array (physical units; rows top -> bottom as displayed)
    units: dict
    scan_um: float
    pixel_nm: float
    header: dict = field(default_factory=dict)
    path: str = ""

    def __getitem__(self, k):
        return self.channels[k]


def _sections(h: str):
    out = []
    for sec in re.split(r"\\\*", h):
        title, _, body = sec.partition("\n")
        kv = {}
        for line in body.splitlines():
            m = re.match(r"\\(@?(?:\d:)?[^:]+):\s*(.*)", line.strip())
            if m:
                kv[m.group(1).strip()] = m.group(2).strip()
        out.append((title.strip(), kv))
    return out


def read_spm(path) -> SPM:
    """Bruker NanoScope (v5-v9) image file.  Integer data are scaled to the full-range value of the
    '@2:Z scale' entry: NanoScope 9 stretches each image over the whole integer range (one pixel at the
    maximum), and that value (V) times the sensitivity (nm/V) is the physical range."""
    b = Path(path).read_bytes()
    end = b.find(b"\\*File list end")
    h = b[:end].decode("latin-1")
    secs = _sections(h)
    glob = {}
    for title, kv in secs:
        if not title.startswith("Ciao image list"):
            glob.update(kv)
    chans, units = {}, {}
    scan_um = None
    for title, kv in secs:
        if not title.startswith("Ciao image list"):
            continue
        name = re.search(r'"(.*)"', kv.get("@2:Image Data", '"?"')).group(1)
        n = int(kv["Samps/line"])
        lines = int(kv.get("Number of lines", kv.get("Lines")))
        off, ln = int(kv["Data offset"]), int(kv["Data length"])
        bpp = ln // (n * lines)
        raw = np.frombuffer(b, dtype={2: "<i2", 4: "<i4"}[bpp], count=n * lines, offset=off).reshape(lines, n).astype(np.float64)
        m = re.match(r"V \[(.*?)\] \(([-0-9.eE]+) [^)]*\) ([-0-9.eE]+)", kv["@2:Z scale"])
        sens_key, full = m.group(1), float(m.group(3))
        sens_s = glob.get("@" + sens_key, "")
        sm = re.match(r"V ([-0-9.eE]+) (\S+)", sens_s)
        sens, unit = (float(sm.group(1)), sm.group(2).split("/")[0]) if sm else (1.0, kv["@2:Z scale"].split()[-1])
        z = raw / 2.0 ** (8 * bpp) * full * sens
        z[raw >= 2 ** (8 * bpp - 1) - 1] = np.nan          # the clipped maximum
        z[raw <= -2 ** (8 * bpp - 1)] = np.nan
        chans[name] = np.flipud(z)                          # NanoScope stores the bottom line first
        units[name] = unit
        if scan_um is None:
            sz = kv.get("Scan Size", glob.get("Scan Size", "0 0 nm")).split()
            val, u = float(sz[0]), sz[-1]
            scan_um = val / 1000.0 if u == "nm" else val    # '~m' = um
    return SPM(channels=chans, units=units, scan_um=scan_um, pixel_nm=scan_um * 1000.0 / n, header=glob, path=str(path))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _robust_sigma(v):
    v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else np.nan


def bad_mask(z, k=6.0, floor_nm=2.0, grow=2):
    """Clipped / spike / streak pixels (True = bad).

    A spike or streak (the tip jumping off or dragging along a line) sticks out on the same side relative to
    both the rows above and the rows below.  An edge running along the scan lines is high on one side and low
    on the other, so it is kept."""
    bad = ~np.isfinite(z)
    zf = np.where(bad, np.nanmedian(z), z)
    zl = zf - np.median(zf, axis=1, keepdims=True)           # line offsets are not defects
    up = np.median(np.stack([np.roll(zl, s, axis=0) for s in (1, 2)]), axis=0)      # rows above
    dn = np.median(np.stack([np.roll(zl, -s, axis=0) for s in (1, 2)]), axis=0)     # rows below
    du, dd = zl - up, zl - dn
    s = _robust_sigma(np.r_[du[2:-2].ravel(), dd[2:-2].ravel()])
    T = max(k * s, floor_nm)
    spike = (np.sign(du) == np.sign(dd)) & (np.minimum(np.abs(du), np.abs(dd)) > T)
    spike[:2] = spike[-2:] = False
    # within a line too: a narrow excursion against both horizontal neighbours (particles, single bad pixels)
    l = np.median(np.stack([np.roll(zl, s, axis=1) for s in (1, 2)]), axis=0)
    r = np.median(np.stack([np.roll(zl, -s, axis=1) for s in (1, 2)]), axis=0)
    dl, dr = zl - l, zl - r
    spike |= (np.sign(dl) == np.sign(dr)) & (np.minimum(np.abs(dl), np.abs(dr)) > T)
    bad |= spike
    # streaks span runs of a line: close the gaps along the line, and drop lines that are mostly bad
    bad = ndi.binary_closing(bad, structure=np.ones((1, 5), bool)) | bad
    frac = bad.mean(axis=1)
    bad[frac > 0.5, :] = True
    if grow:
        bad = ndi.binary_dilation(bad, iterations=grow)
    return bad


def _regions(z, good, line_off, min_frac=0.02):
    """Flat regions = connected areas between edges.  Edges: within-line slope far above the typical slope (a step
    of even 1 nm over a pixel or two stands out against tilt and noise), and the same between lines after the
    line offsets are removed.  Independent of the overall tilt, which shifts all slopes equally."""
    H, W = z.shape
    zl = z - line_off[:, None]
    zf = np.where(good, zl, np.nan)
    gx = np.diff(zf, axis=1)
    gy = np.diff(zf, axis=0)
    gx -= np.nanmedian(gx)                                   # tilt
    gy -= np.nanmedian(gy)
    sx, sy = _robust_sigma(gx), _robust_sigma(gy)
    ex = np.zeros((H, W), bool)
    ex[:, 1:] |= np.abs(np.nan_to_num(gx)) > max(6 * sx, 0.6)
    ex[:, :-1] |= np.abs(np.nan_to_num(gx)) > max(6 * sx, 0.6)
    ey = np.zeros((H, W), bool)
    ey[1:, :] |= np.abs(np.nan_to_num(gy)) > max(8 * sy, 1.0)
    edge = ndi.binary_dilation(ex | ey, iterations=2)
    flat = good & ~edge
    lab, n = ndi.label(flat)
    sizes = ndi.sum(flat, lab, np.arange(1, n + 1))
    keep = [k + 1 for k in np.argsort(sizes)[::-1] if sizes[k] >= min_frac * H * W]
    return [ndi.binary_erosion(lab == k, iterations=1) for k in keep], edge


def _joint(z, regions, order, smooth=0.05):
    """One least-squares fit to every flat-region pixel:
         z[i, j] = line offset[i] + background along the line (u, and u^2, u v for order 2) + offset[region]
    plus a weak penalty smooth * W * (line offset[i+1] - line offset[i])^2.  Lines that contain two regions fix
    the step between them from their own data; where no line contains both, the penalty (no jump unless the
    data need one) links them.  Returns (line offsets, background image, region offsets with region 0 at 0)."""
    from scipy.sparse import coo_matrix, vstack as svstack
    from scipy.sparse.linalg import lsqr
    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W]
    u, v = (xx - W / 2) / W, (yy - H / 2) / H
    bas = [u] + ([u * u, u * v] if order >= 2 else [])
    K = len(regions)
    nb = len(bas)
    rows, cols, vals, rhs = [], [], [], []
    eq = 0
    for k, m in enumerate(regions):
        ii, jj = np.nonzero(m)
        n = ii.size
        e = np.arange(eq, eq + n)
        rows += [e]; cols += [ii]; vals += [np.ones(n)]
        for q, b in enumerate(bas):
            rows += [e]; cols += [np.full(n, H + q)]; vals += [b[ii, jj]]
        if k > 0:
            rows += [e]; cols += [np.full(n, H + nb + k - 1)]; vals += [np.ones(n)]
        rhs.append(z[ii, jj])
        eq += n
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(eq, H + nb + K - 1))
    lam = np.sqrt(smooth * W)
    i = np.arange(H - 1)
    P = coo_matrix((np.r_[-np.ones(H - 1), np.ones(H - 1)] * lam, (np.r_[i, i], np.r_[i, i + 1])), shape=(H - 1, H + nb + K - 1))
    sol = lsqr(svstack([A, P]).tocsr(), np.r_[np.concatenate(rhs), np.zeros(H - 1)], atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    lo = sol[:H]
    bg = sum(c * b for c, b in zip(sol[H:H + nb], bas))
    offs = np.r_[0.0, sol[H + nb:]]
    return lo, bg, offs


def _fit(z, regions, basis, line_off):
    """z - line offsets = plane(basis) + offset per region (least squares).  Returns (plane, offsets)."""
    zr = z - line_off[:, None]
    rows, rhs = [], []
    for k, m in enumerate(regions):
        one = np.zeros((m.sum(), len(regions)))
        one[:, k] = 1
        rows.append(np.hstack([np.stack([b[m] for b in basis], 1), one]))
        rhs.append(zr[m])
    sol, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    return sum(ci * b for ci, b in zip(sol[:len(basis)], basis)), sol[len(basis):]


def analyse_height(z, pixel_nm, order=1, iterations=4, min_frac=0.02, merge_nm=None, bad=None):
    """Mask, level and find the step heights of a height image z (nm).  Returns a dict (see module doc)."""
    H, W = z.shape
    if bad is None:
        bad = bad_mask(z)
    good = ~bad
    yy, xx = np.mgrid[0:H, 0:W]
    u, v = (xx - W / 2) / W, (yy - H / 2) / H
    basis = [u, v] + ([u * u, u * v, v * v] if order >= 2 else [])
    # line offsets start at zero: the closed-loop Z sensor has little line-to-line drift, and a per-line median
    # would flatten away a step that runs along the scan lines (no substrate on those lines to compare with)
    line_off = np.zeros(H)
    regions, edge = [], np.zeros_like(good)
    plane = np.zeros_like(z)
    offs = np.zeros(0)
    # first pass: a plane through all good pixels, so that the edge finder sees the steps
    A0 = np.stack([b[good] for b in basis] + [np.ones(good.sum())], 1)
    c0, *_ = np.linalg.lstsq(A0, z[good], rcond=None)
    plane = sum(ci * b for ci, b in zip(c0[:-1], basis))
    for it in range(iterations):
        regions, edge = _regions(z - plane, good, line_off, min_frac)
        if not regions:
            break
        line_off, plane, offs = _joint(z, regions, order)
    if not regions:
        return dict(z=np.where(good, z, np.nan), bad=bad, levels=[], level_masks=[], regions=[], noise_nm=np.nan,
                    bad_frac=float(bad.mean()), order=order, edge=edge)
    zc = z - line_off[:, None] - plane
    zc[bad] = np.nan
    # blocks: bands of rows with no usable data (the tip jumped or lost contact across the whole width) can hide
    # an offset, so heights are only compared within a block
    rowbad = bad.mean(axis=1) > 0.9
    blk = np.cumsum(np.r_[0, (rowbad[1:] & ~rowbad[:-1]).astype(int)])
    blk[rowbad] = -1
    region_block = [int(np.bincount(blk[np.nonzero(m)[0]][blk[np.nonzero(m)[0]] >= 0]).argmax()) for m in regions]
    noise = float(np.nanmedian([_robust_sigma(zc[m] - np.median(zc[m])) for m in regions]))
    tol = merge_nm if merge_nm is not None else max(3 * noise, 0.5)
    hts = [float(np.median(zc[m])) for m in regions]
    groups = []                                              # (block, [regions]) at one height
    for b in sorted(set(region_block)):
        ks = sorted((k for k in range(len(regions)) if region_block[k] == b), key=lambda k: hts[k])
        for k in ks:
            if groups and groups[-1][0] == b and hts[k] - hts[groups[-1][1][-1]] < tol:
                groups[-1][1].append(k)
            else:
                groups.append((b, [k]))
    level_masks = [np.any([regions[k] for k in g], axis=0) for _, g in groups]
    blocks = sorted(set(region_block))
    area_b = {b: sum(level_masks[i].sum() for i, (bb, _) in enumerate(groups) if bb == b) for b in blocks}
    main = max(blocks, key=lambda b: area_b[b])
    refs = {b: next(i for i, (bb, _) in enumerate(groups) if bb == b) for b in blocks}      # lowest level per block
    zc = zc - np.median(zc[level_masks[refs[main]]])         # the main block's lowest level at 0
    out_levels = []
    for li, m in enumerate(level_masks):
        b = groups[li][0]
        ref = level_masks[refs[b]]
        vals = zc[m]
        rec = dict(index=li, block=b, main_block=b == main, height_nm=float(np.median(vals) - np.median(zc[ref])),
                   rms_nm=float(_robust_sigma(vals - np.median(vals))),
                   area_frac=float(m.sum() / (H * W)), n_px=int(m.sum()), n_regions=len(groups[li][1]))
        if li != refs[b]:
            per = []
            for r in range(H):
                a, bb = ref[r], m[r]
                if a.sum() >= 8 and bb.sum() >= 8:
                    per.append(np.median(zc[r][bb]) - np.median(zc[r][a]))
            per = np.array(per)
            rec["step_nm"] = rec["height_nm"]                 # above the lowest level of its block
            if per.size >= 3:
                rec["step_lines_nm"] = float(np.median(per))
                rec["step_err_nm"] = float(1.2533 * 1.4826 * np.median(np.abs(per - np.median(per))) / np.sqrt(per.size))
                rec["n_lines"] = int(per.size)
                rec["per_line"] = per
        out_levels.append(rec)
    return dict(z=zc, bad=bad, levels=out_levels, level_masks=level_masks, regions=regions, noise_nm=noise,
                bad_frac=float(bad.mean()), order=order, edge=edge, line_off=line_off, plane=plane, n_blocks=len(blocks),
                row_block=blk, main_block=main)


def junk_mask(r, far_nm=20.0, k=10.0):
    """Second pass on a levelled result: pixels far from every flat level (not at an edge) are bad data that the
    first mask missed (stretches of a line where the tip lost the surface); rows that are mostly such pixels go."""
    z = r["z"]
    if not r["levels"]:
        return r["bad"]
    hts = np.array([np.median(z[m]) for m in r["level_masks"]])
    d = np.min(np.abs(z[..., None] - hts[None, None, :]), axis=2)
    far = np.isfinite(z) & (d > max(far_nm, k * r["noise_nm"])) & ~r["edge"]
    bad = r["bad"] | far
    bad[far.mean(axis=1) > 0.5, :] = True                     # rows mostly off the surface (not just debris)
    return ndi.binary_dilation(bad, iterations=1)


def analyse_auto(z, pixel_nm, **kw):
    """analyse_height with a plane and with a quadratic background (scanner bow), after a second bad-data pass.
    The quadratic is used when it lowers the residual noise of the flat areas by more than 5 % (two more
    parameters cannot over-fit tens of thousands of pixels)."""
    res = []
    for order in (1, 2):
        r = analyse_height(z, pixel_nm, order=order, **kw)
        if r["levels"]:
            bad2 = junk_mask(r)
            if bad2.sum() > r["bad"].sum() * 1.02:
                r = analyse_height(z, pixel_nm, order=order, bad=bad2, **kw)
        res.append(r)
    r1, r2 = res
    if not r1["levels"]:
        return r2
    if not r2["levels"]:
        return r1
    r = r2 if r2["noise_nm"] < 0.95 * r1["noise_nm"] else r1
    r["alt"] = r1 if r is r2 else r2
    return r
