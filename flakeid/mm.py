"""Micro-Manager acquisitions: tile reader, flat-field, stitching and stage map.

A Micro-Manager Multi-D acquisition with a tile grid is saved as

    <dataset>/Pos-2-CCC_RRR/img_channel000_positionNNN_time000000000_z000.tif
    <dataset>/Pos-2-CCC_RRR/metadata.txt        (JSON; stage X/Y/Z, camera properties)

(or a single ``Default/`` folder for a snap).  This module

* reads the frames as linear RGB: RAW16 (XIMEA XC10 Bayer mosaic, R at the
  top-left pixel, 12-bit) is demosaiced here; RGB32 is accepted but is only
  linear when the camera gammas are 1;
* applies a dark frame and a flat field built from frames of bare substrate
  (``build_flatfield``);
* registers neighbouring tiles, fits the pixel -> stage matrix (which also
  absorbs the camera rotation / image flip) and places every tile in a
  global mosaic by least squares, so stage step errors do not create seams;
* composes a seam-free mosaic in which every pixel comes from the tile whose
  centre is nearest (no blending, hence no doubled features);
* maps any mosaic pixel back to the stage position that puts it in the centre
  of the field of view, using the tile it came from (``StageMap``), which
  keeps systematic stage errors local.

    python -m flakeid.mm flat  tiles_blank_1 -o flat_6.3x.npz [--dark dark_1]
    python -m flakeid.mm stitch tiles_v2_1 --flat flat_6.3x.npz
    python find_flakes.py tiles_v2_1 hbn 285 --flat flat_6.3x.npz --viewer
    python -m flakeid.mm offset offset_10x offset_100x --stagemap tiles_v2_1_stagemap.json --flat flat_6.3x.npz
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

RAW_FULL_SCALE = 4095          # 12-bit device output
# OpenCV names the Bayer pattern by the 2x2 block starting at pixel (1, 1);
# BG = R at (0, 0), as determined for the XC10 against its own RGB32 output.
BAYER_CODE = cv2.COLOR_BayerBG2RGB
# pixel -> stage matrix (um/px) of the BX51 10x + 0.63x adapter + XC10, from tile registration;
# used to predict tile overlaps (the fit on each scan replaces it)
DEFAULT_M = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class Tile:
    path: Path
    label: str
    stage_xy: np.ndarray            # (2,) um, stage position when acquired
    z_um: float | None              # as recorded: real um for script-acquired tiles, drive units for Multi-D
    grid: tuple | None              # (col, row) if known
    props: dict                     # camera / device properties
    z_units: str = "drive"          # "um" (script-acquired tiles) or "drive" (Micro-Manager Multi-D)

    @property
    def data_format(self) -> str:
        return next((v for k, v in self.props.items() if k.endswith("-Data format")), "")


def _frame_metadata(md: dict, tif_name: str) -> dict:
    for k, v in md.items():
        if k.startswith("Metadata") and k.endswith(tif_name):
            return v
    for k, v in md.items():                     # older layouts: keyed by file name only
        if k.endswith(tif_name) and isinstance(v, dict) and "XPositionUm" in v:
            return v
    raise KeyError(f"no metadata for {tif_name}")


def is_dataset(path) -> bool:
    path = Path(path)
    return path.is_dir() and any(path.glob("*/metadata.txt"))


def read_tile(mdf) -> Tile | None:
    """The first frame of one position folder (<pos>/metadata.txt + TIFF), or None if it has no TIFF."""
    mdf = Path(mdf)
    md = json.loads(mdf.read_text(encoding="utf-8", errors="replace"))
    tifs = sorted(mdf.parent.glob("img_channel000_*_time000000000_z000.tif")) or sorted(mdf.parent.glob("*.tif"))
    if not tifs:
        return None
    fm = _frame_metadata(md, tifs[0].name)
    m = re.search(r"(\d+)_(\d+)$", mdf.parent.name)
    grid = (int(m.group(1)), int(m.group(2))) if m else None
    z = fm.get("ZPositionUm")
    # tiles written by mm_chip_scan.bsh record Z in real um (Summary has Z_SCALE); Micro-Manager's
    # own acquisitions record the focus-drive reading
    real = isinstance(md.get("Summary"), dict) and "Z_SCALE" in md["Summary"]
    return Tile(path=tifs[0], label=mdf.parent.name,
                stage_xy=np.array([float(fm["XPositionUm"]), float(fm["YPositionUm"])]),
                z_um=None if z is None else float(z), grid=grid,
                props={k: v for k, v in fm.items() if isinstance(v, (str, int, float))},
                z_units="um" if real else "drive")


def read_dataset(path) -> list:
    """All frames of an acquisition folder, as Tile records (first z/time/channel only)."""
    path = Path(path)
    tiles = [t for t in (read_tile(mdf) for mdf in sorted(path.glob("*/metadata.txt"))) if t is not None]
    if not tiles:
        raise FileNotFoundError(f"no Micro-Manager frames under {path}")
    return tiles


READ_THREADS = 8      # tiles read / demosaiced in parallel (file reading and OpenCV release the GIL)


def scan_params(folder) -> dict | None:
    """Material and oxide entered at the microscope for a tile folder (mm_chip_scan.bsh): <tiles>/scan_params.json,
    else the piece's <name>_chip.json next to the folder.  {material, oxide_nm, ...} or None."""
    folder = Path(folder)
    cands = [folder / "scan_params.json",
             folder.parent / (re.sub(r"_tiles(_\d+)?$", "", folder.name) + "_chip.json")]
    for c in cands:
        try:
            d = json.loads(c.read_text())
        except (OSError, ValueError):
            continue
        if d.get("material") and d.get("oxide_nm"):
            m = str(d["material"]).lower()
            return dict(d, material="hbn" if m in ("bn", "hbn") else m, oxide_nm=float(d["oxide_nm"]), source=str(c))
    return None


def resolve_material(folder, material, oxide):
    """(material, oxide_nm) for an analysis: "auto" (or None) takes the value stored by the microscope script."""
    sp = scan_params(folder) if (material in (None, "auto") or oxide in (None, "auto")) else None
    if material in (None, "auto"):
        if not sp:
            raise ValueError(f"no material given and no scan_params.json / _chip.json with one for {folder}: "
                             f"give MATERIAL and OXIDE_NM on the command line")
        material = sp["material"]
    if oxide in (None, "auto"):
        if not sp:
            raise ValueError(f"no oxide thickness given and none stored for {folder}: give OXIDE_NM on the command line")
        oxide = sp["oxide_nm"]
    return material, float(oxide)


def read_frame(tile: Tile, shift=None) -> np.ndarray:
    """Linear float32 RGB frame (raw counts, dark not subtracted).  shift (dy, dx): a raw frame the camera
    delivered one pixel off (its colours come out swapped) is moved back before demosaicing."""
    a = cv2.imread(str(tile.path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"could not read {tile.path}")
    if a.ndim == 2:
        tile.glitch = raw_glitch(a)                  # camera garbage frame? (checked on the raw mosaic)
    if a.ndim == 2 and shift:
        a = np.roll(a, shift, axis=(0, 1))
    if a.ndim == 2:
        fmt = tile.data_format.upper()
        if fmt.startswith("MONO"):
            raise ValueError(f"{tile.path.name}: {fmt} has no colour; acquire RAW16 (or RGB32)")
        return cv2.cvtColor(a, BAYER_CODE).astype(np.float32)
    return cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)


def full_scale(tile: Tile) -> float:
    if tile.data_format.upper().startswith("RGB32"):
        return 255.0
    bits = next((v for k, v in tile.props.items() if k.endswith("-Image data bit depth")), None)
    return float(2 ** int(bits) - 1) if bits else float(RAW_FULL_SCALE)


def check_linear(tile: Tile) -> list:
    """Warnings for camera settings that break the linear contrast model."""
    warn = []
    for key, want in (("Luminosity gamma", 1.0), ("Chromaticity gamma", 1.0), ("Gain", 0.0)):
        v = next((v for k, v in tile.props.items() if k.endswith("-" + key)), None)
        if v is not None and abs(float(v) - want) > 1e-3 and tile.data_format.upper().startswith("RGB32"):
            warn.append(f"{key} = {v} (expected {want:g} for a linear response)")
    if next((v for k, v in tile.props.items() if k.endswith("-Auto white balance")), "Off") != "Off":
        warn.append("auto white balance is on; colours change between frames")
    return warn


# ---------------------------------------------------------------------------
# Dark / flat field
# ---------------------------------------------------------------------------

@dataclass
class FlatField:
    flat: np.ndarray                # (H, W, 3) float32, 1.0 at the brightest (centre) area
    dark: np.ndarray                # (3,) or (H, W, 3) counts
    info: dict = field(default_factory=dict)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return (frame - self.dark) / self.flat

    def save(self, path):
        np.savez_compressed(path, flat=self.flat.astype(np.float16), dark=np.asarray(self.dark, np.float32),
                            info=json.dumps(self.info))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(flat=z["flat"].astype(np.float32), dark=z["dark"].astype(np.float32),
                   info=json.loads(str(z["info"])))


def _median_frame(tiles) -> np.ndarray:
    return np.median(np.stack([read_frame(t) for t in tiles]), axis=0).astype(np.float32)


def build_flatfield(blank, dark=None, smooth_px: float = 2.0, mask_tol: float = 0.04) -> FlatField:
    """Flat field from frames of bare substrate at different positions.

    The per-pixel median over the frames removes dust and particles (which move
    with the sample); dirt on the optics or sensor (which does not move) stays
    in the flat and is corrected.  ``dark``: folder of frames with the light
    blocked at the same exposure (default: no offset).
    """
    tiles = read_dataset(blank)
    if len(tiles) < 5:
        print(f"warning: only {len(tiles)} blank frames; >= 9 at different positions recommended")
    if dark is not None:
        dk = _median_frame(read_dataset(dark))
        dark_v = cv2.GaussianBlur(dk, (0, 0), 5)
    else:
        dark_v = np.zeros(3, np.float32)
    frames = np.stack([read_frame(t) - dark_v for t in tiles]).astype(np.float32)
    med = np.median(frames, axis=0)
    # second pass: in each frame, mask what differs from the first median by more than mask_tol after
    # matching the brightness (flakes, residue, specks on the sample: they move between frames, while dirt
    # on the optics does not), then take the median of the rest.  Few blank areas are perfectly clean.
    if len(tiles) >= 5 and mask_tol > 0:
        base = cv2.GaussianBlur(med, (0, 0), 1.5)
        for k in range(len(frames)):
            fb = cv2.GaussianBlur(frames[k], (0, 0), 1.5)
            q = fb / np.maximum(base, 1.0)
            q /= np.median(q.reshape(-1, 3), axis=0)
            bad = (np.abs(q - 1.0).max(axis=2) > mask_tol).astype(np.uint8)
            bad = cv2.dilate(bad, np.ones((7, 7), np.uint8)).astype(bool)
            frames[k][bad] = np.nan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # pixels masked in every frame: filled below
            med2 = np.nanmedian(frames, axis=0)
        n_ok = np.isfinite(frames[..., 1]).sum(axis=0)
        thin = (n_ok < max(3, len(tiles) // 4)) | ~np.isfinite(med2).all(axis=2)
        med = np.where(thin[..., None], med, med2)
    del frames
    if smooth_px > 0:
        med = cv2.GaussianBlur(med, (0, 0), smooth_px)
    ref = np.percentile(cv2.GaussianBlur(med, (0, 0), 25).reshape(-1, 3), 99.5, axis=0)
    flat = np.maximum(med / ref, 0.05).astype(np.float32)
    # consistency: each frame / median, smooth, should be ~1 everywhere
    spread = []
    for t in tiles:
        q = cv2.GaussianBlur(read_frame(t) - dark_v, (0, 0), 15) / np.maximum(cv2.GaussianBlur(med, (0, 0), 15), 1)
        spread.append(float(np.percentile(q[..., 1], 99) - np.percentile(q[..., 1], 1)))
    info = dict(source=str(blank), n_frames=len(tiles), dark=str(dark) if dark else None,
                exposure_ms=tiles[0].props.get("Exposure-ms"), data_format=tiles[0].data_format,
                bit_depth=int(round(np.log2(full_scale(tiles[0]) + 1))),
                substrate_counts=[round(float(v), 1) for v in ref],
                flat_min=[round(float(v), 3) for v in flat.reshape(-1, 3).min(0)],
                frame_spread=round(float(np.median(spread)), 4))
    return FlatField(flat=flat, dark=dark_v, info=info)


# ---------------------------------------------------------------------------
# Registration and stage map
# ---------------------------------------------------------------------------

def _prep(frame: np.ndarray) -> np.ndarray:
    """Band-passed, normalised grey frame for registration (flake edges, not noise or shading)."""
    g = frame.mean(axis=2)
    g = g / np.maximum(cv2.GaussianBlur(g, (0, 0), 40), 1.0)
    g = cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 8)
    return np.clip(g, -0.3, 0.3).astype(np.float32)


def _overlap_ncc(a, b, dx, dy):
    """NCC of a and b where a feature at p in `a` is at p + (dx, dy) in `b`."""
    h, w = a.shape
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    if x1 - x0 < 48 or y1 - y0 < 48:
        return -1.0
    pa = a[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    pb = b[y0:y1, x0:x1]
    pa, pb = pa - pa.mean(), pb - pb.mean()
    den = float(np.sqrt((pa * pa).sum() * (pb * pb).sum()))
    return float((pa * pb).sum()) / den if den > 0 else -1.0


def register_pair(a: np.ndarray, b: np.ndarray, frac: float = 0.16):
    """Shift d (px) such that a feature at pixel p in `a` is at p + d in `b`.

    The overlap of neighbouring tiles is a thin strip (or a corner), where
    phase correlation aliases.  Instead, patches along each edge and corner of
    `b` are located anywhere in `a` (template matching at half resolution);
    the candidate whose whole implied overlap correlates best wins, and is
    refined at full resolution.  Returns (d, ncc).
    """
    h, w = a.shape
    pw, ph = max(int(frac * w), 32), max(int(frac * h), 32)
    lw, lh = int(0.6 * w), int(0.6 * h)
    m = 4
    patches = [  # (x, y, width, height) in b
        (m, (h - lh) // 2, pw, lh), (w - pw - m, (h - lh) // 2, pw, lh),
        ((w - lw) // 2, m, lw, ph), ((w - lw) // 2, h - ph - m, lw, ph),
        (m, m, pw, ph), (w - pw - m, m, pw, ph), (m, h - ph - m, pw, ph), (w - pw - m, h - ph - m, pw, ph)]
    a2 = cv2.resize(a, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cands = set()
    for x, y, pw_, ph_ in patches:
        tpl = b2[y // 2:(y + ph_) // 2, x // 2:(x + pw_) // 2]
        if tpl.std() < 1e-4:
            continue
        r = cv2.matchTemplate(a2, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        if score > 0.3:
            cands.add((x - 2 * loc[0], y - 2 * loc[1]))
    if not cands:
        return None, -1.0
    ncc, (dx, dy) = max((_overlap_ncc(a, b, dx, dy), (dx, dy)) for dx, dy in cands)
    for _ in range(4):
        c, (ndx, ndy) = max((_overlap_ncc(a, b, dx + i, dy + j), (dx + i, dy + j))
                            for i in (-1, 0, 1) for j in (-1, 0, 1))
        if (ndx, ndy) == (dx, dy):
            break
        dx, dy, ncc = ndx, ndy, c
    return np.array([dx, dy], float), ncc


def register_predicted(a: np.ndarray, b: np.ndarray, d0, search: int = 50):
    """Like register_pair, but around a predicted shift d0 (from the stage positions).

    Only the predicted overlap strip is used, so this works for small overlaps
    (15 % = ~150-200 px) where the edge patches of register_pair do not fit.
    Returns (d, ncc) or (None, -1).
    """
    h, w = a.shape
    dx0, dy0 = int(round(d0[0])), int(round(d0[1]))
    S = int(search)
    x0, x1 = max(0, dx0), min(w, w + dx0)          # overlap, in b's pixels
    y0, y1 = max(0, dy0), min(h, h + dy0)
    tx0, tx1, ty0, ty1 = x0 + S, x1 - S, y0 + S, y1 - S
    if tx1 - tx0 < 32 or ty1 - ty0 < 32:
        return None, -1.0
    tpl = b[ty0:ty1, tx0:tx1]
    if tpl.std() < 1e-4:
        return None, -1.0
    wx0, wy0 = max(0, tx0 - dx0 - S), max(0, ty0 - dy0 - S)
    wx1, wy1 = min(w, tx1 - dx0 + S), min(h, ty1 - dy0 + S)
    win = a[wy0:wy1, wx0:wx1]
    if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return None, -1.0
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(r)
    dx, dy = tx0 - (wx0 + loc[0]), ty0 - (wy0 + loc[1])
    ncc = _overlap_ncc(a, b, dx, dy)
    for _ in range(4):
        c, (ndx, ndy) = max((_overlap_ncc(a, b, dx + i, dy + j), (dx + i, dy + j))
                            for i in (-1, 0, 1) for j in (-1, 0, 1))
        if (ndx, ndy) == (dx, dy):
            break
        dx, dy, ncc = ndx, ndy, c
    return np.array([dx, dy], float), ncc


@dataclass
class StageMap:
    """Mosaic pixel <-> stage coordinates.

    For tile k (stage position P_k, top-left at mosaic pixel q_k) a mosaic
    pixel x maps to the stage position that centres it in the field of view:
        stage = P_k + M @ (x - q_k - c),   c = frame centre (px)
    with M the 2x2 um/px matrix (pixel size, rotation and axis flips).  The
    tile used is the one whose centre is nearest to x.
    """
    M: np.ndarray                   # (2, 2) um per px
    P: np.ndarray                   # (N, 2) tile stage positions
    q: np.ndarray                   # (N, 2) tile top-left in the mosaic (px)
    frame_wh: tuple                 # (W, H)
    labels: list
    z: list
    z_units: str = "drive"

    @property
    def centres(self):
        return self.q + np.array(self.frame_wh, float) / 2.0

    @property
    def pixel_um(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.M))))

    def owner(self, x, y) -> int:
        return int(np.argmin(((self.centres - np.array([x, y])) ** 2).sum(1)))

    def to_stage(self, x, y, tile: int | None = None) -> tuple:
        k = self.owner(x, y) if tile is None else tile
        c = np.array(self.frame_wh, float) / 2.0
        s = self.P[k] + self.M @ (np.array([x, y], float) - self.q[k] - c)
        return float(s[0]), float(s[1])

    def to_pixel(self, sx, sy) -> tuple:
        """Mosaic pixel of a stage position (nearest tile by stage distance)."""
        k = int(np.argmin(((self.P - np.array([sx, sy])) ** 2).sum(1)))
        c = np.array(self.frame_wh, float) / 2.0
        p = self.q[k] + c + np.linalg.solve(self.M, np.array([sx, sy]) - self.P[k])
        return float(p[0]), float(p[1])

    def to_dict(self) -> dict:
        return dict(M=self.M.tolist(), P=self.P.tolist(), q=self.q.tolist(), frame_wh=list(self.frame_wh),
                    labels=self.labels, z=self.z, z_units=self.z_units, pixel_um=self.pixel_um,
                    rotation_deg=float(np.degrees(np.arctan2(self.M[1, 0], self.M[0, 0]))))

    @classmethod
    def from_dict(cls, d):
        return cls(M=np.array(d["M"]), P=np.array(d["P"]), q=np.array(d["q"]), frame_wh=tuple(d["frame_wh"]),
                   labels=list(d["labels"]), z=list(d["z"]), z_units=d.get("z_units", "drive"))


def _neighbour_pairs(P: np.ndarray):
    """Pairs of tiles adjacent on the acquisition grid (by stage distance)."""
    n = len(P)
    if n < 2:
        return []
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    step = np.median(d.min(1))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # direct neighbours and diagonals
            if d[i, j] < 1.6 * step:
                pairs.append((i, j))
    return pairs


def fit_layout(tiles, frames_prep, min_ncc: float = 0.3, max_resid_px: float = 25.0, verbose=True, M_prior=None):
    """Register neighbours, fit M and solve the global tile positions."""
    P = np.array([t.stage_xy for t in tiles])
    h, w = frames_prep[0].shape
    pairs = _neighbour_pairs(P)
    from .progress import Progress
    pr = Progress("registering tiles", len(pairs), verbose)
    meas = []
    # 1) around the overlap predicted from the stage positions and the usual calibration
    Minv0 = np.linalg.inv(DEFAULT_M if M_prior is None else np.asarray(M_prior))
    done = set()
    for i, j in pairs:
        d0 = -Minv0 @ (P[j] - P[i])
        if abs(d0[0]) >= w or abs(d0[1]) >= h:
            continue
        d, ncc = register_predicted(frames_prep[i].astype(np.float32), frames_prep[j].astype(np.float32), d0)
        pr.update()
        if d is not None and ncc >= min_ncc:
            meas.append((i, j, d, ncc)); done.add((i, j))
    pr.done()
    # 2) blind search for the rest if the prediction did not work (other objective / camera)
    if len(meas) < max(2, 0.3 * len(pairs)):
        for i, j in pairs:
            if (i, j) in done:
                continue
            d, ncc = register_pair(frames_prep[i].astype(np.float32), frames_prep[j].astype(np.float32))
            if d is not None and ncc >= min_ncc:
                meas.append((i, j, d, ncc))
    if len(meas) < 2:
        raise RuntimeError("too few tile pairs could be registered (need overlapping tiles with some features)")

    # stage: M (q_j - q_i) = P_j - P_i, and q_j - q_i = -d (content moves opposite to the frame)
    def fit_M(ms):
        A = np.array([-m[2] for m in ms])
        B = np.array([P[m[1]] - P[m[0]] for m in ms])
        X, *_ = np.linalg.lstsq(A, B, rcond=None)
        return X.T
    good = meas
    for _ in range(3):
        M = fit_M(good)
        Minv = np.linalg.inv(M)
        res = [np.linalg.norm(-m[2] - Minv @ (P[m[1]] - P[m[0]])) for m in meas]
        new = [m for m, r in zip(meas, res) if r < max_resid_px]
        if len(new) < 2 or len(new) == len(good):
            good = new if len(new) >= 2 else good
            break
        good = new
    M = fit_M(good)
    Minv = np.linalg.inv(M)

    # global positions: pair constraints (weight ~ ncc) + weak stage prior
    n = len(tiles)
    rows, rhs, wts = [], [], []
    for i, j, d, ncc in good:
        r = np.zeros(n); r[j], r[i] = 1, -1
        rows.append(r); rhs.append(-d); wts.append(ncc)
    prior = (Minv @ (P - P[0]).T).T
    for k in range(n):
        r = np.zeros(n); r[k] = 1
        rows.append(r); rhs.append(prior[k]); wts.append(0.01)
    A = np.array(rows) * np.array(wts)[:, None]
    B = np.array(rhs) * np.array(wts)[:, None]
    q, *_ = np.linalg.lstsq(A, B, rcond=None)
    q -= q.min(0)
    stage_err = [(Minv @ (P[j] - P[i])) - (q[j] - q[i]) for i, j, _, _ in good]
    stats = dict(pairs=len(pairs), registered=len(meas), used=len(good),
                 ncc_median=float(np.median([m[3] for m in good])),
                 stage_error_px_rms=float(np.sqrt(np.mean(np.sum(np.square(stage_err), 1)))),
                 stage_error_px_max=float(np.max(np.linalg.norm(stage_err, axis=1))),
                 pair_residual_px_rms=float(np.sqrt(np.mean([np.sum((q[j] - q[i] + d) ** 2) for i, j, d, _ in good]))))
    smap = StageMap(M=M, P=P, q=q, frame_wh=(w, h), labels=[t.label for t in tiles], z=[t.z_um for t in tiles],
                    z_units=tiles[0].z_units)
    if verbose:
        print(f"tiles: {n}, pairs registered {len(meas)}/{len(pairs)} (used {len(good)}, median NCC {stats['ncc_median']:.2f}); "
              f"pixel {smap.pixel_um:.4f} um/px, rotation {smap.to_dict()['rotation_deg']:+.2f} deg; "
              f"stage error vs registration rms {stats['stage_error_px_rms']:.1f} px (max {stats['stage_error_px_max']:.1f})")
    return smap, stats


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------

def raw_glitch(raw: np.ndarray, bits: int | None = None) -> str | None:
    """Camera garbage frame (XC10, ~1 in 100-200 frames), judged on the raw Bayer mosaic (R at the top-left):
    random values (median difference of same-colour neighbours > 4 % of full scale; ~10 % seen; real
    frames, bright or dark: ~0.3 %), or scrambled data (the two green sites of the 2x2 cell differ by > 15 %;
    real frames, on or off the chip: < 3 %).  Returns the reason or None."""
    ev = np.asarray(raw[8:-8:4, 8:-8], np.float32)                 # even rows: R G R G ...
    od = np.asarray(raw[9:-8:4, 8:-8], np.float32)                 # odd rows:  G B G B ...
    fs = float(2 ** bits - 1) if bits else float(2 ** int(np.ceil(np.log2(max(float(ev.max()), 2.0) + 1))) - 1)
    g = ev[:, 1::2]
    d = float(np.median(np.abs(np.diff(g, axis=1))))
    if d > 0.04 * fs:
        return "noise frame"
    g2a = od[:, 0::2]
    n = min(g.shape[0], g2a.shape[0])
    g1a, g2a = g[:n], g2a[:n]
    # in strips (8 across, 8 down): a frame can be scrambled in part only (bigbntest_2 Pos-005_001: 33 % in one
    # strip, 13 % over the frame; good frames <= 1.2 % in any strip)
    for ax in (0, 1):
        for idx in np.array_split(np.arange(g1a.shape[ax]), 8):
            a1 = float(np.take(g1a, idx, axis=ax).mean()); a2 = float(np.take(g2a, idx, axis=ax).mean())
            if abs(a1 - a2) > 0.15 * max(a1, a2, 1.0) and max(a1, a2) > 0.02 * fs:
                return "scrambled frame"
    return None


def glitch_frame(tile) -> bool:
    """True if the last read_frame of this tile found a camera garbage frame."""
    return bool(getattr(tile, "glitch", None))


def compose(frames, smap: StageMap, dtype=np.float32, fill=0.0, verbose=False, skip=(), prefetch=0):
    """Nearest-centre mosaic (every pixel from the tile whose centre is closest).

    frames: list of frames, or a function k -> frame (tiles are then loaded one at a time).
    skip: tile indices left out (garbage frames); their neighbours cover what they can, the rest stays `fill`.
    prefetch: with a function, load this many tiles ahead in threads (file reading and demosaicing overlap)."""
    skip = set(skip)
    if prefetch and callable(frames):
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(prefetch)
        futs = {}
        order = [k for k in range(len(smap.labels)) if k not in skip]
        load = frames

        def frames(k):
            i = order.index(k)
            for j in order[i:i + prefetch + 1]:
                if j not in futs:
                    futs[j] = ex.submit(load, j)
            return futs.pop(k).result()
    get = frames if callable(frames) else (lambda k: frames[k])
    n = len(smap.labels)
    w, h = smap.frame_wh
    qi = np.round(smap.q).astype(int)
    W, H = int(qi[:, 0].max()) + w, int(qi[:, 1].max()) + h
    out = np.full((H, W, 3), fill, dtype)
    covered = np.zeros((H, W), bool)
    # pixel p belongs to tile k where |p - c_k|^2 < |p - c_j|^2 for every overlapping j (ties: the lower index).  In
    # integers (C = 2 c): s = |2p - C_k|^2 - |2p - C_j|^2 = a x + r(y), linear in x along a row, so tile k owns one span
    # of every row: the same pixels as comparing whole distance maps, without them (that was 0.4 s per tile, a quarter
    # of a live block)
    C2 = 2 * qi.astype(np.int64) + np.array([w, h], np.int64)     # (64 bit: numpy 1.x int is 32 bit on Windows)
    xs = np.arange(w)
    from .progress import Progress
    pr = Progress("building mosaic", n, verbose)
    for k in range(n):
        pr.update()
        if k in skip:
            continue
        f = get(k)
        x0, y0 = int(qi[k, 0]), int(qi[k, 1])
        lo, hi = np.zeros(h, np.int64), np.full(h, w, np.int64)
        Y = 2 * (np.arange(h, dtype=np.int64) + y0)
        for j in range(n):
            if j == k or j in skip or abs(qi[j, 0] - x0) >= w or abs(qi[j, 1] - y0) >= h:
                continue
            a = 4 * int(C2[j, 0] - C2[k, 0])
            r = 4 * x0 * (C2[j, 0] - C2[k, 0]) + 2 * Y * (C2[j, 1] - C2[k, 1]) \
                + int(C2[k] @ C2[k]) - int(C2[j] @ C2[j])
            m = (1 if k < j else 0) - r                  # own where a x < m (x = 0 .. w-1 within the tile)
            if a > 0:
                hi = np.minimum(hi, -(-m // a))          # x < m / a  <=>  x < ceil(m / a)
            elif a < 0:
                lo = np.maximum(lo, m // a + 1)          # x > m / a  <=>  x >= floor(m / a) + 1
            else:
                hi = np.where(m > 0, hi, 0)
        own = (xs[None, :] >= lo[:, None]) & (xs[None, :] < hi[:, None])
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        np.copyto(out[sl], f, where=own[..., None], casting="unsafe")
        covered[sl] |= own
    return out, covered


def prepare_flat(flat, tile0: Tile, fs: float):
    """Load / check a flat field against the scan's first tile; returns (FlatField or None, flat minimum)."""
    if isinstance(flat, (str, Path)):
        flat = FlatField.load(flat)
    if flat is None:
        print("warning: no flat field given (--flat); tile vignetting will leave seams and bias the contrast")
        return None, 1.0
    first = read_frame(tile0)
    if flat.flat.shape != first.shape:
        raise ValueError(f"flat field {flat.flat.shape[:2]} does not match frames {first.shape[:2]}")
    fe, te = flat.info.get("exposure_ms"), tile0.props.get("Exposure-ms")
    if fe is not None and te is not None and abs(float(fe) - float(te)) > 1e-6:
        print(f"warning: flat field taken at {fe} ms, scan at {te} ms; take a flat field with the scan settings")
    # the black level scales with the output bit depth (the XC10 12-bit output is the 14-bit
    # data shifted by 2 bits); flat fields from before this was recorded are 12-bit
    fb = int(flat.info.get("bit_depth", 12 if flat.info.get("data_format", "").upper().startswith("RAW") else 8))
    sb = int(round(np.log2(fs + 1)))
    if fb != sb:
        print(f"warning: flat field / dark taken at {fb} bit, scan at {sb} bit; dark level scaled by {2 ** (sb - fb)}x "
              f"(measure a dark frame with the scan settings)")
        flat = FlatField(flat=flat.flat, dark=flat.dark * float(2 ** (sb - fb)), info=dict(flat.info, bit_depth=sb))
    return flat, float(flat.flat.min())


def load_dataset(path, flat: FlatField | str | None = None, verbose: bool = True):
    """Stitched, flat-field corrected mosaic of an acquisition folder.

    Returns (ImageData, StageMap).  The image is uint16 with raw-saturated
    pixels set to 65535 (so they are recognised as clipped), and
    ``clip_value`` set to the level a raw-saturated pixel reaches in the
    centre of the field (the most conservative clipping level).
    """
    from .io import ImageData

    tiles = read_dataset(path)
    for msg in check_linear(tiles[0]):
        print("warning:", msg)
    fs = full_scale(tiles[0])
    flat, fmin = prepare_flat(flat, tiles[0], fs)

    # tiles are read twice (registration, then the mosaic) instead of being held in memory:
    # a 1-inch piece is ~500 tiles (~17 MB each as float RGB)
    shifts = {}

    def corrected(k):
        r = read_frame(tiles[k], shifts.get(k))
        return r, (r if flat is None else flat.apply(r))

    from .progress import Progress
    pr = Progress("reading tiles", len(tiles), verbose)
    prep, bad = [], []
    from concurrent.futures import ThreadPoolExecutor

    def prepped(k_):                                 # read + demosaic + flat + registration copy, in threads
        r_, c_ = corrected(k_)
        return _prep(c_).astype(np.float16), glitch_frame(tiles[k_])
    with ThreadPoolExecutor(READ_THREADS) as ex:
        for k_, (pp, g) in enumerate(ex.map(prepped, range(len(tiles)))):
            if g:                                    # camera garbage frames are left out of the mosaic
                bad.append(k_)
            prep.append(pp)
            pr.update()
    if bad:
        print(f"warning: {len(bad)} garbage tile(s) left out of the mosaic (camera glitch frames): "
              + ", ".join(tiles[i].label for i in bad))
    smap, stats = fit_layout(tiles, prep, verbose=verbose)
    del prep

    # scale so that no unsaturated pixel can reach the clipping threshold
    dark = 0.0 if flat is None else float(np.max(flat.dark))
    k = 0.96 * 65535.0 * fmin / max(fs - dark, 1.0)

    def tile16(i):
        r, c = corrected(i)
        v = np.clip(c * k, 0, 0.965 * 65535).astype(np.uint16)
        v[r >= fs - 0.5] = 65535
        return v

    mosaic, covered = compose(tile16, smap, dtype=np.uint16, verbose=verbose, skip=bad, prefetch=READ_THREADS)
    del covered

    meta = dict(source="micromanager", dataset=str(path), n_tiles=len(tiles), data_format=tiles[0].data_format,
                exposure_ms=tiles[0].props.get("Exposure-ms"), stage_map=smap.to_dict(), registration=stats,
                flatfield=None if flat is None else flat.info, counts_scale=k,
                camera={k_.split("-", 1)[1]: v for k_, v in tiles[0].props.items() if k_.startswith(tiles[0].props.get("Camera", "~") + "-")},
                glitch_tiles={tiles[i].label: tiles[i].glitch for i in bad})
    v = mosaic[::7, ::7].reshape(-1, 3)
    sub = np.median(v[v.min(1) > 0], axis=0)
    # display: substrate -> neutral grey at 31 % (white = 3.2x the substrate; it was 62 %, which made flakes up to
    # only 1.6x the substrate look clipped)
    gain = tuple(float(g) for g in (0.31 * 65535.0 / np.maximum(sub, 1.0)))
    img = ImageData(rgb=mosaic, path=Path(path), pixel_um=smap.pixel_um,
                    pixel_um_source=f"Micro-Manager tile registration vs stage ({stats['used']} pairs)",
                    metadata=meta, max_value=65535, clip_value=k * (fs - dark), stage_map=smap,
                    display_gain=gain)
    return img, smap


def preview(rgb16: np.ndarray, substrate=None, scale: float = 0.25) -> np.ndarray:
    """White-balanced 8-bit preview (substrate -> neutral grey), sRGB-encoded."""
    small = cv2.resize(rgb16, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
    if substrate is None:
        v = small.reshape(-1, 3)
        substrate = np.median(v[v.min(1) > 0], axis=0)
    lin = np.clip(small / np.asarray(substrate, np.float32) * 0.42, 0, 1)
    return (lin ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Objective offset (scan objective -> high-magnification objective)
# ---------------------------------------------------------------------------

def _flake_mask(frame, blur, thr=0.25, min_px=0):
    g = cv2.GaussianBlur(frame, (0, 0), blur)
    h, w = g.shape[:2]
    b = max(8, h // 25)
    border = np.concatenate([g[:b].reshape(-1, 3), g[-b:].reshape(-1, 3), g[:, :b].reshape(-1, 3), g[:, -b:].reshape(-1, 3)])
    c = np.abs(g / np.maximum(np.median(border, axis=0), 1) - 1).max(axis=2)
    m = cv2.morphologyEx((c > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if min_px:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        m = np.isin(lab, [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_px]).astype(np.uint8)
    return m.astype(np.float32)


def objective_offset(low, high, M, dark=None, ratio_range=(8.0, 12.0), search_px=350):
    """Offset between two objectives from images of the same flake.

    low / high: acquisition folders (single snaps) with the scan and the
    high-magnification objective.  M: 2x2 um/px matrix of the scan objective
    (from a stitched scan).  The high-magnification field is located in the
    low-magnification image by matching flake outlines over a range of scale
    ratios, near the centre (the flake only needs to be roughly centred).

    Returns dict(offset_um, ratio, high_pixel_um, score, centre_px): add
    offset_um to a scan-objective target to centre it with the other objective.
    The recorded stage positions are used (offset += P_high - P_low); they are
    only right if Micro-Manager saw every stage move.
    """
    tl, th = read_dataset(low)[0], read_dataset(high)[0]
    d = 0.0 if dark is None else dark
    a, b = read_frame(tl) - d, read_frame(th) - d
    h, w = a.shape[:2]
    ma = _flake_mask(a, 0.7)
    mb = _flake_mask(b, 3.0, min_px=int(0.002 * b.shape[0] * b.shape[1]))
    if mb.sum() == 0:
        raise RuntimeError("no flake found in the high-magnification image")
    cx, cy = w / 2.0, h / 2.0
    x0, y0 = max(0, int(cx - search_px)), max(0, int(cy - search_px))
    roi = ma[y0:y0 + 2 * search_px, x0:x0 + 2 * search_px]
    best = (-2.0, None, None)
    for r in np.arange(ratio_range[0], ratio_range[1] + 1e-9, 0.02):
        t = cv2.resize(mb, None, fx=1 / r, fy=1 / r, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= roi.shape[0] or t.shape[1] >= roi.shape[1]:
            continue
        res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(res)
        if v > best[0]:
            best = (v, r, loc)
    v, r, loc = best
    hh, hw = b.shape[:2]
    centre = np.array([x0 + loc[0] + (hw / 2.0) / r, y0 + loc[1] + (hh / 2.0) / r])   # high-mag centre, low-mag px
    p = centre - np.array([cx, cy])
    M = np.asarray(M, float)
    off = (th.stage_xy - tl.stage_xy) - M @ p
    return dict(offset_um=off.tolist(), ratio=float(r), high_pixel_um=float(np.sqrt(abs(np.linalg.det(M))) / r),
                score=float(v), centre_px=centre.tolist(), stage_moved_um=(th.stage_xy - tl.stage_xy).tolist())


def _load_M(path):
    d = json.loads(Path(path).read_text())
    sm = d.get("stage_map") or d.get("metadata", {}).get("stage_map")
    if not sm:
        raise ValueError(f"{path} has no stage_map (use a *_stagemap.json or *_flakes.json from a tile scan)")
    return np.array(sm["M"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m flakeid.mm", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flat", help="build a flat field from frames of bare substrate")
    f.add_argument("blank"); f.add_argument("-o", "--out", required=True)
    f.add_argument("--dark", help="folder of dark frames (light blocked, same exposure)")
    s = sub.add_parser("stitch", help="stitch a tile acquisition; writes a preview JPEG and the stage map")
    s.add_argument("dataset"); s.add_argument("--flat"); s.add_argument("-o", "--out-dir")
    s.add_argument("--tiff", action="store_true", help="also write the 16-bit mosaic as TIFF")
    o = sub.add_parser("offset", help="objective offset from snaps of one flake with both objectives")
    o.add_argument("low"); o.add_argument("high")
    o.add_argument("--stagemap", required=True, help="*_stagemap.json or *_flakes.json of a tile scan (gives M)")
    o.add_argument("--flat", help="flat field .npz (its dark frame is subtracted)")
    a = p.parse_args(argv)
    if a.cmd == "offset":
        dark = FlatField.load(a.flat).dark if a.flat else None
        r = objective_offset(a.low, a.high, _load_M(a.stagemap), dark=dark)
        print(f"match score {r['score']:.2f}; magnification ratio {r['ratio']:.2f} -> "
              f"high-mag pixel {r['high_pixel_um']:.4f} um/px")
        if any(abs(v) > 0.01 for v in r["stage_moved_um"]):
            print(f"recorded stage move between the snaps: {np.round(r['stage_moved_um'], 1)} um (included)")
        print(f"offset: --offset {r['offset_um'][0]:.1f} {r['offset_um'][1]:.1f}")
        return
    if a.cmd == "flat":
        ff = build_flatfield(a.blank, a.dark)
        ff.save(a.out)
        print(json.dumps(ff.info, indent=1))
        print(f"wrote {a.out}")
    else:
        img, smap = load_dataset(a.dataset, a.flat)
        ds = Path(a.dataset)
        out = Path(a.out_dir) if a.out_dir else ds.parent
        out.mkdir(parents=True, exist_ok=True)
        pj = out / f"{ds.name}_mosaic.jpg"
        cv2.imwrite(str(pj), cv2.cvtColor(preview(img.rgb, scale=0.5), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / f"{ds.name}_stagemap.json").write_text(json.dumps(img.metadata, indent=1, default=str))
        print(f"wrote {pj}\nwrote {out / f'{ds.name}_stagemap.json'}")
        if a.tiff:
            cv2.imwrite(str(out / f"{ds.name}_mosaic.tif"), cv2.cvtColor(img.rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
