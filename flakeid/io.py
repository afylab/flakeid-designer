"""Image loading and (best-effort) Olympus cellSens metadata extraction.

Supported inputs
----------------
* JPEG / PNG / TIFF exports of the stitched overview (read with OpenCV).
* Micro-Manager tile acquisitions (a folder of Pos-*/ frames), stitched by
  ``flakeid.mm`` with flat-field correction and an exact pixel -> stage map.
* Olympus .vsi: the .vsi file itself only contains a small preview plus
  metadata; the full-resolution pixels live in the companion folder
  ``_<name>_/stack*/frame_t*.ets``, which is read natively here (tiled,
  uncompressed 14-bit BGR as written by cellSens for the BX51/XC10).  The
  cellSens JPEG export of the same scan is the 14-bit data scaled linearly to
  8 bits (no gamma, no colour matrix; verified on fullchipbn23).

If a .vsi with the same stem sits next to a JPEG/TIFF export, its metadata
(pixel size, stage position, objective) is used automatically.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Default calibration for the Young Lab BX51 + XC10 at 10x objective / 0.63x
# camera adapter ("6.3x"), as stored by cellSens and confirmed by the 1 mm
# scale bar burned into the exports.
DEFAULT_PIXEL_UM = 1.005176

_UNIT_UM = "10^-6m^1".encode("utf-16-le")


@dataclass
class ImageData:
    rgb: np.ndarray                     # (H, W, 3) uint8 or uint16, RGB order
    path: Path
    pixel_um: float
    pixel_um_source: str
    metadata: dict = field(default_factory=dict)
    max_value: int = 255                # full-scale value (clipping level)
    clip_value: float | None = None     # level a clipped pixel reaches, if below max_value (flat-fielded data)
    stage_map: object = None            # mm.StageMap for Micro-Manager mosaics (pixel -> stage)
    display_gain: tuple | None = None   # per-channel gain for 8-bit display (white balance), not used for analysis

    def to_uint8(self, arr=None, bgr: bool = False) -> np.ndarray:
        """8-bit version (linear scaling, as the cellSens JPEG export; white-balanced if display_gain is set).

        Converted in blocks of rows, so a multi-GB mosaic needs no full-size float copy."""
        a = self.rgb if arr is None else arr
        if a.dtype == np.uint8:
            return a[..., ::-1].copy() if bgr else a
        g = (255.0 / self.max_value) * (np.asarray(self.display_gain, np.float32) if self.display_gain is not None else np.ones(3, np.float32))
        g = np.asarray(g, np.float32)
        if bgr:
            g = g[::-1]
        out = np.empty(a.shape, np.uint8)
        for r in range(0, a.shape[0], 1024):
            blk = a[r:r + 1024]
            if bgr:
                blk = blk[..., ::-1]
            out[r:r + 1024] = np.clip(blk.astype(np.float32) * g + 0.5, 0, 255).astype(np.uint8)
        return out


def parse_vsi_metadata(vsi_path: Path) -> dict:
    """Best-effort extraction of calibration and stage info from a .vsi file.

    cellSens stores values as little-endian doubles next to UTF-16 unit
    strings.  This is a heuristic parser validated on BX51/cellSens files;
    anything it cannot find is simply omitted.
    """
    data = Path(vsi_path).read_bytes()
    meta: dict = {}
    units = [m.start() for m in re.finditer(re.escape(_UNIT_UM), data)]

    def dbl(off):
        if 0 <= off <= len(data) - 8:
            return struct.unpack_from("<d", data, off)[0]
        return None

    # Image calibration block: origin_x, origin_y (stage um of the image
    # origin) followed shortly by the pixel size pair (pixel_x == pixel_y).
    for u in units:
        for off in range(u, min(u + 256, len(data) - 16)):
            px, py = struct.unpack_from("<2d", data, off)
            if not (px == py and 0.01 < px < 100):
                continue
            for back in range(16, 48, 8):
                ox, oy = dbl(off - back - 8), dbl(off - back)
                if (ox and oy and np.isfinite(ox) and np.isfinite(oy)
                        and 100 < abs(ox) < 1e6 and 100 < abs(oy) < 1e6):
                    meta["pixel_um"] = px
                    meta["origin_x_um"], meta["origin_y_um"] = ox, oy
                    break
            if "pixel_um" in meta:
                break
        if "pixel_um" in meta:
            break
    if "pixel_um" not in meta:
        # fallback: identical (x, y) pixel-size pair just before a unit string
        for u in units:
            for off in range(u - 64, u - 8):
                a, b = dbl(off), dbl(off + 8)
                if a and b and a == b and 0.01 < a < 100:
                    meta["pixel_um"] = a
                    break
            if "pixel_um" in meta:
                break

    # Stage position shown on the burned-in label: two consecutive unit
    # strings, 128 bytes apart, each followed (+64 bytes) by a coordinate.
    for u1, u2 in zip(units, units[1:]):
        if u2 - u1 == 128:
            x, y = dbl(u1 + 64), dbl(u2 + 64)
            if x and y and 100 < abs(x) < 1e6 and 100 < abs(y) < 1e6:
                meta["stage_x_um"], meta["stage_y_um"] = x, y
                break

    text = data.decode("utf-16-le", errors="ignore")
    m = re.search(r"(\d+(?:\.\d+)?)x", text)
    if m:
        meta["objective"] = m.group(0)
    for key, pat in (("objective_name", r"MPLFLN[ A-Za-z]*|LMPLFLN[ A-Za-z]*|MPLAPON[ A-Za-z]*"),):
        m = re.search(pat, text)
        if m:
            meta[key] = m.group(0).strip()
    return meta


def _ets_tiles(ets_path: Path):
    """Parse an Olympus .ets tile container.  Returns (header, {coord: (offset, nbytes)})."""
    with open(ets_path, "rb") as f:
        d = f.read(0x200)
        magic, _hs, _ver, ndim, aoff, _asz, _pad, uoff, nused, _ = struct.unpack_from("<4sIIIQIIQII", d, 0)
        if magic != b"SIS\x00":
            raise RuntimeError(f"{ets_path} is not an ETS file")
        ets = struct.unpack_from("<4sIIIIIIIII", d, aoff)
        if ets[0] != b"ETS\x00":
            raise RuntimeError(f"{ets_path}: missing ETS header")
        hdr = dict(pixel_type=ets[2], channels=ets[3], colorspace=ets[4], compression=ets[5],
                   quality=ets[6], tile_w=ets[7], tile_h=ets[8], tile_z=ets[9], ndim=ndim)
        # background (fill) value per channel follows the ETS header block
        hdr["fill"] = struct.unpack_from("<3H", d, aoff + 0x6c)
        f.seek(uoff)
        rec = 4 + 4 * ndim + 8 + 4 + 4
        tab = f.read(nused * rec)
    tiles = {}
    for i in range(nused):
        coord = struct.unpack_from("<" + "i" * ndim, tab, i * rec + 4)
        off, nb = struct.unpack_from("<QI", tab, i * rec + 4 + 4 * ndim)
        tiles[coord] = (off, nb)
    return hdr, tiles


_ETS_DTYPES = {2: np.uint8, 4: np.uint16}     # pixel types seen in cellSens files


def read_ets(ets_path: Path, level: int = 0) -> tuple:
    """Assemble one pyramid level of an .ets file.  Returns (rgb, max_value)."""
    hdr, tiles = _ets_tiles(ets_path)
    if hdr["compression"] == 0:
        decode = None
    elif hdr["compression"] in (2, 3):          # JPEG / JPEG2000 tiles
        decode = hdr["compression"]
    else:
        raise RuntimeError(f"unsupported ETS compression {hdr['compression']}")
    dt = _ETS_DTYPES.get(hdr["pixel_type"])
    if dt is None:
        raise RuntimeError(f"unsupported ETS pixel type {hdr['pixel_type']}")
    tw, th, nc = hdr["tile_w"], hdr["tile_h"], hdr["channels"]
    lv = [c for c in tiles if c[-1] == level]
    if not lv:
        raise RuntimeError(f"level {level} not in {ets_path}")
    nx = max(c[0] for c in lv) + 1
    ny = max(c[1] for c in lv) + 1
    img = np.zeros((ny * th, nx * tw, nc), dt)
    mm = np.memmap(ets_path, dtype=np.uint8, mode="r")
    for c in lv:
        off, nb = tiles[c]
        buf = mm[off:off + nb]
        if decode is None:
            t = np.frombuffer(buf, dtype=np.dtype(dt).newbyteorder("<")).reshape(th, tw, nc)
        else:
            t = cv2.imdecode(np.asarray(buf), cv2.IMREAD_UNCHANGED)
            if t is None:
                raise RuntimeError("could not decode ETS tile")
            t = t.reshape(th, tw, -1)
        img[c[1] * th:(c[1] + 1) * th, c[0] * tw:(c[0] + 1) * tw] = t[..., :nc]
    del mm
    # cellSens stores BGR; tiles are padded to whole tiles with the fill value
    img = img[..., ::-1] if nc == 3 else np.repeat(img[..., :1], 3, axis=2)
    fill = np.array(hdr["fill"][:3][::-1] if nc == 3 else hdr["fill"][:1] * 3)
    is_fill = np.all(img == fill[None, None, :], axis=2)
    rows = np.flatnonzero(~is_fill.all(axis=1))
    cols = np.flatnonzero(~is_fill.all(axis=0))
    img = np.ascontiguousarray(img[: rows[-1] + 1, : cols[-1] + 1])
    max_value = 255 if dt == np.uint8 else int(max(fill.max(), 2 ** int(np.ceil(np.log2(img.max() + 1))) - 1))
    return img, max_value


def find_ets(vsi_path: Path) -> Path:
    """Largest frame of the companion folder _<stem>_/stack*/frame_t*.ets."""
    folder = vsi_path.parent / f"_{vsi_path.stem}_"
    cands = sorted(folder.glob("stack*/frame_t*.ets"))
    if not cands:
        raise RuntimeError(
            f"{vsi_path.name} only contains a small preview; the full-resolution data "
            f"({folder.name}/stack*/frame_t*.ets) was not found next to it.")
    return max(cands, key=lambda p: p.stat().st_size)


def export_offset(export_rgb8: np.ndarray, ets_path: Path) -> tuple:
    """Pixel offset (dx, dy) of a cellSens export inside the full-resolution
    VSI frame (exports can be cropped).  Coarse match on pyramid level 3,
    refined by phase correlation at full resolution."""
    lv3, _ = read_ets(ets_path, level=3)
    g_v = lv3[..., 1].astype(np.float32)
    small = cv2.resize(export_rgb8[..., 1], None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA).astype(np.float32)
    g_v /= max(np.median(g_v), 1.0)
    small /= max(np.median(small), 1.0)
    res = cv2.matchTemplate(g_v, small[16:-16, 16:-16], cv2.TM_CCOEFF_NORMED) if (
        g_v.shape[0] >= small.shape[0] - 32 and g_v.shape[1] >= small.shape[1] - 32) else None
    if res is None:
        return (0, 0)
    _, _, _, (mx, my) = cv2.minMaxLoc(res)
    dx0, dy0 = (mx - 16) * 8, (my - 16) * 8
    full, _ = read_ets(ets_path, level=0)
    h, w = export_rgb8.shape[:2]
    cy, cx = h // 2 - 1024, w // 2 - 1024
    a = export_rgb8[cy:cy + 2048, cx:cx + 2048, 1].astype(np.float32)
    b = full[cy + dy0:cy + dy0 + 2048, cx + dx0:cx + dx0 + 2048, 1].astype(np.float32)
    if a.shape != b.shape:
        return (dx0, dy0)
    (sx, sy), _ = cv2.phaseCorrelate(np.log1p(a), np.log1p(b))
    return (int(round(dx0 + sx)), int(round(dy0 + sy)))


def load_image(path, pixel_um: float | None = None, flat=None) -> ImageData:
    """Load a stitched image, a .vsi, or a Micro-Manager tile folder (``flat``: flat-field .npz)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        from .mm import is_dataset, load_dataset
        if not is_dataset(path):
            raise ValueError(f"{path} is not a Micro-Manager acquisition folder")
        img, _ = load_dataset(path, flat)
        if pixel_um is not None:
            img.pixel_um, img.pixel_um_source = float(pixel_um), "command line"
        return img

    meta: dict = {}
    vsi = path if path.suffix.lower() == ".vsi" else path.with_suffix(".vsi")
    if vsi.exists():
        try:
            meta.update(parse_vsi_metadata(vsi))
            meta["vsi_file"] = str(vsi)
        except Exception as exc:  # metadata is optional
            meta["vsi_error"] = str(exc)

    max_value = 255
    if path.suffix.lower() == ".vsi":
        ets = find_ets(path)
        rgb, max_value = read_ets(ets)
        meta["ets_file"] = str(ets)
        meta["bit_depth"] = int(np.log2(max_value + 1))
    else:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # an export next to its .vsi may be cropped: locate it in the VSI frame
        # so that the stage origin refers to this image's top-left pixel
        if "origin_x_um" in meta:
            try:
                ets = find_ets(vsi)
                dx, dy = export_offset(rgb, ets)
                meta["export_offset_px"] = [dx, dy]
                px = meta.get("pixel_um", DEFAULT_PIXEL_UM)
                meta["origin_x_um"] += dx * px
                meta["origin_y_um"] += dy * px
            except RuntimeError:
                meta["origin_note"] = ("VSI origin used as-is; the export may be cropped relative to "
                                       "the VSI frame (full-resolution data not available to check)")

    if pixel_um is not None:
        src = "command line"
    elif "pixel_um" in meta:
        pixel_um, src = meta["pixel_um"], f"VSI metadata ({vsi.name})"
    else:
        pixel_um, src = DEFAULT_PIXEL_UM, "default (BX51 6.3x)"
    return ImageData(rgb=rgb, path=path, pixel_um=float(pixel_um), pixel_um_source=src, metadata=meta,
                     max_value=max_value)
