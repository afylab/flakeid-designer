"""The look of the pictures people see (Avram, 2026-10-27): the substrate at the colour of a reference picture (hBN on
285 nm SiO2 at 100x, local_imgs/example_BN_285nm/1_3_dirty.jpg: substrate (176, 122, 148)), bright colours rolled off
without changing their hue.

Only for copies that are looked at (<stem>_look.jpg next to the 100x pictures, the viewer's tiles/look layer, the
designer's natural view): the analysis keeps reading the original, linear pictures (the 100x JPEGs and the viewer's
tiles/image are read back by terraces100, defects100, outline100, the designer's catalog, the IR alignment and the
re-centring), so no thickness or other measurement changes.

The filter, from a picture whose substrate colour is known in its own units:
1. a gain per channel that takes the substrate to LOOK_SUBSTRATE (a white balance; on the scope's 100x hBN pictures it
   is ~1.6 in every channel: mostly brightness);
2. a soft shoulder on the largest channel above LOOK_KNEE, applied to all three channels alike (the hue is kept): a bright
   flake becomes a bright saturated colour instead of being clipped to white or cyan.
"""
from __future__ import annotations

import numpy as np

LOOK_SUBSTRATE = (176.0, 122.0, 148.0)      # 8-bit RGB of the substrate in the reference picture
LOOK_KNEE = 0.80                            # the shoulder starts here (fraction of white)
CANVAS_SUBSTRATE = 0.31 * 255.0             # the substrate in the 10x canvas / viewer tiles/image (grey, livecanvas g8)


def look(img, substrate, bgr=False):
    """img (H, W, 3; any units, RGB or BGR with bgr=True) with its substrate colour (RGB, same units) -> uint8 picture in
    the same channel order with the reference look."""
    sub = np.maximum(np.asarray(substrate, np.float32).reshape(3), 1e-6)
    g = np.asarray(LOOK_SUBSTRATE, np.float32) / 255.0 / sub
    if bgr:
        g = g[::-1]
    x = np.asarray(img, np.float32) * g
    m = np.maximum(x.max(axis=2, keepdims=True), 1e-6)
    k = LOOK_KNEE
    mm = np.where(m > k, k + (1.0 - k) * np.tanh((m - k) / (1.0 - k)), m)
    return np.clip(x * (mm / m) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def look_tile(img_bgr):
    """A viewer tile (8-bit BGR, the canvas's grey substrate CANVAS_SUBSTRATE) with the look."""
    return look(img_bgr, (CANVAS_SUBSTRATE,) * 3, bgr=True)
