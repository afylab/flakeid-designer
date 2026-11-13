"""Make the scan program's window theme: Sun Valley (sv-ttk, MIT licence) with its blue accent turned into the violet
of the FlakeIdentifier icon.  Writes flakeid/scope/theme_sv/ (loaded by flakeid/scope/ui.py: apply_theme).

    python tools/make_purple_theme.py            (needs sv-ttk installed: pip install sv-ttk==2.6.0)

Every blue in the theme's sprite sheets and colour settings (hue 180-235 deg, some saturation) gets the icon's hue;
saturation is scaled a little so the accent sits with the icon's muted violet (#6c5cb4).
"""
import colorsys
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_HUE = 252 / 360            # the icon's oxide violet (#6c5cb4: 251 deg)
SAT = {"dark": 0.80, "light": 0.88}
OUT = Path(__file__).resolve().parents[1] / "flakeid" / "scope" / "theme_sv"


def shift_rgb(rgb, sat):
    """(N, 3) uint8 -> the same with blue hues moved to violet."""
    a = rgb.astype(np.float64) / 255.0
    mx, mn = a.max(1), a.min(1)
    d = mx - mn
    h = np.zeros(len(a))
    nz = d > 1e-9
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    m = nz & (mx == r); h[m] = ((g[m] - b[m]) / d[m]) % 6
    m = nz & (mx == g); h[m] = (b[m] - r[m]) / d[m] + 2
    m = nz & (mx == b); h[m] = (r[m] - g[m]) / d[m] + 4
    h = h / 6.0
    s = np.where(mx > 0, d / np.maximum(mx, 1e-9), 0)
    v = mx
    sel = (h >= 180 / 360) & (h <= 235 / 360) & (s > 0.15)
    out = rgb.copy()
    for i in np.nonzero(sel)[0]:
        rr, gg, bb = colorsys.hsv_to_rgb(TARGET_HUE, s[i] * sat, v[i])
        out[i] = np.round(np.array([rr, gg, bb]) * 255)
    return out


def hexshift(m, sat):
    c = m.group(0)
    rgb = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)]], np.uint8)
    o = shift_rgb(rgb, sat)[0]
    return "#%02x%02x%02x" % tuple(int(x) for x in o)


def main():
    import sv_ttk
    src = Path(sv_ttk.__file__).parent
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "theme").mkdir(parents=True)
    shutil.copy(src / "sv.tcl", OUT / "sv.tcl")
    lic = next(src.parent.glob("sv_ttk-*.dist-info/LICENSE"), None)
    if lic:
        shutil.copy(lic, OUT / "LICENSE_sv_ttk.txt")
    for mode in ("dark", "light"):
        for f in (f"{mode}.tcl", f"sprites_{mode}.tcl"):
            t = (src / "theme" / f).read_text()
            t = re.sub(r"#[0-9a-fA-F]{6}\b", lambda m: hexshift(m, SAT[mode]), t)
            (OUT / "theme" / f).write_text(t)
        im = np.array(Image.open(src / "theme" / f"spritesheet_{mode}.png").convert("RGBA"))
        flat = im[..., :3].reshape(-1, 3)
        im[..., :3] = shift_rgb(flat, SAT[mode]).reshape(im.shape[:2] + (3,))
        Image.fromarray(im).save(OUT / "theme" / f"spritesheet_{mode}.png", optimize=True)
    (OUT / "README.txt").write_text("Sun Valley ttk theme (sv-ttk 2.6.0, MIT licence: LICENSE_sv_ttk.txt) with the accent recoloured "
                                    "to the FlakeIdentifier violet by tools/make_purple_theme.py.\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
