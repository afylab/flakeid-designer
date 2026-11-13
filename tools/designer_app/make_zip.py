"""Zip the built app folder and Install.bat for handing out (Windows build), and check the zip reads back.

    python make_zip.py <out.zip> <app folder> <Install.bat>
"""
import sys
import zipfile
from pathlib import Path

out, folder, inst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if out.exists():
    out.unlink()
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:     # (pre-1980 dates: clamped)
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            z.write(p, Path(folder.name) / p.relative_to(folder))
            n += 1
    z.write(inst, inst.name)
with zipfile.ZipFile(out) as z:
    bad = z.testzip()
if bad:
    sys.exit(f"the zip is damaged at {bad}")
print(f"{out}: {n + 1} files, {out.stat().st_size / 2**20:.0f} MB, checked")
