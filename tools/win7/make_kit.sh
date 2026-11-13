#!/bin/sh
# Build the offline install kit for the microscope PC (Windows 7 SP1, 64-bit).
#
#   sh tools/win7/make_kit.sh [OUT]          (default OUT: ./scope_pc_kit)
#   PYTHON=.../python.exe NO_FLATS=1 sh tools/win7/make_kit.sh    (Windows Git Bash: a real Python; an update kit
#                                             without flat fields: setup's xcopy keeps the ones already on the PC)
#
# Copy OUT to the PC (USB stick or the share) and run setup_scope_pc.bat there.  Contents: the Python 3.8.10
# installer from python.org, Windows wheels of requirements-py38.txt for Python 3.8, and the code with the flat fields.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/scope_pc_kit}
PY=${PYTHON:-python3}
mkdir -p "$OUT/wheels" "$OUT/FlakeIdentifier/flats"

[ -f "$OUT/python-3.8.10-amd64.exe" ] || curl -fL -o "$OUT/python-3.8.10-amd64.exe" \
    https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe
"$PY" -m pip download --only-binary=:all: --platform win_amd64 --python-version 38 --implementation cp --abi cp38 \
    -d "$OUT/wheels" -r "$ROOT/tools/win7/requirements-py38.txt" -r "$ROOT/tools/win7/requirements-mm-py38.txt"

# code: the package, the scan script and tools (not the 2.5 GB Micro-Manager mock), the flat fields
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude __pycache__ "$ROOT/flakeid" "$OUT/FlakeIdentifier/"
    rsync -a --delete --exclude __pycache__ --exclude mm_mock "$ROOT/tools" "$OUT/FlakeIdentifier/"
else   # Git Bash on Windows has no rsync
    rm -rf "$OUT/FlakeIdentifier/flakeid" "$OUT/FlakeIdentifier/tools"
    cp -Rp "$ROOT/flakeid" "$ROOT/tools" "$OUT/FlakeIdentifier/"
    rm -rf "$OUT/FlakeIdentifier/tools/mm_mock"
    find "$OUT/FlakeIdentifier" -name __pycache__ -type d -prune -exec rm -rf {} +
fi
cp -p "$ROOT/find_flakes.py" "$ROOT/README.md" "$OUT/FlakeIdentifier/"
mkdir -p "$OUT/FlakeIdentifier/docs" && cp -p "$ROOT/docs/manual.md" "$ROOT/docs/scan_guide.md" "$ROOT/docs/scan_guide.pdf" "$OUT/FlakeIdentifier/docs/" && cp -Rp "$ROOT/docs/scan_guide_images" "$OUT/FlakeIdentifier/docs/"
# flat fields: instrument calibration, not in git (see flats/README.md); flat_10x_3ms.npz, flat_100x_285nm.npz, flat_100x_90nm.npz
if ls "$ROOT"/flats/*.npz >/dev/null 2>&1; then
    cp -p "$ROOT"/flats/*.npz "$OUT/FlakeIdentifier/flats/"
elif [ -n "$NO_FLATS" ]; then
    echo "NO_FLATS: kit without flat fields (only for a PC that already has them)"
else
    echo "no flat fields in $ROOT/flats (see flats/README.md); NO_FLATS=1 for an update kit"; exit 1
fi
cp -p "$ROOT/tools/win7/setup_scope_pc.bat" "$OUT/"
# every dependency present for Windows / Python 3.8 (pip on the Mac leaves out "Windows only" ones)
"$PY" "$ROOT/tools/win7/check_kit.py" "$OUT/wheels" "$ROOT/tools/win7/requirements-py38.txt" "$ROOT/tools/win7/requirements-mm-py38.txt"
echo "kit: $OUT ($(du -sh "$OUT" | cut -f1))"
