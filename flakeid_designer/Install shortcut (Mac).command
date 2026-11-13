#!/bin/bash
# vdW Stack Designer on this Mac: everything it needs on a fresh macOS (11 or newer), then its app icon ("FlakeIdentifier
# Designer" in Applications and on the Desktop).  Double-click it; safe to run again (it then only checks, updates the
# packages and makes the icon again).
#   1. Python: one already on this Mac (3.10 - 3.13: python.org's, Homebrew's), else Python 3.12.10 from python.org
#      (checked against its known SHA-256; macOS asks for your password to install it).
#   2. A Python environment of the designer's own (~/.flakeid_designer_venv): pip and the packages the designer's builds
#      and tools need (requirements.txt beside the designer, and certifi for Update...).
#   3. The app icon, started with that environment.
# Log: ~/.flakeid_designer_install_log.txt.  For tests: FLAKEID_VENV (another environment folder),
# FLAKEID_INSTALL_NO_SHORTCUT=1 (no app icon).
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
PYVER=3.12.10
PYURL="https://www.python.org/ftp/python/$PYVER/python-$PYVER-macos11.pkg"
PYSHA=8373e58da4ea146b3eb1c1f9834f19a319440b6b679b06050b1f9ee3237aa8e4
OKV="3.12 3.13 3.11 3.10"                  # (in this order of preference)
VENV="${FLAKEID_VENV:-$HOME/.flakeid_designer_venv}"
LOG="$HOME/.flakeid_designer_install_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "==== $(date '+%Y-%m-%d %H:%M:%S') install"

fail() {
  echo
  echo "NOT INSTALLED: $1"
  echo "(the log: $LOG)"
  [ -t 0 ] && read -r -p "Press Return to close this window. " _
  exit 1
}
pyver() {   # 'major.minor' of a Python, or nothing.  (/usr/bin/python3 only with Apple's command line tools: without
            # them it opens a dialog to install them; it is 3.9 anyway)
  case "$1" in /usr/bin/python3) xcode-select -p >/dev/null 2>&1 || return 0 ;; esac
  [ -x "$1" ] && "$1" -c 'import sys;print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}
find_python() {
  local want c v
  for want in $OKV; do
    for c in "/Library/Frameworks/Python.framework/Versions/$want/bin/python$want" "/opt/homebrew/bin/python$want" \
             "/usr/local/bin/python$want" "$(command -v "python$want" 2>/dev/null)" /opt/homebrew/bin/python3 \
             /usr/local/bin/python3 "$(command -v python3 2>/dev/null)"; do
      [ -n "$c" ] || continue
      v="$(pyver "$c")"
      [ "$v" = "$want" ] && { echo "$c"; return 0; }
    done
  done
  return 1
}

echo "vdW Stack Designer: installing what it needs (a few minutes the first time)"
echo "  the designer: $HERE"
[ -f "$HERE/__main__.py" ] || fail "this file must stay in the designer's folder (flakeid_designer), next to __main__.py"

# 1. Python
PY="$(find_python)"
if [ -n "$PY" ]; then
  echo "1. Python: $PY ($(pyver "$PY"))"
else
  echo "1. Python: none usable on this Mac; downloading Python $PYVER from python.org ..."
  PKG="$(mktemp -d)/python-$PYVER-macos11.pkg"
  curl -fL --progress-bar -o "$PKG" "$PYURL" || fail "the download failed; is this Mac online?"
  H="$(shasum -a 256 "$PKG" | cut -d' ' -f1)"
  [ "$H" = "$PYSHA" ] || { rm -f "$PKG"; fail "the downloaded Python installer is not the expected file (SHA-256 $H)"; }
  echo "   installing Python $PYVER (macOS asks for your password: an administrator's) ..."
  osascript -e "do shell script \"installer -pkg \" & quoted form of \"$PKG\" & \" -target /\" with administrator privileges" \
    || { rm -f "$PKG"; fail "Python was not installed (no administrator password given?)"; }
  rm -f "$PKG"
  PY="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
  [ -n "$(pyver "$PY")" ] || fail "Python was installed but cannot be found at $PY"
  echo "   installed: $PY"
fi

# 2. the designer's own environment, pip and the packages
VPY="$VENV/bin/python"
V="$(pyver "$VPY")"
case " $OKV " in *" $V "*) ;; *) V="" ;; esac
if [ -z "$V" ]; then
  rm -rf "$VENV"                                # (missing, broken e.g. its Python removed, or a Python not used any more)
  echo "2. the designer's Python environment: $VENV"
  "$PY" -m venv "$VENV" && [ -n "$(pyver "$VPY")" ] || fail "the environment could not be made ($PY -m venv)"
else
  echo "2. the designer's Python environment: $VENV (there already)"
fi
echo "   pip ..."
"$VPY" -m pip install --disable-pip-version-check --upgrade pip || fail "pip could not be updated (is this Mac online?)"
echo "   the packages (numpy, scipy, OpenCV, scikit-image, ...) ..."
if [ -f "$HERE/../requirements.txt" ]; then
  "$VPY" -m pip install --disable-pip-version-check --upgrade -r "$HERE/../requirements.txt" certifi
else
  "$VPY" -m pip install --disable-pip-version-check --upgrade numpy scipy opencv-python scikit-image scikit-learn pillow \
    tifffile matplotlib gdstk certifi
fi || fail "the packages could not be installed (see above)"
"$VPY" -c "import numpy, scipy, cv2, skimage, sklearn, PIL, tifffile, matplotlib, gdstk, certifi; print('   packages: all import (numpy ' + numpy.__version__ + ', OpenCV ' + cv2.__version__ + ')')" \
  || fail "the packages were installed but do not all import (see above)"

# 3. the app icon, started with that environment
if [ "${FLAKEID_INSTALL_NO_SHORTCUT:-}" = "1" ]; then
  echo "3. the app icon: skipped (FLAKEID_INSTALL_NO_SHORTCUT)"
else
  echo "3. the app icon ..."
  "$VPY" "$HERE" --install-shortcut || fail "the app icon could not be made (see above)"
fi
echo
echo "Done: start the designer from \"FlakeIdentifier Designer\" in Applications or on the Desktop."
exit 0
