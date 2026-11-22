#!/bin/sh
# Builds the designer app for Apple-silicon Macs and a disk image to hand out:
#   sh tools/designer_app/build_mac.sh            office version  -> dist_app/vdW Stack Designer <version>.dmg
#   sh tools/designer_app/build_mac.sh stacker    with the live camera (OpenCV; on a Mac: system and simulated cameras)
set -e
cd "$(dirname "$0")/../.."
if [ "$1" = "stacker" ]; then
  VENV=.venv-stacker; NAME="vdW Stack Designer (Stacker)"; STACKER=1
else
  VENV=.venv-app; NAME="vdW Stack Designer"; STACKER=0
fi
PKGS="pywebview pyinstaller opencv-python-headless numpy scipy scikit-image gdstk"
PY=$VENV/bin/python
if [ ! -x "$PY" ]; then
  python3 -m venv $VENV
  $VENV/bin/pip install --no-cache-dir -q --upgrade pip
fi
$VENV/bin/pip install --no-cache-dir -q $PKGS          # (every build: packages added since the environment was made)
rm -rf build_app "dist_app/$NAME.app" "dist_app/$NAME"
mkdir -p build_app
# the version of the code the app carries (tools/designer_app/launcher.py: copied to the code folder when that has an
# older commit); the commit's date, as update.py writes it after an update
"$PY" - "$(git rev-parse HEAD)" "$(TZ=UTC git log -1 --date=format-local:%Y-%m-%dT%H:%M:%SZ --format=%cd)" "$(git log -1 --format=%s)" "$(git status --porcelain --untracked-files=no | grep -c . || true)" > build_app/installed_version.json <<'PYEOF'
import json, sys, time
sha, date, msg, dirty = sys.argv[1:5]
print(json.dumps(dict(sha=sha, date=date, message=msg, installed=time.strftime("%Y-%m-%d %H:%M:%S"), build="app",
                      changed=dirty != "0"), indent=1))
PYEOF
# a thin app (tools/designer_app/designer_app.spec): the designer's code runs from a folder Update... can write
DESIGNER_NAME="$NAME" DESIGNER_STACKER=$STACKER DESIGNER_VERSION_JSON="$PWD/build_app/installed_version.json" \
  "$PY" -m PyInstaller --noconfirm --workpath build_app --distpath dist_app tools/designer_app/designer_app.spec
VER=$(grep '^VERSION = ' flakeid_designer/__main__.py | sed 's/.*"\(.*\)".*/\1/')
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VER" "dist_app/$NAME.app/Contents/Info.plist" || true
# signed again (ad hoc) after the version is set: else macOS reports the app as modified ("invalid Info.plist")
codesign --force --deep --sign - "dist_app/$NAME.app" >/dev/null 2>&1 || true
rm -rf dist_app/dmg && mkdir -p dist_app/dmg && cp -R "dist_app/$NAME.app" dist_app/dmg/ && ln -s /Applications dist_app/dmg/Applications
hdiutil create -volname "$NAME" -srcfolder dist_app/dmg -ov -format UDZO "dist_app/$NAME $VER.dmg" >/dev/null
rm -rf dist_app/dmg
echo "built: dist_app/$NAME.app and dist_app/$NAME $VER.dmg"
