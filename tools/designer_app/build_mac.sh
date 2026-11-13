#!/bin/sh
# Builds the designer app for Apple-silicon Macs and a disk image to hand out:
#   sh tools/designer_app/build_mac.sh            office version  -> dist_app/vdW Stack Designer <version>.dmg
#   sh tools/designer_app/build_mac.sh stacker    with the live camera (OpenCV; on a Mac: system and simulated cameras)
set -e
cd "$(dirname "$0")/../.."
if [ "$1" = "stacker" ]; then
  VENV=.venv-stacker; NAME="vdW Stack Designer (Stacker)"; PKGS="pywebview pyinstaller opencv-python-headless numpy scipy scikit-image gdstk"
  EXTRA="--hidden-import flakeid_designer.camera --hidden-import flakeid_designer.marks_to_klayout --hidden-import cv2 --hidden-import numpy --hidden-import gdstk"
else
  VENV=.venv-app; NAME="vdW Stack Designer"; PKGS="pywebview pyinstaller opencv-python-headless numpy scipy scikit-image gdstk"
  EXTRA="--hidden-import flakeid_designer.camera --hidden-import flakeid_designer.marks_to_klayout --hidden-import gdstk --exclude-module imagingcontrol4"
fi
# the builder of runs' designer files (tools/make_designer_bundle.py and the flakeid modules it uses, with its calibrations)
BUILDER="--paths $PWD/tools --hidden-import make_designer_bundle --hidden-import make_catalog_100x --hidden-import make_afm_set --hidden-import make_catalog --hidden-import make_catalog_live --hidden-import make_stack_bundle --hidden-import terraces100 --hidden-import flakeid.background --hidden-import flakeid.branch --hidden-import flakeid.flakes --hidden-import flakeid.graphite --hidden-import flakeid.hundred --hidden-import flakeid.io --hidden-import flakeid.mm --hidden-import flakeid.optics --hidden-import flakeid.outline --hidden-import flakeid.pipeline --hidden-import flakeid.scorer --hidden-import flakeid.segment --hidden-import flakeid.targets --hidden-import flakeid.thickness --hidden-import flakeid.screen100 --hidden-import flakeid.progress --hidden-import flakeid.afm --exclude-module sklearn --add-data $PWD/flakeid/profiles:flakeid/profiles --exclude-module matplotlib --exclude-module flakeid.scope"
PY=$VENV/bin/python
if [ ! -x "$PY" ]; then
  python3 -m venv $VENV
  $VENV/bin/pip install --no-cache-dir -q --upgrade pip
fi
$VENV/bin/pip install --no-cache-dir -q $PKGS          # (every build: packages added since the environment was made)
rm -rf build_app "dist_app/$NAME.app" "dist_app/$NAME"
"$PY" -m PyInstaller --noconfirm --windowed --name "$NAME" \
  --icon "$PWD/flakeid_designer/static/designer.icns" \
  --osx-bundle-identifier "edu.ucsb.younglab.vdwdesigner$( [ "$1" = stacker ] && echo .stacker)" \
  --add-data "$PWD/flakeid_designer/index.html:flakeid_designer" \
  --add-data "$PWD/flakeid_designer/static:flakeid_designer/static" \
  --add-data "$PWD/flakeid_designer/README.txt:flakeid_designer" \
  --hidden-import flakeid_designer.__main__ --hidden-import flakeid_designer.stacks --hidden-import flakeid_designer.manual --hidden-import flakeid_designer.afmplan --hidden-import flakeid_designer.stackdir --hidden-import flakeid_designer.kept $EXTRA $BUILDER --paths "$PWD" \
  --workpath build_app --distpath dist_app --specpath build_app \
  "$PWD/flakeid_designer/app.py"
VER=$(grep '^VERSION = ' flakeid_designer/__main__.py | sed 's/.*"\(.*\)".*/\1/')
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VER" "dist_app/$NAME.app/Contents/Info.plist" || true
rm -rf dist_app/dmg && mkdir -p dist_app/dmg && cp -R "dist_app/$NAME.app" dist_app/dmg/ && ln -s /Applications dist_app/dmg/Applications
hdiutil create -volname "$NAME" -srcfolder dist_app/dmg -ov -format UDZO "dist_app/$NAME $VER.dmg" >/dev/null
rm -rf dist_app/dmg
echo "built: dist_app/$NAME.app and dist_app/$NAME $VER.dmg"
