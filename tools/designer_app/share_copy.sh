#!/bin/sh
# The designer for everyone on the share (the browser version: "Install shortcut" there), with the builder of runs'
# designer files beside it (flakeid_designer/builder/: the flakeid package and the tools it uses), which the designer
# runs on the user's computer when a new run arrives.
#   sh tools/designer_app/share_copy.sh "/Volumes/<share>/<your name>/flakeid_designer"   (or set FLAKEID_DESIGNER_SHARE)
set -e
cd "$(dirname "$0")/../.."
DEST=${1:-${FLAKEID_DESIGNER_SHARE:-}}
[ -n "$DEST" ] || { echo "give the designer folder on the share (or set FLAKEID_DESIGNER_SHARE)"; exit 1; }
[ -d "$DEST" ] || { echo "no folder $DEST (is the share mounted?)"; exit 1; }
for f in index.html __main__.py __init__.py camera.py marks_to_klayout.py stacks.py manual.py afmplan.py stackdir.py offline.py kept.py app.py README.txt; do
  [ -f "flakeid_designer/$f" ] && rsync -t "flakeid_designer/$f" "$DEST/"     # (only what changed: the share is slow)
done
rsync -a --exclude __pycache__ flakeid_designer/static/ "$DEST/static/"
mkdir -p "$DEST/builder/tools"
rsync -a --delete --exclude __pycache__ --exclude scope flakeid/ "$DEST/builder/flakeid/"
for m in make_designer_bundle make_catalog_100x make_afm_set make_catalog make_catalog_live make_stack_bundle terraces100; do
  rsync -t "tools/$m.py" "$DEST/builder/tools/"
done
echo "designer and builder copied to $DEST"
