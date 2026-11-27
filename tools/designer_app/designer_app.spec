# -*- mode: python ; coding: utf-8 -*-
# The packaged designer as a "thin app" (2026-11-21, tools/designer_app/launcher.py): Python and the libraries in the
# app; the designer's own code (flakeid_designer/, flakeid/, tools/) as plain files (SEED), copied to a folder this
# account can write and run from there, so the designer's Update... can replace it.
#
# The analysis starts from flakeid_designer/app.py with the designer's modules as hidden imports, as before, so every
# library they use is collected; the designer's own modules are then taken out of the archive (Python reads the
# archive first: left in, they would hide the updated files).
#   ../../.venv-app/bin/python -m PyInstaller --noconfirm --workpath build_app --distpath dist_app tools/designer_app/designer_app.spec
# Environment: DESIGNER_NAME (the app's name), DESIGNER_STACKER=1 (the live camera's modules too),
# DESIGNER_VERSION_JSON (installed_version.json of the code bundled: written by build_mac.sh).
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent.parent
NAME = os.environ.get("DESIGNER_NAME", "vdW Stack Designer")
STACKER = os.environ.get("DESIGNER_STACKER") == "1"
VJSON = os.environ.get("DESIGNER_VERSION_JSON")
TOOLS_MODS = ["make_designer_bundle", "make_catalog_100x", "make_afm_set", "make_catalog", "make_catalog_live",
              "make_stack_bundle", "terraces100"]
OURS = ("flakeid_designer", "flakeid", "update", "offline", "app") + tuple(TOOLS_MODS)
OWN_TOOLS = {p.stem for p in (ROOT / "tools").glob("*.py")}

hidden = ["flakeid_designer.__main__", "flakeid_designer.stacks", "flakeid_designer.manual", "flakeid_designer.afmplan",
          "flakeid_designer.stackdir", "flakeid_designer.kept", "flakeid_designer.camera", "flakeid_designer.marks_to_klayout",
          "flakeid_designer.update", "flakeid_designer.offline", "flakeid_designer.sharegate", "flakeid_designer.app", "gdstk",
          "flakeid.background", "flakeid.branch", "flakeid.flakes", "flakeid.graphite", "flakeid.hundred", "flakeid.io",
          "flakeid.mm", "flakeid.optics", "flakeid.outline", "flakeid.pipeline", "flakeid.scorer", "flakeid.segment",
          "flakeid.targets", "flakeid.thickness", "flakeid.screen100", "flakeid.progress", "flakeid.afm",
          "webview", "multiprocessing", "concurrent.futures"] + TOOLS_MODS
if STACKER:
    hidden += ["cv2", "numpy"]

# the code as plain files (the seed): flakeid_designer/ (with its page), flakeid/ (with its calibrations), tools/
def _tree(sub):
    out = []
    for p in sorted((ROOT / sub).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",) and "scope" not in p.relative_to(ROOT / sub).parts[:1]:
            out.append((str(p), str(Path("designer_code") / p.parent.relative_to(ROOT))))
    return out

seed = _tree("flakeid_designer") + _tree("flakeid") + _tree("tools")
for f in ("requirements.txt", "README.md", "LICENSE"):
    if (ROOT / f).is_file():
        seed.append((str(ROOT / f), "designer_code"))
if VJSON:
    seed.append((VJSON, "designer_code"))

a = Analysis(
    [str(ROOT / "tools" / "designer_app" / "launcher.py"), str(ROOT / "flakeid_designer" / "app.py")],
    pathex=[str(ROOT / "tools"), str(ROOT)],
    binaries=[],
    datas=seed,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["sklearn", "matplotlib", "flakeid.scope"] + ([] if STACKER else ["imagingcontrol4"]),
    noarchive=False,
    optimize=0,
)


def ours(modname):
    top = modname.split(".")[0]
    return top in OURS or top in OWN_TOOLS


a.pure = [x for x in a.pure if not ours(x[0])]
# (only the launcher runs: app.py was given to the analysis for its imports)
a.scripts = [s for s in a.scripts if Path(s[1]).name != "app.py"]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(ROOT / "flakeid_designer" / "static" / ("designer.icns" if sys.platform == "darwin" else "designer.ico"))],
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, upx_exclude=[], name=NAME)
if sys.platform == "darwin":
    app = BUNDLE(coll, name=NAME + ".app",
                 icon=str(ROOT / "flakeid_designer" / "static" / "designer.icns"),
                 bundle_identifier="edu.ucsb.younglab.vdwdesigner" + (".stacker" if STACKER else ""))
