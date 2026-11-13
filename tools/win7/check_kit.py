"""Check an install kit's wheels folder resolves completely for Windows 64-bit, Python 3.8 (run on the Mac).

    python3 tools/win7/check_kit.py scope_pc_kit/wheels tools/win7/requirements-py38.txt tools/win7/requirements-mm-py38.txt

pip download on the Mac evaluates "Windows only" markers for the Mac and leaves those dependencies out (colorama for
click, 2026-10-03); this walks every requirement with the Windows / Python 3.8 environment instead.
"""
import glob
import os
import re
import sys
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ENV = dict(default_environment(), platform_system="Windows", sys_platform="win32", os_name="nt", platform_machine="AMD64",
           python_version="3.8", python_full_version="3.8.10", implementation_name="cpython",
           platform_python_implementation="CPython", extra="")
wheels = {}
for w in glob.glob(sys.argv[1] + "/*.whl"):
    name, ver = os.path.basename(w).split("-")[:2]
    z = zipfile.ZipFile(w)
    meta = z.read([n for n in z.namelist() if n.endswith(".dist-info/METADATA")][0]).decode("utf8", "replace")
    reqs = [l[len("Requires-Dist:"):].strip() for l in meta.splitlines() if l.startswith("Requires-Dist:")]
    wheels[canonicalize_name(name)] = (Version(ver), reqs)
todo = [Requirement(l.split("#")[0].strip()) for f in sys.argv[2:] for l in open(f) if l.split("#")[0].strip()]
seen, missing = set(), []
while todo:
    r = todo.pop()
    if r.marker is not None and not r.marker.evaluate(ENV):
        continue
    n = canonicalize_name(r.name)
    if n not in wheels:
        missing.append(str(r))
        continue
    v, reqs = wheels[n]
    if r.specifier and v not in r.specifier:
        missing.append(f"{r} (kit has {v})")
    if n in seen:
        continue
    seen.add(n)
    todo += [Requirement(q) for q in reqs]
print(f"{len(seen)} packages resolved for Windows / Python 3.8")
print("MISSING: " + "; ".join(sorted(set(missing))) if missing else "nothing missing")
sys.exit(1 if missing else 0)
