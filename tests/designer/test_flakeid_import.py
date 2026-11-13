"""Started as `python flakeid_designer` (how an install runs, on Update too) only that folder is on the path: the flakeid
package beside it must still be importable, or the Thickness cursor's server call dies (ModuleNotFoundError)."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROBE = """
import runpy, sys
sys.argv = ["flakeid_designer", "--help"]
try:
    runpy.run_path({d!r}, run_name="__main__")
except SystemExit:
    pass
"""


def _run(tmp_path, tail):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["HOME"] = str(tmp_path)
    code = PROBE.format(d=str(REPO / "flakeid_designer")) + tail
    return subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60)


def test_flakeid_is_found_as_an_install_starts(tmp_path):
    r = _run(tmp_path, "\nimport flakeid, flakeid.thick100\nprint(flakeid.__file__)")
    assert r.returncode == 0, r.stderr
    assert str(REPO / "flakeid") in r.stdout
