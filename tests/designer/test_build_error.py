"""A failed build shows its whole error (flakeid_designer Builder: detail, log, last_build_error.txt; build_error)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import flakeid_designer.__main__ as D  # noqa: E402

NUMPY_ERR = """Traceback (most recent call last):
  File "numpy/__init__.py", line 114, in <module>
    from numpy.__config__ import show_config
ImportError: dlopen(_multiarray_umath.so): incompatible architecture

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "tools/make_designer_bundle.py", line 30, in <module>
ImportError: Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""


def test_build_error_keeps_the_whole_message():
    e = D.build_error(NUMPY_ERR.splitlines())
    assert e.startswith("ImportError: Error importing numpy") and e.endswith("from there.")
    assert D.build_error(["PROGRESS 0.1 x", "just a line"]) == "just a line"


def test_failed_build_detail(tmp_path, monkeypatch):
    fake = tmp_path / "builder"
    (fake / "flakeid").mkdir(parents=True)
    (fake / "flakeid" / "__init__.py").write_text("")
    (fake / "tools").mkdir()
    (fake / "tools" / "make_designer_bundle.py").write_text(
        "import sys\nprint('PROGRESS 0.1 reading')\nprint(" + repr(NUMPY_ERR) + ")\nsys.exit(1)\n")
    monkeypatch.setenv("FLAKEID_BUILDER", str(fake))
    monkeypatch.setattr(D, "HERE", tmp_path / "src" / "flakeid_designer")
    (tmp_path / "src" / "flakeid_designer").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    run = tmp_path / "run_13"
    run.mkdir()
    b = D.Builder()
    b.want(run)
    for _ in range(200):
        st = {x["dir"]: x for x in b.status()}.get(str(run), {})
        if st.get("state") == "failed":
            break
        time.sleep(0.05)
    assert st["state"] == "failed"
    assert "Error importing numpy" in st["msg"] and "incompatible architecture" in st["detail"]
    assert Path(st["log"]).is_file()
    assert "incompatible architecture" in (tmp_path / "src" / "last_build_error.txt").read_text()
