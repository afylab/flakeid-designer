"""Check a FlakeIdentifier install on the microscope PC: versions, the library calls the pipeline relies on, and a
quick speed test.

    C:\\FlakeID\\python38\\python.exe tools\\win7\\check_env.py

Prints OK / FAILED per item and a rough analysis speed relative to the analysis Mac (M-series, 2026), so it is clear
whether the PC can keep up with a scan before relying on it.
"""
import platform
import sys
import time
import traceback

ok = True


def item(name, fn):
    global ok
    t0 = time.time()
    try:
        r = fn()
        print(f"  OK      {name}{(': ' + str(r)) if r is not None else ''}  ({time.time() - t0:.1f} s)")
    except Exception:
        ok = False
        print(f"  FAILED  {name}")
        traceback.print_exc()


print(f"Python {sys.version.split()[0]} ({platform.architecture()[0]}) on {platform.platform()}")
print(f"  executable {sys.executable}")
if any("anaconda" in p.lower() for p in sys.path):
    print("  WARNING: an Anaconda folder is on sys.path; run through run_flakeid.bat, which isolates this Python")

import numpy as np  # noqa: E402

item("numpy", lambda: np.__version__)
item("scipy", lambda: __import__("scipy").__version__)
item("opencv", lambda: __import__("cv2").__version__)
item("scikit-image", lambda: __import__("skimage").__version__)
item("scikit-learn", lambda: __import__("sklearn").__version__)
item("tifffile", lambda: __import__("tifffile").__version__)


def calls():
    import cv2
    from scipy import ndimage
    from skimage.morphology import remove_small_holes, skeletonize
    from skimage.segmentation import watershed
    rng = np.random.default_rng(0)
    a = (rng.random((400, 400, 3)) * 0.05).astype(np.float32)
    a[100:300, 120:260] += 0.6
    b = cv2.bilateralFilter(a, d=5, sigmaColor=0.08, sigmaSpace=1.5)
    m = np.linalg.norm(b, axis=2) > 0.3
    m = remove_small_holes(m, 25)
    lab, n = ndimage.label(m)
    sk = skeletonize(m)
    w = watershed(cv2.GaussianBlur(b[..., 0], (0, 0), 1), lab)
    bay = cv2.cvtColor((rng.random((64, 64)) * 4000).astype(np.uint16), cv2.COLOR_BayerRG2RGB)
    assert n == 1 and sk.any() and w.max() == 1 and bay.shape == (64, 64, 3)
    return "bilateral / label / skeleton / watershed / demosaic"


item("library calls used by the pipeline", calls)


def speed():
    import cv2
    rng = np.random.default_rng(1)
    a = (rng.random((2000, 2000, 3))).astype(np.float32)
    t0 = time.time()
    for _ in range(3):
        cv2.bilateralFilter(a, d=5, sigmaColor=0.08, sigmaSpace=1.5)
        cv2.GaussianBlur(a, (0, 0), 6)
        np.linalg.norm(a, axis=2)
    dt = (time.time() - t0) / 3
    ref = 0.14                                  # the analysis Mac (Python 3.8, these versions), same test
    return f"{dt:.2f} s per pass, ~{dt / ref:.1f}x the analysis Mac's time"


item("speed", speed)
sys.path.insert(0, __file__.rsplit("tools", 1)[0])
item("flakeid package", lambda: (__import__("flakeid.pipeline"), __import__("flakeid.live"), __import__("flakeid.hundred"),
                                 __import__("flakeid.screen100"), "imports")[-1])
print("ALL OK" if ok else "SOME CHECKS FAILED (see above)")
sys.exit(0 if ok else 1)
