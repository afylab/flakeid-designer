"""Can IC Imaging Control 4 see and use The Imaging Source camera on this PC?  A check without the designer.

    python -m pip install imagingcontrol4 numpy
    python tools\\ic4_check.py                  (close the designer, IC Capture and anything else using the camera)

Prints each step (library version, the GenTL drivers IC4 found, the cameras, the camera's pixel formats, one picture)
and writes ic4_check.txt (and ic4_check.png, the picture) in the current folder: send ic4_check.txt.  No window opens.
"""
import os
import platform
import sys
import time
import traceback

OUT = []


def say(s):
    print(s)
    OUT.append(str(s))


def main():
    say(f"python {sys.version.split()[0]} {platform.platform()}")
    try:
        import imagingcontrol4 as ic4
    except Exception as e:
        say(f"FAIL: imagingcontrol4 does not import ({e}): python -m pip install imagingcontrol4")
        return
    say(f"imagingcontrol4 {getattr(ic4, '__version__', '?')}")
    try:
        ic4.Library.init(api_log_level=ic4.LogLevel.INFO, internal_log_level=ic4.LogLevel.WARNING,
                         log_targets=ic4.LogTarget.FILE, log_file=os.path.abspath("ic4_check_log.txt"))
    except Exception:
        try:
            ic4.Library.init()
        except Exception as e:
            say(f"FAIL: IC4 library init: {e}")
            return
    # the GenTL producers ("drivers") IC4 found: none = the IC4 driver for USB3 Vision cameras is not installed (the
    # older IC Imaging Control 3 / DirectShow driver alone is not enough for IC4)
    try:
        itfs = ic4.DeviceEnum.interfaces()
        say(f"interfaces (GenTL drivers): {len(itfs)}")
        for i in itfs:
            say("  " + ", ".join(f"{k}={getattr(i, k, '?')}" for k in
                                 ("display_name", "transport_layer_name", "transport_layer_type", "transport_layer_version")))
    except Exception as e:
        say(f"interfaces not listed ({e})")
    devs = ic4.DeviceEnum.devices()
    say(f"cameras: {len(devs)}")
    for d in devs:
        say("  " + ", ".join(f"{k}={getattr(d, k, '?')}" for k in ("model_name", "serial", "version", "user_id")))
    if not devs:
        say("FAIL: no camera.  If no interfaces either: install The Imaging Source's 'IC4 GenTL Driver for USB3Vision "
            "devices' (theimagingsource.com, Support > Downloads > IC Imaging Control 4); then unplug / plug the camera. "
            "If interfaces but no camera: another program has it, or USB (use a USB 3 port, the camera's own cable).")
        return
    g = ic4.Grabber()
    try:
        g.device_open(devs[0])
    except Exception as e:
        say(f"FAIL: opening the camera: {e} (in use by another program?)")
        return
    pm = g.device_property_map
    try:
        say(f"pixel format {pm.get_value_str(ic4.PropId.PIXEL_FORMAT)}; available: " +
            ", ".join(str(e.name) for e in pm.find_enumeration(ic4.PropId.PIXEL_FORMAT).entries))
    except Exception as e:
        say(f"pixel formats not readable ({e})")
    for p in ("WIDTH", "HEIGHT", "EXPOSURE_TIME", "EXPOSURE_AUTO", "TRIGGER_MODE", "ACQUISITION_FRAME_RATE"):
        try:
            say(f"  {p} = {pm.get_value_str(getattr(ic4.PropId, p))}")
        except Exception as e:
            say(f"  {p}: {e}")
    try:
        pm.try_set_value(ic4.PropId.TRIGGER_MODE, "Off")
    except Exception:
        pass
    try:
        sink = ic4.SnapSink(accepted_pixel_formats=[ic4.PixelFormat.BGR8])
    except Exception:
        sink = ic4.SnapSink()
    try:
        t = time.time()
        g.stream_setup(sink)
        say(f"stream started ({time.time() - t:.2f} s)")
        ts = []
        for k in range(5):
            t = time.time()
            buf = sink.snap_single(3000)
            ts.append(time.time() - t)
        a = buf.numpy_copy()
        say(f"5 pictures: {', '.join(f'{x:.3f}' for x in ts)} s; shape {a.shape} {a.dtype}; "
            f"min {int(a.min())} mean {float(a.mean()):.1f} max {int(a.max())}")
        if a.ndim == 3 and a.shape[2] >= 3:
            m = a[..., :3].reshape(-1, 3).mean(0)
            say(f"channel means B G R: {m.round(1).tolist()}" + ("  (all equal: grey picture)" if abs(m[0] - m[2]) < 0.5 else ""))
        try:
            import cv2
            cv2.imwrite("ic4_check.png", a)
            say("picture written: ic4_check.png")
        except Exception:
            pass
        say("OK: IC4 works with this camera")
    except Exception as e:
        say(f"FAIL: streaming / snapping: {e}")
        say(traceback.format_exc())
    finally:
        try:
            g.stream_stop()
        except Exception:
            pass
        g.device_close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say(traceback.format_exc())
    with open("ic4_check.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("written: ic4_check.txt")
