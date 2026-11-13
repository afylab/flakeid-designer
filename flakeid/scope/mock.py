"""Simulated microscope for testing the Python scan (flakeid.scope): a Python port of tools/mm_mock/src/MockCore.java.

The camera is a RAW16 Bayer XC10 (R at the top-left pixel), vignetted and noisy, looking at stitched chip images that
lie on the stage at their own positions, rotations and heights, on a tilted and bowed surface.  Frame pixel p (from
the frame centre) shows the sample point at stage + M (p - c).  100x (mockObjective(100)): finer pixels, the
objective's XY and focus offsets and more defocus blur; the stacking-domain configuration (mockDomain): its own camera
"DomainCam" (mono, 16 bit), offsets and blur.  The mock* methods stand in for the user (focusing, changing the
objective, blocking the light).  Method names follow pymmcore's CMMCore.
"""
from __future__ import annotations

import numpy as np

M10 = np.array([[-1.0905, -0.0067], [0.0066, -1.0917]])


class MockCore:
    W0, H0 = 1376, 1038
    DOMAIN = 999

    def __init__(self, um_per_px=1.09, M=M10, x0=0.0, y0=0.0, seed=1):
        self.um_per_px = um_per_px
        self.M = np.array(M, float)
        self.x, self.y, self.z = float(x0), float(y0), -90.0
        self.rng = np.random.default_rng(seed)
        self.chips = []                     # (image RGB uint8, cx, cy, rot rad, dz)
        self.patches = []                   # (image, cx, cy)
        self.patch_um_per_px = None
        self.objective = 10
        self.ratio100, self.off100, self.zoff100, self.blur100 = 9.54, np.array([24.0, -18.3]), -41.0, 3.0
        # other objectives (device photos, objective offsets tool): XY and focus offsets as the 100x one's
        self.off_obj, self.zoff_obj = {50: np.array([8.0, -5.5]), 5: np.array([-30.0, 42.0])}, {50: -25.0, 5: 60.0}
        self.offD, self.zoffD, self.blurD, self.signalD = np.array([36.5, -25.3]), -34.7, 2.5, 15000.0
        self.MD = np.array([[0.0, 0.108], [-0.108, 0.0]])
        self.WD = self.HD = 1024
        self.camera = "MockCam"
        self.signal, self.dark, self.noise, self.vignette = 2200.0, 128.0, 10.0, 0.38
        self.exposure = 10.0
        self.blur_per_um = 0.15
        self.z0, self.tilt, self.bowK, self.bowc = -90.0, np.zeros(2), 0.0, np.zeros(2)
        self.focus_noise = 0.5
        self.z_scale = 12.0
        self.true_step, self.step_size = 0.025 / 12, 0.025
        self.dark_mode = False
        self.field_stop = None              # mockFieldStop: radius (px) of the closed field-stop image
        self.ill_dust = None                # [(u, v, r_px)]: dust in the light path, sharp at ill_dust_dz um from focus
        self.ill_dust_dz, self.ill_dust_depth = 3.0, 0.08
        self.glitch_at = -1
        self.snaps = 0
        # a camera that stops giving pictures (the XC10, 2026-10-26/28: Error in device "16501874" (25) on every snap while
        # its properties still answer): cam_stuck = True until it is unloaded and loaded again (Scanner.reconnect_camera);
        # cam_dead = True: reloading does not help either
        self.cam_stuck, self.cam_dead, self.reloads = False, False, 0
        self.unloaded = set()
        self.frame = None
        self.props = {"Data format": "RAW16", "Device output data bit depth": "14", "AEAG": "Off", "Auto white balance": "Off",
                      "Gain": "0.0000", "Luminosity gamma": "0.4700", "Chromaticity gamma": "0.8000", "Sharpness": "0.0000",
                      "Downsampling": "1x1", "Binning": "1", "Image data bit depth": "14"}

    # ---- scene
    def add_chip(self, img_rgb, cx, cy, rot_deg, dz):
        self.chips.append((np.ascontiguousarray(img_rgb), float(cx), float(cy), np.radians(rot_deg), float(dz)))

    def add_patch(self, img_rgb, cx, cy):
        self.patches.append((np.ascontiguousarray(img_rgb), float(cx), float(cy)))

    def _obj_offset(self):
        return (self.off100 if self.objective == 100 else self.offD if self.objective == self.DOMAIN
                else self.off_obj.get(self.objective, np.zeros(2)))

    def _chip_index(self, sx, sy):
        for k, (im, cx, cy, rot, dz) in enumerate(self.chips):
            c, s = np.cos(rot), np.sin(rot)
            tx, ty = sx - cx, sy - cy
            u = im.shape[1] / 2.0 + (c * tx - s * ty) / self.um_per_px
            v = im.shape[0] / 2.0 + (s * tx + c * ty) / self.um_per_px
            if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
                return k
        return -1

    def true_focus(self, px, py):
        z = self.z0 + self.tilt[0] * px + self.tilt[1] * py + self.bowK * ((px - self.bowc[0]) ** 2 + (py - self.bowc[1]) ** 2)
        o = self._obj_offset()
        k = self._chip_index(px - o[0], py - o[1])
        if k >= 0:
            z += self.chips[k][4]
        return z + (self.zoff100 if self.objective == 100 else self.zoffD if self.objective == self.DOMAIN
                    else self.zoff_obj.get(self.objective, 0.0))

    def _render_affine(self, W, H, A):
        """RGB float32 image of the sample seen by a W x H camera centred on the stage position, pixel (i, j) at stage
        (x + A[0] . (i - W/2, j - H/2), y + A[1] . (...)).  Each chip / patch is one affine warp (nearest pixel), painted
        from the lowest priority up (the first patch wins, then the first chip), as the per-pixel lookup did."""
        import cv2
        o = self._obj_offset()
        x0, y0 = self.x - o[0], self.y - o[1]
        out = np.full((H, W, 3), 20, np.uint8)                       # dark surround (0x141414)
        # stage point of camera pixel (i, j): S = s0 + A @ (i, j), s0 = (x0, y0) - A @ (W/2, H/2)
        s0 = np.array([x0, y0]) - A @ np.array([W / 2.0, H / 2.0])
        layers = []
        ps = self.patch_um_per_px or self.um_per_px
        for im, cx, cy in self.patches:
            R = np.eye(2) / ps
            layers.append((im, R, np.array([im.shape[1] / 2.0 - cx / ps, im.shape[0] / 2.0 - cy / ps])))
        for im, cx, cy, rot, dz in self.chips:
            c, sn = np.cos(rot), np.sin(rot)
            R = np.array([[c, -sn], [sn, c]]) / self.um_per_px
            layers.append((im, R, np.array([im.shape[1] / 2.0, im.shape[0] / 2.0]) - R @ np.array([cx, cy])))
        for im, R, t in reversed(layers):
            # image pixel (u, v) = R @ S + t = (R @ A) @ (i, j) + (R @ s0 + t); the old lookup took floor(u): nearest of u - 0.5
            Mi = np.hstack([R @ A, (R @ s0 + t - 0.5)[:, None]])
            # quick reject: does the frame touch this image at all?
            cs = Mi[:, :2] @ np.array([[0, W, 0, W], [0, 0, H, H]], float) + Mi[:, 2:]
            if cs[0].max() < -1 or cs[1].max() < -1 or cs[0].min() > im.shape[1] or cs[1].min() > im.shape[0]:
                continue
            cv2.warpAffine(im, Mi, (W, H), dst=out, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_TRANSPARENT)
        return out.astype(np.float32)

    def _vignette(self, W, H):
        key = (W, H, self.vignette)
        if getattr(self, "_vig_key", None) != key:
            jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)
            ox, oy = ii - W / 2.0, jj - H / 2.0
            self._vig = (1 - self.vignette * (ox * ox + oy * oy) / ((W / 2.0) ** 2 + (H / 2.0) ** 2)).astype(np.float32)
            self._vig_key = key
        return self._vig

    # ---- CMMCore-like API
    def getXYStageDevice(self): return "XY"
    def getFocusDevice(self): return "Z"
    def getCameraDevice(self): return self.camera
    def getLoadedDevices(self): return [d for d in ("MockCam", "DomainCam", "XY", "Z") if d not in self.unloaded]
    def getVersionInfo(self): return "MockCore (Python)"
    def getAPIVersionInfo(self): return "Device API version 71, Module API version 10 (simulated)"

    def setCameraDevice(self, c):
        if c not in ("MockCam", "DomainCam") or c in self.unloaded:
            raise RuntimeError(f"no camera {c}")
        self.camera = c

    # unloading and loading a device again (Scanner.reconnect_camera)
    def getDeviceLibrary(self, d): return "MockLib"
    def getDeviceName(self, d): return d
    def getParentLabel(self, d): return ""
    def isPropertyPreInit(self, d, name): return False
    def isPropertyReadOnly(self, d, name): return name == "Image data bit depth"

    def _cam_err(self, code, msg):
        raise RuntimeError(f'Error in device "{self.camera}": {msg} ({code})')

    def unloadDevice(self, d):
        self.unloaded.add(d)
        if d == self.camera:
            self.camera, self.streaming = "", False

    def loadDevice(self, d, lib, name):
        if d not in self.unloaded or lib != "MockLib":
            raise RuntimeError(f"cannot load {d}")
        self.unloaded.discard(d)

    def initializeDevice(self, d):
        self.reloads += 1
        if d == "MockCam":                  # (the driver's defaults again: apply_camera sets them back)
            self.props.update({"Luminosity gamma": "0.4700", "Chromaticity gamma": "0.8000", "Gain": "0.0000"})
            self.cam_stuck = self.cam_dead

    def _dom(self): return self.camera == "DomainCam"
    def hasProperty(self, dev, name): return name == "StepSize_um" if dev == "Z" else (name in self.props and not self._dom())
    def getProperty(self, dev, name): return f"{self.step_size:.4f}" if dev == "Z" else self.props[name]

    def setProperty(self, dev, name, v):
        if dev == "Z":
            self.step_size = round(float(v) * 1e4) / 1e4
            self.z_scale = self.step_size / self.true_step
            return
        if name == "Device output data bit depth":
            if str(v) not in ("8", "10", "12", "14"):
                raise RuntimeError(f"Invalid property value: {v}")
            self.props["Image data bit depth"] = str(v)
        if name == "Image data bit depth":
            raise RuntimeError("read-only")
        self.props[name] = f"{float(v):.4f}" if ("gamma" in name or name in ("Gain", "Sharpness")) else str(v)

    def getDevicePropertyNames(self, dev): return [] if dev != self.camera or self._dom() else list(self.props)
    def setExposure(self, ms): self.exposure = float(ms)
    def getExposure(self): return self.exposure
    def getImageWidth(self): return self.WD if self._dom() else self.W0
    def getImageHeight(self): return self.HD if self._dom() else self.H0
    def getBytesPerPixel(self): return 2
    def getImageBitDepth(self): return 16 if self._dom() else 12
    def getNumberOfComponents(self): return 1
    def getXPosition(self, dev=None): return self.x
    def getYPosition(self, dev=None): return self.y
    def getPosition(self, dev=None): return self.z * self.z_scale
    def setPosition(self, *a): self.z = float(a[-1]) / self.z_scale
    def setXYPosition(self, *a): self.x, self.y = float(a[-2]), float(a[-1])
    def waitForDevice(self, dev): pass
    def waitForSystem(self): pass
    # continuous acquisition (live view): every poll delivers a new frame
    streaming = False
    def isSequenceRunning(self): return self.streaming
    def stopSequenceAcquisition(self): self.streaming = False
    nbuf = 0
    def startContinuousSequenceAcquisition(self, interval_ms=0):
        if self.cam_stuck and not self._dom():
            self._cam_err(1, "Unknown error in the device")
        if self.streaming:
            raise RuntimeError("a sequence acquisition is already running")
        self.streaming, self.nbuf = True, 0
    def getRemainingImageCount(self):
        if not self.streaming:
            return 0
        self.nbuf += 1                      # every poll: a new frame has arrived (drawn when it is read)
        self._stale = True
        return self.nbuf
    def getLastImage(self):
        if getattr(self, "_stale", False) or self.frame is None:
            self.snapImage()
            self._stale = False
        return self.getImage()              # (the --glitch frame can come from the stream too)
    def clearCircularBuffer(self): self.nbuf = 0

    def snapImage(self):
        import cv2
        if self.cam_stuck and not self._dom():
            self._cam_err(25, "(Error message unavailable)")
        self.snaps += 1
        mono = self._dom()
        W, H = (self.WD, self.HD) if mono else (self.W0, self.H0)
        A = self.MD if mono else self.M
        img = self._render_affine(W, H, A)
        dz = self.z - self.true_focus(self.x, self.y)
        sigma = self.blur_per_um * abs(dz)
        if sigma > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), min(sigma, 60.0))
        if self.ill_dust and not mono:
            m = np.zeros((H, W), np.float32)
            for u, v, r in self.ill_dust:
                cv2.circle(m, (int(u), int(v)), int(r), 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 0.7 + min(self.blur_per_um * abs(dz - self.ill_dust_dz), 60.0))
            img = (img.astype(np.float32) * (1 - self.ill_dust_depth * m)[..., None]).astype(img.dtype)
        if self.field_stop:
            m = np.zeros((H, W), np.float32)
            cv2.circle(m, (W // 2, H // 2), int(self.field_stop), 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 0.7 + min(2 * sigma, 60.0))      # projected on the surface and imaged: twice the blur
            img = (img.astype(np.float32) * m[..., None]).astype(img.dtype)
        vig = self._vignette(W, H)
        if mono:
            val = img.mean(axis=2)
            sig = self.signalD
            full = 65535
        else:
            val = np.empty((H, W), np.float32)
            val[0::2, 0::2] = img[0::2, 0::2, 0]
            val[0::2, 1::2] = img[0::2, 1::2, 1]
            val[1::2, 0::2] = img[1::2, 0::2, 1]
            val[1::2, 1::2] = img[1::2, 1::2, 2]
            sig = self.signal
            full = 4095
        cnt = self.dark + (0.0 if self.dark_mode else sig * (self.exposure / 3.0) * val / 255.0 * vig) \
            + self.noise * self.rng.standard_normal((H, W), dtype=np.float32)
        self.frame = np.clip(np.round(cnt), 0, full).astype(np.uint16)

    def getImage(self):
        if self.snaps == self.glitch_at:
            return self.rng.integers(0, 4096, self.frame.shape).astype(np.uint16)
        return self.frame

    # ---- the simulated user
    def mockFocus(self):
        self.z = self.true_focus(self.x, self.y) + self.focus_noise * self.rng.standard_normal()

    def mockObjective(self, mag):
        if mag == self.objective:
            return
        if not hasattr(self, "_base"):                # (the 10x state: the other objectives are scaled from it)
            self._base = (self.M.copy(), self.blur_per_um, self.signal) if self.objective == 10 else None
        M10, b10, s10 = self._base or (self.M * (self.ratio100 if self.objective == 100 else 1.0), 0.15, self.signal)
        if mag == 100:
            self.M, self.blur_per_um, self.signal = M10 / self.ratio100, self.blur100, s10 / 9.0
        else:                                         # (5x, 10x, 50x: magnification scaled, light as 1 / mag^2 roughly)
            self.M, self.blur_per_um, self.signal = M10 * 10.0 / mag, b10 * (mag / 10.0), s10 * min(4.0, (10.0 / mag) ** 0.5)
        self.objective = mag

    def mockDomain(self):
        self.objective = self.DOMAIN
        self.blur_per_um = self.blurD
        self.camera = "DomainCam"

    def mockDark(self, on):
        self.dark_mode = bool(on)

    def mockFieldStop(self, closed):
        self.field_stop = 0.3 * min(self.W0, self.H0) if closed else None
