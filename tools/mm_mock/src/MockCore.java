import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Random;
import javax.imageio.ImageIO;

/** Stand-in for mmcorej.CMMCore: a RAW16 Bayer camera (R at top-left) and an XY/Z stage
 *  looking at a stitched chip image.  Frame pixel p (from the frame centre) shows the sample
 *  point whose stage coordinate is  stage + M (p - c);  sample stage coordinate T is image pixel
 *  u = cimg + R(rot) (T - O) / umPerPx. */
public class MockCore {
    BufferedImage chip;
    final java.util.ArrayList<BufferedImage> patches = new java.util.ArrayList<BufferedImage>();
    final java.util.ArrayList<Double> pcx = new java.util.ArrayList<Double>(), pcy = new java.util.ArrayList<Double>();
    /** more chips (multi-piece runs): image, stage coordinate of its centre, rotation, focus height offset */
    final java.util.ArrayList<BufferedImage> chips = new java.util.ArrayList<BufferedImage>();
    final java.util.ArrayList<double[]> chipPose = new java.util.ArrayList<double[]>();      // {o0, o1, rot (rad), dz}
    public void addChip(String path, double cx, double cy, double rotDeg, double dz) throws Exception {
        chips.add(ImageIO.read(new File(path))); chipPose.add(new double[] { cx, cy, Math.toRadians(rotDeg), dz });
    }
    public void addPatch(String path, double cx, double cy) throws Exception {
        patches.add(ImageIO.read(new File(path))); pcx.add(cx); pcy.add(cy);
    }
    /** chip index whose image covers stage point (sx, sy) on a non-dark pixel, or -1 */
    int chipAt(double sx, double sy, int[] rgbOut) {
        for (int k = 0; k < chips.size(); k++) {
            BufferedImage im = chips.get(k); double[] p = chipPose.get(k);
            double c = Math.cos(p[2]), s = Math.sin(p[2]), tx = sx - p[0], ty = sy - p[1];
            int u = (int) Math.floor(im.getWidth() / 2.0 + (c * tx - s * ty) / umPerPx);
            int v = (int) Math.floor(im.getHeight() / 2.0 + (s * tx + c * ty) / umPerPx);
            if (u >= 0 && v >= 0 && u < im.getWidth() && v < im.getHeight()) {
                int rgb = im.getRGB(u, v);
                if (rgbOut != null) rgbOut[0] = rgb;
                return k;
            }
        }
        return -1;
    }
    public double patchUmPerPx = -1;                      // scale of the patch images (default: umPerPx)
    int sample(double sx, double sy) {                    // sample stage point -> RGB
        if (objective == 100) { sx -= offX100; sy -= offY100; }
        if (objective == DOMAIN) { sx -= offXD; sy -= offYD; }
        if (!patches.isEmpty() && (chip != null || !chips.isEmpty())) {      // 100x patches on top of chips
            double ps = patchUmPerPx > 0 ? patchUmPerPx : umPerPx;
            for (int k = 0; k < patches.size(); k++) {
                BufferedImage im = patches.get(k);
                int u = (int) Math.floor(im.getWidth() / 2.0 + (sx - pcx.get(k)) / ps);
                int v = (int) Math.floor(im.getHeight() / 2.0 + (sy - pcy.get(k)) / ps);
                if (u >= 0 && v >= 0 && u < im.getWidth() && v < im.getHeight()) return im.getRGB(u, v);
            }
        }
        if (!chips.isEmpty()) {
            int[] o = new int[1];
            return chipAt(sx, sy, o) >= 0 ? o[0] : 0x141414;
        }
        if (chip != null) {
            double c = Math.cos(rot), s = Math.sin(rot), tx = sx - o0, ty = sy - o1;
            int u = (int) Math.floor(chip.getWidth() / 2.0 + (c * tx - s * ty) / umPerPx);
            int v = (int) Math.floor(chip.getHeight() / 2.0 + (s * tx + c * ty) / umPerPx);
            return (u >= 0 && v >= 0 && u < chip.getWidth() && v < chip.getHeight()) ? chip.getRGB(u, v) : 0x141414;
        }
        for (int k = 0; k < patches.size(); k++) {
            BufferedImage im = patches.get(k);
            int u = (int) Math.floor(im.getWidth() / 2.0 + (sx - pcx.get(k)) / umPerPx);
            int v = (int) Math.floor(im.getHeight() / 2.0 + (sy - pcy.get(k)) / umPerPx);
            if (u >= 0 && v >= 0 && u < im.getWidth() && v < im.getHeight()) return im.getRGB(u, v);
        }
        return 0x8a8a8a;                                    // bare substrate between the flakes
    }
    final double umPerPx, rot, o0, o1;
    double m00, m01, m10, m11;
    /** objective: 10 (as constructed) or 100 (mockObjective): finer pixels, XY and focus offsets, more defocus blur */
    public int objective = 10;
    public double ratio100 = 9.54, offX100 = 24.0, offY100 = -18.3, zOff100 = -41.0, blur100 = 3.0;
    /** stacking-domain configuration (mockDomain): its own objective, camera "DomainCam" (mono 16 bit,
     *  pixel matrix mD: rotated against the scan camera), XY and focus offsets relative to the sample */
    public static final int DOMAIN = 999;
    public double offXD = 36.5, offYD = -25.3, zOffD = -34.7, blurD = 2.5, signalD = 15000;
    public double mD00 = 0.0, mD01 = 0.108, mD10 = -0.108, mD11 = 0.0;
    public int WD = 1024, HD = 1024;
    public String camera = "MockCam";
    public void setCameraDevice(String c) throws Exception {
        if (!c.equals("MockCam") && !c.equals("DomainCam")) throw new Exception("no camera " + c);
        camera = c; System.out.println("MOCK camera -> " + c);
    }
    public void mockDomain() { objective = DOMAIN; blurPerUm = blurD; camera = "DomainCam"; System.out.println("MOCK domain objective, camera and filter in"); }
    boolean dom() { return camera.equals("DomainCam"); }
    public void mockObjective(int mag) {
        if (mag == objective) return;
        // 100x: finer pixels, more defocus blur, and less light per pixel (the scripts use 27 ms at 100x, 3 ms at 10x)
        if (mag == 100) { m00 /= ratio100; m01 /= ratio100; m10 /= ratio100; m11 /= ratio100; blurPerUm = blur100; signal /= 9.0; }
        objective = mag;
        System.out.println("MOCK objective -> " + mag + "x");
    }
    final int W0 = 1376, H0 = 1038;
    int W = W0, H = H0;
    public double x, y, z = -90.0, travel = 0;
    int snaps = 0, moves = 0;
    short[] frame;
    final Random rng;
    public double signal = 2200, dark = 128, noise = 10, vignette = 0.38;

    public MockCore(String path, double umPerPx, double o0, double o1, double rotDeg,
                    double m00, double m01, double m10, double m11, double x0, double y0, long seed) throws Exception {
        if (path.endsWith(".txt")) {                       // patch list: "x_um y_um image" per line (stage of the image centre)
            chip = null;
            for (String line : java.nio.file.Files.readAllLines(new File(path).toPath())) {
                String[] f = line.trim().split("\\s+", 3);
                if (f.length < 3) continue;
                patches.add(ImageIO.read(new File(f[2]))); pcx.add(Double.parseDouble(f[0])); pcy.add(Double.parseDouble(f[1]));
            }
        } else chip = ImageIO.read(new File(path));
        this.umPerPx = umPerPx; this.o0 = o0; this.o1 = o1; this.rot = Math.toRadians(rotDeg);
        this.m00 = m00; this.m01 = m01; this.m10 = m10; this.m11 = m11;
        this.x = x0; this.y = y0; rng = new Random(seed);
    }
    public String getXYStageDevice() { return "XY"; }
    public String getFocusDevice() { return "Z"; }
    public String getCameraDevice() { return camera; }
    public java.util.LinkedHashMap<String, String> props = new java.util.LinkedHashMap<String, String>();
    { props.put("Data format", "RAW16"); props.put("Device output data bit depth", "14"); props.put("AEAG", "Off");
      props.put("Auto white balance", "Off"); props.put("Gain", "0.0000"); props.put("Luminosity gamma", "0.4700");
      props.put("Chromaticity gamma", "0.8000"); props.put("Sharpness", "0.0000"); props.put("Downsampling", "1x1");
      props.put("Binning", "1"); props.put("Image data bit depth", "14"); }
    public double exposure = 10.0;
    public boolean lockGamma = false;     // simulate a property that refuses to change
    /** focus drive: controller counts steps of trueStep um; the adapter converts with StepSize_um (4 decimals) */
    public double trueStep = 0.025 / 12, stepSize = 0.025;
    public boolean stepReadOnly = false;
    public boolean hasProperty(String dev, String name) { return dev.equals("Z") ? name.equals("StepSize_um") : props.containsKey(name); }
    public String getProperty(String dev, String name) { return dev.equals("Z") ? String.format("%.4f", stepSize) : props.get(name); }
    public void setProperty(String dev, String name, String v) throws Exception {
        if (dev.equals("Z") && stepReadOnly) throw new Exception("property is read-only");
        if (dev.equals("Z")) { stepSize = Math.round(Double.parseDouble(v) * 1e4) / 1e4; zScale = stepSize / trueStep; return; }
        if (lockGamma && name.equals("Luminosity gamma")) return;           // silently ignored
        if (name.equals("Device output data bit depth") && !(v.equals("8") || v.equals("10") || v.equals("12") || v.equals("14")))
            throw new Exception("Invalid property value: " + v);
        if (name.equals("Device output data bit depth")) props.put("Image data bit depth", v);   // follows, like the XC10
        if (name.equals("Image data bit depth")) throw new Exception("read-only");
        props.put(name, name.contains("gamma") || name.equals("Gain") || name.equals("Sharpness") ? String.format("%.4f", Double.parseDouble(v)) : v);
    }
    public void setExposure(double ms) { exposure = ms; }
    public double getExposure() { return exposure; }
    public long getImageWidth() { return dom() ? WD : W0; }
    public long getImageHeight() { return dom() ? HD : H0; }
    public long getBytesPerPixel() { return 2; }
    public long getImageBitDepth() { return dom() ? 16 : 12; }
    public double getXPosition(String d) { return x; }
    public double getYPosition(String d) { return y; }
    public double zScale = 12.0;           // drive units per real um (the BX51 + Oasis Z: ~12)
    public double getPosition(String d) { return z * zScale; }
    public void setPosition(String d, double nz) { z = nz / zScale; zMoves++; }
    public double trueFocusHere() { return trueFocus(x, y); }
    /** true focus of the chip surface: z0 + tiltX * x + tiltY * y (um); tilt in um/um */
    public double z0 = -90.0, tiltX = 0.0, tiltY = 0.0, focusNoise = 0.5;
    public int zMoves = 0;
    /** bow: + bowK * ((x - bowX)^2 + (y - bowY)^2) (um per um^2); each chip of a multi-piece run adds its dz */
    public double bowK = 0.0, bowX = 0.0, bowY = 0.0;
    public double trueFocus(double px, double py) {
        double z = z0 + tiltX * px + tiltY * py + bowK * ((px - bowX) * (px - bowX) + (py - bowY) * (py - bowY));
        if (!chips.isEmpty()) {
            double sx = px, sy = py;
            if (objective == 100) { sx -= offX100; sy -= offY100; }
            if (objective == DOMAIN) { sx -= offXD; sy -= offYD; }
            int k = chipAt(sx, sy, null);
            if (k >= 0) z += chipPose.get(k)[3];
        }
        return z + (objective == 100 ? zOff100 : objective == DOMAIN ? zOffD : 0.0);
    }
    /** the simulated user: focuses at the current XY (with some error) */
    public void mockFocus() { z = trueFocus(x, y) + focusNoise * rng.nextGaussian(); }
    public void waitForDevice(String d) { }
    public int getSnaps() { return snaps; }
    public int getMoves() { return moves; }
    public double getTravel() { return travel; }
    /** fault: X moves to faultScaleX x the commanded value (readout and moves disagree) */
    public double faultScaleX = 1.0;
    public void setXYPosition(String d, double nx, double ny) {
        nx *= faultScaleX;
        travel += Math.hypot(nx - x, ny - y); moves++; x = nx; y = ny;
    }
    /** defocus blur: Gaussian sigma (frame px) per um of |z - true focus| (10x ~0.15, 100x ~3) */
    public double blurPerUm = 0.15;
    static void boxBlur(float[] a, int w, int h, int r) {
        if (r < 1) return;
        float[] t = new float[a.length];
        for (int y = 0; y < h; y++) {                        // horizontal
            double s = 0; int o = y * w;
            for (int x = -r; x <= r; x++) s += a[o + Math.max(0, Math.min(w - 1, x))];
            for (int x = 0; x < w; x++) {
                t[o + x] = (float) (s / (2 * r + 1));
                s += a[o + Math.min(w - 1, x + r + 1)] - a[o + Math.max(0, x - r)];
            }
        }
        for (int x = 0; x < w; x++) {                        // vertical
            double s = 0;
            for (int y = -r; y <= r; y++) s += t[Math.max(0, Math.min(h - 1, y)) * w + x];
            for (int y = 0; y < h; y++) {
                a[y * w + x] = (float) (s / (2 * r + 1));
                s += t[Math.min(h - 1, y + r + 1) * w + x] - t[Math.max(0, y - r) * w + x];
            }
        }
    }
    static void gaussBlur(float[] a, int w, int h, double sigma) {
        int r = Math.max(1, (int) Math.ceil(3 * sigma));
        double[] g = new double[2 * r + 1]; double sum = 0;
        for (int i = -r; i <= r; i++) { g[i + r] = Math.exp(-0.5 * i * i / (sigma * sigma)); sum += g[i + r]; }
        for (int i = 0; i < g.length; i++) g[i] /= sum;
        float[] t = new float[a.length];
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            double s = 0; for (int i = -r; i <= r; i++) s += g[i + r] * a[y * w + Math.max(0, Math.min(w - 1, x + i))];
            t[y * w + x] = (float) s;
        }
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            double s = 0; for (int i = -r; i <= r; i++) s += g[i + r] * t[Math.max(0, Math.min(h - 1, y + i)) * w + x];
            a[y * w + x] = (float) s;
        }
    }
    public void snapImage() {
        snaps++;
        boolean mono = dom();
        W = mono ? WD : W0; H = mono ? HD : H0;
        double a00 = mono ? mD00 : m00, a01 = mono ? mD01 : m01, a10 = mono ? mD10 : m10, a11 = mono ? mD11 : m11;
        frame = new short[W * H];
        float[][] ch3 = new float[3][W * H];
        for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
            double ox = i - W / 2.0, oy = j - H / 2.0;
            int rgb = sample(x + a00 * ox + a01 * oy, y + a10 * ox + a11 * oy);
            ch3[0][j * W + i] = (rgb >> 16) & 0xff; ch3[1][j * W + i] = (rgb >> 8) & 0xff; ch3[2][j * W + i] = rgb & 0xff;
        }
        double sigma = blurPerUm * Math.abs(z - trueFocus(x, y));
        if (sigma > 4.0) {                                    // large blur: 3 box passes ~ Gaussian
            int r = (int) Math.round((Math.sqrt(12 * sigma * sigma / 3 + 1) - 1) / 2);
            for (int k = 0; k < 3; k++) for (int p = 0; p < 3; p++) boxBlur(ch3[k], W, H, r);
        } else if (sigma > 0.05) {                            // small blur: exact separable Gaussian
            for (int k = 0; k < 3; k++) gaussBlur(ch3[k], W, H, sigma);
        }
        double rmax2 = (W / 2.0) * (W / 2.0) + (H / 2.0) * (H / 2.0);
        for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
            double ox = i - W / 2.0, oy = j - H / 2.0;
            int ch = ((j & 1) << 1) | (i & 1);                 // 0 R, 1 G, 2 G, 3 B
            float val = mono ? (ch3[0][j * W + i] + ch3[1][j * W + i] + ch3[2][j * W + i]) / 3f
                             : ch == 0 ? ch3[0][j * W + i] : ch == 3 ? ch3[2][j * W + i] : ch3[1][j * W + i];
            double vig = 1 - vignette * (ox * ox + oy * oy) / rmax2;
            double cnt = dark + (darkMode ? 0.0 : (mono ? signalD : signal) * (exposure / 3.0) * val / 255.0 * vig) + noise * rng.nextGaussian();
            frame[j * W + i] = (short) Math.max(0, Math.min(mono ? 65535 : 4095, (int) Math.round(cnt)));
        }
    }
    /** the simulated user blocks the light (dark frames) */
    public boolean darkMode = false;
    public void mockDark(boolean on) { darkMode = on; }
    public int glitchAt = -1;              // this snap returns a garbage frame (XC10 glitch)
    public Object getImage() {
        if (snaps == glitchAt) { short[] g = frame.clone(); for (int i = 0; i < g.length; i++) g[i] = (short) rng.nextInt(4096); return g; }
        return frame;
    }
}
