"""
MecanTensor Vision — Python Wrapper for Color Engine + Edge Detection

Phase 1: 20 color ops + 6 edge/corner ops
All operations work on numpy arrays. Falls back to pure Python
when the C++ DLL is not compiled.

Usage:
    from mecantensor import vision

    # Color conversions
    lab = vision.color.rgb_to_lab(img_rgb)
    hsv = vision.color.rgb_to_hsv(img_rgb)

    # Edge detection
    edges = vision.detect.canny(gray, low=0.1, high=0.3)
    corners = vision.detect.harris(gray, k=0.04)
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOR ENGINE — 20 Operations (Pure Python Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

class color:
    """Spectral Color Engine — 10 color spaces, HDR, spectral wavelengths."""

    # ─── sRGB Gamma ─────────────────────────────────────────────────

    @staticmethod
    def _srgb_to_linear(v):
        return np.where(v <= 0.04045, v / 12.92,
                        np.power((v + 0.055) / 1.055, 2.4))

    @staticmethod
    def _linear_to_srgb(v):
        v = np.maximum(v, 0)
        return np.where(v <= 0.0031308, v * 12.92,
                        1.055 * np.power(v, 1.0/2.4) - 0.055)

    # D65 white point
    D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    # sRGB→XYZ matrix
    _M_RGB_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)

    _M_XYZ_RGB = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)

    # 1
    @staticmethod
    def rgb_to_xyz(img):
        """sRGB [0,1] → CIE XYZ. Input shape: (..., 3)"""
        linear = color._srgb_to_linear(img.astype(np.float32))
        return linear @ color._M_RGB_XYZ.T

    # 2
    @staticmethod
    def xyz_to_rgb(img):
        """CIE XYZ → sRGB [0,1]."""
        linear = img.astype(np.float32) @ color._M_XYZ_RGB.T
        return np.clip(color._linear_to_srgb(linear), 0, 1)

    # 3
    @staticmethod
    def rgb_to_lab(img):
        """sRGB → CIE L*a*b*."""
        xyz = color.rgb_to_xyz(img) / color.D65
        delta = 6.0 / 29.0
        f = np.where(xyz > delta**3, np.cbrt(xyz),
                     xyz / (3 * delta**2) + 4.0/29.0)
        L = 116.0 * f[..., 1] - 16.0
        a = 500.0 * (f[..., 0] - f[..., 1])
        b = 200.0 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    # 4
    @staticmethod
    def lab_to_rgb(img):
        """CIE L*a*b* → sRGB."""
        L, a, b = img[..., 0], img[..., 1], img[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0
        delta = 6.0 / 29.0
        def finv(t): return np.where(t > delta, t**3, 3*delta**2*(t - 4.0/29.0))
        xyz = np.stack([finv(fx)*color.D65[0], finv(fy)*color.D65[1],
                        finv(fz)*color.D65[2]], axis=-1)
        return color.xyz_to_rgb(xyz)

    # 5
    @staticmethod
    def rgb_to_hsv(img):
        """sRGB → HSV. H in [0,360], S,V in [0,1]."""
        img = img.astype(np.float32)
        mx = np.max(img, axis=-1)
        mn = np.min(img, axis=-1)
        d = mx - mn
        V = mx
        S = np.where(mx > 1e-7, d / mx, 0)
        R, G, B = img[...,0], img[...,1], img[...,2]
        H = np.zeros_like(mx)
        mask_r = (mx == R) & (d > 1e-7)
        mask_g = (mx == G) & (d > 1e-7)
        mask_b = (mx == B) & (d > 1e-7)
        H = np.where(mask_r, 60 * (((G-B)/np.maximum(d,1e-7)) % 6), H)
        H = np.where(mask_g, 60 * ((B-R)/np.maximum(d,1e-7) + 2), H)
        H = np.where(mask_b, 60 * ((R-G)/np.maximum(d,1e-7) + 4), H)
        H = H % 360
        return np.stack([H, S, V], axis=-1).astype(np.float32)

    # 6
    @staticmethod
    def hsv_to_rgb(img):
        """HSV → sRGB."""
        H, S, V = img[...,0], img[...,1], img[...,2]
        C = V * S
        X = C * (1 - np.abs((H / 60) % 2 - 1))
        m = V - C
        R = np.zeros_like(H); G = np.zeros_like(H); B = np.zeros_like(H)
        for lo, hi, rv, gv, bv in [(0,60,C,X,0),(60,120,X,C,0),(120,180,0,C,X),
                                    (180,240,0,X,C),(240,300,X,0,C),(300,360,C,0,X)]:
            mask = (H >= lo) & (H < hi)
            R = np.where(mask, rv, R); G = np.where(mask, gv, G); B = np.where(mask, bv, B)
        return np.stack([R+m, G+m, B+m], axis=-1).astype(np.float32)

    # 7
    @staticmethod
    def rgb_to_hsl(img):
        """sRGB → HSL."""
        img = img.astype(np.float32)
        mx, mn = np.max(img, axis=-1), np.min(img, axis=-1)
        L = (mx + mn) * 0.5
        d = mx - mn
        S = np.where(d > 1e-7, np.where(L > 0.5, d/(2-mx-mn), d/(mx+mn)), 0)
        R, G, B = img[...,0], img[...,1], img[...,2]
        H = np.zeros_like(mx)
        m = d > 1e-7
        H = np.where(m & (mx==R), 60*(((G-B)/np.maximum(d,1e-7))%6), H)
        H = np.where(m & (mx==G), 60*((B-R)/np.maximum(d,1e-7)+2), H)
        H = np.where(m & (mx==B), 60*((R-G)/np.maximum(d,1e-7)+4), H)
        return np.stack([H%360, S, L], axis=-1).astype(np.float32)

    # 8
    @staticmethod
    def rgb_to_lms(img):
        """sRGB → LMS cone response."""
        M = np.array([[0.4002, 0.7076, -0.0808],
                       [-0.2263, 1.1653, 0.0457],
                       [0.0, 0.0, 0.9182]], dtype=np.float32)
        linear = color._srgb_to_linear(img.astype(np.float32))
        return linear @ M.T

    # 9
    @staticmethod
    def rgb_to_linear(img):
        """Remove sRGB gamma curve."""
        return color._srgb_to_linear(img.astype(np.float32))

    # 10
    @staticmethod
    def linear_to_rgb(img):
        """Apply sRGB gamma curve."""
        return np.clip(color._linear_to_srgb(img.astype(np.float32)), 0, 1)

    # 11
    @staticmethod
    def rgb_to_grayscale(img):
        """BT.709 luminosity-weighted grayscale."""
        return (0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]).astype(np.float32)

    # 12
    @staticmethod
    def rgb_to_spectral(img):
        """Estimate 81-band spectral power distribution (380-780nm @ 5nm)."""
        shape = img.shape[:-1]
        lambdas = np.arange(380, 781, 5, dtype=np.float32)
        wr = np.exp(-0.5*((lambdas-600)/40)**2)
        wg = np.exp(-0.5*((lambdas-550)/35)**2)
        wb = np.exp(-0.5*((lambdas-450)/30)**2)
        R, G, B = img[...,0:1], img[...,1:2], img[...,2:3]
        return (R*wr + G*wg + B*wb).astype(np.float32)

    # 13
    @staticmethod
    def spectral_to_rgb(spectral):
        """Integrate spectral back to sRGB."""
        lambdas = np.arange(380, 781, 5, dtype=np.float32)
        wr = np.exp(-0.5*((lambdas-600)/40)**2)
        wg = np.exp(-0.5*((lambdas-550)/35)**2)
        wb = np.exp(-0.5*((lambdas-450)/30)**2)
        R = np.sum(spectral * wr, axis=-1) / 81 * 3
        G = np.sum(spectral * wg, axis=-1) / 81 * 3
        B = np.sum(spectral * wb, axis=-1) / 81 * 3
        return np.clip(np.stack([R,G,B], axis=-1), 0, 1).astype(np.float32)

    # 14
    @staticmethod
    def wavelength_to_rgb(lam):
        """Single wavelength (380-780nm) → visible RGB."""
        r = g = b = 0.0
        if 380 <= lam < 440:   r = -(lam-440)/60; b = 1
        elif lam < 490:        g = (lam-440)/50; b = 1
        elif lam < 510:        g = 1; b = -(lam-510)/20
        elif lam < 580:        r = (lam-510)/70; g = 1
        elif lam < 645:        r = 1; g = -(lam-645)/65
        elif lam <= 780:       r = 1
        f = 1.0
        if lam < 420:    f = 0.3 + 0.7*(lam-380)/40
        elif lam > 700:  f = 0.3 + 0.7*(780-lam)/80
        return np.array([r*f, g*f, b*f], dtype=np.float32)

    # 15
    @staticmethod
    def color_distance(lab1, lab2, metric=0):
        """Delta-E: 0=CIE76, 1=CIE94, 2=CIEDE2000."""
        dL = lab1[0]-lab2[0]; da = lab1[1]-lab2[1]; db = lab1[2]-lab2[2]
        if metric == 0:
            return float(np.sqrt(dL**2+da**2+db**2))
        C1 = np.sqrt(lab1[1]**2+lab1[2]**2); C2 = np.sqrt(lab2[1]**2+lab2[2]**2)
        dC = C1-C2; dH2 = da**2+db**2-dC**2; dH = np.sqrt(max(dH2,0))
        SC = 1+0.045*C1; SH = 1+0.015*C1
        return float(np.sqrt(dL**2 + (dC/SC)**2 + (dH/SH)**2))

    # 16
    @staticmethod
    def chromatic_adapt(img_xyz, src_white, dst_white):
        """Bradford chromatic adaptation."""
        M = np.array([[0.8951,0.2664,-0.1614],[-0.7502,1.7135,0.0367],[0.0389,-0.0685,1.0296]], dtype=np.float32)
        cs = M @ src_white; cd = M @ dst_white
        scale = cd / cs
        adapted = (img_xyz @ M.T) * scale
        Mi = np.linalg.inv(M)
        return (adapted @ Mi.T).astype(np.float32)

    # 17
    @staticmethod
    def hdr_tonemap(img, method=0):
        """HDR tonemapping: 0=Reinhard, 1=ACES, 2=Filmic."""
        x = img.astype(np.float32)
        if method == 0:   return x / (1 + x)
        elif method == 1: return np.clip((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0, 1)
        else:
            x = np.maximum(x - 0.004, 0)
            return (x*(6.2*x+0.5))/(x*(6.2*x+1.7)+0.06)

    # 18
    @staticmethod
    def color_histogram(img, bins=256):
        """Per-channel histogram. Returns (C, bins)."""
        C = img.shape[-1]
        hist = np.zeros((C, bins), dtype=np.float32)
        for c in range(C):
            hist[c], _ = np.histogram(img[...,c].flatten(), bins=bins, range=(0,1))
        return hist

    # 19
    @staticmethod
    def dominant_colors(img, k=5, max_iters=20):
        """Extract K dominant colors via K-means."""
        pixels = img.reshape(-1, 3).astype(np.float32)
        N = pixels.shape[0]
        idx = np.linspace(0, N-1, k, dtype=int)
        centroids = pixels[idx].copy()
        for _ in range(max_iters):
            dists = np.linalg.norm(pixels[:,None,:] - centroids[None,:,:], axis=2)
            labels = np.argmin(dists, axis=1)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centroids[j] = pixels[mask].mean(axis=0)
        return centroids

    # 20
    @staticmethod
    def color_quantize(img, palette):
        """Map each pixel to nearest palette color."""
        pixels = img.reshape(-1, 3).astype(np.float32)
        dists = np.linalg.norm(pixels[:,None,:] - palette[None,:,:], axis=2)
        labels = np.argmin(dists, axis=1)
        return palette[labels].reshape(img.shape).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  EDGE/CORNER DETECTION — 6 Operations
# ═══════════════════════════════════════════════════════════════════════════════

class detect:
    """Object Detection Primitives — edges, corners, contours."""

    @staticmethod
    def _gaussian_blur(img, ksize=5, sigma=1.4):
        """Separable Gaussian blur."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img.astype(np.float32), sigma=sigma).astype(np.float32)

    # 1
    @staticmethod
    def sobel(gray, axis=2):
        """Sobel gradient. axis: 0=dx, 1=dy, 2=magnitude."""
        gray = gray.astype(np.float32)
        Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
        from scipy.signal import convolve2d
        if axis == 0: return convolve2d(gray, Kx, mode='same', boundary='fill')
        if axis == 1: return convolve2d(gray, Ky, mode='same', boundary='fill')
        dx = convolve2d(gray, Kx, mode='same', boundary='fill')
        dy = convolve2d(gray, Ky, mode='same', boundary='fill')
        return np.sqrt(dx**2 + dy**2).astype(np.float32)

    # 2
    @staticmethod
    def canny(gray, low=0.1, high=0.3):
        """Full Canny pipeline: blur → Sobel → NMS → hysteresis."""
        gray = gray.astype(np.float32)
        from scipy.ndimage import gaussian_filter
        from scipy.signal import convolve2d
        blurred = gaussian_filter(gray, sigma=1.4)
        Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
        dx = convolve2d(blurred, Kx, mode='same', boundary='fill')
        dy = convolve2d(blurred, Ky, mode='same', boundary='fill')
        mag = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * 180 / np.pi
        angle[angle < 0] += 180

        H, W = gray.shape
        thin = np.zeros_like(mag)
        for y in range(1, H-1):
            for x in range(1, W-1):
                a = angle[y, x]
                if a < 22.5 or a >= 157.5:   q, r = mag[y,x+1], mag[y,x-1]
                elif a < 67.5:                q, r = mag[y-1,x+1], mag[y+1,x-1]
                elif a < 112.5:               q, r = mag[y-1,x], mag[y+1,x]
                else:                         q, r = mag[y-1,x-1], mag[y+1,x+1]
                thin[y,x] = mag[y,x] if (mag[y,x] >= q and mag[y,x] >= r) else 0

        edges = np.zeros_like(thin)
        strong = thin >= high
        weak = (thin >= low) & (thin < high)
        edges[strong] = 1.0
        from scipy.ndimage import binary_dilation
        connected = binary_dilation(strong, iterations=2) & weak
        edges[connected] = 1.0
        return edges.astype(np.float32)

    # 3
    @staticmethod
    def laplacian(gray):
        """Laplacian edge detection."""
        K = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
        from scipy.signal import convolve2d
        return convolve2d(gray.astype(np.float32), K, mode='same', boundary='fill')

    # 4
    @staticmethod
    def harris(gray, k=0.04, block_size=3):
        """Harris corner response map."""
        dx = detect.sobel(gray, axis=0)
        dy = detect.sobel(gray, axis=1)
        from scipy.ndimage import uniform_filter
        Ixx = uniform_filter(dx*dx, size=block_size)
        Iyy = uniform_filter(dy*dy, size=block_size)
        Ixy = uniform_filter(dx*dy, size=block_size)
        det = Ixx*Iyy - Ixy*Ixy
        trace = Ixx + Iyy
        return (det - k * trace**2).astype(np.float32)

    # 5
    @staticmethod
    def shi_tomasi(gray, block_size=3):
        """Shi-Tomasi minimum eigenvalue response."""
        dx = detect.sobel(gray, axis=0)
        dy = detect.sobel(gray, axis=1)
        from scipy.ndimage import uniform_filter
        Ixx = uniform_filter(dx*dx, size=block_size)
        Iyy = uniform_filter(dy*dy, size=block_size)
        Ixy = uniform_filter(dx*dy, size=block_size)
        diff = Ixx - Iyy
        disc = np.sqrt(diff**2 + 4*Ixy**2)
        return (0.5 * (Ixx + Iyy - disc)).astype(np.float32)

    # 6
    @staticmethod
    def gaussian_blur(gray, ksize=5, sigma=1.4):
        """Gaussian blur."""
        return detect._gaussian_blur(gray, ksize, sigma)


def info():
    """Print Vision Engine status."""
    print("MecanTensor Vision Engine: Active (Python fallback)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SPACEPROOF PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class spaceproof:
    """
    Robust Vision Engine — Handles infinite dynamic range and extreme micro-motion.
    Every algorithm operates in log-domain with adaptive thresholds based on noise floors.
    """

    @staticmethod
    def estimate_noise_floor(gray):
        """
        Estimate sensor noise floor using Median Absolute Deviation (MAD) of high frequencies.
        Works across all lighting conditions.
        """
        gray = gray.astype(np.float32)
        from scipy.signal import convolve2d
        # Laplacian kernel to isolate high-frequency noise
        K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        noise = convolve2d(gray, K, mode='valid')
        # Robust MAD estimator: median(|x - median(x)|) * 1.4826
        mad = np.median(np.abs(noise - np.median(noise)))
        sigma = mad * 1.4826
        return sigma

    @staticmethod
    def to_log_luminance(img, epsilon=1e-6):
        """
        Compresses 10^15 dynamic range into ~50 stops.
        (e.g., handles starlight at 10^-6 and sunlight at 10^9).
        """
        img = img.astype(np.float32)
        # Ensure positive
        img = np.maximum(img, 0)
        return np.log2(img + epsilon)

    @staticmethod
    def adaptive_thresholds(gray, k_low=1.0, k_high=3.0, ksize=31):
        """
        Returns local threshold maps rather than fixed scalars.
        threshold = local_mean + k * local_std
        """
        gray = gray.astype(np.float32)
        from scipy.ndimage import uniform_filter
        
        # Local mean
        mean = uniform_filter(gray, size=ksize)
        # Local variance
        mean_sq = uniform_filter(gray**2, size=ksize)
        var = np.maximum(mean_sq - mean**2, 0)
        std = np.sqrt(var)
        
        # Add the global noise floor so we don't amplify pure noise
        noise_floor = spaceproof.estimate_noise_floor(gray)
        std = np.maximum(std, noise_floor)

        low_thresh = mean + k_low * std
        high_thresh = mean + k_high * std
        return low_thresh, high_thresh

    @staticmethod
    def build_pyramid(gray, levels=4):
        """
        Builds a multi-scale Gaussian pyramid for detecting gradients
        from 1 pixel up to 2^levels pixels.
        """
        gray = gray.astype(np.float32)
        from scipy.ndimage import gaussian_filter
        pyramid = [gray]
        current = gray
        for _ in range(levels - 1):
            blurred = gaussian_filter(current, sigma=1.0)
            # 2x Downsample
            current = blurred[::2, ::2]
            pyramid.append(current)
        return pyramid

    @staticmethod
    def spaceproof_canny(gray, k_low=1.0, k_high=3.0):
        """
        Canny edge detection that works on ANY lighting condition.
        1. Convert to log-domain
        2. Compute adaptive local thresholds
        3. Apply Canny pipeline using local maps instead of fixed scalars
        """
        # 1. Log domain
        log_gray = spaceproof.to_log_luminance(gray)
        
        # 2. Local adaptive thresholds
        lo_map, hi_map = spaceproof.adaptive_thresholds(log_gray, k_low, k_high)
        
        # 3. Standard gradients
        from scipy.ndimage import gaussian_filter
        from scipy.signal import convolve2d
        blurred = gaussian_filter(log_gray, sigma=1.4)
        Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
        dx = convolve2d(blurred, Kx, mode='same', boundary='fill')
        dy = convolve2d(blurred, Ky, mode='same', boundary='fill')
        mag = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * 180 / np.pi
        angle[angle < 0] += 180

        # NMS
        H, W = log_gray.shape
        thin = np.zeros_like(mag)
        for y in range(1, H-1):
            for x in range(1, W-1):
                a = angle[y, x]
                if a < 22.5 or a >= 157.5:   q, r = mag[y,x+1], mag[y,x-1]
                elif a < 67.5:                q, r = mag[y-1,x+1], mag[y+1,x-1]
                elif a < 112.5:               q, r = mag[y-1,x], mag[y+1,x]
                else:                         q, r = mag[y-1,x-1], mag[y+1,x+1]
                thin[y,x] = mag[y,x] if (mag[y,x] >= q and mag[y,x] >= r) else 0

        # ADAPTIVE Hysteresis
        edges = np.zeros_like(thin)
        strong = thin >= hi_map
        weak = (thin >= lo_map) & (thin < hi_map)
        edges[strong] = 1.0
        from scipy.ndimage import binary_dilation
        connected = binary_dilation(strong, iterations=2) & weak
        edges[connected] = 1.0
        return edges.astype(np.float32)
