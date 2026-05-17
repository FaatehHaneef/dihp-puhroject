"""
Preprocessing — histogram equalization, adaptive Gaussian blur, BGR/HSV/YCrCb conversions.
"""

import cv2
import math
import numpy as np

def _clamp(v, lo=0, hi=255):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _bgr_pixel_to_ycrcb(b, g, r):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return y, cr, cb


def _ycrcb_pixel_to_bgr(y, cr, cb):
    cr_c = cr - 128.0
    cb_c = cb - 128.0
    r = y + 1.403 * cr_c
    g = y - 0.344 * cb_c - 0.714 * cr_c
    b = y + 1.773 * cb_c
    return _clamp(int(round(b))), _clamp(int(round(g))), _clamp(int(round(r)))


def histogram_equalizer(frame):
    """Histogram equalization on the Y channel of YCrCb."""
    h, w = frame.shape[:2]
    y_plane = np.zeros((h, w), dtype=np.float32)
    cr_plane = np.zeros((h, w), dtype=np.float32)
    cb_plane = np.zeros((h, w), dtype=np.float32)

    # Step 1: BGR -> YCrCb
    for yy in range(h):
        for xx in range(w):
            b = int(frame[yy, xx, 0])
            g = int(frame[yy, xx, 1])
            r = int(frame[yy, xx, 2])
            y_val, cr_val, cb_val = _bgr_pixel_to_ycrcb(b, g, r)
            y_plane[yy, xx] = y_val
            cr_plane[yy, xx] = cr_val
            cb_plane[yy, xx] = cb_val

    # Step 2: histogram of Y
    hist = [0] * 256
    for yy in range(h):
        for xx in range(w):
            hist[_clamp(int(y_plane[yy, xx]), 0, 255)] += 1

    # Step 3: CDF
    cdf = [0] * 256
    running = 0
    for i in range(256):
        running += hist[i]
        cdf[i] = running
    total = cdf[255]
    if total == 0:
        return frame.copy()

    # Step 4: CDF -> LUT
    lut = [0] * 256
    for i in range(256):
        lut[i] = int(round(255.0 * cdf[i] / total))

    # Step 5: map Y through LUT and convert back to BGR
    out = np.zeros_like(frame)
    for yy in range(h):
        for xx in range(w):
            y_new = lut[_clamp(int(y_plane[yy, xx]), 0, 255)]
            b_new, g_new, r_new = _ycrcb_pixel_to_bgr(y_new, cr_plane[yy, xx], cb_plane[yy, xx])
            out[yy, xx, 0] = b_new
            out[yy, xx, 1] = g_new
            out[yy, xx, 2] = r_new
    return out


def _convolve2d_manual(plane, kernel):
    h, w = plane.shape
    k = len(kernel)
    pad = k // 2
    out = np.zeros((h, w), dtype=np.float32)
    for yy in range(h):
        for xx in range(w):
            acc = 0.0
            for dy in range(k):
                py = yy + dy - pad
                if py < 0 or py >= h:
                    continue
                for dx in range(k):
                    px = xx + dx - pad
                    if px < 0 or px >= w:
                        continue
                    acc += plane[py, px] * kernel[dy][dx]
            out[yy, xx] = acc
    return out


def _build_gaussian_kernel(ksize, sigma):
    pad = ksize // 2
    kernel = [[0.0] * ksize for _ in range(ksize)]
    total = 0.0
    for dy in range(ksize):
        for dx in range(ksize):
            x = dx - pad
            y = dy - pad
            val = math.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            kernel[dy][dx] = val
            total += val
    for dy in range(ksize):
        for dx in range(ksize):
            kernel[dy][dx] /= total
    return kernel


def adaptive_blur_manual(frame, base_kernel=5):
    """
    Steps:
      1. Grayscale luminance.
      2. Laplacian convolution.
      3. Variance of Laplacian.
      4. Pick kernel size from variance.
      5. Build Gaussian kernel.
      6. Convolve each BGR channel.
    """
    h, w = frame.shape[:2]

    # Step 1
    gray = np.zeros((h, w), dtype=np.float32)
    for yy in range(h):
        for xx in range(w):
            b = int(frame[yy, xx, 0])
            g = int(frame[yy, xx, 1])
            r = int(frame[yy, xx, 2])
            gray[yy, xx] = 0.299 * r + 0.587 * g + 0.114 * b

    # Step 2
    laplacian_kernel = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    lap = _convolve2d_manual(gray, laplacian_kernel)

    # Step 3
    n = h * w
    mean = 0.0
    for yy in range(h):
        for xx in range(w):
            mean += lap[yy, xx]
    mean /= n
    var = 0.0
    for yy in range(h):
        for xx in range(w):
            d = lap[yy, xx] - mean
            var += d * d
    var /= n

    # Step 4
    ksize = base_kernel
    if var < 100:
        ksize = 3
    elif var < 500:
        ksize = 5
    else:
        ksize = 7

    # Step 5
    kernel = _build_gaussian_kernel(ksize, sigma=1.0)

    # Step 6
    out = np.zeros_like(frame)
    for c in range(3):
        plane = np.zeros((h, w), dtype=np.float32)
        for yy in range(h):
            for xx in range(w):
                plane[yy, xx] = float(frame[yy, xx, c])
        blurred = _convolve2d_manual(plane, kernel)
        for yy in range(h):
            for xx in range(w):
                out[yy, xx, c] = _clamp(int(round(blurred[yy, xx])))
    return out


def bgr_to_hsv_func(frame):
    """Per-pixel BGR -> HSV. Output ranges match OpenCV: H [0,179], S/V [0,255]."""
    h, w = frame.shape[:2]
    out = np.zeros_like(frame)
    for yy in range(h):
        for xx in range(w):
            b = frame[yy, xx, 0] / 255.0
            g = frame[yy, xx, 1] / 255.0
            r = frame[yy, xx, 2] / 255.0

            mx = r
            if g > mx: mx = g
            if b > mx: mx = b
            mn = r
            if g < mn: mn = g
            if b < mn: mn = b

            delta = mx - mn
            v = mx
            s = 0.0 if mx <= 0 else delta / mx

            if delta <= 0:
                hue = 0.0
            elif mx == r:
                hue = 60.0 * (((g - b) / delta) % 6.0)
            elif mx == g:
                hue = 60.0 * ((b - r) / delta + 2.0)
            else:
                hue = 60.0 * ((r - g) / delta + 4.0)
            if hue < 0:
                hue += 360.0

            out[yy, xx, 0] = _clamp(int(round(hue / 2.0)), 0, 179)
            out[yy, xx, 1] = _clamp(int(round(s * 255.0)), 0, 255)
            out[yy, xx, 2] = _clamp(int(round(v * 255.0)), 0, 255)
    return out


# =============================================================================


def histogram_equalize(frame):
    """Histogram equalization on the Y channel of YCrCb (cv2 path)."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def adaptive_blur(frame, base_kernel=5):
    """Gaussian blur with kernel size selected by Laplacian variance (cv2 path)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    kernel_size = base_kernel
    if variance < 100:
        kernel_size = 3
    elif variance < 500:
        kernel_size = 5
    else:
        kernel_size = 7

    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 1.0)


def bgr_to_hsv(frame):
    """BGR -> HSV (cv2 path)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def bgr_to_ycbcr(frame):
    """BGR -> YCrCb (cv2 path)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)


def normalize_brightness(frame, target_mean=127):
    """Scale luminance so mean Y matches target_mean."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)

    current_mean = y_channel.mean()
    if current_mean > 0:
        scale = target_mean / current_mean
        y_channel = np.clip(y_channel * scale, 0, 255)

    ycrcb[:, :, 0] = y_channel.astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


