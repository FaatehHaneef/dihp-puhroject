"""
Preprocessing — Image enhancement and color conversion utilities.

DIP Concepts: Spatial filtering, histogram equalization, color space conversion
"""

import cv2
import numpy as np


def histogram_equalize(frame):
    """
    Apply histogram equalization to the luminance channel for lighting normalization.
    
    Args:
        frame: BGR image (uint8)
    
    Returns:
        Equalized BGR image (uint8)
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def adaptive_blur(frame, base_kernel=5):
    """
    Adaptive Gaussian blur based on frame brightness variance.
    Higher variance = smoother blur, lower variance = minimal blur.
    
    Args:
        frame: BGR image (uint8)
        base_kernel: Base kernel size (must be odd)
    
    Returns:
        Blurred BGR image (uint8)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Adjust kernel based on variance
    kernel_size = base_kernel
    if variance < 100:
        kernel_size = 3
    elif variance < 500:
        kernel_size = 5
    else:
        kernel_size = 7
    
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 1.0)


def bgr_to_hsv(frame):
    """
    Convert BGR to HSV color space.
    
    Args:
        frame: BGR image (uint8)
    
    Returns:
        HSV image (uint8)
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def bgr_to_ycbcr(frame):
    """
    Convert BGR to YCbCr color space.
    
    Args:
        frame: BGR image (uint8)
    
    Returns:
        YCbCr image (uint8)
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)


def normalize_brightness(frame, target_mean=127):
    """
    Normalize brightness by scaling the image to match a target mean luminance.
    
    Args:
        frame: BGR image (uint8)
        target_mean: Target mean luminance (0-255)
    
    Returns:
        Normalized BGR image (uint8)
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    
    current_mean = y_channel.mean()
    if current_mean > 0:
        scale = target_mean / current_mean
        y_channel = np.clip(y_channel * scale, 0, 255)
    
    ycrcb[:, :, 0] = y_channel.astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
