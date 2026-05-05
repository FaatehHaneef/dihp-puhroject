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
    # Convert to YCbCr, equalize Y channel, convert back
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2BGR)


def adaptive_blur(frame, base_kernel=5):
    """
    Adaptive Gaussian blur based on frame brightness variance.
    
    Args:
        frame: BGR image (uint8)
        base_kernel: Base kernel size (must be odd)
    
    Returns:
        Blurred BGR image (uint8)
    """
    # TODO: Compute variance of luminance, adjust kernel size
    return cv2.GaussianBlur(frame, (base_kernel, base_kernel), 1.0)


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
    # TODO: Compute mean luminance, scale accordingly
    return frame
