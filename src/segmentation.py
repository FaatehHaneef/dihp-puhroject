"""
Segmentation — Skin detection, morphological operations, connected component analysis.

DIP Concepts: Color thresholding, morphological operations, connected component analysis
"""

import cv2
import numpy as np


def get_skin_mask(hsv_frame, h_range=(0, 20), s_range=(10, 150), v_range=(60, 255)):
    """
    Detect skin regions using HSV thresholding.
    Tuned for typical skin tones in indoor lighting.
    
    Args:
        hsv_frame: HSV image (uint8)
        h_range: Hue range for skin (0-179 in OpenCV HSV)
        s_range: Saturation range for skin (0-255)
        v_range: Value (brightness) range for skin (0-255)
    
    Returns:
        Binary mask (uint8, 0/255)
    """
    lower = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
    upper = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower, upper)
    
    # Handle hue wrap-around (skin can be near 180 for darker tones)
    lower_wrap = np.array([170, s_range[0], v_range[0]], dtype=np.uint8)
    upper_wrap = np.array([180, s_range[1], v_range[1]], dtype=np.uint8)
    mask_wrap = cv2.inRange(hsv_frame, lower_wrap, upper_wrap)
    mask = cv2.bitwise_or(mask, mask_wrap)
    
    return mask


def apply_morphology(mask, morph_kernel_size=5):
    """
    Apply morphological operations to clean the mask.
    Pipeline: Opening (erosion → dilation) → Closing (dilation → erosion) → Dilation
    
    Args:
        mask: Binary mask (uint8, 0/255)
        morph_kernel_size: Kernel size for morphological ops
    
    Returns:
        Cleaned binary mask (uint8, 0/255)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    
    # Opening: remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Dilation: expand regions slightly
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def connected_component_analysis(mask):
    """
    Label connected components in the binary mask.
    
    Args:
        mask: Binary mask (uint8, 0/255)
    
    Returns:
        Tuple: (labeled_map, num_labels, centroids, blob_sizes)
        - labeled_map: Each pixel labeled with component ID (0 = background)
        - num_labels: Number of components found
        - centroids: List of (x, y) centroids per component
        - blob_sizes: List of pixel counts per component
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    blob_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background (index 0)
    
    return labels, num_labels, centroids[1:], blob_sizes


def extract_hand_regions(mask, top_n=2):
    """
    Extract the top N largest connected components as hand regions.
    
    Args:
        mask: Binary mask (uint8, 0/255)
        top_n: Number of hand regions to extract (usually 2)
    
    Returns:
        List of tuples: [(contour_left, bbox_left), (contour_right, bbox_right)]
    """
    labels, num_labels, centroids, blob_sizes = connected_component_analysis(mask)
    
    if len(blob_sizes) == 0:
        return []
    
    # Sort by size descending, take top N
    top_indices = np.argsort(blob_sizes)[::-1][:top_n]
    
    hand_regions = []
    for idx in top_indices:
        label_id = idx + 1  # Component IDs start at 1
        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            bbox = cv2.boundingRect(contour)
            hand_regions.append((contour, bbox))
    
    return hand_regions


def extract_face_region(mask):
    """
    Extract the largest connected component as the face region.
    
    Args:
        mask: Binary mask (uint8, 0/255)
    
    Returns:
        Tuple: (contour, bbox) or (None, None) if no face found
    """
    labels, num_labels, centroids, blob_sizes = connected_component_analysis(mask)
    
    if len(blob_sizes) == 0:
        return None, None
    
    # Get largest component
    largest_idx = np.argmax(blob_sizes)
    label_id = largest_idx + 1
    component_mask = (labels == label_id).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(contour)
        return contour, bbox
    
    return None, None
