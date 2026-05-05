"""
Hand Analysis — Finger detection, hand shape analysis, gesture features.

DIP Concepts: Convex hull, convexity defects, feature extraction
"""

import cv2
import numpy as np


def get_convex_hull(hand_contour):
    """
    Compute convex hull of a hand blob contour.
    
    Args:
        hand_contour: OpenCV contour array
    
    Returns:
        Convex hull points (numpy array)
    """
    if len(hand_contour) < 3:
        return None
    return cv2.convexHull(hand_contour)


def get_convexity_defects(hand_contour, hull):
    """
    Detect convexity defects (finger valleys) in a hand contour.
    
    Args:
        hand_contour: OpenCV contour array
        hull: Convex hull of the contour
    
    Returns:
        Defects array from cv2.convexityDefects (or None if insufficient points)
    """
    if len(hand_contour) < 5 or hull is None:
        return None
    
    defects = cv2.convexityDefects(hand_contour, hull)
    return defects


def compute_finger_state(landmarks, hand_side="right"):
    """
    Determine which fingers are extended or curled using landmark distances.
    
    Finger layout (MediaPipe hand landmarks):
    - Thumb: 0 (wrist), 1-4 (joints/tip)
    - Index: 5-8
    - Middle: 9-12
    - Ring: 13-16
    - Pinky: 17-20
    
    Args:
        landmarks: MediaPipe hand landmarks (21 points)
        hand_side: "right" or "left" (affects thumb logic)
    
    Returns:
        Dict: {
            "thumb": bool,
            "index": bool,
            "middle": bool,
            "ring": bool,
            "pinky": bool
        }
        True = extended, False = curled
    """
    # TODO: Implement landmark-based finger extension detection
    # Compare PIP joint (second knuckle) vs. tip Y-coordinate
    # If tip Y < PIP Y significantly, finger is extended
    
    return {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False
    }


def count_fingers(landmarks, hand_side="right"):
    """
    Count extended fingers using finger state.
    
    Args:
        landmarks: MediaPipe hand landmarks
        hand_side: "right" or "left"
    
    Returns:
        Integer: 0-5 (number of extended fingers)
    """
    finger_state = compute_finger_state(landmarks, hand_side)
    return sum(finger_state.values())


def get_hand_shape_features(hull, contour, landmarks=None):
    """
    Extract hand shape descriptor features.
    
    Args:
        hull: Convex hull of hand contour
        contour: Hand contour
        landmarks: MediaPipe hand landmarks (optional)
    
    Returns:
        Dict: {
            "hull_area": float,
            "contour_area": float,
            "solidity": float,
            "aspect_ratio": float,
            "extent": float
        }
    """
    if hull is None or len(contour) < 5:
        return {}
    
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    
    # Solidity = contour area / hull area
    solidity = float(contour_area) / hull_area if hull_area > 0 else 0
    
    # Aspect ratio from bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Extent = contour area / bounding rect area
    rect_area = w * h
    extent = float(contour_area) / rect_area if rect_area > 0 else 0
    
    return {
        "hull_area": hull_area,
        "contour_area": contour_area,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "extent": extent
    }


def compute_hand_angle(landmarks):
    """
    Compute the angle of the hand (rotation around wrist).
    
    Args:
        landmarks: MediaPipe hand landmarks (21 points)
    
    Returns:
        Angle in degrees (0-360)
    """
    # TODO: Use landmark positions to estimate hand orientation
    # E.g., wrist to middle finger direction
    return 0.0
