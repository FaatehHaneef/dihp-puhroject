"""
Face Analysis — Head pose, mouth state, gaze direction.

DIP Concepts: Feature extraction, spatial relationships
"""

import cv2
import numpy as np


def compute_head_tilt_angle(face_landmarks):
    """
    Estimate head tilt angle (roll) from face mesh landmarks.
    Uses the line connecting both eye outer corners as reference.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks (478 points)
    
    Returns:
        Angle in degrees (-90 to 90, where 0 is upright)
    """
    if face_landmarks is None or len(face_landmarks.landmark) < 468:
        return 0.0
    
    lm = face_landmarks.landmark
    
    # Eye outer corners: right eye (133), left eye (362)
    right_eye = lm[133]
    left_eye = lm[362]
    
    # Compute angle of line between eyes
    dx = left_eye.x - right_eye.x
    dy = left_eye.y - right_eye.y
    
    import math
    angle = math.degrees(math.atan2(dy, dx))
    
    return angle


def is_mouth_open(face_landmarks, threshold=20):
    """
    Determine if mouth is open and return openness magnitude.
    Uses vertical distance between lips.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
        threshold: Minimum pixel distance to consider "open" (in normalized coords)
    
    Returns:
        Tuple: (is_open: bool, openness: float)
    """
    if face_landmarks is None or len(face_landmarks.landmark) < 468:
        return False, 0.0
    
    lm = face_landmarks.landmark
    
    # Lip landmarks: top (61), bottom (291)
    top_lip = lm[61]
    bottom_lip = lm[291]
    
    # Distance in normalized coordinates (0-1)
    openness = abs(bottom_lip.y - top_lip.y)
    
    # Threshold for "open" (tune this value based on testing)
    is_open = openness > 0.05
    
    return is_open, openness


def compute_gaze_direction(face_landmarks):
    """
    Estimate gaze direction (which direction eyes are looking).
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        String: "left", "right", "center", "up", "down"
    """
    # TODO: Use iris landmarks relative to eye corners
    # Iris position within eye bounds → direction
    return "center"


def get_face_orientation(face_landmarks):
    """
    Compute head yaw angle (left/right turn).
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        Dict: {
            "yaw": float (degrees),
            "pitch": float (degrees),
            "roll": float (degrees)
        }
    """
    # TODO: Use nose, eye, and mouth landmarks for Euler angles
    return {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0
    }


def lip_to_nose_distance(face_landmarks):
    """
    Compute Euclidean distance from lip centroid to nose tip.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        Float: Distance in pixels
    """
    # TODO: Extract lip and nose landmarks, compute distance
    return 0.0


def lip_to_finger_distance(face_landmarks, hand_landmarks):
    """
    Compute minimum distance from any lip landmark to any hand landmark.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
        hand_landmarks: MediaPipe hand landmarks (21 points)
    
    Returns:
        Float: Minimum distance in pixels
    """
    # TODO: Iterate all lip landmarks and hand landmarks, find minimum
    if face_landmarks is None or hand_landmarks is None:
        return float('inf')
    return float('inf')


def eye_aspect_ratio(face_landmarks):
    """
    Compute eye aspect ratio (blink detector).
    Lower ratio = eyes more closed.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        Float: Average EAR of both eyes
    """
    # TODO: Use eye landmark distances
    return 1.0
