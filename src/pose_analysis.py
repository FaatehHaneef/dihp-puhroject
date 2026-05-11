"""
Pose Analysis — Body posture, hand-to-face proximity, symmetry.

DIP Concepts: Spatial relationships, feature extraction
"""

import cv2
import numpy as np


def compute_shoulder_angle(pose_landmarks):
    """
    Compute shoulder and torso angle for posture detection.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks (33 points)
    
    Returns:
        Float: Angle in degrees
    """
    # TODO: Use shoulder, hip, and neck landmarks
    # Compute angle of torso relative to vertical
    return 0.0


def is_hand_near_face(pose_landmarks, hand_landmarks, threshold=100):
    """
    Check if hand is near face (wrist or hand points within threshold distance).
    
    Args:
        pose_landmarks: MediaPipe pose landmarks
        hand_landmarks: MediaPipe hand landmarks (21 points)
        threshold: Maximum distance in normalized coordinates (0-1 scale)
    
    Returns:
        Bool: True if any hand point is near face
    """
    if hand_landmarks is None or pose_landmarks is None:
        return False
    
    if len(pose_landmarks.landmark) < 10:
        return False
    
    # Face center (use nose as reference)
    nose = pose_landmarks.landmark[0]
    face_x, face_y = nose.x, nose.y
    
    # Check distance from all hand landmarks to face
    threshold_norm = threshold / 1000  # Convert pixel threshold to normalized
    
    for hand_lm in hand_landmarks.landmark:
        dx = hand_lm.x - face_x
        dy = hand_lm.y - face_y
        distance = (dx**2 + dy**2) ** 0.5
        
        if distance < 0.3:  # Normalized distance threshold
            return True
    
    return False


def body_symmetry_score(pose_landmarks):
    """
    Compute symmetry score of body (0-1).
    1 = perfectly symmetric, 0 = fully asymmetric.
    Uses corresponding left/right joint angles.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks
    
    Returns:
        Float: 0-1 symmetry score
    """
    # TODO: Compare left arm angles vs. right arm angles
    # Compare left leg angles vs. right leg angles
    # Average the similarity
    return 0.5


def get_body_posture_vector(pose_landmarks):
    """
    Extract body posture as a feature vector.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks
    
    Returns:
        Dict: {
            "head_forward": bool,
            "shoulders_level": bool,
            "upright": bool,
            "leaning_forward": bool,
            "arms_raised": bool
        }
    """
    # TODO: Use shoulder, hip, neck, and wrist landmarks
    return {
        "head_forward": True,
        "shoulders_level": True,
        "upright": True,
        "leaning_forward": False,
        "arms_raised": False
    }


def compute_arm_angle(pose_landmarks, arm_side="right"):
    """
    Compute the angle of one arm (shoulder to elbow to wrist).
    
    Args:
        pose_landmarks: MediaPipe pose landmarks
        arm_side: "right" or "left"
    
    Returns:
        Float: Angle in degrees
    """
    # TODO: Extract shoulder, elbow, wrist for the specified arm
    # Compute angle at elbow joint
    return 0.0


def hands_touching(left_hand_landmarks, right_hand_landmarks, threshold=50):
    """
    Check if left and right hand landmarks are within threshold distance.
    
    Args:
        left_hand_landmarks: MediaPipe left hand landmarks
        right_hand_landmarks: MediaPipe right hand landmarks
        threshold: Maximum distance in pixels
    
    Returns:
        Bool: True if hands are close
    """
    if left_hand_landmarks is None or right_hand_landmarks is None:
        return False
    
    # TODO: Compute minimum distance between all pairs of landmarks
    return False
