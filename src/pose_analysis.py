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
    if pose_landmarks is None or len(pose_landmarks.landmark) < 14:
        return 0.0
    
    import math
    lm = pose_landmarks.landmark
    
    # Shoulder landmarks: 11 (left), 12 (right)
    left_shoulder = lm[11]
    right_shoulder = lm[12]
    
    # Vector from left to right shoulder
    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y
    
    # Angle relative to horizontal
    angle = math.atan2(dy, dx) * 180 / math.pi
    angle = abs(angle)  # Take absolute value
    
    return angle


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
    if pose_landmarks is None or len(pose_landmarks.landmark) < 14:
        return 0.5
    
    import math
    lm = pose_landmarks.landmark
    
    # Symmetric joint pairs: (left_idx, right_idx)
    pairs = [
        (11, 12),  # shoulders
        (13, 14),  # elbows
        (15, 16),  # wrists
        (23, 24),  # hips
        (25, 26),  # knees
        (27, 28),  # ankles
    ]
    
    total_distance = 0
    valid_pairs = 0
    
    for left_idx, right_idx in pairs:
        if left_idx < len(lm) and right_idx < len(lm):
            left_point = lm[left_idx]
            right_point = lm[right_idx]
            
            # Distance between left and right joints
            dist = math.sqrt((left_point.x - right_point.x)**2 + (left_point.y - right_point.y)**2)
            total_distance += dist
            valid_pairs += 1
    
    if valid_pairs == 0:
        return 0.5
    
    avg_distance = total_distance / valid_pairs
    # Convert distance to symmetry score (smaller distance = higher symmetry)
    symmetry = max(0.0, 1.0 - avg_distance * 2)
    
    return symmetry


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
    if pose_landmarks is None or len(pose_landmarks.landmark) < 28:
        return {
            "head_forward": True,
            "shoulders_level": True,
            "upright": True,
            "leaning_forward": False,
            "arms_raised": False
        }
    
    import math
    lm = pose_landmarks.landmark
    
    # Spine angle: shoulder to hip
    shoulder_mid_x = (lm[11].x + lm[12].x) / 2
    shoulder_mid_y = (lm[11].y + lm[12].y) / 2
    hip_mid_x = (lm[23].x + lm[24].x) / 2
    hip_mid_y = (lm[23].y + lm[24].y) / 2
    
    spine_angle = math.atan2(hip_mid_y - shoulder_mid_y, hip_mid_x - shoulder_mid_x) * 180 / math.pi
    
    # Leg angle: hip to knee (right side)
    leg_angle = math.atan2(lm[26].y - lm[24].y, lm[26].x - lm[24].x) * 180 / math.pi
    
    # Arm angle: shoulder to wrist (right side)
    arm_angle = math.atan2(lm[16].y - lm[12].y, lm[16].x - lm[12].x) * 180 / math.pi
    
    # Shoulder level: compare Y coordinates
    shoulder_diff = abs(lm[11].y - lm[12].y)
    shoulders_level = shoulder_diff < 0.05
    
    # Upright: spine angle near vertical (-90 to -85 range)
    upright = abs(spine_angle + 90) < 5
    
    # Leaning forward: spine angle between -80 and -70
    leaning_forward = -80 < spine_angle < -70
    
    # Arms raised: wrist Y < shoulder Y
    arms_raised = (lm[15].y < lm[11].y) and (lm[16].y < lm[12].y)
    
    return {
        "head_forward": True,  # Simplified: always forward
        "shoulders_level": shoulders_level,
        "upright": upright,
        "leaning_forward": leaning_forward,
        "arms_raised": arms_raised
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
    if pose_landmarks is None or len(pose_landmarks.landmark) < 17:
        return 0.0
    
    import math
    lm = pose_landmarks.landmark
    
    # Right side: 12=shoulder, 14=elbow, 16=wrist
    # Left side: 11=shoulder, 13=elbow, 15=wrist
    if arm_side == "right":
        shoulder = lm[12]
        elbow = lm[14]
        wrist = lm[16]
    else:
        shoulder = lm[11]
        elbow = lm[13]
        wrist = lm[15]
    
    # Vectors: shoulder->elbow and elbow->wrist
    v1_x = elbow.x - shoulder.x
    v1_y = elbow.y - shoulder.y
    v2_x = wrist.x - elbow.x
    v2_y = wrist.y - elbow.y
    
    # Dot product and magnitudes
    dot = v1_x * v2_x + v1_y * v2_y
    mag1 = math.sqrt(v1_x**2 + v1_y**2)
    mag2 = math.sqrt(v2_x**2 + v2_y**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Angle between vectors
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
    angle = math.acos(cos_angle) * 180 / math.pi
    
    return angle


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
    
    import math
    
    # Use wrist landmarks (landmark 0 in hand coords corresponds to wrist)
    left_wrist = left_hand_landmarks.landmark[0]
    right_wrist = right_hand_landmarks.landmark[0]
    
    # Euclidean distance between wrists
    dist = math.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
    
    # Threshold for "touching"
    return dist < 0.1
