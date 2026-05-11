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
    # Handle both Tasks API (list) and old API (.landmark)
    lm = face_landmarks if isinstance(face_landmarks, list) else (face_landmarks.landmark if face_landmarks else None)
    if lm is None or len(lm) < 468:
        return 0.0
    
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
    # Handle both Tasks API (list) and old API (.landmark)
    lm = face_landmarks if isinstance(face_landmarks, list) else (face_landmarks.landmark if face_landmarks else None)
    if lm is None or len(lm) < 468:
        return False, 0.0
    
    # Inner-lip vertical: 13 = upper inner lip, 14 = lower inner lip
    # (61 / 291 are mouth CORNERS — using them measures lip width, not openness)
    top_lip = lm[13]
    bottom_lip = lm[14]

    openness = abs(bottom_lip.y - top_lip.y)

    # Inner-lip distance is small even when "open" — drop threshold accordingly
    is_open = openness > 0.02

    return is_open, openness


def compute_gaze_direction(face_landmarks):
    """
    Estimate gaze direction (which direction eyes are looking).
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        String: "left", "right", "center", "up", "down"
    """
    # Handle both Tasks API (list) and old API (.landmark)
    lm = face_landmarks if isinstance(face_landmarks, list) else (face_landmarks.landmark if face_landmarks else None)
    if lm is None or len(lm) < 468:
        return "center"
    
    import math
    
    # Right eye inner/outer corners: 133 (right), 33 (left)
    # Left eye inner/outer corners: 362 (right), 263 (left)
    right_eye_inner = lm[133]
    right_eye_outer = lm[33]
    left_eye_inner = lm[362]
    left_eye_outer = lm[263]
    
    # Eye centers
    right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
    left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
    avg_eye_center_x = (right_eye_center_x + left_eye_center_x) / 2
    
    # Estimate iris center (approximate from eye region midpoint)
    avg_iris_x = (right_eye_center_x + left_eye_center_x) / 2
    
    # Determine gaze direction
    gaze_offset = avg_iris_x - avg_eye_center_x
    
    if gaze_offset < -0.05:
        return "left"
    elif gaze_offset > 0.05:
        return "right"
    else:
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
    # Handle both Tasks API (list) and old API (.landmark)
    lm = face_landmarks if isinstance(face_landmarks, list) else (face_landmarks.landmark if face_landmarks else None)
    if lm is None or len(lm) < 468:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    import math

    # Key landmarks: nose (1), left/right eye outer (33/263), chin (199)
    nose = lm[1]
    left_eye = lm[33]
    right_eye = lm[263]
    chin = lm[199]
    
    # Roll: tilt of eyes relative to horizontal
    roll = math.atan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x) * 180 / math.pi
    
    # Yaw: left-right rotation (nose to eye midpoint horizontal)
    eye_center_x = (left_eye.x + right_eye.x) / 2
    yaw_offset = nose.x - eye_center_x
    yaw = yaw_offset * 180  # Scale to degrees
    
    # Pitch: up-down rotation (nose to chin vertical)
    eye_center_y = (left_eye.y + right_eye.y) / 2
    pitch_offset = nose.y - eye_center_y
    pitch = pitch_offset * 180  # Scale to degrees
    
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def lip_to_nose_distance(face_landmarks):
    """
    Compute Euclidean distance from lip centroid to nose tip.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        Float: Distance in pixels
    """
    # Handle both Tasks API (list) and old API (.landmark)
    lm = face_landmarks if isinstance(face_landmarks, list) else (face_landmarks.landmark if face_landmarks else None)
    if lm is None or len(lm) < 300:
        return 0.0
    
    import math
    
    # Nose tip (1), lip center (inner lip midpoints 13/14, not corners)
    nose = lm[1]
    lip_top = lm[13]
    lip_bottom = lm[14]
    lip_center_x = (lip_top.x + lip_bottom.x) / 2
    lip_center_y = (lip_top.y + lip_bottom.y) / 2
    
    # Euclidean distance
    dist = math.sqrt((nose.x - lip_center_x)**2 + (nose.y - lip_center_y)**2)
    
    # Normalize by typical face region (~0.4 in normalized coords)
    normalized_dist = min(dist / 0.4, 1.0)
    return normalized_dist


def lip_to_finger_distance(face_landmarks, hand_landmarks):
    """
    Compute minimum distance from any lip landmark to any hand landmark.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
        hand_landmarks: MediaPipe hand landmarks (21 points)
    
    Returns:
        Float: Minimum distance in pixels
    """
    if face_landmarks is None or hand_landmarks is None:
        return 1.0
    
    import math
    # Handle both Tasks API (list) and old API (.landmark)
    face_lm = face_landmarks if isinstance(face_landmarks, list) else face_landmarks.landmark
    hand_lm = hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks.landmark
    
    # Lip center (inner lip top/bottom, not corners)
    lip_top = face_lm[13]
    lip_bottom = face_lm[14]
    lip_x = (lip_top.x + lip_bottom.x) / 2
    lip_y = (lip_top.y + lip_bottom.y) / 2
    
    # Find minimum distance from any finger tip to lips
    min_dist = float('inf')
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
    
    for tip_idx in finger_tips:
        tip = hand_lm[tip_idx]
        dist = math.sqrt((lip_x - tip.x)**2 + (lip_y - tip.y)**2)
        min_dist = min(min_dist, dist)
    
    # Normalize
    normalized_dist = min(min_dist / 0.2, 1.0)  # 0.2 is typical finger-to-mouth distance
    return normalized_dist


def eye_aspect_ratio(face_landmarks):
    """
    Compute eye aspect ratio (blink detector).
    Lower ratio = eyes more closed.
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        Float: Average EAR of both eyes
    """
    # Handle both Tasks API (list) and old API (.landmark)
    lm = face_landmarks if isinstance(face_landmarks, list) else (face_landmarks.landmark if face_landmarks else None)
    if lm is None or len(lm) < 468:
        return 1.0
    
    import math
    
    # Right eye landmarks (upper/lower lids, inner/outer corners)
    # Using approximations: 33=right corner, 133=left corner, 159=upper lid, 145=lower lid
    right_p1 = lm[159]  # Top lid
    right_p2 = lm[145]  # Bottom lid
    right_p3 = lm[33]   # Right corner
    right_p4 = lm[133]  # Left corner
    
    # Left eye landmarks
    left_p1 = lm[386]   # Top lid
    left_p2 = lm[374]   # Bottom lid
    left_p3 = lm[263]   # Right corner
    left_p4 = lm[362]   # Left corner
    
    def calc_ear(p1, p2, p3, p4):
        """Calculate EAR for one eye"""
        vert_dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        horiz_dist = math.sqrt((p3.x - p4.x)**2 + (p3.y - p4.y)**2)
        ear = vert_dist / horiz_dist if horiz_dist > 0 else 0
        return ear
    
    right_ear = calc_ear(right_p1, right_p2, right_p3, right_p4)
    left_ear = calc_ear(left_p1, left_p2, left_p3, left_p4)
    avg_ear = (right_ear + left_ear) / 2

    return avg_ear


def _get_lm(face_landmarks):
    """Internal helper: unwrap Tasks API list or old API .landmark."""
    if face_landmarks is None:
        return None
    return face_landmarks if isinstance(face_landmarks, list) else face_landmarks.landmark


def mouth_aspect_ratio(face_landmarks):
    """
    Width / height of the mouth.
    Small (<1.5) = round-shocked-O (surprised_pikachu).
    Large (>2.5) = wide angry/yelling (woman_yelling_at_cat).
    """
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return 0.0

    import math
    # Mouth corners: 61 (left), 291 (right)
    # Mouth center top/bottom: 13 (upper inner), 14 (lower inner)
    left_corner = lm[61]
    right_corner = lm[291]
    top_center = lm[13]
    bottom_center = lm[14]

    width = math.sqrt((left_corner.x - right_corner.x) ** 2 + (left_corner.y - right_corner.y) ** 2)
    height = math.sqrt((top_center.x - bottom_center.x) ** 2 + (top_center.y - bottom_center.y) ** 2)

    if height < 1e-4:
        return 99.0  # mouth closed -> very large ratio
    return width / height


def classify_mouth_state(face_landmarks):
    """
    Classify mouth into discrete states based on openness and aspect ratio.
    Returns one of: "closed", "smile", "open_round", "open_wide", "open_neutral".
    """
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return "closed"

    is_open, openness = is_mouth_open(face_landmarks)
    smile = is_smiling(face_landmarks)
    ar = mouth_aspect_ratio(face_landmarks)

    if not is_open:
        return "smile" if smile else "closed"

    # Open mouth — classify by shape
    if ar < 1.8:
        return "open_round"   # tall/narrow opening — shocked O
    if ar > 2.6:
        return "open_wide"    # short/wide opening — angry yell
    return "open_neutral"


def is_smiling(face_landmarks):
    """
    Smile detector. A smile pulls mouth corners up and out, so:
      1) average corner Y is above the lip midline by a meaningful fraction
         of face height, OR
      2) mouth width is large relative to face width (toothy smile).
    Either condition triggers True.
    """
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return False

    left_corner = lm[61]
    right_corner = lm[291]
    upper_mid = lm[0]
    lower_mid = lm[17]
    nose = lm[1]
    chin = lm[152]
    left_cheek = lm[234]
    right_cheek = lm[454]

    face_height = abs(chin.y - nose.y)
    face_width = abs(left_cheek.x - right_cheek.x)
    if face_height < 1e-4 or face_width < 1e-4:
        return False

    mid_y = (upper_mid.y + lower_mid.y) / 2
    corner_y_avg = (left_corner.y + right_corner.y) / 2
    # Positive when corners are ABOVE the mouth midline (smile upturn)
    corner_elevation = (mid_y - corner_y_avg) / face_height

    mouth_width = abs(left_corner.x - right_corner.x)
    width_ratio = mouth_width / face_width

    # Either a clear corner upturn OR a wide mouth qualifies
    return corner_elevation > 0.025 or width_ratio > 0.45


def classify_eye_state(face_landmarks):
    """
    Discretize eye_aspect_ratio into a categorical state.
    "closed" / "narrow" / "normal" / "wide".
    Suspicious_cat = narrow. Chloe / pikachu = wide.
    """
    ear = eye_aspect_ratio(face_landmarks)
    if ear < 0.18:
        return "closed"
    if ear < 0.27:
        return "narrow"
    if ear < 0.36:
        return "normal"
    return "wide"


def eyebrow_raise(face_landmarks):
    """
    Vertical distance between eyebrow midpoint and eye top, normalized by face height.
    Larger = eyebrow raised (surprise). Smaller = eyebrow lowered (anger/suspicion).
    """
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return 0.0

    # Right eyebrow midpoint ~ 105, right eye top ~ 159
    # Left  eyebrow midpoint ~ 334, left  eye top ~ 386
    right_brow = lm[105]
    right_eye_top = lm[159]
    left_brow = lm[334]
    left_eye_top = lm[386]

    # Eyebrow-to-eye gap (positive normal value; bigger when raised)
    right_gap = right_eye_top.y - right_brow.y
    left_gap = left_eye_top.y - left_brow.y
    avg_gap = (right_gap + left_gap) / 2

    # Normalize by inter-ocular distance
    import math
    iod = math.sqrt((lm[33].x - lm[263].x) ** 2 + (lm[33].y - lm[263].y) ** 2)
    if iod < 1e-4:
        return 0.0
    return avg_gap / iod


def is_looking_at_camera(face_landmarks, yaw_tolerance=12.0, pitch_tolerance=12.0):
    """
    True when head yaw and pitch are both near zero AND gaze is central.
    Used to distinguish memes that engage camera (drake_yes, zesty) from those that look away.
    """
    if face_landmarks is None:
        return False
    orient = get_face_orientation(face_landmarks)
    gaze = compute_gaze_direction(face_landmarks)
    return (abs(orient["yaw"]) < yaw_tolerance
            and abs(orient["pitch"]) < pitch_tolerance
            and gaze == "center")
