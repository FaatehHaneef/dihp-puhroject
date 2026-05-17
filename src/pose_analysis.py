"""
Pose Analysis — shoulder/arm angles, body symmetry, head-yaw fallback, sideways-arm geometry.
"""

import math


def _unwrap(pose_landmarks):
    if pose_landmarks is None:
        return None
    return pose_landmarks if isinstance(pose_landmarks, list) else pose_landmarks.landmark


def _pose_vis(point, default=1.0):
    """Visibility/presence value for a landmark (Tasks API uses 'presence', old API 'visibility')."""
    return getattr(point, "visibility", getattr(point, "presence", default))


def compute_shoulder_angle(pose_landmarks):
    """Shoulder-line angle vs horizontal, in degrees."""
    lm = _unwrap(pose_landmarks)
    if lm is None or len(lm) < 14:
        return 0.0
    dx = lm[12].x - lm[11].x
    dy = lm[12].y - lm[11].y
    return abs(math.atan2(dy, dx) * 180 / math.pi)


def is_hand_near_face(pose_landmarks, hand_landmarks, threshold=100):
    """Any hand landmark within 0.3 normalized of the nose."""
    if hand_landmarks is None or pose_landmarks is None:
        return False
    pose_lm = _unwrap(pose_landmarks)
    hand_lm_list = hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks.landmark
    if len(pose_lm) < 10:
        return False

    nose = pose_lm[0]
    for hand_lm in hand_lm_list:
        dx = hand_lm.x - nose.x
        dy = hand_lm.y - nose.y
        if (dx * dx + dy * dy) ** 0.5 < 0.3:
            return True
    return False


def body_symmetry_score(pose_landmarks):
    """Left/right joint-pair separation score in [0, 1]; 1 = symmetric."""
    lm = _unwrap(pose_landmarks)
    if lm is None or len(lm) < 14:
        return 0.5

    pairs = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]
    total = 0.0
    valid = 0
    for left_idx, right_idx in pairs:
        if left_idx < len(lm) and right_idx < len(lm):
            lp = lm[left_idx]
            rp = lm[right_idx]
            total += math.sqrt((lp.x - rp.x)**2 + (lp.y - rp.y)**2)
            valid += 1
    if valid == 0:
        return 0.5
    avg = total / valid
    return max(0.0, 1.0 - avg * 2)


def get_body_posture_vector(pose_landmarks):
    """Dict of posture booleans: shoulders_level / upright / leaning_forward / arms_raised."""
    lm = _unwrap(pose_landmarks)
    if lm is None or len(lm) < 28:
        return {"head_forward": True, "shoulders_level": True, "upright": True,
                "leaning_forward": False, "arms_raised": False}

    shoulder_mid_y = (lm[11].y + lm[12].y) / 2
    hip_mid_x = (lm[23].x + lm[24].x) / 2
    hip_mid_y = (lm[23].y + lm[24].y) / 2
    shoulder_mid_x = (lm[11].x + lm[12].x) / 2

    spine_angle = math.atan2(hip_mid_y - shoulder_mid_y, hip_mid_x - shoulder_mid_x) * 180 / math.pi

    shoulder_diff = abs(lm[11].y - lm[12].y)
    shoulders_level = shoulder_diff < 0.05
    upright = abs(spine_angle + 90) < 5
    leaning_forward = -80 < spine_angle < -70
    arms_raised = (lm[15].y < lm[11].y) and (lm[16].y < lm[12].y)

    return {"head_forward": True, "shoulders_level": shoulders_level, "upright": upright,
            "leaning_forward": leaning_forward, "arms_raised": arms_raised}


def compute_arm_angle(pose_landmarks, arm_side="right"):
    """Shoulder->elbow->wrist angle in degrees."""
    lm = _unwrap(pose_landmarks)
    if lm is None or len(lm) < 17:
        return 0.0

    if arm_side == "right":
        shoulder, elbow, wrist = lm[12], lm[14], lm[16]
    else:
        shoulder, elbow, wrist = lm[11], lm[13], lm[15]

    v1x = elbow.x - shoulder.x
    v1y = elbow.y - shoulder.y
    v2x = wrist.x - elbow.x
    v2y = wrist.y - elbow.y

    mag1 = math.sqrt(v1x*v1x + v1y*v1y)
    mag2 = math.sqrt(v2x*v2x + v2y*v2y)
    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))
    return math.acos(cos_angle) * 180 / math.pi


def hands_touching(left_hand_landmarks, right_hand_landmarks, threshold=50):
    """Wrists within 0.1 normalized of each other."""
    if left_hand_landmarks is None or right_hand_landmarks is None:
        return False
    left_wrist = left_hand_landmarks[0] if isinstance(left_hand_landmarks, list) else left_hand_landmarks.landmark[0]
    right_wrist = right_hand_landmarks[0] if isinstance(right_hand_landmarks, list) else right_hand_landmarks.landmark[0]
    dist = math.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
    return dist < 0.1


def estimate_head_yaw_from_pose(pose_landmarks):
    """
    Pose-based yaw estimate for when the face mesh fails on profile views.
    Compares nose x against the shoulder midpoint, normalized by shoulder width.
    """
    lm = _unwrap(pose_landmarks)
    if lm is None or len(lm) < 13:
        return 0.0

    nose = lm[0]
    left_shoulder = lm[11]
    right_shoulder = lm[12]

    if _pose_vis(left_shoulder) < 0.5 or _pose_vis(right_shoulder) < 0.5:
        return 0.0

    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_width < 1e-3:
        return 0.0

    offset_ratio = (nose.x - shoulder_mid_x) / shoulder_width
    return offset_ratio * 80.0


def is_arm_extended_sideways(pose_landmarks, arm_side="right", min_horizontal=0.20):
    """Wrist far to the side of the shoulder AND near shoulder height. Gated on visibility."""
    lm = _unwrap(pose_landmarks)
    if lm is None or len(lm) < 17:
        return False

    if arm_side == "right":
        shoulder, wrist = lm[12], lm[16]
    else:
        shoulder, wrist = lm[11], lm[15]

    if _pose_vis(wrist) < 0.6 or _pose_vis(shoulder) < 0.6:
        return False

    horizontal_reach = abs(wrist.x - shoulder.x)
    vertical_offset = abs(wrist.y - shoulder.y)
    return horizontal_reach > min_horizontal and vertical_offset < 0.20
