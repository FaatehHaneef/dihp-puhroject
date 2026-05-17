"""
Face Analysis — head tilt/yaw/pitch, mouth/eye state, smile, gaze, eyebrow raise.
"""

import math


def _get_lm(face_landmarks):
    if face_landmarks is None:
        return None
    return face_landmarks if isinstance(face_landmarks, list) else face_landmarks.landmark


def compute_head_tilt_angle(face_landmarks):
    """Roll angle in degrees from the eye-corner baseline."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return 0.0

    right_eye = lm[133]
    left_eye = lm[362]
    dx = left_eye.x - right_eye.x
    dy = left_eye.y - right_eye.y
    return math.degrees(math.atan2(dy, dx))


def is_mouth_open(face_landmarks, threshold=20):
    """Vertical inner-lip gap; returns (is_open, openness)."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return False, 0.0

    # 13/14 are inner upper/lower lip (61/291 are corners, would measure width)
    top_lip = lm[13]
    bottom_lip = lm[14]
    openness = abs(bottom_lip.y - top_lip.y)
    is_open = openness > 0.02
    return is_open, openness


def compute_gaze_direction(face_landmarks):
    """Returns 'left', 'right', or 'center' based on iris offset from eye center."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return "center"

    right_eye_inner = lm[133]
    right_eye_outer = lm[33]
    left_eye_inner = lm[362]
    left_eye_outer = lm[263]

    right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
    left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
    avg_eye_center_x = (right_eye_center_x + left_eye_center_x) / 2
    avg_iris_x = (right_eye_center_x + left_eye_center_x) / 2
    gaze_offset = avg_iris_x - avg_eye_center_x

    if gaze_offset < -0.05:
        return "left"
    if gaze_offset > 0.05:
        return "right"
    return "center"


def get_face_orientation(face_landmarks):
    """Returns dict with yaw/pitch/roll in degrees."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    nose = lm[1]
    left_eye = lm[33]
    right_eye = lm[263]

    roll = math.atan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x) * 180 / math.pi

    eye_center_x = (left_eye.x + right_eye.x) / 2
    yaw = (nose.x - eye_center_x) * 180

    eye_center_y = (left_eye.y + right_eye.y) / 2
    pitch = (nose.y - eye_center_y) * 180

    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def lip_to_nose_distance(face_landmarks):
    """Distance from inner-lip centroid to nose tip, normalized."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 300:
        return 0.0

    nose = lm[1]
    lip_top = lm[13]
    lip_bottom = lm[14]
    lip_center_x = (lip_top.x + lip_bottom.x) / 2
    lip_center_y = (lip_top.y + lip_bottom.y) / 2

    dist = math.sqrt((nose.x - lip_center_x)**2 + (nose.y - lip_center_y)**2)
    return min(dist / 0.4, 1.0)


def lip_to_finger_distance(face_landmarks, hand_landmarks):
    """Minimum distance from any fingertip to the inner-lip centroid, normalized."""
    if face_landmarks is None or hand_landmarks is None:
        return 1.0

    face_lm = face_landmarks if isinstance(face_landmarks, list) else face_landmarks.landmark
    hand_lm = hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks.landmark

    lip_top = face_lm[13]
    lip_bottom = face_lm[14]
    lip_x = (lip_top.x + lip_bottom.x) / 2
    lip_y = (lip_top.y + lip_bottom.y) / 2

    min_dist = float('inf')
    for tip_idx in (4, 8, 12, 16, 20):
        tip = hand_lm[tip_idx]
        dist = math.sqrt((lip_x - tip.x)**2 + (lip_y - tip.y)**2)
        if dist < min_dist:
            min_dist = dist

    return min(min_dist / 0.2, 1.0)


def eye_aspect_ratio(face_landmarks):
    """Average EAR across both eyes (low = closed, high = wide)."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return 1.0

    # Right eye: 159 top lid, 145 bottom lid, 33/133 corners.
    # Left eye:  386 top lid, 374 bottom lid, 263/362 corners.
    right_p1, right_p2, right_p3, right_p4 = lm[159], lm[145], lm[33], lm[133]
    left_p1, left_p2, left_p3, left_p4 = lm[386], lm[374], lm[263], lm[362]

    def calc_ear(p1, p2, p3, p4):
        vert = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        horiz = math.sqrt((p3.x - p4.x)**2 + (p3.y - p4.y)**2)
        return vert / horiz if horiz > 0 else 0

    return (calc_ear(right_p1, right_p2, right_p3, right_p4)
            + calc_ear(left_p1, left_p2, left_p3, left_p4)) / 2


def mouth_aspect_ratio(face_landmarks):
    """Width / height of the mouth. Closed mouth returns a sentinel large value."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return 0.0

    left_corner = lm[61]
    right_corner = lm[291]
    top_center = lm[13]
    bottom_center = lm[14]

    width = math.sqrt((left_corner.x - right_corner.x) ** 2 + (left_corner.y - right_corner.y) ** 2)
    height = math.sqrt((top_center.x - bottom_center.x) ** 2 + (top_center.y - bottom_center.y) ** 2)

    if height < 1e-4:
        return 99.0
    return width / height


def classify_mouth_state(face_landmarks):
    """Returns 'closed' / 'smile' / 'open_round' / 'open_wide' / 'open_neutral'."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return "closed"

    is_open, _ = is_mouth_open(face_landmarks)
    smile = is_smiling(face_landmarks)
    ar = mouth_aspect_ratio(face_landmarks)

    if not is_open:
        return "smile" if smile else "closed"
    if ar < 1.8:
        return "open_round"
    if ar > 2.6:
        return "open_wide"
    return "open_neutral"


def is_smiling(face_landmarks):
    """True on corner upturn relative to face height OR wide mouth relative to face width."""
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
    corner_elevation = (mid_y - corner_y_avg) / face_height

    mouth_width = abs(left_corner.x - right_corner.x)
    width_ratio = mouth_width / face_width

    return corner_elevation > 0.025 or width_ratio > 0.45


def classify_eye_state(face_landmarks):
    """Discretize EAR into 'closed' / 'narrow' / 'normal' / 'wide'."""
    ear = eye_aspect_ratio(face_landmarks)
    if ear < 0.18:
        return "closed"
    if ear < 0.27:
        return "narrow"
    if ear < 0.36:
        return "normal"
    return "wide"


def eyebrow_raise(face_landmarks):
    """Eyebrow-to-eye-top gap, normalized by inter-ocular distance."""
    lm = _get_lm(face_landmarks)
    if lm is None or len(lm) < 468:
        return 0.0

    right_brow = lm[105]
    right_eye_top = lm[159]
    left_brow = lm[334]
    left_eye_top = lm[386]

    right_gap = right_eye_top.y - right_brow.y
    left_gap = left_eye_top.y - left_brow.y
    avg_gap = (right_gap + left_gap) / 2

    iod = math.sqrt((lm[33].x - lm[263].x) ** 2 + (lm[33].y - lm[263].y) ** 2)
    if iod < 1e-4:
        return 0.0
    return avg_gap / iod


def is_looking_at_camera(face_landmarks, yaw_tolerance=12.0, pitch_tolerance=12.0):
    """Head yaw + pitch near zero AND gaze centered."""
    if face_landmarks is None:
        return False
    orient = get_face_orientation(face_landmarks)
    gaze = compute_gaze_direction(face_landmarks)
    return (abs(orient["yaw"]) < yaw_tolerance
            and abs(orient["pitch"]) < pitch_tolerance
            and gaze == "center")
