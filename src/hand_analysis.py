"""
Hand Analysis — finger states, finger count, hand angle, pointing/above-head/finger-in-mouth checks.
"""

import math


def _hand_lm(hand_landmarks):
    if hand_landmarks is None:
        return None
    return hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks.landmark


def compute_finger_state(landmarks, hand_side="right"):
    """Per-finger extended/curled flags. Tip Y above PIP Y => extended."""
    lm = _hand_lm(landmarks)
    if lm is None or len(lm) < 21:
        return {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}

    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 7, 11, 15, 19]
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]

    state = {}
    for i, name in enumerate(finger_names):
        tip = lm[finger_tips[i]]
        pip = lm[finger_pips[i]]
        state[name] = (pip.y - tip.y) > 0.05
    return state


def count_fingers(landmarks, hand_side="right"):
    """Number of extended fingers (0-5)."""
    if landmarks is None:
        return 0
    return sum(compute_finger_state(landmarks, hand_side).values())


def compute_hand_angle(landmarks):
    """Wrist-to-middle-MCP orientation in degrees [0, 360)."""
    lm = _hand_lm(landmarks)
    if lm is None or len(lm) < 13:
        return 0.0
    wrist = lm[0]
    middle_mcp = lm[9]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    return (math.atan2(dy, dx) * 180 / math.pi + 360) % 360


def is_hand_pointing_forward(hand_landmarks, z_threshold=0.08):
    """Index tip noticeably closer to camera than wrist (drake_yes 'this' gesture)."""
    lm = _hand_lm(hand_landmarks)
    if lm is None or len(lm) < 21:
        return False
    wrist_z = getattr(lm[0], "z", 0.0)
    tip_z = getattr(lm[8], "z", 0.0)
    return (wrist_z - tip_z) > z_threshold


def is_hand_above_head(hand_landmarks, face_landmarks):
    """Wrist above the brow line (105) — covers temple-level hand-in-hair too."""
    hlm = _hand_lm(hand_landmarks)
    if hlm is None or len(hlm) < 1 or face_landmarks is None:
        return False
    flm = face_landmarks if isinstance(face_landmarks, list) else face_landmarks.landmark
    if len(flm) < 106:
        return False
    return hlm[0].y < flm[105].y


def is_finger_in_mouth(face_landmarks, hand_landmarks, threshold=0.05):
    """Any fingertip within `threshold` of the inner-lip centroid."""
    if face_landmarks is None or hand_landmarks is None:
        return False
    flm = face_landmarks if isinstance(face_landmarks, list) else face_landmarks.landmark
    hlm = _hand_lm(hand_landmarks)
    if hlm is None or len(flm) < 468 or len(hlm) < 21:
        return False

    lip_top = flm[13]
    lip_bot = flm[14]
    lip_x = (lip_top.x + lip_bot.x) / 2
    lip_y = (lip_top.y + lip_bot.y) / 2

    for tip_idx in (4, 8, 12, 16, 20):
        tip = hlm[tip_idx]
        dist = math.sqrt((lip_x - tip.x) ** 2 + (lip_y - tip.y) ** 2)
        if dist < threshold:
            return True
    return False
