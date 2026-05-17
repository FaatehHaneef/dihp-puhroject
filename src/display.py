"""
Display — meme image/video overlay, landmark drawing, debug panels.
"""

import cv2
import time
import numpy as np
from pathlib import Path


_meme_window_open = False
_meme_window_x = 1400
_meme_window_y = 100

_overlay_cache = {"path": None, "img": None}

# Video meme cache; advances by wall-clock so it plays at native fps.
_video_cache = {
    "path": None,
    "cap": None,
    "fps": 30.0,
    "total_frames": 0,
    "start_time": None,
    "last_frame": None,
}


def overlay_meme_on_frame(frame, meme_path, max_w=320, max_h=320, margin=15):
    """Composite a meme image onto the top-right of `frame`. Cached across frames."""
    if meme_path is None:
        return False

    meme_path = str(meme_path)
    if _overlay_cache["path"] != meme_path:
        img = cv2.imread(meme_path)
        if img is None:
            return False
        _overlay_cache["path"] = meme_path
        _overlay_cache["img"] = img

    meme_img = _overlay_cache["img"]
    if meme_img is None:
        return False

    mh, mw = meme_img.shape[:2]
    scale = min(max_w / mw, max_h / mh, 1.0)
    if scale < 1.0:
        meme_img = cv2.resize(meme_img, (int(mw * scale), int(mh * scale)))

    fh, fw = frame.shape[:2]
    rh, rw = meme_img.shape[:2]
    x1 = fw - rw - margin
    y1 = margin
    x2 = x1 + rw
    y2 = y1 + rh
    if x1 < 0 or y1 < 0:
        return False

    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
    frame[y1:y2, x1:x2] = meme_img
    return True


def clear_overlay_cache():
    _overlay_cache["path"] = None
    _overlay_cache["img"] = None


def overlay_video_on_frame(frame, video_path, max_w=320, max_h=320, margin=15):
    """Composite the current video-meme frame onto the top-right of `frame`, advancing by wall-clock."""
    if video_path is None:
        return False

    video_path = str(video_path)

    if _video_cache["path"] != video_path:
        if _video_cache["cap"] is not None:
            _video_cache["cap"].release()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        _video_cache.update({
            "path": video_path,
            "cap": cap,
            "fps": fps,
            "total_frames": total,
            "start_time": time.time(),
            "last_frame": None,
        })

    cap = _video_cache["cap"]
    fps = _video_cache["fps"]
    total = _video_cache["total_frames"]

    elapsed = time.time() - _video_cache["start_time"]
    target_idx = int(elapsed * fps) % total
    current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Seek only when drifted — H.264 keyframe seeks are expensive.
    if target_idx != current_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

    ret, vframe = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, vframe = cap.read()
        if not ret:
            if _video_cache["last_frame"] is None:
                return False
            vframe = _video_cache["last_frame"]
    else:
        _video_cache["last_frame"] = vframe

    mh, mw = vframe.shape[:2]
    scale = min(max_w / mw, max_h / mh, 1.0)
    if scale < 1.0:
        vframe = cv2.resize(vframe, (int(mw * scale), int(mh * scale)))

    fh, fw = frame.shape[:2]
    rh, rw = vframe.shape[:2]
    x1 = fw - rw - margin
    y1 = margin
    x2 = x1 + rw
    y2 = y1 + rh
    if x1 < 0 or y1 < 0:
        return False

    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
    frame[y1:y2, x1:x2] = vframe
    return True


def clear_video_cache():
    if _video_cache["cap"] is not None:
        _video_cache["cap"].release()
    _video_cache["path"] = None
    _video_cache["cap"] = None
    _video_cache["start_time"] = None
    _video_cache["last_frame"] = None


def draw_landmarks(frame, landmarks, landmark_type="pose"):
    """Draw skeleton connections and joint points on `frame`."""
    if landmarks is None:
        return

    h, w, _ = frame.shape

    pose_connections = [
        (11, 12), (12, 24), (24, 23), (23, 11),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (24, 26), (26, 28), (23, 25), (25, 27),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    ]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    lm = landmarks if isinstance(landmarks, list) else landmarks.landmark

    if landmark_type == "pose":
        for start, end in pose_connections:
            if start < len(lm) and end < len(lm):
                p1, p2 = lm[start], lm[end]
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
    elif landmark_type == "hand":
        for start, end in hand_connections:
            if start < len(lm) and end < len(lm):
                p1, p2 = lm[start], lm[end]
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)), (255, 0, 0), 1)

    color = (0, 255, 0) if landmark_type == "pose" else (255, 0, 0)
    for pt in lm:
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 3, color, -1)


def _draw_panel(frame, x, y, w, h, alpha=0.55):
    """Translucent dark panel with thin border."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)


def _draw_section(frame, x, y, w, title, rows, title_color=(0, 200, 255)):
    """Labelled panel with header strip and rows. Returns the section's bottom y."""
    pad_x = 12
    line_h = 20
    title_h = 26
    bottom_pad = 8
    h = title_h + len(rows) * line_h + bottom_pad

    _draw_panel(frame, x, y, w, h)

    cv2.rectangle(frame, (x, y), (x + w, y + title_h), (35, 35, 35), -1)
    cv2.line(frame, (x, y + title_h), (x + w, y + title_h), (90, 90, 90), 1)
    cv2.putText(frame, title, (x + pad_x, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_color, 1, cv2.LINE_AA)

    for i, row in enumerate(rows):
        label, value, color = row
        text = f"{label}: {value}" if label else str(value)
        y_pos = y + title_h + i * line_h + 16
        cv2.putText(frame, text, (x + pad_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return y + h


def _draw_meme_banner(frame, matched_meme, confidence):
    """Bottom-center banner with matched meme name and confidence bar."""
    fh, fw, _ = frame.shape
    bw = 360
    bh = 56
    x = (fw - bw) // 2
    y = fh - bh - 18

    _draw_panel(frame, x, y, bw, bh, alpha=0.70)

    cv2.rectangle(frame, (x, y), (x + bw, y + 22), (40, 40, 40), -1)
    cv2.putText(frame, "MATCHED MEME", (x + 10, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, matched_meme, (x + 10, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    bar_x1 = x + bw - 110
    bar_x2 = x + bw - 12
    bar_y1 = y + 36
    bar_y2 = y + 48
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), 1)
    fill = int((bar_x2 - bar_x1) * max(0.0, min(1.0, confidence)))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill, bar_y2), (0, 255, 100), -1)
    cv2.putText(frame, f"{confidence:.0%}", (bar_x1, bar_y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


def draw_debug_info(frame, features, matched_meme=None, confidence=0.0, system_info=None):
    """Stacked SYSTEM / FACE / HANDS panels on the left, meme banner at bottom-center."""
    if features is None:
        features = {}
    system_info = system_info or {}

    panel_x = 10
    panel_y = 10
    panel_w = 230
    gap = 8
    green = (0, 255, 120)
    white = (230, 230, 230)
    orange = (0, 165, 255)

    system_rows = [
        ("FPS",   f"{system_info.get('fps', 0.0):.1f}", green),
        ("Frame", f"{system_info.get('frame', 0)}",     white),
        ("Pose",  "Detected" if system_info.get('pose_detected') else "—",
                  green if system_info.get('pose_detected') else (120, 120, 120)),
    ]
    bottom = _draw_section(frame, panel_x, panel_y, panel_w, "SYSTEM", system_rows)

    face_rows = [
        ("Mouth",  features.get("mouth_state", "?"),                white),
        ("Smile",  str(features.get("is_smiling", False)),          white),
        ("Eyes",   features.get("eye_state", "?"),                  white),
        ("Gaze",   features.get("gaze_direction", "?"),             white),
        ("AtCam",  str(features.get("looking_at_camera", False)),   white),
        ("Tilt",   f"{features.get('head_tilt_angle', 0.0):.1f}°",  white),
        ("Yaw",    f"{features.get('head_yaw', 0.0):.1f}°",         white),
    ]
    bottom = _draw_section(frame, panel_x, bottom + gap, panel_w, "FACE", face_rows)

    hand_rows = [
        ("Count",   str(features.get("hand_count", 0)),       white),
        ("Fingers", str(features.get("fingers_extended", 0)), white),
    ]
    events = []
    if features.get("any_finger_in_mouth"):       events.append("FingerInMouth")
    if features.get("any_hand_above_head"):       events.append("HandAboveHead")
    fwd_l = features.get("hand_pointing_forward_left", False)
    fwd_r = features.get("hand_pointing_forward_right", False)
    if fwd_l and fwd_r:   events.append("PointFwd:LR")
    elif fwd_l:           events.append("PointFwd:L")
    elif fwd_r:           events.append("PointFwd:R")
    if features.get("any_arm_extended_sideways"): events.append("ArmSideways")
    if events:
        hand_rows.append((None, " | ".join(events), orange))
    else:
        hand_rows.append((None, "(no events)", (120, 120, 120)))

    _draw_section(frame, panel_x, bottom + gap, panel_w, "HANDS", hand_rows)

    if matched_meme:
        _draw_meme_banner(frame, matched_meme, confidence)


def show_main_window(frame, window_name="PoseMeme"):
    cv2.imshow(window_name, frame)


def show_meme_popup(meme_path, x=1400, y=100, window_name="Meme", max_height=400):
    """Standalone meme image popup window (legacy; in-frame overlay is preferred)."""
    global _meme_window_open, _meme_window_x, _meme_window_y

    try:
        meme_img = cv2.imread(str(meme_path))
        if meme_img is None:
            print(f"Warning: Could not load meme image {meme_path}")
            return False

        h, w = meme_img.shape[:2]
        if h > max_height:
            scale = max_height / h
            meme_img = cv2.resize(meme_img, (int(w * scale), max_height))

        cv2.imshow(window_name, meme_img)
        cv2.moveWindow(window_name, x, y)
        _meme_window_open = True
        _meme_window_x = x
        _meme_window_y = y
        return True
    except Exception as e:
        print(f"Error displaying meme: {e}")
        return False


def show_meme_video(video_path, x=1400, y=100, window_name="Meme"):
    """Standalone meme video popup (legacy; in-frame overlay_video_on_frame is preferred)."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            if h > 400:
                scale = 400 / h
                frame = cv2.resize(frame, (int(w * scale), 400))
            cv2.imshow(window_name, frame)
            cv2.moveWindow(window_name, x, y)
        return cap
    except Exception as e:
        print(f"Error displaying meme video: {e}")
        return None


def close_meme_window(window_name="Meme"):
    global _meme_window_open
    try:
        cv2.destroyWindow(window_name)
        _meme_window_open = False
    except Exception as e:
        print(f"Warning: Error closing meme window: {e}")


def cleanup_windows():
    cv2.destroyAllWindows()


def get_meme_window_state():
    return _meme_window_open
