"""
Display — Window management, landmark overlays, meme popup handling.

DIP Concepts: Image synthesis, overlay rendering
"""

import cv2
import numpy as np
from pathlib import Path


# Global state for meme window
_meme_window_open = False
_meme_window_x = 1400
_meme_window_y = 100

# Cache for the currently-displayed meme image so we can overlay it onto each frame
_overlay_cache = {"path": None, "img": None}


def overlay_meme_on_frame(frame, meme_path, max_w=320, max_h=320, margin=15):
    """
    Composite a meme image onto the top-right corner of `frame`.

    Replaces the separate popup window which could land off-screen.
    Caches the loaded image so we don't re-read from disk every frame.
    Returns True on success, False otherwise.
    """
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

    # Scale meme to fit within max_w x max_h while keeping aspect ratio
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

    # Border + paste
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
    frame[y1:y2, x1:x2] = meme_img

    # Caption strip
    label_path = Path(meme_path).stem
    cv2.rectangle(frame, (x1, y2 - 28), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, label_path, (x1 + 6, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return True


def clear_overlay_cache():
    _overlay_cache["path"] = None
    _overlay_cache["img"] = None


def draw_landmarks(frame, landmarks, landmark_type="pose"):
    """
    Draw landmark points and connections on the frame.
    
    Args:
        frame: BGR frame to draw on (modified in-place)
        landmarks: MediaPipe landmarks object
        landmark_type: "pose", "hand", or "face"
    """
    if landmarks is None:
        return
    
    h, w, _ = frame.shape
    
    # Pose skeleton connections
    pose_connections = [
        (11, 12), (12, 24), (24, 23), (23, 11),  # Torso
        (11, 13), (13, 15),  # Right arm
        (12, 14), (14, 16),  # Left arm
        (24, 26), (26, 28),  # Right leg
        (23, 25), (25, 27),  # Left leg
        (5, 6), (5, 7), (7, 9),  # Right hand
        (6, 8), (8, 10),  # Left hand
    ]
    
    # Hand skeleton connections
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    ]
    
    # Handle both Tasks API (list) and old API (.landmark)
    lm = landmarks if isinstance(landmarks, list) else landmarks.landmark
    
    # Draw connections
    if landmark_type == "pose":
        for start, end in pose_connections:
            if start < len(lm) and end < len(lm):
                p1 = lm[start]
                p2 = lm[end]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    elif landmark_type == "hand":
        for start, end in hand_connections:
            if start < len(lm) and end < len(lm):
                p1 = lm[start]
                p2 = lm[end]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    # Draw circles at each landmark point
    for i, pt in enumerate(lm):
        x, y = int(pt.x * w), int(pt.y * h)
        color = (0, 255, 0) if landmark_type == "pose" else (255, 0, 0)
        cv2.circle(frame, (x, y), 3, color, -1)


def draw_debug_info(frame, features, matched_meme=None, confidence=0.0):
    """
    Draw debug text overlays on the frame.
    
    Args:
        frame: BGR frame to draw on (modified in-place)
        features: Dict of extracted features
        matched_meme: Name of matched meme (or None)
        confidence: Confidence score (0-1)
    """
    if features is None:
        features = {}
    
    h, w, _ = frame.shape
    
    # Text position and font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (0, 255, 0)
    y_offset = 30
    x_pos = 10
    
    def line(text, col=color):
        nonlocal y_offset
        cv2.putText(frame, text, (x_pos, y_offset), font, font_scale, col, thickness)
        y_offset += 22

    line(f"Hands: {features.get('hand_count', 0)}  Fingers: {features.get('fingers_extended', 0)}")
    line(f"Mouth: {features.get('mouth_state', '?')}  Smile: {features.get('is_smiling', False)}")
    line(f"Eyes: {features.get('eye_state', '?')}")
    line(f"Gaze: {features.get('gaze_direction', '?')}  AtCam: {features.get('looking_at_camera', False)}")
    line(f"Tilt: {features.get('head_tilt_angle', 0.0):.1f}  Yaw: {features.get('head_yaw', 0.0):.1f}")
    if features.get("any_finger_in_mouth"):
        line("FingerInMouth", (0, 200, 255))
    if features.get("any_hand_above_head"):
        line("HandAboveHead", (0, 200, 255))
    if features.get("any_hand_pointing_forward"):
        line("PointingFwd", (0, 200, 255))
    if features.get("any_arm_extended_sideways"):
        line("ArmSideways", (0, 200, 255))

    if matched_meme:
        line(f"Meme: {matched_meme} ({confidence:.0%})", (0, 0, 255))


def show_main_window(frame, window_name="PoseMeme"):
    """
    Display the main camera feed window with overlays.
    
    Args:
        frame: BGR frame to display
        window_name: Window title
    """
    cv2.imshow(window_name, frame)


def show_meme_popup(meme_path, x=1400, y=100, window_name="Meme", max_height=400):
    """
    Display meme image in a separate popup window.
    
    Args:
        meme_path: Path to meme image file
        x: Window X position on screen
        y: Window Y position on screen
        window_name: Window title
        max_height: Maximum height to resize to
    
    Returns:
        Bool: True if successfully displayed, False otherwise
    """
    global _meme_window_open, _meme_window_x, _meme_window_y
    
    try:
        meme_img = cv2.imread(str(meme_path))
        if meme_img is None:
            print(f"Warning: Could not load meme image {meme_path}")
            return False
        
        # Resize meme if it's too large
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
    """
    Display video meme (loops while pose holds).
    Call repeatedly in main loop to advance frames.
    
    Args:
        video_path: Path to video file
        x: Window X position on screen
        y: Window Y position on screen
        window_name: Window title
    
    Returns:
        VideoCapture object for continuous frame reading
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        ret, frame = cap.read()
        if ret:
            # Resize if needed
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
    """
    Close the meme popup window.
    
    Args:
        window_name: Window to close
    """
    global _meme_window_open
    try:
        cv2.destroyWindow(window_name)
        _meme_window_open = False
    except Exception as e:
        print(f"Warning: Error closing meme window: {e}")


def cleanup_windows():
    """
    Destroy all cv2 windows gracefully.
    """
    cv2.destroyAllWindows()


def get_meme_window_state():
    """
    Check if meme window is currently open.
    
    Returns:
        Bool: True if meme window exists
    """
    global _meme_window_open
    return _meme_window_open
