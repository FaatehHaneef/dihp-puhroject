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


def draw_landmarks(frame, landmarks, landmark_type="pose"):
    """
    Draw landmark points and connections on the frame.
    
    Args:
        frame: BGR frame to draw on (modified in-place)
        landmarks: MediaPipe landmarks object
        landmark_type: "pose", "hand", or "face"
    """
    h, w, _ = frame.shape
    
    if landmark_type == "pose":
        # TODO: Draw pose skeleton (specific connections for pose)
        pass
    elif landmark_type == "hand":
        # TODO: Draw hand connections
        pass
    elif landmark_type == "face":
        # TODO: Draw face mesh (if needed, can be sparse)
        pass
    
    # TODO: Draw circles at each landmark point


def draw_debug_info(frame, features, matched_meme=None, confidence=0.0):
    """
    Draw debug text overlays on the frame.
    
    Args:
        frame: BGR frame to draw on (modified in-place)
        features: Dict of extracted features
        matched_meme: Name of matched meme (or None)
        confidence: Confidence score (0-1)
    """
    # TODO: Draw text at top-left showing:
    # - Finger count (left and right)
    # - Head orientation
    # - Matched meme name and confidence
    # - FPS counter
    pass


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
    # TODO: Load video, read next frame, display
    # Returns a VideoCapture object that can be read from each iteration
    pass


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
