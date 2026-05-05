"""
PoseMeme — Main Entry Point
Real-time webcam pose/gesture/expression detection with meme matching.

Phase 1: Foundation — Camera loop, preprocessing, segmentation, landmark overlay
"""

import cv2
import json
import mediapipe as mp
from pathlib import Path
from preprocessing import histogram_equalize, adaptive_blur, bgr_to_hsv
from segmentation import get_skin_mask, apply_morphology, connected_component_analysis
from hand_analysis import compute_finger_state, count_fingers
from face_analysis import compute_head_tilt_angle, is_mouth_open
from pose_analysis import is_hand_near_face, body_symmetry_score
from meme_matcher import load_meme_config, match_best_meme
from display import show_main_window, show_meme_popup, draw_landmarks, draw_debug_info


def main():
    """Main camera loop."""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True
    )
    
    # Load meme database
    config_path = Path(__file__).parent / "meme_config.json"
    try:
        with open(config_path, 'r') as f:
            meme_db = json.load(f)
        print(f"✓ Loaded {len(meme_db)} memes from config")
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, running without memes")
        meme_db = {}
    
    # Meme popup state (debouncing)
    current_meme = None
    meme_lock_frames = 0
    MEME_LOCK_DURATION = 30  # ~1 second at 30fps
    
    print("Starting camera loop... Press ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Preprocess
        frame_blur = adaptive_blur(frame)
        frame_hsv = bgr_to_hsv(frame_blur)
        
        # Segment skin
        skin_mask = get_skin_mask(frame_hsv)
        skin_mask = apply_morphology(skin_mask)
        
        # Extract landmarks (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Draw landmarks on frame (debug)
        if results.pose_landmarks:
            draw_landmarks(frame, results.pose_landmarks, "pose")
        if results.hand_landmarks:
            draw_landmarks(frame, results.hand_landmarks, "hand")
        if results.face_landmarks:
            draw_landmarks(frame, results.face_landmarks, "face")
        
        # Update meme lock
        if meme_lock_frames > 0:
            meme_lock_frames -= 1
        else:
            # Try to match new meme
            # TODO: Extract features and call match_best_meme()
            pass
        
        # Display main window
        show_main_window(frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Exiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    main()
