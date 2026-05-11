"""
PoseMeme — Main Entry Point
Real-time webcam pose/gesture/expression detection with meme matching.

Phase 1: Foundation — Camera loop, preprocessing, segmentation, landmark overlay
"""

import cv2
import json
import time
import mediapipe as mp
import numpy as np
from pathlib import Path

from preprocessing import histogram_equalize, adaptive_blur, bgr_to_hsv
from segmentation import get_skin_mask, apply_morphology, connected_component_analysis
from hand_analysis import compute_finger_state, count_fingers
from face_analysis import compute_head_tilt_angle, is_mouth_open
from pose_analysis import is_hand_near_face, body_symmetry_score
from meme_matcher import load_meme_config, match_best_meme
from display import show_main_window, show_meme_popup, draw_landmarks, draw_debug_info


def draw_pose_skeleton(frame, landmarks):
    """Draw pose skeleton on frame."""
    h, w, _ = frame.shape
    
    # Define pose connections (body skeleton)
    connections = [
        (11, 12),  # shoulders
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (11, 23), (12, 24),  # torso
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]
    
    # Draw connections
    for start, end in connections:
        if landmarks.landmark[start].visibility > 0.5 and landmarks.landmark[end].visibility > 0.5:
            x1 = int(landmarks.landmark[start].x * w)
            y1 = int(landmarks.landmark[start].y * h)
            x2 = int(landmarks.landmark[end].x * w)
            y2 = int(landmarks.landmark[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw landmarks (joints)
    for lm in landmarks.landmark:
        if lm.visibility > 0.5:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def draw_hand_skeleton(frame, landmarks, hand_side="right"):
    """Draw hand skeleton on frame."""
    h, w, _ = frame.shape
    
    # Hand connections (21 points)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17),  # knuckles
    ]
    
    # Draw connections
    for start, end in connections:
        if start < len(landmarks.landmark) and end < len(landmarks.landmark):
            x1 = int(landmarks.landmark[start].x * w)
            y1 = int(landmarks.landmark[start].y * h)
            x2 = int(landmarks.landmark[end].x * w)
            y2 = int(landmarks.landmark[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    # Draw landmarks
    for lm in landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)


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
    
    print("✓ Camera initialized")
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True
    )
    
    print("✓ MediaPipe loaded")
    
    # Load meme database
    config_path = Path(__file__).parent / "meme_config.json"
    try:
        with open(config_path, 'r') as f:
            meme_db = json.load(f)
        print(f"✓ Loaded {len(meme_db)} memes")
    except FileNotFoundError:
        print(f"Warning: {config_path} not found")
        meme_db = {}
    
    # Meme popup state (debouncing)
    current_meme = None
    meme_lock_frames = 0
    MEME_LOCK_DURATION = 45  # ~1.5 seconds at 30fps
    
    # FPS tracking
    fps_start = time.time()
    frame_count = 0
    fps = 0
    
    print("✓ Starting camera loop (press ESC to exit)...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        frame_count += 1
        
        # FPS calculation
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = 30 / elapsed
            fps_start = time.time()
        
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
            draw_pose_skeleton(frame, results.pose_landmarks)
        
        if results.left_hand_landmarks:
            draw_hand_skeleton(frame, results.left_hand_landmarks, "left")
        
        if results.right_hand_landmarks:
            draw_hand_skeleton(frame, results.right_hand_landmarks, "right")
        
        # Update meme lock
        if meme_lock_frames > 0:
            meme_lock_frames -= 1
        else:
            # Try to match new meme (placeholder for now)
            pass
        
        # Draw debug info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if results.pose_landmarks:
            cv2.putText(frame, "Pose: Detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display main window
        cv2.imshow("DIP Project - Live Pose Detection", frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\nExiting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Done.")


if __name__ == "__main__":
    main()
