"""
PoseMeme — Main Entry Point
Real-time webcam pose/gesture/expression detection with meme matching.

Phase 1: Foundation — Camera loop, preprocessing, segmentation, landmark overlay
"""

import cv2
import json
import time
import threading
import platform
import ctypes
from ctypes import create_unicode_buffer
import numpy as np
from pathlib import Path
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options
from mediapipe import Image as MPImage
import mediapipe as mp

if platform.system().lower() == "windows":
    _winmm = ctypes.WinDLL("winmm")
else:
    _winmm = None

from preprocessing import histogram_equalize, adaptive_blur, bgr_to_hsv
from segmentation import get_skin_mask, apply_morphology, connected_component_analysis
from hand_analysis import (compute_finger_state, count_fingers, compute_hand_angle,
                           get_hand_shape_features, is_hand_pointing_forward,
                           is_hand_above_head, is_finger_in_mouth)
from face_analysis import (compute_head_tilt_angle, is_mouth_open, compute_gaze_direction,
                           get_face_orientation, lip_to_nose_distance, lip_to_finger_distance, eye_aspect_ratio,
                           mouth_aspect_ratio, classify_mouth_state, is_smiling,
                           classify_eye_state, eyebrow_raise, is_looking_at_camera)
from pose_analysis import (is_hand_near_face, body_symmetry_score, compute_shoulder_angle,
                          get_body_posture_vector, compute_arm_angle, hands_touching,
                          is_arm_extended_sideways, estimate_head_yaw_from_pose)
from meme_matcher import (load_meme_config, match_best_meme, evaluate_triggers, compute_confidence,
                         match_best_meme_by_template)
from display import (show_main_window, show_meme_popup, draw_landmarks, draw_debug_info,
                     close_meme_window, cleanup_windows, overlay_meme_on_frame, clear_overlay_cache)


def _mci_send(command):
    buffer = create_unicode_buffer(255)
    error = _winmm.mciSendStringW(command, buffer, 254, 0)
    return error, buffer.value


def _mci_error(error):
    buffer = create_unicode_buffer(255)
    _winmm.mciGetErrorStringW(error, buffer, 254)
    return buffer.value


def _play_audio(paths, state):
    try:
        if _winmm is None:
            raise RuntimeError("Windows audio not available")
        last_error = ""
        for path in paths:
            alias = f"meme_{int(time.time() * 1000)}"
            err, _ = _mci_send(f'open "{path}" alias {alias}')
            if err:
                last_error = _mci_error(err)
                _mci_send(f"close {alias}")
                continue
            err, _ = _mci_send(f"play {alias} wait")
            _mci_send(f"close {alias}")
            if err:
                last_error = _mci_error(err)
                continue
            state["playing"] = False
            return
        raise RuntimeError(last_error or "No playable audio file found")
    except Exception as exc:
        print(f"Warning: Failed to play audio: {exc}")
    state["playing"] = False


def extract_all_features(results, skin_mask):
    """
    Build a rich feature dict from MediaPipe results.

    Produces both low-level metrics (angles, ratios) and HIGH-LEVEL
    categorical states (expression, mouth_state, eye_state, etc.)
    that the rule-based matcher consumes.
    """
    f = {}

    rh = results.right_hand_landmarks
    lh = results.left_hand_landmarks
    face = results.face_landmarks
    pose = results.pose_landmarks

    # ---- presence flags ----
    f["face_visible"] = face is not None
    f["pose_visible"] = pose is not None
    f["hand_count"] = (1 if rh else 0) + (1 if lh else 0)
    f["right_hand_visible"] = rh is not None
    f["left_hand_visible"] = lh is not None

    # ---- segmentation stats (skin mask) ----
    if skin_mask is not None:
        total_pixels = skin_mask.size
        skin_pixels = int(np.count_nonzero(skin_mask))
        f["skin_coverage"] = (skin_pixels / total_pixels) if total_pixels else 0.0
        _, num_labels, _, blob_sizes = connected_component_analysis(skin_mask)
        f["skin_blob_count"] = max(0, num_labels - 1)
        f["skin_largest_blob_ratio"] = (max(blob_sizes) / total_pixels) if (blob_sizes.size > 0 and total_pixels) else 0.0
    else:
        f["skin_coverage"] = 0.0
        f["skin_blob_count"] = 0
        f["skin_largest_blob_ratio"] = 0.0

    # ---- per-hand low-level ----
    if rh:
        f["fingers_extended_right"] = count_fingers(rh, "right")
        f["hand_angle_right"] = compute_hand_angle(rh)
        f["hand_near_face_right"] = is_hand_near_face(pose, rh)
        f["hand_above_head_right"] = is_hand_above_head(rh, face)
        f["hand_pointing_forward_right"] = is_hand_pointing_forward(rh)
        f["finger_in_mouth_right"] = is_finger_in_mouth(face, rh)
    else:
        f["fingers_extended_right"] = 0
        f["hand_angle_right"] = 0.0
        f["hand_near_face_right"] = False
        f["hand_above_head_right"] = False
        f["hand_pointing_forward_right"] = False
        f["finger_in_mouth_right"] = False

    if lh:
        f["fingers_extended_left"] = count_fingers(lh, "left")
        f["hand_angle_left"] = compute_hand_angle(lh)
        f["hand_near_face_left"] = is_hand_near_face(pose, lh)
        f["hand_above_head_left"] = is_hand_above_head(lh, face)
        f["hand_pointing_forward_left"] = is_hand_pointing_forward(lh)
        f["finger_in_mouth_left"] = is_finger_in_mouth(face, lh)
    else:
        f["fingers_extended_left"] = 0
        f["hand_angle_left"] = 0.0
        f["hand_near_face_left"] = False
        f["hand_above_head_left"] = False
        f["hand_pointing_forward_left"] = False
        f["finger_in_mouth_left"] = False

    f["fingers_extended"] = f["fingers_extended_right"] + f["fingers_extended_left"]

    # ---- aggregate hand states (either-hand convenience flags) ----
    f["any_hand_near_face"] = f["hand_near_face_left"] or f["hand_near_face_right"]
    f["both_hands_near_face"] = f["hand_near_face_left"] and f["hand_near_face_right"]
    f["any_hand_above_head"] = f["hand_above_head_left"] or f["hand_above_head_right"]
    f["any_finger_in_mouth"] = f["finger_in_mouth_left"] or f["finger_in_mouth_right"]
    f["any_hand_pointing_forward"] = f["hand_pointing_forward_left"] or f["hand_pointing_forward_right"]

    # ---- face low-level + high-level ----
    if face:
        f["head_tilt_angle"] = compute_head_tilt_angle(face)
        is_open, openness = is_mouth_open(face)
        f["mouth_open"] = is_open
        f["mouth_openness"] = openness
        f["mouth_aspect_ratio"] = mouth_aspect_ratio(face)
        f["mouth_state"] = classify_mouth_state(face)        # closed | smile | open_round | open_wide | open_neutral
        f["is_smiling"] = is_smiling(face)
        f["eye_state"] = classify_eye_state(face)            # closed | narrow | normal | wide
        f["eye_aspect_ratio"] = eye_aspect_ratio(face)
        f["eyebrow_raise"] = eyebrow_raise(face)
        f["gaze_direction"] = compute_gaze_direction(face)
        orient = get_face_orientation(face)
        f["face_yaw"] = orient["yaw"]
        f["face_pitch"] = orient["pitch"]
        f["face_roll"] = orient["roll"]
        f["looking_at_camera"] = is_looking_at_camera(face)
        f["lip_to_nose_distance"] = lip_to_nose_distance(face)
    else:
        f["head_tilt_angle"] = 0.0
        f["mouth_open"] = False
        f["mouth_openness"] = 0.0
        f["mouth_aspect_ratio"] = 0.0
        f["mouth_state"] = "unknown"
        f["is_smiling"] = False
        f["eye_state"] = "unknown"
        f["eye_aspect_ratio"] = 0.0
        f["eyebrow_raise"] = 0.0
        f["gaze_direction"] = "unknown"
        f["face_yaw"] = 0.0
        f["face_pitch"] = 0.0
        f["face_roll"] = 0.0
        f["looking_at_camera"] = False
        f["lip_to_nose_distance"] = 0.0

    # ---- pose ----
    if pose:
        f["shoulder_angle"] = compute_shoulder_angle(pose)
        f["body_symmetry"] = body_symmetry_score(pose)
        posture = get_body_posture_vector(pose)
        f["posture_upright"] = posture.get("upright", False)
        f["arms_raised"] = posture.get("arms_raised", False)
        f["arm_angle_right"] = compute_arm_angle(pose, "right")
        f["arm_angle_left"] = compute_arm_angle(pose, "left")
        f["arm_extended_sideways_right"] = is_arm_extended_sideways(pose, "right")
        f["arm_extended_sideways_left"] = is_arm_extended_sideways(pose, "left")
        f["any_arm_extended_sideways"] = f["arm_extended_sideways_left"] or f["arm_extended_sideways_right"]
    else:
        f["shoulder_angle"] = 0.0
        f["body_symmetry"] = 0.5
        f["posture_upright"] = False
        f["arms_raised"] = False
        f["arm_angle_right"] = 0.0
        f["arm_angle_left"] = 0.0
        f["arm_extended_sideways_right"] = False
        f["arm_extended_sideways_left"] = False
        f["any_arm_extended_sideways"] = False

    # ---- both-hands interactions ----
    f["hands_touching"] = hands_touching(lh, rh) if (lh and rh) else False

    # ---- unified head yaw: prefer face mesh, fall back to pose-based estimate ----
    # MediaPipe's face mesh fails on profile views, so drake_no etc. need a
    # pose-based fallback that still works when the face is half-turned.
    if face:
        f["head_yaw"] = f["face_yaw"]
    elif pose:
        f["head_yaw"] = estimate_head_yaw_from_pose(pose)
    else:
        f["head_yaw"] = 0.0

    return f


def draw_pose_skeleton(frame, landmarks):
    """Draw pose skeleton on frame. Handles both old and Tasks API formats."""
    h, w, _ = frame.shape
    
    # Handle both landmark list formats (Tasks API returns list directly, old API uses .landmark)
    lm_list = landmarks if isinstance(landmarks, list) else landmarks.landmark
    
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
        if start < len(lm_list) and end < len(lm_list):
            start_lm = lm_list[start]
            end_lm = lm_list[end]
            # Handle both visibility/presence attributes
            start_vis = getattr(start_lm, 'visibility', getattr(start_lm, 'presence', 0.5))
            end_vis = getattr(end_lm, 'visibility', getattr(end_lm, 'presence', 0.5))
            
            if start_vis > 0.5 and end_vis > 0.5:
                x1 = int(start_lm.x * w)
                y1 = int(start_lm.y * h)
                x2 = int(end_lm.x * w)
                y2 = int(end_lm.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw landmarks (joints)
    for lm in lm_list:
        vis = getattr(lm, 'visibility', getattr(lm, 'presence', 0.5))
        if vis > 0.5:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def draw_hand_skeleton(frame, landmarks, hand_side="right"):
    """Draw hand skeleton on frame. Handles both old and Tasks API formats."""
    h, w, _ = frame.shape
    
    # Handle both landmark list formats (Tasks API returns list directly, old API uses .landmark)
    lm_list = landmarks if isinstance(landmarks, list) else landmarks.landmark
    
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
        if start < len(lm_list) and end < len(lm_list):
            x1 = int(lm_list[start].x * w)
            y1 = int(lm_list[start].y * h)
            x2 = int(lm_list[end].x * w)
            y2 = int(lm_list[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    # Draw landmarks
    for lm in lm_list:
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
    
    # Initialize MediaPipe Holistic (Tasks API with downloaded model)
    model_path = Path.home() / ".cache" / "mediapipe" / "holistic_landmarker.task"
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        print("Run: python download_model.py")
        raise FileNotFoundError(f"Model file required: {model_path}")
    
    options = HolisticLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        # Lower confidence floors so hands pressed against face / partially visible still register.
        min_hand_landmarks_confidence=0.3,
        min_face_detection_confidence=0.3,
        min_pose_detection_confidence=0.3,
    )
    holistic = HolisticLandmarker.create_from_options(options)
    
    print("✓ MediaPipe loaded")
    
    # Load meme templates (feature-based matching)
    template_path = Path(__file__).parent / "meme_templates.json"
    try:
        with open(template_path, 'r') as f:
            meme_templates = json.load(f)
        print(f"✓ Loaded {len(meme_templates)} meme templates")
    except FileNotFoundError:
        print(f"Warning: {template_path} not found")
        meme_templates = {}
    
    # Also load meme_config.json for image paths and metadata
    config_path = Path(__file__).parent / "meme_config.json"
    try:
        with open(config_path, 'r') as f:
            meme_config = json.load(f)
    except FileNotFoundError:
        meme_config = {}
    
    # Meme popup state (debouncing)
    current_meme = None
    current_meme_path = None
    meme_lock_frames = 0
    MEME_LOCK_DURATION = 20  # ~0.7 seconds at 30fps — shorter so stale memes clear quickly

    # Audio playback state
    audio_dir = Path(__file__).parent.parent / "audios"
    audio_state = {"playing": False}
    missing_audio_warned = set()
    
    # FPS tracking
    fps_start = time.time()
    frame_count = 0
    fps = 0
    start_time = time.time()  # For video timestamp in Tasks API
    
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
        frame_eq = histogram_equalize(frame)
        frame_blur = adaptive_blur(frame_eq)
        frame_hsv = bgr_to_hsv(frame_blur)
        
        # Segment skin
        skin_mask = get_skin_mask(frame_hsv)
        skin_mask = apply_morphology(skin_mask)
        
        # Extract landmarks (MediaPipe Tasks API - VIDEO mode)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=1, data=frame_rgb)  # Format 1 = RGB
        timestamp_ms = int((time.time() - start_time) * 1000)
        results = holistic.detect_for_video(mp_image, timestamp_ms)
        
        # Draw landmarks on frame (debug)
        if results.pose_landmarks:
            draw_pose_skeleton(frame, results.pose_landmarks)
        
        if results.left_hand_landmarks:
            draw_hand_skeleton(frame, results.left_hand_landmarks, "left")
        
        if results.right_hand_landmarks:
            draw_hand_skeleton(frame, results.right_hand_landmarks, "right")
        
        # Anything to analyze this frame?
        has_any = (results.face_landmarks
                   or results.left_hand_landmarks
                   or results.right_hand_landmarks
                   or results.pose_landmarks)

        features = extract_all_features(results, skin_mask) if has_any else None

        audio_playing = audio_state["playing"]

        # Update meme lock
        if meme_lock_frames > 0 and not audio_playing:
            meme_lock_frames -= 1
            if meme_lock_frames == 0:
                # Lock expired — clear current meme so debug overlay stops showing stale name
                current_meme = None
                current_meme_path = None
                clear_overlay_cache()
        elif features is not None and not audio_playing:
            matched_meme, confidence = match_best_meme_by_template(features, meme_templates, threshold=0.55)

            if matched_meme:
                current_meme = matched_meme
                meme_lock_frames = MEME_LOCK_DURATION

                template = meme_templates.get(matched_meme, {})
                meme_file = template.get("image", "")
                meme_path = Path(__file__).parent.parent / "memes" / meme_file

                if meme_path.exists():
                    current_meme_path = meme_path
                    clear_overlay_cache()  # force reload for new meme
                else:
                    current_meme_path = None
                    print(f"Warning: Meme file not found: {meme_path}")

                audio_paths = []
                for ext in (".wav", ".mp3", ".m4a"):
                    candidate = audio_dir / f"{matched_meme}{ext}"
                    if candidate.exists():
                        audio_paths.append(candidate)

                if audio_paths:
                    audio_state["playing"] = True
                    thread = threading.Thread(
                        target=_play_audio,
                        args=(audio_paths, audio_state),
                        daemon=True,
                    )
                    thread.start()
                elif matched_meme not in missing_audio_warned:
                    print(f"Warning: Audio file not found: {audio_paths}")
                    missing_audio_warned.add(matched_meme)

        # In-frame meme overlay (top-right corner)
        if current_meme_path is not None and meme_lock_frames > 0:
            overlay_meme_on_frame(frame, current_meme_path)

        # Draw debug info
        if features is not None:
            if meme_lock_frames > 0 and current_meme:
                from meme_matcher import score_meme
                rules = meme_templates.get(current_meme, {}).get("rules", {})
                _, lock_score = score_meme(features, rules)
                draw_debug_info(frame, features, current_meme, lock_score)
            else:
                draw_debug_info(frame, features)
        
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
