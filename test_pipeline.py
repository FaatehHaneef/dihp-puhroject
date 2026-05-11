"""Test the full pipeline with one frame"""
import sys
sys.path.insert(0, r'C:\Users\sfate\OneDrive\Desktop\dip project\src')

import cv2
from pathlib import Path
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options
from mediapipe import Image as MPImage

print("Importing feature extraction modules...")
from preprocessing import adaptive_blur, bgr_to_hsv
from segmentation import get_skin_mask, apply_morphology

print("Initializing camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Loading model...")
model_path = Path.home() / ".cache" / "mediapipe" / "holistic_landmarker.task"
options = HolisticLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
    running_mode=RunningMode.VIDEO
)
holistic = HolisticLandmarker.create_from_options(options)

print("Reading frame...")
ret, frame = cap.read()
if not ret:
    print("ERROR: Cannot read frame")
    sys.exit(1)

print("Processing pipeline...")
frame_blur = adaptive_blur(frame)
frame_hsv = bgr_to_hsv(frame_blur)
skin_mask = get_skin_mask(frame_hsv)
skin_mask = apply_morphology(skin_mask)

print("Running MediaPipe...")
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = MPImage(image_format=1, data=frame_rgb)
results = holistic.detect_for_video(mp_image, 0)

print(f"Results:")
print(f"  pose_landmarks: {len(results.pose_landmarks) if results.pose_landmarks else 0}")
print(f"  left_hand_landmarks: {len(results.left_hand_landmarks) if results.left_hand_landmarks else 0}")
print(f"  right_hand_landmarks: {len(results.right_hand_landmarks) if results.right_hand_landmarks else 0}")
print(f"  face_landmarks: {len(results.face_landmarks) if results.face_landmarks else 0}")

print("SUCCESS - Full pipeline works!")
cap.release()
