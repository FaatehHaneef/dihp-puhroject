"""Test MediaPipe Tasks API HolisticLandmarker"""
import cv2
from pathlib import Path
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options
from mediapipe import Image as MPImage

# Initialize model
model_path = Path.home() / ".cache" / "mediapipe" / "holistic_landmarker.task"
print(f"Model path: {model_path}")
print(f"Exists: {model_path.exists()}")

if not model_path.exists():
    print("ERROR: Model not found!")
    exit(1)

print("Creating HolisticLandmarker...")
options = HolisticLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
    running_mode=RunningMode.VIDEO
)
holistic = HolisticLandmarker.create_from_options(options)
print("SUCCESS: HolisticLandmarker created!")

# Test with a frame
print("\nOpening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

print("Reading frame...")
ret, frame = cap.read()
if not ret:
    print("ERROR: Cannot read frame")
    exit(1)

print(f"Frame shape: {frame.shape}")

# Convert to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(f"RGB frame shape: {frame_rgb.shape}")

# Create MediaPipe Image
mp_image = MPImage(image_format=1, data=frame_rgb)
print(f"MP Image created: {mp_image}")

# Run detection
try:
    print("Running detect_for_video...")
    results = holistic.detect_for_video(mp_image, 0)
    print(f"Results type: {type(results)}")
    print(f"Results dir: {[x for x in dir(results) if not x.startswith('_')]}")
    print(f"Has pose_landmarks: {hasattr(results, 'pose_landmarks')}")
    print(f"Has left_hand_landmarks: {hasattr(results, 'left_hand_landmarks')}")
    print(f"Has right_hand_landmarks: {hasattr(results, 'right_hand_landmarks')}")
    print(f"Has face_landmarks: {hasattr(results, 'face_landmarks')}")
    
    if results.pose_landmarks:
        print(f"Pose landmarks count: {len(results.pose_landmarks)}")
        print(f"First landmark: {results.pose_landmarks[0]}")
        print(f"Has x, y, z: {hasattr(results.pose_landmarks[0], 'x')}, {hasattr(results.pose_landmarks[0], 'y')}, {hasattr(results.pose_landmarks[0], 'z')}")
    else:
        print("No pose landmarks detected")
        
    print("\nSUCCESS!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
