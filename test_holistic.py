from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options

# Try with minimal options first
try:
    options = HolisticLandmarkerOptions(
        base_options=base_options.BaseOptions(),
        running_mode=RunningMode.VIDEO
    )
    landmarker = HolisticLandmarker.create_from_options(options)
    print("✓ Created HolisticLandmarker!")
except Exception as e:
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
    
    # Try to figure out what model path is needed
    import os
    import urllib.request
    print("\nTrying to download model...")
    # MediaPipe typically uses preset models
    try:
        # Check if there's a preset/auto-download mechanism
        from mediapipe.tasks.python.vision import holistic_landmarker
        print("Available in holistic_landmarker:", [x for x in dir(holistic_landmarker) if 'model' in x.lower() or 'preset' in x.lower()])
    except:
        pass
