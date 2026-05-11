import sys
sys.path.insert(0, r'C:\Users\sfate\AppData\Roaming\Python\Python311\site-packages')
import mediapipe
print("MediaPipe path:", mediapipe.__file__)
print("Dir:", [x for x in dir(mediapipe) if not x.startswith('_')])

# Try to access solutions
try:
    from mediapipe import solutions
    print("✓ Got solutions directly!")
    print("Solutions dir:", [x for x in dir(solutions) if not x.startswith('_')][:10])
except Exception as e:
    print(f"✗ Could not import solutions: {e}")
    
try:
    import mediapipe.solutions as solutions
    print("✓ Got mediapipe.solutions!")
except Exception as e:
    print(f"✗ Could not import mediapipe.solutions: {e}")
        
# Try tasks
try:
    from mediapipe.tasks.python.vision import HolisticLandmarker
    print("✓ Got HolisticLandmarker")
except Exception as e:
    print(f"✗ Could not get HolisticLandmarker: {e}")
