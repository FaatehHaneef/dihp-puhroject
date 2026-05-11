import mediapipe as mp
print(f"MediaPipe module: {mp}")
print(f"Has solutions attr: {hasattr(mp, 'solutions')}")

# Try different ways
import sys
print(f"Python: {sys.version}")

# See what happens when we access it
try:
    x = mp.solutions
    print(f"Got solutions: {x}")
except Exception as e:
    print(f"Error accessing solutions: {e}")
    # Try to force import
    try:
        from mediapipe.python import solutions
        print(f"Got solutions from mediapipe.python: {solutions}")
    except Exception as e2:
        print(f"Failed mediapipe.python: {e2}")
        try:
            from mediapipe import solutions
            print(f"Got solutions from mediapipe: {solutions}")
            print(f"Has holistic: {hasattr(solutions, 'holistic')}")
        except Exception as e3:
            print(f"Failed mediapipe import: {e3}")
