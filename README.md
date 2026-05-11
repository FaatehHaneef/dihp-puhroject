# PoseMeme - Real-Time Pose/Gesture/Expression to Meme Matching

## Quick Start

### Prerequisites
- Python 3.11+
- Webcam connected
- Dependencies installed: `pip install opencv-python mediapipe numpy`

### Setup (One-Time)

1. **Download MediaPipe model** (13 MB):
```bash
python download_model.py
```
This creates `~/.cache/mediapipe/holistic_landmarker.task`

2. **Verify installation**:
```bash
python src/main.py
```
Should initialize and open a camera feed.

### Architecture

**Offline (Once)**:
- Meme templates extracted → stored in `src/meme_templates.json`
- Each meme: image file + pre-calculated feature signatures

**Runtime (Continuous)**:
- Capture frame from webcam
- Extract landmarks (MediaPipe: 33 pose, 21 hand each, 468 face points)
- Calculate 20+ features (fingers extended, mouth open, head tilt, etc.)
- Compare to meme templates using distance metrics
- Show matched meme if confidence > threshold (0.4)

### System Components

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point - camera loop, orchestration |
| `src/preprocessing.py` | Frame enhancement (blur, brightness, equalization) |
| `src/segmentation.py` | Skin detection via HSV thresholding |
| `src/hand_analysis.py` | Finger detection, hand angle, gestures |
| `src/face_analysis.py` | Head tilt, mouth state, gaze direction |
| `src/pose_analysis.py` | Body posture, symmetry, arm angle |
| `src/meme_matcher.py` | Distance metrics, template-based matching |
| `src/display.py` | Skeleton rendering, meme popups |
| `src/meme_templates.json` | Pre-extracted meme feature signatures |
| `download_model.py` | Fetch holistic_landmarker.task model |

### Running

```bash
cd "c:\Users\sfate\OneDrive\Desktop\dip project"
python src/main.py
```

**Controls**:
- ESC: Exit
- Meme appears when detected pose matches template

### Feature Extraction

All features calculated from MediaPipe landmarks (NO machine learning):
- `fingers_extended_left/right`: 0-5 finger count
- `mouth_open`: boolean, `mouth_openness`: 0-1 degree
- `head_tilt_angle`: -90 to +90 degrees
- `shoulder_angle`: 0-180 degrees
- `body_symmetry`: 0-1 score
- `arm_angle_left/right`: 0-180 degrees  
- `hands_touching`: boolean
- `hand_near_face_left/right`: boolean
- Plus 10+ more...

### Meme Database

8 memes with feature templates:
- `thinking_monkey`, `zesty`, `drake_yes`, `drake_no`
- `woman_yelling_at_cat`, `surprised_pikachu`, `side_eye_chloe`, `suspicious_cat`

### Template-Based Matching Algorithm

1. **Weighted Distance** between user features and each meme template:
   - `mouth_open` (weight: 2.0) - most important
   - `fingers_extended_left/right` (weight: 1.0)
   - `head_tilt_angle` (weight: 0.8)
   - Other features: 0.5 weight

2. **Confidence Conversion**: `confidence = max(0.0, 1.0 - distance)`

3. **Match**: Best meme where `confidence > 0.4` (configurable)

### Debugging

**Test individual components**:
```bash
python test_pipeline.py         # Full pipeline with one frame
python test_tasks_api.py        # MediaPipe Tasks API functionality  
python test_init_only.py        # Initialization only
```

**Known Limitations**:
- Static camera positioning (not mobile)
- Threshold tuning needed for different environments/people
- Outdoor lighting may affect skin detection

### MediaPipe Upgrade Note

Previously used deprecated `mp.solutions.holistic` (old API).
Now uses `HolisticLandmarker` from Tasks API (current standard).
Model: `holistic_landmarker.task` (13 MB, downloaded once)

---

**Status**: MVP complete, template-based matching ready, camera loop functional.
