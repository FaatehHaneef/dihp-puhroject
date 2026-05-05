# DIP Project — Complete Dependencies Guide

This document lists all dependencies required to run and develop the DIP project (webcam pose/gesture/meme matching system). Share this with collaborators to ensure everyone has the same environment.

---

## Quick Setup (All at Once)

```bash
pip install opencv-python numpy mediapipe pytest matplotlib pillow
```

Then verify:
```bash
python -c "import cv2, numpy, mediapipe, pytest, matplotlib, PIL; print('✓ All dependencies installed')"
```

---

## CORE DEPENDENCIES (Required to Run)

### 1. **opencv-python** (≥ 4.0)
- **What it is:** Open Source Computer Vision library
- **Why we need it:** 
  - Camera capture (`cv2.VideoCapture`) for real-time webcam input
  - Image preprocessing (blur, histogram equalization, color conversion)
  - Morphological operations (erode, dilate, open, close) for segmentation
  - Window display (`cv2.imshow`, `cv2.moveWindow`) for main feed and meme popups
  - Contour extraction and convex hull computation
  - Edge detection and feature extraction
- **Install:** `pip install opencv-python`
- **Verify:** `python -c "import cv2; print(cv2.__version__)"`

### 2. **numpy** (≥ 1.21)
- **What it is:** Numerical computing library for Python
- **Why we need it:**
  - Array operations for image data (frames are numpy arrays)
  - Landmark coordinate math (Euclidean distances, angles, vectors)
  - Feature vector computation (distances, angles, thresholds)
  - Batch operations for efficiency
  - Linear algebra for pose/angle calculations
- **Install:** `pip install numpy`
- **Verify:** `python -c "import numpy; print(numpy.__version__)"`

### 3. **mediapipe** (≥ 0.10.0)
- **What it is:** Google's framework for building multimodal machine perception pipelines
- **Why we need it:**
  - Pose Holistic landmark detection (33 pose points + 21-point hand landmarks + 468 face mesh)
  - Provides the **skeleton** (landmarks) that our DIP algorithms analyze
  - NOT used for classification — only as a landmark extractor
  - Pretrained models handle robustness across lighting/angles
- **Install:** `pip install mediapipe`
- **Verify:** `python -c "import mediapipe; print(mediapipe.__version__)"`
- **Note:** First run downloads ~150MB of model files to `~/.cache/mediapipe/`

---

## OPTIONAL DEPENDENCIES (Development & Testing)

### 4. **pytest** (≥ 9.0.0)
- **What it is:** Python testing framework
- **Why we might need it:** Unit testing individual modules (preprocessing, segmentation, feature extraction)
- **Install:** `pip install pytest`
- **Usage:** `pytest tests/` (when test suite is created)

### 5. **matplotlib** (≥ 3.10.0)
- **What it is:** Plotting and visualization library
- **Why we might need it:**
  - Debug visualizations (plot histograms, segmentation masks, landmarks)
  - Feature validation before hooking into meme matching
  - Analyze confidence scores and trigger distributions
- **Install:** `pip install matplotlib`
- **Usage:** `python debug_scripts/visualize_features.py` (when created)

### 6. **pillow** (≥ 12.0.0)
- **What it is:** Python Imaging Library fork
- **Why we might need it:**
  - Advanced image manipulation if needed
  - Image resizing for meme assets
  - Meme overlay blending (alpha compositing)
- **Install:** `pip install pillow`
- **Verify:** `python -c "from PIL import Image; print('Pillow OK')"`

---

## Full Installation (All at Once)

```bash
# Core only (minimum to run project)
pip install opencv-python numpy mediapipe

# Core + development tools
pip install opencv-python numpy mediapipe pytest matplotlib pillow

# Or create a requirements.txt and install from it
echo "opencv-python>=4.0
numpy>=1.21
mediapipe>=0.10.0
pytest>=9.0.0
matplotlib>=3.10.0
pillow>=12.0.0" > requirements.txt

pip install -r requirements.txt
```

---

## Verification Checklist

Run this to verify all dependencies are ready:

```python
# save as check_deps.py
import sys

dependencies = {
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'mediapipe': 'mediapipe',
    'pytest': 'pytest',
    'matplotlib': 'matplotlib',
    'PIL': 'pillow',
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} (missing)")
        missing.append(package)

if missing:
    print(f"\nInstall missing packages with:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\n✓ All dependencies ready!")
    sys.exit(0)
```

Run with: `python check_deps.py`

---

## System Requirements

- **Python:** 3.8 or later
- **OS:** Windows, macOS, Linux (tested on Windows)
- **Camera:** USB webcam or built-in camera (must have /dev/video0 on Linux or equivalent on Windows)
- **Disk Space:** ~500MB (for cached models + meme images)
- **RAM:** 4GB minimum (8GB recommended)

---

## Troubleshooting

### MediaPipe won't install
```bash
# On Windows with older pip:
pip install --upgrade pip
pip install mediapipe

# If still fails, try:
pip install mediapipe --no-cache-dir
```

### OpenCV not finding camera
```bash
# Check if camera is accessible
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera NOT found')"
```

### ImportError after installation
```bash
# Reinstall in case of corruption
pip uninstall opencv-python numpy mediapipe -y
pip install opencv-python numpy mediapipe
```

---

## Notes

- **First run:** MediaPipe will download pretrained pose/hand/face models (~150MB total) on first use. This is one-time only.
- **Camera permissions:** On macOS/Linux, may need to grant camera access to terminal/IDE.
- **GPU support (optional):** OpenCV can use GPU acceleration if CUDA/cuDNN are installed, but not required.
- **Virtual environment (recommended):** Use `python -m venv venv` to create an isolated environment before installing packages.
