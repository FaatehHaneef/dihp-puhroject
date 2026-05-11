# DIP Project — Dependencies

## Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.13.0 | Camera capture, image processing, display |
| numpy | 2.3.5 | Array operations, math for landmarks |
| mediapipe | 0.10.35 | Pose/hand/face landmark extraction |
| pytest | 9.0.2 | Testing (optional) |
| matplotlib | 3.10.7 | Visualization (optional) |
| pillow | 12.0.0 | Image manipulation (optional) |

## Install

```bash
pip install opencv-python numpy mediapipe pytest matplotlib pillow
```

## Verify

```bash
python -c "import cv2, numpy, mediapipe; print('✓ All ready')"
```

---

## CORE DEPENDENCIES (Required to Run)

### opencv-python
- Camera capture & image processing (blur, color conversion, morphology)
- Window display (`cv2.imshow`, `cv2.moveWindow`)

### numpy  
- Array operations & math for landmark coordinates

### mediapipe
- Pose/hand/face landmark extraction (skeleton provider only)
- First run downloads ~150MB models to `~/.cache/mediapipe/`

## OPTIONAL DEPENDENCIES

### pytest
- Unit testing framework

### matplotlib
- Debug visualizations

### pillow
- Image manipulation
