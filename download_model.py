import urllib.request
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "mediapipe"
cache_dir.mkdir(parents=True, exist_ok=True)

model_url = "https://storage.googleapis.com/mediapipe-assets/holistic_landmarker.task"
output_path = cache_dir / "holistic_landmarker.task"

print(f"Downloading to {output_path}...")
try:
    urllib.request.urlretrieve(model_url, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Downloaded ({size_mb:.1f} MB)")
except Exception as e:
    print(f"✗ Error: {e}")
