"""
Meme Templates — Extract and store reference landmarks from meme images.

DIP Concepts: Template-based matching, reference feature extraction
"""

import cv2
import json
import math
import numpy as np
from pathlib import Path


def extract_meme_template(image_path):
    """
    Extract template features from a single meme image.
    
    Args:
        image_path: Path to meme image file
    
    Returns:
        Dict: {
            "image": "filename.jpg",
            "landmarks": {...},
            "features": {...}
        }
        Returns None if no landmarks detected
    """
    # Import MediaPipe inside function (lazy import)
    import mediapipe as mp
    
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        
        # Get image dimensions
        h, w, _ = img.shape
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe Holistic
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1
        )
        
        # Process image
        results = holistic.process(img_rgb)
        holistic.close()
        
        # Extract landmarks (convert to normalized list format for JSON serialization)
        landmarks_data = {}
        
        if results.pose_landmarks:
            landmarks_data["pose"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]
        
        if results.left_hand_landmarks:
            landmarks_data["left_hand"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in results.left_hand_landmarks.landmark
            ]
        
        if results.right_hand_landmarks:
            landmarks_data["right_hand"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in results.right_hand_landmarks.landmark
            ]
        
        if results.face_landmarks:
            landmarks_data["face"] = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in results.face_landmarks.landmark
            ]
        
        # Extract high-level features directly
        features = {}
        
        # Hand features
        if results.right_hand_landmarks:
            lm = results.right_hand_landmarks.landmark
            # Count extended fingers: thumb(4), index(8), middle(12), ring(16), pinky(20)
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [3, 7, 11, 15, 19]
            extended_count = 0
            for tip_idx, pip_idx in zip(finger_tips, finger_pips):
                is_extended = (lm[pip_idx].y - lm[tip_idx].y) > 0.05
                extended_count += int(is_extended)
            features["fingers_extended_right"] = extended_count
            
            # Hand angle
            wrist = lm[0]
            middle_mcp = lm[9]
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            hand_angle = math.atan2(dy, dx) * 180 / math.pi
            features["hand_angle_right"] = (hand_angle + 360) % 360
        
        if results.left_hand_landmarks:
            lm = results.left_hand_landmarks.landmark
            # Count extended fingers
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [3, 7, 11, 15, 19]
            extended_count = 0
            for tip_idx, pip_idx in zip(finger_tips, finger_pips):
                is_extended = (lm[pip_idx].y - lm[tip_idx].y) > 0.05
                extended_count += int(is_extended)
            features["fingers_extended_left"] = extended_count
            
            # Hand angle
            wrist = lm[0]
            middle_mcp = lm[9]
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            hand_angle = math.atan2(dy, dx) * 180 / math.pi
            features["hand_angle_left"] = (hand_angle + 360) % 360
        
        # Face features
        if results.face_landmarks:
            lm = results.face_landmarks.landmark
            
            # Head tilt angle
            right_eye = lm[133]
            left_eye = lm[362]
            dx = left_eye.x - right_eye.x
            dy = left_eye.y - right_eye.y
            head_tilt = math.degrees(math.atan2(dy, dx))
            features["head_tilt_angle"] = head_tilt
            
            # Mouth open
            top_lip = lm[61]
            bottom_lip = lm[291]
            mouth_openness = abs(bottom_lip.y - top_lip.y)
            features["mouth_open"] = mouth_openness > 0.05
            features["mouth_openness"] = mouth_openness
            
            # Gaze direction
            right_eye_inner = lm[133]
            right_eye_outer = lm[33]
            right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2
            left_eye_inner = lm[362]
            left_eye_outer = lm[263]
            left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2
            
            features["gaze_direction"] = "center"  # Simplified
        
        # Pose features
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Shoulder angle
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            dx = right_shoulder.x - left_shoulder.x
            dy = right_shoulder.y - left_shoulder.y
            shoulder_angle = math.atan2(dy, dx) * 180 / math.pi
            features["shoulder_angle"] = abs(shoulder_angle)
            
            # Body symmetry (simplified)
            total_distance = 0
            valid_pairs = 0
            pairs = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]
            for left_idx, right_idx in pairs:
                if left_idx < len(lm) and right_idx < len(lm):
                    left_pt = lm[left_idx]
                    right_pt = lm[right_idx]
                    dist = math.sqrt((left_pt.x - right_pt.x)**2 + (left_pt.y - right_pt.y)**2)
                    total_distance += dist
                    valid_pairs += 1
            if valid_pairs > 0:
                avg_dist = total_distance / valid_pairs
                features["body_symmetry"] = max(0.0, 1.0 - avg_dist * 2)
        
        # Hand interaction
        if results.left_hand_landmarks and results.right_hand_landmarks:
            left_wrist = results.left_hand_landmarks.landmark[0]
            right_wrist = results.right_hand_landmarks.landmark[0]
            dist = math.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
            features["hands_touching"] = dist < 0.1
        
        template = {
            "image": Path(image_path).name,
            "landmarks": landmarks_data,
            "features": features
        }
        
        return template
    
    except Exception as e:
        print(f"Error extracting template from {image_path}: {e}")
        return None


def extract_all_meme_templates(meme_dir):
    """
    Extract templates from all meme images in a directory.
    
    Args:
        meme_dir: Path to memes directory
    
    Returns:
        Dict: {meme_name (without extension): template, ...}
    """
    meme_dir = Path(meme_dir)
    templates = {}
    
    # Supported extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    for image_file in meme_dir.iterdir():
        if image_file.suffix.lower() not in image_extensions:
            continue
        
        print(f"Extracting template from {image_file.name}...")
        template = extract_meme_template(image_file)
        
        if template:
            # Use filename without extension as key
            meme_name = image_file.stem
            templates[meme_name] = template
            print(f"  [OK] Extracted: {meme_name}")
        else:
            print(f"  [FAILED] {image_file.name}")
    
    return templates


def save_templates(templates, output_path):
    """
    Save templates to JSON file.
    
    Args:
        templates: Dict of templates
        output_path: Path to save JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(templates, f, indent=2)
    
    print(f"\n✓ Saved {len(templates)} templates to {output_path}")


def load_templates(template_path):
    """
    Load templates from JSON file.
    
    Args:
        template_path: Path to template JSON
    
    Returns:
        Dict: {meme_name: template, ...}
    """
    try:
        with open(template_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Template file not found: {template_path}")
        return {}


if __name__ == "__main__":
    # Extract templates from memes folder
    meme_dir = Path(__file__).parent.parent / "memes"
    template_path = Path(__file__).parent / "meme_templates.json"
    
    print("Extracting meme templates...")
    templates = extract_all_meme_templates(meme_dir)
    
    if templates:
        save_templates(templates, template_path)
        print(f"\nExtracted {len(templates)} meme templates")
        for name in templates.keys():
            print(f"  - {name}")
    else:
        print("No memes extracted!")
