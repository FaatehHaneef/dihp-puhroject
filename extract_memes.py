#!/usr/bin/env python3
"""
Extract meme templates once and save to file.
Run this ONCE to initialize meme_templates.json, then use for runtime matching.
"""

import cv2
import json
import math
from pathlib import Path

def extract_features_from_image(image_path):
    """Extract features from a single image using MediaPipe."""
    try:
        # Import MediaPipe only when needed
        import mediapipe as mp
        
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Access holistic via direct import
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1
        )
        results = holistic.process(img_rgb)
        holistic.close()
        
        features = {}
        
        # Hand features
        if results.right_hand_landmarks:
            lm = results.right_hand_landmarks.landmark
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [3, 7, 11, 15, 19]
            extended_count = sum(1 for tip, pip in zip(finger_tips, finger_pips) 
                                if (lm[pip].y - lm[tip].y) > 0.05)
            features["fingers_extended_right"] = extended_count
            
            wrist = lm[0]
            middle_mcp = lm[9]
            hand_angle = math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x) * 180 / math.pi
            features["hand_angle_right"] = (hand_angle + 360) % 360
        
        if results.left_hand_landmarks:
            lm = results.left_hand_landmarks.landmark
            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [3, 7, 11, 15, 19]
            extended_count = sum(1 for tip, pip in zip(finger_tips, finger_pips) 
                                if (lm[pip].y - lm[tip].y) > 0.05)
            features["fingers_extended_left"] = extended_count
            
            wrist = lm[0]
            middle_mcp = lm[9]
            hand_angle = math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x) * 180 / math.pi
            features["hand_angle_left"] = (hand_angle + 360) % 360
        
        # Face features
        if results.face_landmarks:
            lm = results.face_landmarks.landmark
            
            # Head tilt
            right_eye = lm[133]
            left_eye = lm[362]
            head_tilt = math.degrees(math.atan2(left_eye.y - right_eye.y, left_eye.x - right_eye.x))
            features["head_tilt_angle"] = head_tilt
            
            # Mouth
            top_lip = lm[61]
            bottom_lip = lm[291]
            mouth_openness = abs(bottom_lip.y - top_lip.y)
            features["mouth_open"] = mouth_openness > 0.05
            features["mouth_openness"] = mouth_openness
        
        # Pose features
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            shoulder_angle = math.atan2(right_shoulder.y - left_shoulder.y, 
                                       right_shoulder.x - left_shoulder.x) * 180 / math.pi
            features["shoulder_angle"] = abs(shoulder_angle)
            
            # Body symmetry
            total_dist = 0
            pairs = [(11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28)]
            for left_idx, right_idx in pairs:
                left_pt = lm[left_idx]
                right_pt = lm[right_idx]
                dist = math.sqrt((left_pt.x - right_pt.x)**2 + (left_pt.y - right_pt.y)**2)
                total_dist += dist
            avg_dist = total_dist / len(pairs)
            features["body_symmetry"] = max(0.0, 1.0 - avg_dist * 2)
        
        # Hand interaction
        if results.left_hand_landmarks and results.right_hand_landmarks:
            left_wrist = results.left_hand_landmarks.landmark[0]
            right_wrist = results.right_hand_landmarks.landmark[0]
            dist = math.sqrt((left_wrist.x - right_wrist.x)**2 + (left_wrist.y - right_wrist.y)**2)
            features["hands_touching"] = dist < 0.1
        
        return features
    
    except Exception as e:
        print(f"  Error: {e}")
        return None


def extract_all_templates(meme_dir):
    """Extract templates from all memes."""
    meme_dir = Path(meme_dir)
    templates = {}
    
    image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    for img_file in sorted(meme_dir.iterdir()):
        if img_file.suffix.lower() not in image_exts:
            continue
        
        print(f"Extracting {img_file.name}...", end=" ")
        features = extract_features_from_image(img_file)
        
        if features:
            meme_name = img_file.stem
            templates[meme_name] = {
                "image": img_file.name,
                "features": features
            }
            print("[OK]")
        else:
            print("[SKIP]")
    
    return templates


if __name__ == "__main__":
    meme_dir = Path(__file__).parent / "memes"
    output_path = Path(__file__).parent / "src" / "meme_templates.json"
    
    print(f"Extracting meme templates from {meme_dir}...")
    templates = extract_all_templates(meme_dir)
    
    if templates:
        with open(output_path, 'w') as f:
            json.dump(templates, f, indent=2)
        print(f"\n[SUCCESS] Extracted {len(templates)} meme templates")
        print(f"Saved to: {output_path}")
    else:
        print("\n[FAILED] No templates extracted")
