"""
Meme Matcher — Rule-based meme matching engine and confidence scoring.

DIP Concepts: Feature matching, rule-based classification
"""

import json
import math
from pathlib import Path


def load_meme_config(json_path):
    """
    Load meme database from JSON config file.
    
    Args:
        json_path: Path to meme_config.json
    
    Returns:
        Dict: {
            "meme_name": {
                "image": "filename.jpg",
                "type": "image" | "video",
                "triggers": {...},
                "priority": int
            },
            ...
        }
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {json_path} not found")
        return {}


def evaluate_triggers(features, meme_triggers):
    """
    Check if all triggers for a meme are satisfied by features.
    
    Trigger formats:
    - "==value": exact match
    - ">value" or "<value": comparison
    - ">=value" or "<=value": comparison
    - true/false: boolean
    
    Args:
        features: Dict of extracted features
        meme_triggers: Dict of meme trigger conditions
    
    Returns:
        Bool: True if all triggers satisfied
    """
    for trigger_key, trigger_value in meme_triggers.items():
        if trigger_key not in features:
            return False
        
        feature_val = features[trigger_key]
        
        # Boolean comparison
        if isinstance(trigger_value, bool):
            if feature_val != trigger_value:
                return False
        
        # String comparison with operators
        elif isinstance(trigger_value, str):
            if trigger_value.startswith(">="):
                threshold = float(trigger_value[2:])
                if not (feature_val >= threshold):
                    return False
            elif trigger_value.startswith("<="):
                threshold = float(trigger_value[2:])
                if not (feature_val <= threshold):
                    return False
            elif trigger_value.startswith(">"):
                threshold = float(trigger_value[1:])
                if not (feature_val > threshold):
                    return False
            elif trigger_value.startswith("<"):
                threshold = float(trigger_value[1:])
                if not (feature_val < threshold):
                    return False
            elif trigger_value.startswith("=="):
                threshold = float(trigger_value[2:])
                if not (feature_val == threshold):
                    return False
            else:
                # Try direct equality
                if str(feature_val) != str(trigger_value):
                    return False
        
        # Numeric comparison
        elif isinstance(trigger_value, (int, float)):
            if feature_val != trigger_value:
                return False
    
    return True


def compute_confidence(features, meme_triggers):
    """
    Compute confidence score (0-1) for a meme match.
    Higher = more confident.
    
    Args:
        features: Dict of extracted features
        meme_triggers: Dict of trigger conditions
    
    Returns:
        Float: 0-1 confidence score
    """
    if not meme_triggers:
        return 0.0
    
    matched_triggers = 0
    
    for trigger_key, trigger_value in meme_triggers.items():
        if trigger_key not in features:
            continue
        
        feature_val = features[trigger_key]
        matched = False
        
        # Boolean comparison
        if isinstance(trigger_value, bool):
            matched = (feature_val == trigger_value)
        
        # String comparison with operators
        elif isinstance(trigger_value, str):
            if trigger_value.startswith(">="):
                threshold = float(trigger_value[2:])
                matched = (feature_val >= threshold)
            elif trigger_value.startswith("<="):
                threshold = float(trigger_value[2:])
                matched = (feature_val <= threshold)
            elif trigger_value.startswith(">"):
                threshold = float(trigger_value[1:])
                matched = (feature_val > threshold)
            elif trigger_value.startswith("<"):
                threshold = float(trigger_value[1:])
                matched = (feature_val < threshold)
            elif trigger_value.startswith("=="):
                threshold = float(trigger_value[2:])
                matched = (feature_val == threshold)
        
        # Numeric comparison
        elif isinstance(trigger_value, (int, float)):
            matched = (feature_val == trigger_value)
        
        if matched:
            matched_triggers += 1
    
    # Confidence = ratio of matched triggers to total triggers
    confidence = matched_triggers / len(meme_triggers) if meme_triggers else 0.0
    return confidence


def match_best_meme(features, meme_db):
    """
    Find the best matching meme from the database.
    
    Args:
        features: Dict of extracted features
        meme_db: Meme database dict (from load_meme_config)
    
    Returns:
        Tuple: (meme_name: str, confidence: float)
        Returns (None, 0.0) if no match found or no memes in DB
    """
    if not meme_db:
        return None, 0.0
    
    best_meme = None
    best_confidence = 0.0
    best_priority = -1
    
    for meme_name, meme_data in meme_db.items():
        triggers = meme_data.get("triggers", {})
        
        # Check if all triggers are satisfied
        if not evaluate_triggers(features, triggers):
            continue
        
        # Compute confidence score
        confidence = compute_confidence(features, triggers)
        priority = meme_data.get("priority", 0)
        
        # Update best if this is better (higher confidence, or same confidence but higher priority)
        if confidence > best_confidence or (confidence == best_confidence and priority > best_priority):
            best_meme = meme_name
            best_confidence = confidence
            best_priority = priority
    
    return best_meme, best_confidence


def get_feature_summary(features):
    """
    Format features dict into a readable summary string for debug display.
    
    Args:
        features: Dict of extracted features
    
    Returns:
        String: Formatted summary
    """
    if not features:
        return ""
    
    parts = []
    
    # Key features to display
    key_features = [
        "fingers_extended",
        "head_tilt_angle",
        "mouth_open",
        "hand_near_face",
        "body_symmetry",
        "arm_angle",
    ]
    
    for key in key_features:
        if key in features:
            val = features[key]
            if isinstance(val, float):
                parts.append(f"{key}: {val:.1f}")
            elif isinstance(val, bool):
                parts.append(f"{key}: {val}")
            else:
                parts.append(f"{key}: {val}")
    
    return " | ".join(parts)


def compute_feature_distance(user_features, template_features):
    """
    Compute distance between user features and template features.
    Lower = better match.
    
    Args:
        user_features: Dict of user's extracted features
        template_features: Dict of template's reference features
    
    Returns:
        Float: Distance (0-1 normalized)
    """
    if not template_features:
        return float('inf')
    
    distance = 0.0
    matched_keys = 0
    
    # Key features with weights (higher weight = more important)
    weighted_features = {
        "fingers_extended_left": 1.0,
        "fingers_extended_right": 1.0,
        "hand_angle_left": 0.5,
        "hand_angle_right": 0.5,
        "mouth_open": 2.0,
        "mouth_openness": 1.0,
        "head_tilt_angle": 0.8,
        "gaze_direction": 0.5,
        "shoulder_angle": 0.5,
        "body_symmetry": 0.3,
        "hands_touching": 1.0,
    }
    
    for key, weight in weighted_features.items():
        if key in template_features and key in user_features:
            template_val = template_features[key]
            user_val = user_features.get(key, 0)
            
            # Different distance metrics based on data type
            if isinstance(template_val, bool):
                # Boolean: 0 if match, 1 if different
                diff = 0 if (template_val == user_val) else 1
            elif isinstance(template_val, str):
                # String (gaze direction): 0 if match, 1 if different
                diff = 0 if (str(template_val) == str(user_val)) else 1
            else:
                # Numeric: normalized absolute difference
                try:
                    diff = abs(float(template_val) - float(user_val))
                    # Normalize to 0-1 range (adjust scale factor as needed)
                    diff = min(diff / 100.0, 1.0)
                except (ValueError, TypeError):
                    diff = 1.0
            
            distance += diff * weight
            matched_keys += weight
    
    # Normalize distance by total weight
    if matched_keys > 0:
        distance = distance / matched_keys
    
    return distance


def match_best_meme_by_template(user_features, meme_templates, threshold=0.3):
    """
    Find best matching meme by comparing features to stored templates.
    
    Args:
        user_features: Dict of user's extracted features
        meme_templates: Dict of {meme_name: template, ...}
        threshold: Maximum distance to consider a match (0-1)
    
    Returns:
        Tuple: (meme_name: str, confidence: float)
        Returns (None, 0.0) if no match found
    """
    if not meme_templates:
        return None, 0.0
    
    best_meme = None
    best_distance = float('inf')
    
    for meme_name, template in meme_templates.items():
        template_features = template.get("features", {})
        
        # Compute distance to this template
        distance = compute_feature_distance(user_features, template_features)
        
        # Update best if closer
        if distance < best_distance:
            best_meme = meme_name
            best_distance = distance
    
    # Convert distance to confidence (inverse: lower distance = higher confidence)
    confidence = max(0.0, 1.0 - best_distance) if best_distance < float('inf') else 0.0
    
    # Only return match if confidence above threshold
    if confidence > threshold:
        return best_meme, confidence
    
    return None, 0.0
