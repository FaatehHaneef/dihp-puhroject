"""
Meme Matcher — Rule-based meme matching engine and confidence scoring.

DIP Concepts: Feature matching, rule-based classification
"""

import json
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
