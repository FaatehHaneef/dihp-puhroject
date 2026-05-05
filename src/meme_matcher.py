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
    # TODO: Iterate meme_triggers, check against features
    # Support string comparisons (">=4", ">10", etc.) and booleans
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
    # TODO: Implement weighted scoring
    # Features that match get higher weights
    # Average the weighted scores
    return 0.0


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
    
    # TODO: Iterate meme_db, score each one, keep track of best
    # Prioritize by meme priority if tied
    
    return best_meme, best_confidence


def get_feature_summary(features):
    """
    Format features dict into a readable summary string for debug display.
    
    Args:
        features: Dict of extracted features
    
    Returns:
        String: Formatted summary
    """
    # TODO: Extract key features and format nicely
    return ""
