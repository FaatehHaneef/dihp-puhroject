"""
Meme Matcher — rule scoring engine for meme_templates.json.

Each meme rule has `required` (all must match), `optional` (each match adds score),
and `forbidden` (any match disqualifies). Conditions support bool/string equality,
numeric operators (">N", "<=N", ...), and dict forms ({"in": [...]}, {"between": [lo,hi]},
{"abs_gt": N}, {"abs_lt": N}).
"""

import json


def _check_condition(value, condition):
    """Return True if `value` satisfies `condition`."""
    if value is None:
        return False

    # Dict-style conditions
    if isinstance(condition, dict):
        if "in" in condition:
            return value in condition["in"]
        if "between" in condition:
            lo, hi = condition["between"]
            try:
                return lo <= float(value) <= hi
            except (TypeError, ValueError):
                return False
        if "abs_gt" in condition:
            try:
                return abs(float(value)) > float(condition["abs_gt"])
            except (TypeError, ValueError):
                return False
        if "abs_lt" in condition:
            try:
                return abs(float(value)) < float(condition["abs_lt"])
            except (TypeError, ValueError):
                return False
        return False

    # Boolean condition
    if isinstance(condition, bool):
        return value == condition

    # Numeric condition
    if isinstance(condition, (int, float)):
        return value == condition

    # String — could be a comparison operator or a literal value
    if isinstance(condition, str):
        for op in (">=", "<=", "==", ">", "<"):
            if condition.startswith(op):
                try:
                    threshold = float(condition[len(op):])
                except ValueError:
                    return str(value) == condition
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return False
                if op == ">":  return v > threshold
                if op == "<":  return v < threshold
                if op == ">=": return v >= threshold
                if op == "<=": return v <= threshold
                if op == "==": return v == threshold
        # Plain string == literal match
        return str(value) == condition

    return False


def _evaluate_block(features, block):
    """
    Evaluate a {feature: condition} block.
    Returns (matched_count, total_count).
    """
    if not block:
        return 0, 0
    matched = 0
    for key, cond in block.items():
        if _check_condition(features.get(key), cond):
            matched += 1
    return matched, len(block)


def score_meme(features, rules):
    """
    Score a single meme's rule set against the current features.

    Returns (eligible: bool, score: float in [0, 1]).
    A meme is eligible iff ALL required rules match AND NO forbidden rules match.
    Score weights required matches more heavily than optional ones.
    """
    required = rules.get("required", {})
    optional = rules.get("optional", {})
    forbidden = rules.get("forbidden", {})

    # Forbidden — any hit disqualifies
    forb_matched, _ = _evaluate_block(features, forbidden)
    if forb_matched > 0:
        return False, 0.0

    # Required — ALL must hit
    req_matched, req_total = _evaluate_block(features, required)
    if req_total > 0 and req_matched < req_total:
        return False, 0.0

    # Eligible. Compute score from required (full credit) + optional (bonus).
    opt_matched, opt_total = _evaluate_block(features, optional)

    # Weighted: required worth 0.7, optional fills up to 0.3.
    req_score = 1.0 if req_total == 0 else (req_matched / req_total)
    opt_score = 0.0 if opt_total == 0 else (opt_matched / opt_total)

    score = 0.7 * req_score + 0.3 * opt_score
    return True, score


def match_best_meme_by_template(features, meme_templates, threshold=0.55):
    """
    Pick the highest-scoring meme that is eligible.

    Args:
        features: dict from extract_all_features()
        meme_templates: dict loaded from meme_templates.json
        threshold: minimum score required to return a match

    Returns:
        (meme_name | None, score: float)
    """
    if not meme_templates:
        return None, 0.0

    best_name = None
    best_score = 0.0

    for name, template in meme_templates.items():
        if template.get("disabled", False):
            continue
        # Support both new ("rules") and legacy ("features") template shapes.
        rules = template.get("rules")
        if rules is None:
            continue  # legacy templates without rules are skipped
        priority = template.get("priority", 0)

        eligible, score = score_meme(features, rules)
        if not eligible:
            continue

        # Priority acts as a tie-breaker boost (small bump).
        effective = score + priority * 0.001

        if effective > best_score:
            best_score = effective
            best_name = name

    if best_score < threshold:
        return None, best_score

    return best_name, min(best_score, 1.0)


# ---- legacy helpers retained so other code keeps importing cleanly ----

def load_meme_config(json_path):
    """Load a JSON file. Returns {} on missing file."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {json_path} not found")
        return {}


def compute_feature_distance(features, template_features):
    """
    Backwards-compat helper used by main.py debug overlay.
    With the rule-based system, "distance" = 1 - score from the
    template's `rules` block. If the template has no rules,
    returns 1.0 (worst).
    """
    rules = template_features if isinstance(template_features, dict) and "required" in template_features else None
    if rules is None:
        # Caller may pass the inner template["features"] from legacy data — try to find rules.
        return 1.0
    eligible, score = score_meme(features, rules)
    return 1.0 - score if eligible else 1.0


def evaluate_triggers(features, triggers):
    """Legacy: check ALL conditions match. Kept for backwards-compat imports."""
    for k, cond in triggers.items():
        if not _check_condition(features.get(k), cond):
            return False
    return True


def compute_confidence(features, triggers):
    """Legacy: ratio of conditions matched."""
    if not triggers:
        return 0.0
    matched, total = _evaluate_block(features, triggers)
    return matched / total if total else 0.0


def match_best_meme(features, meme_db):
    """Legacy wrapper: try `triggers` field first, then fall back to rules."""
    if not meme_db:
        return None, 0.0
    best = (None, 0.0)
    for name, data in meme_db.items():
        if "rules" in data:
            ok, score = score_meme(features, data["rules"])
            if ok and score > best[1]:
                best = (name, score)
        else:
            triggers = data.get("triggers", {})
            if triggers and evaluate_triggers(features, triggers):
                conf = compute_confidence(features, triggers)
                if conf > best[1]:
                    best = (name, conf)
    return best


def get_feature_summary(features):
    """Format key features for debug display."""
    if not features:
        return ""
    keys = [
        "hand_count", "fingers_extended", "mouth_state", "eye_state",
        "gaze_direction", "looking_at_camera", "head_tilt_angle",
    ]
    parts = []
    for k in keys:
        if k in features:
            v = features[k]
            if isinstance(v, float):
                parts.append(f"{k}: {v:.1f}")
            else:
                parts.append(f"{k}: {v}")
    return " | ".join(parts)
