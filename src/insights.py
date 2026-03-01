from __future__ import annotations

from typing import Dict, List, Tuple


def severity_score(pred: str, probs: Dict[str, float]) -> int:
    """
    Simple severity score 0..100.
    Weighted by confidence.
    """
    conf = float(max(probs.values())) if probs else 1.0
    base = 10
    if pred == "OK":
        base = 15
    elif pred == "WARNING":
        base = 60
    elif pred == "FAULT":
        base = 90
    return int(round(base * (0.7 + 0.3 * conf)))


def top_contributors_from_rf(model, feature_row: Dict[str, float], topk: int = 3) -> List[Tuple[str, float]]:
    """
    For RandomForest-like models: feature_importances_ exists.
    Contribution = importance * abs(value).
    """
    if not hasattr(model, "feature_importances_"):
        return []

    importances = list(getattr(model, "feature_importances_"))
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        cols = list(feature_row.keys())

    pairs = []
    for i, name in enumerate(cols):
        v = float(feature_row.get(str(name), 0.0))
        imp = float(importances[i]) if i < len(importances) else 0.0
        pairs.append((str(name), imp * abs(v)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:topk]


def recommended_action(pred: str, top_feats: List[Tuple[str, float]]) -> str:
    """
    Human-readable actions for portfolio.
    """
    feat_names = [f for f, _ in top_feats]

    if pred == "OK":
        return "No action required. Continue monitoring."

    if pred == "WARNING":
        if "vibration_rms" in feat_names or "vibration_peak" in feat_names:
            return "Schedule inspection. Check bearings / imbalance. Plan maintenance within 1–2 weeks."
        if "power_std" in feat_names:
            return "Power instability detected. Check drive load, tool condition, and process parameters."
        return "Monitor more frequently and plan a maintenance check soon."

    # FAULT
    if "vibration_rms" in feat_names or "vibration_peak" in feat_names:
        return "High vibration fault. Stop machine if safe. Inspect bearings, coupling, and balance immediately."
    if "power_mean" in feat_names:
        return "Overload fault. Stop and inspect tool wear / friction and feed parameters."
    return "Fault detected. Stop if safe and perform immediate inspection."
