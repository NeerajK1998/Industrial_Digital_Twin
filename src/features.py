from __future__ import annotations
import numpy as np
from typing import Dict, List


def compute_features(timeseries: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute basic condition-monitoring features.
    Input:
        timeseries = {
            "spindle_rpm": [...],
            "feed_mm_min": [...],
            "vibration": [...],
            "power_kw": [...]
        }
    Returns dictionary of derived features.
    """

    rpm = np.array(timeseries.get("spindle_rpm", []))
    vib = np.array(timeseries.get("vibration", []))
    power = np.array(timeseries.get("power_kw", []))

    features = {}

    if len(vib) > 0:
        features["vibration_rms"] = float(np.sqrt(np.mean(vib**2)))
        features["vibration_peak"] = float(np.max(np.abs(vib)))

    if len(power) > 0:
        features["power_mean"] = float(np.mean(power))
        features["power_std"] = float(np.std(power))

    if len(rpm) > 0:
        features["rpm_mean"] = float(np.mean(rpm))
        features["rpm_std"] = float(np.std(rpm))

    return features
