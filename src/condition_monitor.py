from typing import Dict


def evaluate_condition(features: Dict[str, float]) -> Dict[str, str]:
    """
    Simple rule-based condition evaluation.
    Later this becomes ML-based.
    """

    alerts = []

    if features.get("vibration_rms", 0) > 0.4:
        alerts.append("HIGH_VIBRATION")

    if features.get("power_std", 0) > 0.5:
        alerts.append("POWER_UNSTABLE")

    if features.get("rpm_std", 0) > 200:
        alerts.append("RPM_FLUCTUATION")

    status = "OK" if not alerts else "ALERT"

    return {
        "status": status,
        "alerts": "; ".join(alerts) if alerts else "NONE"
    }
