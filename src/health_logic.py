from __future__ import annotations
from typing import Dict, List

COMPONENTS = ["Fan", "LPC", "HPC", "HPT", "LPT"]

DEFAULT_LIMITS = {
    "N1": (20000, 30000),
    "N2": (4000, 6000),
    "Thrust": (250000, 350000),
    "Fuel": (0.01, 0.1),
}

DEFAULT_THRESHOLDS = {
    "eta_min": 0.88,  # minimum efficiency modifier threshold
}


def decide_status(outputs: dict) -> str:
    """
    Return "FLY" or "GROUND" (framework status).
    Works for both CNC and Turbofan outputs.
    """

    # ---------------- TURBOFAN PATH ----------------
    # Detect turbofan by presence of key signals
    if "T4" in outputs and "TorqueDiff_N2" in outputs:
        T4_max = 1900.0   # K (can move to config later)
        torque_tol = 1.0  # Nm

        # Temperature limit
        if float(outputs.get("T4", 0.0)) > T4_max:
            return "GROUND"

        # Spool balance residuals (should be near zero)
        if abs(float(outputs.get("TorqueDiff_N2", 0.0))) > torque_tol:
            return "GROUND"

        if abs(float(outputs.get("TorqueDiff_N1", 0.0))) > torque_tol:
            return "GROUND"

        return "FLY"

    # ---------------- CNC PATH ----------------
    required_keys = ["N1", "vibration_rms", "power_mean"]
    for k in required_keys:
        if k not in outputs:
            return "GROUND"

    limits = {
        "N1": (0, 999999),
        "vibration_rms": (0, 5.0),
        "power_mean": (0, 50.0),
    }

    for key, (lo, hi) in limits.items():
        x = float(outputs[key])
        if x < lo or x > hi:
            return "GROUND"

    return "FLY"


def check_health_warnings(
    pdm_json: Dict,
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
) -> str:
    """
    Equivalent of checkHealthWarnings() in run_and_log.m
    Looks for:
      - Efficiency_Modifier below eta_min
      - CyclesSinceInstall >= MaxCycles (if MaxCycles exists)
    Returns "OK" or "FAN_EFF↓; HPC_CYC↑; ..."
    """
    warn: List[str] = []
    eta_min = float(thresholds["eta_min"])

    for comp in COMPONENTS:
        part = pdm_json.get(comp, {})

        eff = part.get("Efficiency_Modifier", None)
        if eff is not None:
            eff = float(eff) / 100.0 if float(eff) > 1.5 else float(eff)
            if eff < eta_min:
                warn.append(f"{comp.upper()}_EFF↓")

        max_cycles = part.get("MaxCycles", None)
        cycles = part.get("CyclesSinceInstall", None)
        if max_cycles is not None and cycles is not None:
            if int(cycles) >= int(max_cycles):
                warn.append(f"{comp.upper()}_CYC↑")

    return "OK" if len(warn) == 0 else "; ".join(warn)


if __name__ == "__main__":
    dummy_outputs = {"N1": 25000, "N2": 5000, "Thrust": 300000, "Fuel": 0.05}
    print("Status:", decide_status(dummy_outputs))