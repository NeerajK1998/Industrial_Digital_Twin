# ml/degradation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

@dataclass
class DegradationParams:
    # Eff multipliers (1.00 = healthy, <1 worsens)
    fan_eff_mult: float = 1.00
    lpc_eff_mult: float = 1.00
    hpc_eff_mult: float = 1.00
    hpt_eff_mult: float = 1.00
    lpt_eff_mult: float = 1.00

    # Flow capacity multipliers (optional)
    fan_wc_mult: float = 1.00
    hpc_wc_mult: float = 1.00

    # Sensor noise (applied to observed signals, not internal states)
    sensor_noise_std: float = 0.0

def sample_degradation(rng: random.Random) -> DegradationParams:
    """
    Random but controlled sampling.
    Adjust ranges later without changing feature schema.
    """
    def u(a, b): return rng.uniform(a, b)

    return DegradationParams(
        fan_eff_mult=u(0.92, 1.00),
        lpc_eff_mult=u(0.92, 1.00),
        hpc_eff_mult=u(0.85, 1.00),
        hpt_eff_mult=u(0.90, 1.00),
        lpt_eff_mult=u(0.90, 1.00),
        fan_wc_mult=u(0.97, 1.03),
        hpc_wc_mult=u(0.95, 1.05),
        sensor_noise_std=u(0.0, 0.01),
    )

def label_from_outputs(
    *,
    tsfc_mg_N_s: float,
    thrust_N: float,
    n1_pct: float,
    n2_pct: float,
    degr: DegradationParams,
) -> Tuple[str, str]:
    """
    Returns (label, reason_codes_string).
    Keep label rules explainable.
    Tweak thresholds later — schema stays same.
    """
    reasons: List[str] = []

    # Degradation severity (simple proxy)
    worst_eff = min(degr.fan_eff_mult, degr.lpc_eff_mult, degr.hpc_eff_mult, degr.hpt_eff_mult, degr.lpt_eff_mult)

    if worst_eff < 0.90:
        reasons.append("EFF_SEVERE")
    elif worst_eff < 0.95:
        reasons.append("EFF_MODERATE")

    # KPI-based rules (placeholders; tune after first dataset)
    if tsfc_mg_N_s > 40.0:
        reasons.append("TSFC_HIGH")
    if thrust_N < 15000:
        reasons.append("THRUST_LOW")
    if n2_pct > 105.0 or n1_pct > 105.0:
        reasons.append("OVERSPEED")

    # Decide label
    if "EFF_SEVERE" in reasons or "OVERSPEED" in reasons:
        label = "FAULT"
    elif len(reasons) > 0:
        label = "WARNING"
    else:
        label = "OK"

    return label, ";".join(reasons) if reasons else "NONE"