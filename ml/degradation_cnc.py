from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class CNCDegradation:
    # 0..1
    bearing_wear: float = 0.0      # increases vibration + power jitter
    imbalance: float = 0.0         # increases periodic vibration
    tool_wear: float = 0.0         # increases power mean + vibration a bit
    sensor_noise: float = 0.02     # additive noise


def synthesize_timeseries(
    n: int = 200,
    dt: float = 0.2,
    rpm_cmd: float = 12000.0,
    feed_cmd: float = 2000.0,
    severity: float = 1.0,
    deg: CNCDegradation | None = None,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """
    Returns timeseries dict in the same "shape" your platform expects:
    spindle_rpm, feed_mm_min, vibration, power_kw
    """
    if deg is None:
        deg = CNCDegradation()

    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * dt

    # rpm and feed stable with small noise
    rpm = rpm_cmd + rng.normal(0.0, 30.0 + 120.0 * deg.bearing_wear, size=n)
    feed = feed_cmd + rng.normal(0.0, 10.0, size=n)

    # vibration components
    base_vib = 0.10 + 0.10 * severity
    vib_wear = 0.50 * deg.bearing_wear          # big effect
    vib_tool = 0.15 * deg.tool_wear
    vib_imb = 0.35 * deg.imbalance * np.sin(2 * np.pi * 0.7 * t)

    vib = base_vib + vib_wear + vib_tool + vib_imb
    vib += rng.normal(0.0, deg.sensor_noise, size=n)

    # power components
    base_pwr = 1.0 + 0.00008 * rpm_cmd + 0.0002 * feed_cmd + 0.4 * severity
    pwr_tool = 1.20 * deg.tool_wear           # increases mean load
    pwr_jitter = rng.normal(0.0, 0.10 + 0.50 * deg.bearing_wear, size=n)

    pwr = base_pwr + pwr_tool + pwr_jitter

    return {
        "spindle_rpm": rpm.astype(float).tolist(),
        "feed_mm_min": feed.astype(float).tolist(),
        "vibration": vib.astype(float).tolist(),
        "power_kw": pwr.astype(float).tolist(),
    }


def label_from_degradation(deg: CNCDegradation) -> int:
    """
    0=OK, 1=WARNING, 2=FAULT
    Minimal, generator-aligned mapping:
    - OK: max degradation < 0.20
    - WARNING: 0.20 <= max degradation < 0.60
    - FAULT: max degradation >= 0.60
    """
    m = max(deg.bearing_wear, deg.imbalance, deg.tool_wear)

    if m >= 0.60:
        return 2
    if m >= 0.20:
        return 1
    return 0


def fault_class_from_degradation(deg: CNCDegradation) -> int:
    """
    0=healthy, 1=bearing_wear, 2=imbalance, 3=tool_wear, 4=mixed
    """
    active = []
    if deg.bearing_wear > 0.05:
        active.append("bearing")
    if deg.imbalance > 0.05:
        active.append("imbalance")
    if deg.tool_wear > 0.05:
        active.append("tool")

    if not active:
        return 0
    if len(active) > 1:
        return 4
    if active[0] == "bearing":
        return 1
    if active[0] == "imbalance":
        return 2
    return 3
