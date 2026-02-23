from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable
import numpy as np


@dataclass
class CNCMission:
    """
    Simple CNC "mission profile" over time.
    t in seconds.
    """
    t_end: float = 60.0
    dt: float = 0.1

    # setpoints
    spindle_rpm_cmd: float = 12000.0
    feed_cmd: float = 2000.0  # mm/min

    # process intensity
    severity: float = 1.0  # 0..2 typical


def simulate_cnc(ws: Dict, mission: CNCMission) -> Dict[str, float]:
    """
    Minimal CNC digital twin.
    Returns MATLAB-compatible output keys: N1, N2, Thrust, Fuel
      N1    -> spindle rpm (final)
      N2    -> feed rate (final)
      Thrust-> cutting force proxy (N)
      Fuel  -> power proxy (kW)
    ws: workspace variables from pdm_loader (Eff_Mod_*, DoUpgrade, ...)
    """

    # Interpret "efficiency modifiers" as health multipliers.
    # We'll map them to spindle/axis health in a simple way.
    # Fan/LPC/HPC/HPT/LPT are just placeholders from the turbofan template.
    spindle_health = float(ws.get("Eff_Mod_HPC", 1.0))   # use one as proxy
    axis_health = float(ws.get("Eff_Mod_LPC", 1.0))
    tool_health = float(ws.get("Eff_Mod_HPT", 1.0))

    # Commanded setpoints
    rpm_cmd = mission.spindle_rpm_cmd
    feed_cmd = mission.feed_cmd

    # Dynamics parameters (simple)
    tau_rpm = 1.5  # seconds
    tau_feed = 0.8

    n_steps = int(mission.t_end / mission.dt) + 1
    rpm = 0.0
    feed = 0.0

    # simulate simple first order response to setpoints
    for _ in range(n_steps):
        rpm += (rpm_cmd - rpm) * (mission.dt / tau_rpm)
        feed += (feed_cmd - feed) * (mission.dt / tau_feed)

    # Cutting force proxy (N):
    # increases with feed, severity, and decreases with health (worn tool -> higher force)
    # We clamp health to avoid divide-by-zero.
    tool_health = max(0.3, tool_health)
    axis_health = max(0.3, axis_health)

    # Simple scaling (not physical yet — placeholder)
    cutting_force = (0.08 * feed) * mission.severity * (1.0 / tool_health)

    # Torque proxy (Nm): proportional to cutting force
    torque = 0.02 * cutting_force * (1.0 / axis_health)

    # Power (kW): P = tau * omega ; omega = 2*pi*rpm/60
    omega = 2.0 * np.pi * rpm / 60.0
    power_kw = (torque * omega) / 1000.0

    outputs = {
        "N1": float(rpm),            # spindle rpm
        "N2": float(feed),           # feed rate
        "Thrust": float(cutting_force),
        "Fuel": float(power_kw),
    }
    return outputs


if __name__ == "__main__":
    # quick test
    from pdm_loader import load_pdm
    ws = load_pdm()
    out = simulate_cnc(ws, CNCMission())
    print(out)
