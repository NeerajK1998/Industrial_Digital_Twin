from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TurbineGasProps:
    gamma: float = 1.33
    cp: float = 1150.0  # hot gas cp J/kgK


def hpt_core_calc(
    P4: float,
    T4: float,
    m_gas: float,
    PR_turb: float,
    eta_turb: float,
    N2_RPM: float,
    gas: TurbineGasProps = TurbineGasProps(),
):
    """
    High Pressure Turbine core calc.

    PR_turb = P45 / P4  (must be < 1)
    """

    if PR_turb >= 1.0:
        raise ValueError("Turbine pressure ratio must be < 1")

    gamma = gas.gamma
    cp = gas.cp

    # Ideal outlet temperature
    T45_ideal = T4 * (PR_turb ** ((gamma - 1.0) / gamma))

    # Real outlet temperature
    T45 = T4 - eta_turb * (T4 - T45_ideal)

    # Outlet pressure
    P45 = P4 * PR_turb

    # Power extracted
    power = m_gas * cp * (T4 - T45)

    # Convert RPM to rad/s
    omega = 2.0 * math.pi * N2_RPM / 60.0
    torque = power / max(omega, 1e-6)

    return P45, T45, torque