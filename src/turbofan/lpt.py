from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TurbineGasProps:
    gamma: float = 1.33
    cp: float = 1150.0  # hot gas cp J/kgK


def lpt_core_calc(
    P_in: float,
    T_in: float,
    m_gas: float,
    PR_turb: float,      # P_out / P_in (<1)
    eta_turb: float,
    N1_RPM: float,
    gas: TurbineGasProps = TurbineGasProps(),
):
    if PR_turb >= 1.0:
        raise ValueError("Turbine pressure ratio must be < 1")

    gamma = gas.gamma
    cp = gas.cp

    T_out_ideal = T_in * (PR_turb ** ((gamma - 1.0) / gamma))
    T_out = T_in - eta_turb * (T_in - T_out_ideal)
    P_out = P_in * PR_turb

    power = m_gas * cp * (T_in - T_out)
    omega = 2.0 * math.pi * N1_RPM / 60.0
    torque = power / max(omega, 1e-6)

    return P_out, T_out, torque