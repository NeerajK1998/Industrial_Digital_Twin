from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


@dataclass(frozen=True)
class GasProps:
    gamma: float = 1.4
    cp: float = 1004.5      # J/(kg*K) typical air
    Tref: float = 288.15    # K
    Pref: float = 101325.0  # Pa


def fan_core_calc(
    P0: float,
    T0: float,
    PR_map: float,
    eta_map: float,
    N1_RPM: float,
    Wc_total: float,
    BPR: float,
    eff_mod: float = 1.0,
    gas: GasProps = GasProps(),
) -> Tuple[float, float, float, float, float]:
    """
    Python equivalent of MATLAB Fan_CoreCalc.

    Inputs:
      P0, T0      : inlet total pressure [Pa], total temperature [K]
      PR_map      : map-based pressure ratio (dimensionless)
      eta_map     : map-based efficiency (0..1)
      N1_RPM      : shaft speed [RPM]
      Wc_total    : corrected mass flow (dimensionless in this formulation)
      BPR         : bypass ratio (>=0)
      eff_mod     : efficiency modifier (PDM degradation)
    Returns:
      P1_fan      : outlet total pressure [Pa]
      T1_raw      : outlet total temperature [K]
      Torque_fan  : torque [N*m]
      Wc_core     : corrected core flow
      m_dot_core  : core mass flow [kg/s]
    """

    # 0) Guards
    eps = np.finfo(float).eps
    omega = max(N1_RPM * 2.0 * np.pi / 60.0, eps)  # rad/s

    PR = max(float(PR_map), 1.0)

    # Bound efficiency (MATLAB: min(max(eta_map*eff_mod,0.1),0.98))
    eta = float(eta_map) * float(eff_mod)
    eta = min(max(eta, 0.1), 0.98)

    # 1) Outlet thermodynamics (compressor)
    # T1 = T0 * (1 + (PR^((g-1)/g)-1)/eta)
    g = gas.gamma
    P1_fan = float(P0) * PR
    tau = PR ** ((g - 1.0) / g)
    T1_raw = float(T0) * (1.0 + (tau - 1.0) / eta)

    # 2) Corrected flow -> actual mass flow (and split core/bypass)
    # m_dot = Wc_total * sqrt(T0/Tref) * (P0/Pref)
    # m_dot_core = m_dot / (1 + BPR)
    # Wc_core = m_dot_core / sqrt(T0/Tref) * (Pref/P0)
    sqrt_term = np.sqrt(float(T0) / gas.Tref)
    m_dot = float(Wc_total) * sqrt_term * (float(P0) / gas.Pref)
    m_dot_core = m_dot / (1.0 + max(float(BPR), 0.0))
    Wc_core = m_dot_core / sqrt_term * (gas.Pref / float(P0))

    # 3) Torque from enthalpy rise
    # delta_h = cp * max(T1_raw - T0, 0)
    # Torque = (m_dot * delta_h) / omega
    delta_h = gas.cp * max(T1_raw - float(T0), 0.0)
    Torque_fan = (m_dot * delta_h) / omega

    return P1_fan, T1_raw, Torque_fan, Wc_core, m_dot_core


if __name__ == "__main__":
    # quick sanity test with dummy values
    P1, T1, tau, Wc_core, mdot_core = fan_core_calc(
        P0=101325.0,
        T0=288.15,
        PR_map=1.4,
        eta_map=0.9,
        N1_RPM=12000,
        Wc_total=50.0,
        BPR=5.0,
        eff_mod=0.95,
    )
    print("P1:", P1, "T1:", T1, "Torque:", tau, "Wc_core:", Wc_core, "mdot_core:", mdot_core)
