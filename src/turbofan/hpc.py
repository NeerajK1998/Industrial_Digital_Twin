from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class GasProps:
    gamma: float = 1.4
    cp: float = 1004.0      # J/(kg*K) matches Simulink code
    Tref: float = 288.15    # K
    Pref: float = 101325.0  # Pa


def hpc_core_calc(
    P1: float,
    T1: float,
    PR: float,
    eta: float,
    N2_RPM: float,
    Wc: float,
    gas: GasProps = GasProps(),
) -> Tuple[float, float, float, float]:
    """
    Faithful Python port of Models/HPC.slx MATLAB Function:

      function [P2, T2, Torque_HPC] = HPC_CoreCalc(P1, T1, PR, eta, N2, Wc)

    Returns:
      P2          outlet total pressure [Pa]
      T2          outlet total temperature [K]
      Torque_HPC  shaft torque [N*m]
      m_dot       mass flow [kg/s]
    """
    eps = np.finfo(float).eps
    omega = max(float(N2_RPM) * 2.0 * np.pi / 60.0, eps)

    gamma = gas.gamma
    cp = gas.cp

    # Pressure rise
    P2 = float(P1) * float(PR)

    # Temperature rise
    T2 = float(T1) * (1.0 + (1.0 / float(eta)) * (float(PR) ** ((gamma - 1.0) / gamma) - 1.0))

    # Inverse correction: m_dot = Wc * sqrt(T1/Tref) * (P1/Pref)
    m_dot = float(Wc) * np.sqrt(float(T1) / gas.Tref) * (float(P1) / gas.Pref)

    # Enthalpy rise and torque
    delta_h = cp * (T2 - float(T1))
    Torque_HPC = (m_dot * delta_h) / omega

    return P2, T2, Torque_HPC, m_dot


if __name__ == "__main__":
    P2, T2, tq, mdot = hpc_core_calc(
        P1=200000.0,
        T1=400.0,
        PR=6.0,
        eta=0.88,
        N2_RPM=9000.0,
        Wc=40.0,
    )
    print("P2:", P2)
    print("T2:", T2)
    print("Torque_HPC:", tq)
    print("m_dot:", mdot)
