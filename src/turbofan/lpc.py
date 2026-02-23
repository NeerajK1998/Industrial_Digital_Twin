from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class GasProps:
    gamma: float = 1.4
    cp: float = 1004.0      # J/(kg*K) (matches Simulink LPC_CoreCalc code)
    Tref: float = 288.15    # K
    Pref: float = 101325.0  # Pa


def lpc_core_calc(
    P2_in: float,
    T2_in: float,
    PR_LPC: float,
    eta_LPC: float,
    N1_RPM: float,
    Wc_core: float,
    gas: GasProps = GasProps(),
) -> Tuple[float, float, float, float]:
    """
    Faithful Python port of the MATLAB Function inside Models/LPC.slx:

      function [P3, T3, Torque_LPC] = LPC_CoreCalc(P2_in, T2_in, PR_LPC, eta_LPC, N1, Wc_core)

    Returns:
      P3          outlet total pressure [Pa]
      T3          outlet total temperature [K]
      Torque_LPC  torque on N1 shaft [N*m]
      m_dot_core  core mass flow [kg/s]
    """
    eps = np.finfo(float).eps
    omega1 = max(float(N1_RPM) * 2.0 * np.pi / 60.0, eps)

    gamma = gas.gamma
    cp = gas.cp

    # Outlet conditions
    P3 = float(P2_in) * float(PR_LPC)
    T3 = float(T2_in) * (1.0 + (1.0 / float(eta_LPC)) *
                         (float(PR_LPC) ** ((gamma - 1.0) / gamma) - 1.0))

    # Recover actual core mass flow from corrected Wc_core
    m_dot_core = float(Wc_core) * np.sqrt(float(T2_in) / gas.Tref) * (float(P2_in) / gas.Pref)

    # Enthalpy rise and torque
    delta_h_core = cp * (T3 - float(T2_in))
    Torque_LPC = (m_dot_core * delta_h_core) / omega1

    return P3, T3, Torque_LPC, m_dot_core


if __name__ == "__main__":
    P3, T3, tq, mdot = lpc_core_calc(
        P2_in=180000.0,
        T2_in=330.0,
        PR_LPC=3.5,
        eta_LPC=0.86,
        N1_RPM=12000.0,
        Wc_core=60.0,
    )
    print("P3:", P3)
    print("T3:", T3)
    print("Torque_LPC:", tq)
    print("m_dot_core:", mdot)
