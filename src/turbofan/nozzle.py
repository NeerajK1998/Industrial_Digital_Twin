from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict


@dataclass
class NozzleResult:
    choked: bool
    Me: float
    Pe: float
    Te: float
    rho_e: float
    Ve: float
    thrust: float


def _critical_pressure_ratio(gamma: float) -> float:
    """
    Returns (P*/Pt) for choked flow at M=1, where * denotes critical.
    (P*/Pt) = (2/(gamma+1))^(gamma/(gamma-1))
    """
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def _static_from_total_unchoked(Pt: float, Tt: float, Pe: float, gamma: float) -> tuple[float, float, float]:
    """
    For unchoked expansion to exit static pressure Pe:
    Te = Tt * (Pe/Pt)^((gamma-1)/gamma)
    Me^2 = (2/(gamma-1)) * ((Tt/Te) - 1)
    """
    pr = max(Pe / Pt, 1e-12)
    Te = Tt * (pr ** ((gamma - 1.0) / gamma))
    Me2 = max(0.0, (2.0 / (gamma - 1.0)) * ((Tt / max(Te, 1e-12)) - 1.0))
    Me = sqrt(Me2)
    return Me, Te, Pe


def _static_from_total_choked(Pt: float, Tt: float, gamma: float) -> tuple[float, float, float]:
    """
    For choked flow at nozzle throat/exit (M=1):
    T* = Tt * 2/(gamma+1)
    P* = Pt * (2/(gamma+1))^(gamma/(gamma-1))
    """
    Te = Tt * (2.0 / (gamma + 1.0))
    Pe = Pt * _critical_pressure_ratio(gamma)
    Me = 1.0
    return Me, Te, Pe


def nozzle_calc_isentropic_to_ambient(
    *,
    mdot: float,
    Pt: float,
    Tt: float,
    P0: float,
    A_exit: float,
    gamma: float = 1.33,
    R: float = 287.0,
    V0: float = 0.0,
) -> Dict[str, float]:
    """
    Computes nozzle exit conditions and thrust with choking check.

    Inputs:
      mdot: mass flow [kg/s]
      Pt, Tt: nozzle inlet total conditions [Pa, K]
      P0: ambient static pressure [Pa]
      A_exit: nozzle exit area [m^2]
      gamma, R: gas properties
      V0: flight speed [m/s] (0 for static thrust)

    Returns dict compatible with your turbofan_runner usage.
    """

    # Basic guards
    mdot = float(mdot)
    Pt = float(Pt)
    Tt = float(Tt)
    P0 = float(P0)
    A_exit = float(A_exit)

    if mdot <= 0.0 or Pt <= 0.0 or Tt <= 0.0 or A_exit <= 0.0 or P0 <= 0.0:
        return {
            "choked": 0.0,
            "Me": 0.0,
            "Pe": 0.0,
            "Te": 0.0,
            "rho_e": 0.0,
            "Ve": 0.0,
            "Thrust": 0.0,
        }

    # Choking check: compare ambient to critical pressure
    pcrit_ratio = _critical_pressure_ratio(gamma)  # P*/Pt
    pcrit = Pt * pcrit_ratio

    if P0 <= pcrit:
        # Choked at exit/throat (model exit as choked)
        Me, Te, Pe = _static_from_total_choked(Pt, Tt, gamma)
        choked = True
    else:
        # Unchoked: expand to ambient
        Me, Te, Pe = _static_from_total_unchoked(Pt, Tt, P0, gamma)
        choked = False

    rho_e = Pe / (R * max(Te, 1e-12))

    # Exit velocity from isentropic relation:
    # Ve = sqrt(2*cp*(Tt-Te)), cp = gamma*R/(gamma-1)
    cp = gamma * R / (gamma - 1.0)
    Ve = sqrt(max(0.0, 2.0 * cp * (Tt - Te)))

    # Thrust: momentum + pressure term
    thrust = mdot * (Ve - V0) + (Pe - P0) * A_exit

    return {
        "choked": 1.0 if choked else 0.0,
        "Me": float(Me),
        "Pe": float(Pe),
        "Te": float(Te),
        "rho_e": float(rho_e),
        "Ve": float(Ve),
        "Thrust": float(thrust),
    }
