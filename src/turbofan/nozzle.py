from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class NozzleResult:
    v_exit: float
    thrust: float
    term: float


def nozzle_calc_isentropic_to_ambient(
    Pt: float,
    Tt: float,
    mdot: float,
    P0: float,
    A_exit: float,
    gamma: float = 1.4,
    cp: float = 1004.0,
) -> Dict[str, float]:
    """
    Minimal nozzle thrust model (steady-state, static test stand V0=0):
      v_exit = sqrt( 2 * cp * Tt * (1 - (P0/Pt)^((gamma-1)/gamma)) )

    Report-faithful simple thrust form:
      F = mdot*v_exit + (Pt - P0)*A_exit

    Notes:
    - Uses total inlet conditions Pt, Tt at nozzle inlet.
    - Expands to ambient P0 (no choking handling yet, by design for smallest safe step).
    """
    Pt = float(Pt)
    Tt = float(Tt)
    mdot = float(mdot)
    P0 = float(P0)
    A_exit = float(A_exit)

    if Pt <= 0 or Tt <= 0 or mdot < 0 or P0 <= 0 or A_exit <= 0:
        return {"v_exit": 0.0, "thrust": 0.0, "term": 0.0}

    pr = P0 / Pt
    expo = (gamma - 1.0) / gamma

    # numerical safety: avoid negative inside sqrt if Pt ~ P0
    term = max(0.0, 1.0 - (pr ** expo))

    v_exit = math.sqrt(2.0 * cp * Tt * term)

    # report-style pressure term uses (Pt - P0)*A_exit
    thrust = mdot * v_exit + (Pt - P0) * A_exit

    return {"v_exit": float(v_exit), "thrust": float(thrust), "term": float(term)}
