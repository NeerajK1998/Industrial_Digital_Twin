# src/turbofan/combustor.py
"""
Combustor / burner block.

We keep the report's energy balance but change the control interpretation:

Instead of: throttle -> fuel fraction (which can drive T4 unrealistically high),
we use: throttle -> commanded turbine inlet temperature T4_cmd (within bounds),
then compute fuel from the energy balance:

  m_fuel * LHV * eta_burn = m_air * cp_gas * (T4_cmd - T3)

We still clamp fuel by FAR max: m_fuel <= far_max_factor * m_air

This makes the model stable and keeps T4 in a realistic band for the framework.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CombustorParams:
    cp_gas: float = 1150.0        # J/(kg*K) hot gas cp (use same order as turbine blocks)
    lhv: float = 43e6             # J/kg
    eta_burn: float = 0.99        # burner efficiency
    far_max_factor: float = 0.04  # m_fuel_max = factor * m_air (limit)
    dp: float = 0.0               # Pa (idealized)
    mf_max: float = 1.2          # kg/s, used only in fuel_cmd mode (MATLAB-faithful)
    # throttle -> target T4 mapping
    T4_min: float = 1200.0        # K
    T4_max: float = 2035.0       # K


def combustor_calc(
    P3: float,
    T3: float,
    m_air: float,
    throttle_cmd: float,
    *,
    mode: str = "T4_cmd",  # "T4_cmd" (industrial) or "fuel_cmd" (MATLAB-faithful)
    params: CombustorParams = CombustorParams(),
) -> dict:
    """
    Args:
      P3, T3: combustor inlet total pressure [Pa], total temperature [K]
      m_air: core air mass flow [kg/s]
      throttle_cmd: 0..1 (interpreted as T4 command between T4_min..T4_max)

    Returns:
      dict with P4, T4, m_fuel, m_fuel_max, FAR, m_gas, T4_cmd
    """
    if m_air <= 0:
        raise ValueError("m_air must be > 0")

    throttle = max(0.0, min(float(throttle_cmd), 1.0))

    # Constant total pressure (ideal)
    P4 = P3 - params.dp

    # Limit by FAR max (shared)
    m_fuel_max = params.far_max_factor * m_air

    if mode == "fuel_cmd":
        # MATLAB-faithful: throttle directly scales fuel flow
        m_fuel = throttle * params.mf_max

        # Energy balance gives achieved T4 (no T4_cmd in this mode)
        T4 = float(T3) + (m_fuel * params.lhv * params.eta_burn) / (m_air * params.cp_gas)
        T4_cmd = None

    elif mode == "T4_cmd":
        # Industrial: throttle commands T4 within bounds
        T4_cmd = params.T4_min + throttle * (params.T4_max - params.T4_min)

        # If inlet already hotter than command, no fuel (fail-safe)
        deltaT = max(0.0, float(T4_cmd) - float(T3))

        # Energy balance -> fuel
        m_fuel = (m_air * params.cp_gas * deltaT) / (params.lhv * max(params.eta_burn, 1e-9))

        # Clamp fuel (FAR max) and recompute achieved T4 if clamped
        if m_fuel > m_fuel_max:
            m_fuel = m_fuel_max
            T4 = float(T3) + (m_fuel * params.lhv * params.eta_burn) / (m_air * params.cp_gas)
        else:
            T4 = float(T4_cmd)

    else:
        raise ValueError(f"Unknown combustor mode: {mode}")

    FAR = m_fuel / m_air
    m_gas = m_air + m_fuel

    out = {
        "P4": float(P4),
        "T4": float(T4),
        "m_fuel": float(m_fuel),
        "m_fuel_max": float(m_fuel_max),
        "FAR": float(FAR),
        "m_gas": float(m_gas),
        "mode": mode,
    }
    if mode == "T4_cmd":
        out["T4_cmd"] = float(T4_cmd)
    return out
