from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import root

from src.turbofan.turbofan_runner import run_turbofan_core_given_pr
from src.turbofan.pr_schedule import PRSchedule


# cache schedules by path (so we don't reload JSON every call)
_SCHED_CACHE: dict[str, PRSchedule] = {}


def _get_sched(path: str) -> PRSchedule:
    if path not in _SCHED_CACHE:
        _SCHED_CACHE[path] = PRSchedule(path)
    return _SCHED_CACHE[path]


@dataclass
class SolveRPMResult:
    N1_RPM: float
    N2_RPM: float
    out: Dict[str, float]
    success: bool
    message: str


def solve_n1_n2_scheduled_pr(
    *,
    throttle_cmd: float,
    P0: float,
    T0: float,
    V0: float,
    BPR: float,
    N1_ref_rpm: float,
    N2_ref_rpm: float,
    pr_schedule_path: str = "src/turbofan/pr_schedule.json",  # <--- NEW
    combustor_mode: str = "fuel_cmd",
    nozzle_mode: str = "report_simple",
    A_core_nozzle: float = 0.017,
    A_bypass_nozzle: float = 0.061,
    eff_mod_fan: float = 1.0,
    eff_mod_lpc: float = 1.0,
    eff_mod_hpc: float = 1.0,
    eta_hpt: float = 0.9,
    eta_lpt: float = 0.9,
    N1_guess_pct: float = 76.8,
    N2_guess_pct: float = 93.1,
) -> SolveRPMResult:
    """
    Phase 2B/2C solver:
    solve for N1/N2 torque balance,
    with turbine PRs scheduled vs throttle.
    """

    sched = _get_sched(pr_schedule_path)
    PR_HPT, PR_LPT = sched.get(float(throttle_cmd))

    N1_lo, N1_hi = 0.40 * N1_ref_rpm, 1.05 * N1_ref_rpm
    N2_lo, N2_hi = 0.40 * N2_ref_rpm, 1.05 * N2_ref_rpm

    x0 = np.array([
        (N1_guess_pct / 100.0) * N1_ref_rpm,
        (N2_guess_pct / 100.0) * N2_ref_rpm,
    ], dtype=float)

    def _clamp(n1: float, n2: float) -> Tuple[float, float]:
        return (
            float(np.clip(n1, N1_lo, N1_hi)),
            float(np.clip(n2, N2_lo, N2_hi)),
        )

    def residuals(x: np.ndarray) -> np.ndarray:
        n1, n2 = _clamp(x[0], x[1])

        out = run_turbofan_core_given_pr(
            throttle_cmd=throttle_cmd,
            P0=P0, T0=T0, V0=V0,
            N1_RPM=n1, N2_RPM=n2,
            N1_ref_rpm=N1_ref_rpm, N2_ref_rpm=N2_ref_rpm,
            BPR=BPR,
            combustor_mode=combustor_mode,
            nozzle_mode=nozzle_mode,
            eff_mod_fan=eff_mod_fan,
            eff_mod_lpc=eff_mod_lpc,
            eff_mod_hpc=eff_mod_hpc,
            eta_hpt=eta_hpt,
            eta_lpt=eta_lpt,
            PR_HPT=PR_HPT,
            PR_LPT=PR_LPT,
            A_core_nozzle=A_core_nozzle,
            A_bypass_nozzle=A_bypass_nozzle,
        )

        r1 = float(out["TorqueDiff_N1"])
        r2 = float(out["TorqueDiff_N2"])
        if not np.isfinite(r1) or not np.isfinite(r2):
            return np.array([1e9, 1e9], dtype=float)
        return np.array([r1, r2], dtype=float)

    sol = root(residuals, x0, method="hybr")
    n1_sol, n2_sol = _clamp(sol.x[0], sol.x[1])

    out_final = run_turbofan_core_given_pr(
        throttle_cmd=throttle_cmd,
        P0=P0, T0=T0, V0=V0,
        N1_RPM=n1_sol, N2_RPM=n2_sol,
        N1_ref_rpm=N1_ref_rpm, N2_ref_rpm=N2_ref_rpm,
        BPR=BPR,
        combustor_mode=combustor_mode,
        nozzle_mode=nozzle_mode,
        eff_mod_fan=eff_mod_fan,
        eff_mod_lpc=eff_mod_lpc,
        eff_mod_hpc=eff_mod_hpc,
        eta_hpt=eta_hpt,
        eta_lpt=eta_lpt,
        PR_HPT=PR_HPT,
        PR_LPT=PR_LPT,
        A_core_nozzle=A_core_nozzle,
        A_bypass_nozzle=A_bypass_nozzle,
    )

    out_final["PR_HPT_sched"] = float(PR_HPT)
    out_final["PR_LPT_sched"] = float(PR_LPT)
    out_final["pr_schedule_path"] = pr_schedule_path

    return SolveRPMResult(
        N1_RPM=float(n1_sol),
        N2_RPM=float(n2_sol),
        out=out_final,
        success=bool(sol.success),
        message=str(sol.message),
    )