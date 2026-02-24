from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.turbofan.turbofan_runner import run_turbofan_core_balanced
from src.health_logic import decide_status


FEATURE_KEYS = [
    # speeds
    "N1_RPM", "N2_RPM",
    # temps/flows
    "T4", "m_fuel", "FAR", "m_gas",
    # pressures (core)
    "P0", "P2", "P3", "P4", "P5",
    # thrust
    "Thrust", "Thrust_core", "Thrust_bypass",
    "Vexit_core", "Vexit_bypass",
    # residuals
    "TorqueDiff_N1", "TorqueDiff_N2",
]

DERIVED_KEYS = [
    # pressure ratios
    "PR_fan",           # P2/P0
    "PR_hpc",           # P3/P2
    "PR_core_total",    # P3/P0
    "PR_turb",          # P5/P4

    # performance proxies
    "TSFC",             # m_fuel / Thrust
    "ThrustSplit_bypass",  # Thrust_bypass / Thrust

    # residual summary
    "TorqueResSum",     # abs(TorqueDiff_N1) + abs(TorqueDiff_N2)
    "DeltaT_turb",
]


def _safe_get(d: Dict[str, float], k: str) -> float:
    v = d.get(k, 0.0)
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        a = float(a)
        b = float(b)
        if abs(b) < 1e-12:
            return default
        return a / b
    except Exception:
        return default


def add_derived_features(out: Dict[str, float]) -> None:
    # ----------------------
    # Raw pressures
    # ----------------------
    P0 = out.get("P0", 0.0)
    P2 = out.get("P2", 0.0)
    P3 = out.get("P3", 0.0)
    P4 = out.get("P4", 0.0)
    P5 = out.get("P5", 0.0)

    # ----------------------
    # Temperatures (for turbine drop)
    # ----------------------
    T4 = out.get("T4", 0.0)
    T5 = out.get("T5", 0.0)

    # ----------------------
    # Flows / thrust
    # ----------------------
    thrust = out.get("Thrust", 0.0)
    thrust_bypass = out.get("Thrust_bypass", 0.0)
    m_fuel = out.get("m_fuel", 0.0)

    # ----------------------
    # Torque residuals
    # ----------------------
    td1 = out.get("TorqueDiff_N1", 0.0)
    td2 = out.get("TorqueDiff_N2", 0.0)

    # ======================
    # Pressure ratios
    # ======================
    out["PR_fan"] = _safe_div(P2, P0)
    out["PR_hpc"] = _safe_div(P3, P2)
    out["PR_core_total"] = _safe_div(P3, P0)
    out["PR_turb"] = _safe_div(P4, P5)

    # ======================
    # Performance proxies
    # ======================
    out["TSFC"] = _safe_div(m_fuel, thrust)
    out["ThrustSplit_bypass"] = _safe_div(thrust_bypass, thrust)

    # ======================
    # Residual summary
    # ======================
    out["TorqueResSum"] = abs(float(td1)) + abs(float(td2))

    # ======================
    # Turbine temperature drop
    # ======================
    out["DeltaT_turb"] = float(T4) - float(T5)

def generate_one_sample(
    throttle_cmd: float,
    eff_mod_fan: float,
    eff_mod_lpc: float,
    eff_mod_hpc: float,
    eta_hpt: float,
    eta_lpt: float,
    BPR: float,
) -> Optional[Dict[str, float]]:
    """
    Returns a dict of signals + labels if physics converges,
    otherwise returns None (skip infeasible points).
    """
    try:
        out = run_turbofan_core_balanced(
            throttle_cmd=throttle_cmd,
            eff_mod_fan=eff_mod_fan,
            eff_mod_lpc=eff_mod_lpc,
            eff_mod_hpc=eff_mod_hpc,
            eta_hpt=eta_hpt,
            eta_lpt=eta_lpt,
            BPR=BPR,
        )

        # Stage A label
        status = decide_status(out)
        out["label_status"] = 1 if status == "FLY" else 0

        # Stage B label
        torque_res = abs(out.get("TorqueDiff_N1", 0.0)) + abs(out.get("TorqueDiff_N2", 0.0))
        thrust = out.get("Thrust", 0.0)
        eff_min = min(eff_mod_fan, eff_mod_lpc, eff_mod_hpc)

        if status != "FLY" or torque_res > 5000 or thrust < 50000:
            label_state = 2  # FAULT
        elif eff_min < 0.85 or torque_res > 1000:
            label_state = 1  # WARNING
        else:
            label_state = 0  # OK

        out["label_state"] = label_state

        # Stage C target (continuous)
        fan_loss = 1.0 - eff_mod_fan
        lpc_loss = 1.0 - eff_mod_lpc
        hpc_loss = 1.0 - eff_mod_hpc
        hpt_loss = 1.0 - eta_hpt
        lpt_loss = 1.0 - eta_lpt

        degradation_score = (
            0.4 * fan_loss +
            0.3 * lpc_loss +
            0.3 * hpc_loss +
            0.5 * (hpt_loss + lpt_loss)
        )
        out["degradation_score"] = max(0.0, min(1.0, degradation_score))

        # Derived features
        add_derived_features(out)

        return out

    except Exception:
        return None


def build_dataset(
    n: int = 500,
    out_csv: str = "outputs/ml/turbofan_dataset_A.csv",
) -> str:
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []

    # More operating points (richer dataset)
    throttles = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bprs = [2.0, 5.0, 8.0, 10.0]

    # Multi-severity fault levels
    fault_eff_levels = [0.9, 0.8, 0.7]      # fan/lpc/hpc eff_mod
    fault_eta_levels = [0.88, 0.82, 0.75]   # hpt/lpt eta
    nominal_eta = 0.92

    i = 0

    for th in throttles:
        for bpr in bprs:
            # ------------------
            # NORMAL (class 0)
            # ------------------
            s = generate_one_sample(
                throttle_cmd=th,
                eff_mod_fan=1.0,
                eff_mod_lpc=1.0,
                eff_mod_hpc=1.0,
                eta_hpt=nominal_eta,
                eta_lpt=nominal_eta,
                BPR=bpr,
            )
            if s is not None:
                s["fault_class"] = 0
                rows.append(s)
                i += 1
                if i >= n:
                    break

            # ------------------
            # FAN fault (class 1)
            # ------------------
            for sev in fault_eff_levels:
                s = generate_one_sample(
                    throttle_cmd=th,
                    eff_mod_fan=sev,
                    eff_mod_lpc=1.0,
                    eff_mod_hpc=1.0,
                    eta_hpt=nominal_eta,
                    eta_lpt=nominal_eta,
                    BPR=bpr,
                )
                if s is None:
                    continue
                s["fault_class"] = 1
                rows.append(s)
                i += 1
                if i >= n:
                    break
            if i >= n:
                break

            # ------------------
            # LPC fault (class 2)
            # ------------------
            for sev in fault_eff_levels:
                s = generate_one_sample(
                    throttle_cmd=th,
                    eff_mod_fan=1.0,
                    eff_mod_lpc=sev,
                    eff_mod_hpc=1.0,
                    eta_hpt=nominal_eta,
                    eta_lpt=nominal_eta,
                    BPR=bpr,
                )
                if s is None:
                    continue
                s["fault_class"] = 2
                rows.append(s)
                i += 1
                if i >= n:
                    break
            if i >= n:
                break

            # ------------------
            # HPC fault (class 3)
            # ------------------
            for sev in fault_eff_levels:
                s = generate_one_sample(
                    throttle_cmd=th,
                    eff_mod_fan=1.0,
                    eff_mod_lpc=1.0,
                    eff_mod_hpc=sev,
                    eta_hpt=nominal_eta,
                    eta_lpt=nominal_eta,
                    BPR=bpr,
                )
                if s is None:
                    continue
                s["fault_class"] = 3
                rows.append(s)
                i += 1
                if i >= n:
                    break
            if i >= n:
                break

            # ------------------
            # TURBINE fault (class 4)
            # ------------------
            for sev_eta in fault_eta_levels:
                s = generate_one_sample(
                    throttle_cmd=th,
                    eff_mod_fan=1.0,
                    eff_mod_lpc=1.0,
                    eff_mod_hpc=1.0,
                    eta_hpt=sev_eta,
                    eta_lpt=sev_eta,
                    BPR=bpr,
                )
                if s is None:
                    continue
                s["fault_class"] = 4
                rows.append(s)
                i += 1
                if i >= n:
                    break

            if i >= n:
                break

        if i >= n:
            break

    # Write CSV
    fieldnames = FEATURE_KEYS + DERIVED_KEYS + [
        "label_status",
        "label_state",
        "degradation_score",
        "fault_class",
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _safe_get(r, k) for k in fieldnames})

    return out_csv


if __name__ == "__main__":
    path = build_dataset(n=600)
    print(f"Wrote: {path}")
