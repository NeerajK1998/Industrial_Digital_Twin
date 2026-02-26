from __future__ import annotations

from typing import Dict

from src.turbofan.fan_subsystem import FanSubsystem
from src.turbofan.lpc_subsystem import LPCSubsystem
from src.turbofan.hpc_subsystem import HPCSubsystem
from src.turbofan.combustor import combustor_calc
from src.turbofan.hpt_subsystem import HPTSubsystem
from src.turbofan.lpt_subsystem import LPTSubsystem
from src.turbofan.nozzle import (
    nozzle_calc_isentropic_to_ambient,
    nozzle_calc_report_simple,
)


def run_turbofan_core_balanced(
    throttle_cmd: float = 0.6,
    P0: float = 101325.0,
    T0: float = 288.15,
    V0: float = 0.0,
    N1_RPM: float = 12000.0,
    N2_RPM: float = 9000.0,
    N1_ref_rpm: float = 12000.0,
    N2_ref_rpm: float = 9000.0,
    BPR: float = 5.0,
    combustor_mode: str = "T4_cmd",          # validation: "fuel_cmd"
    nozzle_mode: str = "choked_isentropic",  # validation: "report_simple"
    eff_mod_fan: float = 1.0,
    eff_mod_lpc: float = 1.0,
    eff_mod_hpc: float = 1.0,
    eta_hpt: float = 0.9,
    eta_lpt: float = 0.9,
    A_core_nozzle: float = 0.05,
    A_bypass_nozzle: float = 0.20,
) -> Dict[str, float]:
    """
    Balanced two-spool core (steady-state):
      Fan -> LPC -> HPC -> Combustor -> HPT (match HPC torque) -> LPT (match Fan+LPC torque)

    Phase 1 additions:
      - combustor_mode: "fuel_cmd" or "T4_cmd"
      - nozzle_mode: "report_simple" or "choked_isentropic"
      - V0 flight speed passed into nozzle thrust equation
      - N1_pct / N2_pct based on N1_ref_rpm / N2_ref_rpm
    """

    fan = FanSubsystem.from_default_files()
    lpc = LPCSubsystem.from_default_files()
    hpc = HPCSubsystem.from_default_files()
    hpt = HPTSubsystem()
    lpt = LPTSubsystem()

    # choose corrected flow inside fan map range
    Wc_total = float(fan.fan_map.flow_vec.mean())

    # FAN
    fan_out = fan.step(
        P0=P0,
        T0=T0,
        N1_RPM=N1_RPM,
        Wc_total=Wc_total,
        BPR=BPR,
        eff_mod=eff_mod_fan,
    )
    P2 = fan_out["P1_fan"]
    T2 = fan_out["T1_raw"]
    Wc_core = fan_out["Wc_core"]

    # LPC
    lpc_out = lpc.step(
        P2_in=P2,
        T2_in=T2,
        N1_RPM=N1_RPM,
        Wc_core=Wc_core,
        eff_mod=eff_mod_lpc,
    )

    # HPC
    hpc_out = hpc.step(
        P1=lpc_out["P3"],
        T1=lpc_out["T3"],
        N2_RPM=N2_RPM,
        Wc=lpc_out["m_dot_core"],
        eff_mod=eff_mod_hpc,
    )

    # COMBUSTOR (inlet = HPC outlet)
    comb_out = combustor_calc(
        P3=hpc_out["P2"],
        T3=hpc_out["T2"],
        m_air=hpc_out["m_dot"],
        throttle_cmd=throttle_cmd,
        mode=combustor_mode,
    )

    # HPT BALANCE (match HPC torque)
    hpt_out = hpt.solve_for_balance(
        P4=comb_out["P4"],
        T4=comb_out["T4"],
        m_gas=comb_out["m_gas"],
        N2_RPM=N2_RPM,
        torque_required=hpc_out["Torque_HPC"],
        eta_turb=eta_hpt,
        PR_min=0.1,
        PR_max=0.9,
        tol=1e-2,
        max_iter=80,
    )

    # LPT BALANCE (match Fan+LPC torque)
    torque_required_n1 = fan_out["Torque_fan"] + lpc_out["Torque_LPC"]
    lpt_out = lpt.solve_for_balance(
        P_in=hpt_out["P45"],
        T_in=hpt_out["T45"],
        m_gas=comb_out["m_gas"],
        N1_RPM=N1_RPM,
        torque_required=torque_required_n1,
        eta_turb=eta_lpt,
        PR_min=0.1,
        PR_max=0.95,
        tol=1e-2,
        max_iter=80,
    )

    # Nozzle selector (dual-mode)
    def _noz(*, Pt: float, Tt: float, mdot: float, A_exit: float) -> Dict[str, float]:
        if nozzle_mode == "report_simple":
            return nozzle_calc_report_simple(
                Pt=Pt, Tt=Tt, mdot=mdot, P0=P0, A_exit=A_exit, V0=V0
            )
        if nozzle_mode == "choked_isentropic":
            return nozzle_calc_isentropic_to_ambient(
                Pt=Pt, Tt=Tt, mdot=mdot, P0=P0, A_exit=A_exit, V0=V0
            )
        raise ValueError(f"Unknown nozzle_mode: {nozzle_mode}")

    # --- NOZZLES (core + bypass) ---
    core_noz = _noz(
        Pt=lpt_out["P_out"],
        Tt=lpt_out["T_out"],
        mdot=comb_out["m_gas"],
        A_exit=A_core_nozzle,
    )

    mdot_bypass = fan_out["m_dot_core"] * BPR
    bypass_noz = _noz(
        Pt=fan_out["P1_fan"],
        Tt=fan_out["T1_raw"],
        mdot=mdot_bypass,
        A_exit=A_bypass_nozzle,
    )

    thrust_total = core_noz["Thrust"] + bypass_noz["Thrust"]

    # Percent spool speeds (display only; does not affect physics)
    N1_pct = float(N1_RPM) / float(N1_ref_rpm) * 100.0 if N1_ref_rpm else 0.0
    N2_pct = float(N2_RPM) / float(N2_ref_rpm) * 100.0 if N2_ref_rpm else 0.0

    # Flatten “signals” for the framework
    signals: Dict[str, float] = {
        # inlet
        "P0": float(P0),
        "T0": float(T0),
        "V0": float(V0),
        "N1_RPM": float(N1_RPM),
        "N2_RPM": float(N2_RPM),
        "N1_pct": float(N1_pct),
        "N2_pct": float(N2_pct),
        "BPR": float(BPR),
        "throttle_cmd": float(throttle_cmd),

        # modes (strings aren't floats, but we keep them as separate keys below if needed)
        # keep numeric dict pure; if you want modes in JSON, add them in caller layer

        # compressor path
        "P2": float(P2),
        "T2": float(T2),
        "P3": float(lpc_out["P3"]),
        "T3": float(lpc_out["T3"]),
        "P4": float(comb_out["P4"]),
        "T4": float(comb_out["T4"]),
        "P45": float(hpt_out["P45"]),
        "T45": float(hpt_out["T45"]),
        "P5": float(lpt_out["P_out"]),
        "T5": float(lpt_out["T_out"]),

        # flows / fuel
        "m_air": float(hpc_out["m_dot"]),
        "m_fuel": float(comb_out["m_fuel"]),
        "FAR": float(comb_out["FAR"]),
        "m_gas": float(comb_out["m_gas"]),

        # thrust
        "Thrust_core": float(core_noz["Thrust"]),
        "Thrust_bypass": float(bypass_noz["Thrust"]),
        "thrust_total": float(thrust_total),
        "Thrust": float(thrust_total),  # backward compatible key
        "Vexit_core": float(core_noz.get("Ve", 0.0)),
        "Vexit_bypass": float(bypass_noz.get("Ve", 0.0)),

        # map outputs / efficiencies / PR
        "PR_HPC": float(hpc_out["PR_HPC"]),
        "eta_HPC": float(hpc_out["eta_HPC"]),
        "PR_HPT": float(hpt_out["PR_HPT"]),
        "PR_LPT": float(lpt_out["PR_LPT"]),

        # torque balance
        "Torque_FAN": float(fan_out["Torque_fan"]),
        "Torque_LPC": float(lpc_out["Torque_LPC"]),
        "Torque_HPC": float(hpc_out["Torque_HPC"]),
        "Torque_HPT": float(hpt_out["Torque_HPT"]),
        "Torque_LPT": float(lpt_out["Torque_LPT"]),
        "TorqueReq_N1": float(torque_required_n1),

        # balance residuals (should be ~0)
        "TorqueDiff_N2": float(hpt_out["Torque_HPT"] - hpc_out["Torque_HPC"]),
        "TorqueDiff_N1": float(lpt_out["Torque_LPT"] - torque_required_n1),
    }

    # If you want modes in the returned dict (for logging), you can add them
    # as separate non-float keys in a wrapper layer. For now, keep signals numeric.

    return signals

def run_turbofan_core_given_pr(
    throttle_cmd: float = 0.6,
    P0: float = 101325.0,
    T0: float = 288.15,
    V0: float = 0.0,
    N1_RPM: float = 12000.0,
    N2_RPM: float = 9000.0,
    N1_ref_rpm: float = 12000.0,
    N2_ref_rpm: float = 9000.0,
    BPR: float = 5.0,
    combustor_mode: str = "T4_cmd",
    nozzle_mode: str = "choked_isentropic",
    eff_mod_fan: float = 1.0,
    eff_mod_lpc: float = 1.0,
    eff_mod_hpc: float = 1.0,
    eta_hpt: float = 0.9,
    eta_lpt: float = 0.9,
    PR_HPT: float = 0.8470916748046875,
    PR_LPT: float = 0.8982644081115723,
    A_core_nozzle: float = 0.05,
    A_bypass_nozzle: float = 0.20,
) -> Dict[str, float]:
    """
    Phase 2A runner: Turbine PRs are FIXED (scheduled), and spool speeds are intended
    to be SOLVED externally such that torque residuals go to zero.
    """

    fan = FanSubsystem.from_default_files()
    lpc = LPCSubsystem.from_default_files()
    hpc = HPCSubsystem.from_default_files()
    hpt = HPTSubsystem()
    lpt = LPTSubsystem()

    Wc_total = float(fan.fan_map.flow_vec.mean())

    # FAN
    fan_out = fan.step(P0=P0, T0=T0, N1_RPM=N1_RPM, Wc_total=Wc_total, BPR=BPR, eff_mod=eff_mod_fan)
    P2 = fan_out["P1_fan"]
    T2 = fan_out["T1_raw"]
    Wc_core = fan_out["Wc_core"]

    # LPC
    lpc_out = lpc.step(P2_in=P2, T2_in=T2, N1_RPM=N1_RPM, Wc_core=Wc_core, eff_mod=eff_mod_lpc)

    # HPC
    hpc_out = hpc.step(P1=lpc_out["P3"], T1=lpc_out["T3"], N2_RPM=N2_RPM, Wc=lpc_out["m_dot_core"], eff_mod=eff_mod_hpc)

    # COMBUSTOR
    comb_out = combustor_calc(
        P3=hpc_out["P2"],
        T3=hpc_out["T2"],
        m_air=hpc_out["m_dot"],
        throttle_cmd=throttle_cmd,
        mode=combustor_mode,
    )

    # HPT (FIXED PR)  ✅ you must implement this method in HPTSubsystem if missing
    hpt_out = hpt.step_with_pr(
    P4=comb_out["P4"],
    T4=comb_out["T4"],
    m_gas=comb_out["m_gas"],
    N2_RPM=N2_RPM,
    PR_turb=PR_HPT,
    eta_turb=eta_hpt,
)

    # LPT (FIXED PR) ✅ implement if missing
    lpt_out = lpt.step_with_pr(
    P_in=hpt_out["P45"],
    T_in=hpt_out["T45"],
    m_gas=comb_out["m_gas"],
    N1_RPM=N1_RPM,
    PR_turb=PR_LPT,
    eta_turb=eta_lpt,
)

    # nozzle selector (same as your code)
    def _noz(*, Pt: float, Tt: float, mdot: float, A_exit: float) -> Dict[str, float]:
        if nozzle_mode == "report_simple":
            return nozzle_calc_report_simple(Pt=Pt, Tt=Tt, mdot=mdot, P0=P0, A_exit=A_exit, V0=V0)
        if nozzle_mode == "choked_isentropic":
            return nozzle_calc_isentropic_to_ambient(Pt=Pt, Tt=Tt, mdot=mdot, P0=P0, A_exit=A_exit, V0=V0)
        raise ValueError(f"Unknown nozzle_mode: {nozzle_mode}")

    core_noz = _noz(Pt=lpt_out["P_out"], Tt=lpt_out["T_out"], mdot=comb_out["m_gas"], A_exit=A_core_nozzle)
    mdot_bypass = fan_out["m_dot_core"] * BPR
    bypass_noz = _noz(Pt=fan_out["P1_fan"], Tt=fan_out["T1_raw"], mdot=mdot_bypass, A_exit=A_bypass_nozzle)

    thrust_total = core_noz["Thrust"] + bypass_noz["Thrust"]

    N1_pct = float(N1_RPM) / float(N1_ref_rpm) * 100.0 if N1_ref_rpm else 0.0
    N2_pct = float(N2_RPM) / float(N2_ref_rpm) * 100.0 if N2_ref_rpm else 0.0

    torque_required_n1 = fan_out["Torque_fan"] + lpc_out["Torque_LPC"]

    signals: Dict[str, float] = {
        "P0": float(P0), "T0": float(T0), "V0": float(V0),
        "N1_RPM": float(N1_RPM), "N2_RPM": float(N2_RPM),
        "N1_pct": float(N1_pct), "N2_pct": float(N2_pct),
        "BPR": float(BPR), "throttle_cmd": float(throttle_cmd),

        "P2": float(P2), "T2": float(T2),
        "P3": float(lpc_out["P3"]), "T3": float(lpc_out["T3"]),
        "P4": float(comb_out["P4"]), "T4": float(comb_out["T4"]),
        "P45": float(hpt_out["P45"]), "T45": float(hpt_out["T45"]),
        "P5": float(lpt_out["P_out"]), "T5": float(lpt_out["T_out"]),

        "m_air": float(hpc_out["m_dot"]),
        "m_fuel": float(comb_out["m_fuel"]),
        "FAR": float(comb_out["FAR"]),
        "m_gas": float(comb_out["m_gas"]),

        "Thrust_core": float(core_noz["Thrust"]),
        "Thrust_bypass": float(bypass_noz["Thrust"]),
        "thrust_total": float(thrust_total),
        "Thrust": float(thrust_total),

        "PR_HPC": float(hpc_out["PR_HPC"]),
        "eta_HPC": float(hpc_out["eta_HPC"]),
        "PR_HPT": float(PR_HPT),
        "PR_LPT": float(PR_LPT),

        "Torque_FAN": float(fan_out["Torque_fan"]),
        "Torque_LPC": float(lpc_out["Torque_LPC"]),
        "Torque_HPC": float(hpc_out["Torque_HPC"]),
        "Torque_HPT": float(hpt_out["Torque_HPT"]),
        "Torque_LPT": float(lpt_out["Torque_LPT"]),
        "TorqueReq_N1": float(torque_required_n1),

        "TorqueDiff_N2": float(hpt_out["Torque_HPT"] - hpc_out["Torque_HPC"]),
        "TorqueDiff_N1": float(lpt_out["Torque_LPT"] - torque_required_n1),
    }
    return signals