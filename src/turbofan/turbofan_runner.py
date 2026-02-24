from __future__ import annotations

from typing import Dict

from src.turbofan.fan_subsystem import FanSubsystem
from src.turbofan.lpc_subsystem import LPCSubsystem
from src.turbofan.hpc_subsystem import HPCSubsystem
from src.turbofan.combustor import combustor_calc
from src.turbofan.hpt_subsystem import HPTSubsystem
from src.turbofan.lpt_subsystem import LPTSubsystem
from src.turbofan.nozzle import nozzle_calc_isentropic_to_ambient


from src.turbofan.nozzle import nozzle_calc_isentropic_to_ambient

def run_turbofan_core_balanced(
    throttle_cmd: float = 0.6,
    P0: float = 101325.0,
    T0: float = 288.15,
    N1_RPM: float = 12000.0,
    N2_RPM: float = 9000.0,
    BPR: float = 5.0,
    eff_mod_fan: float = 1.0,
    eff_mod_lpc: float = 1.0,
    eff_mod_hpc: float = 1.0,
    eta_hpt: float = 0.9,
    eta_lpt: float = 0.9,
    A_core_nozzle: float = 0.05,
    A_bypass_nozzle: float = 0.20,
) -> Dict[str, float]:
    """
    Balanced two-spool core:
      Fan -> LPC -> HPC -> Combustor -> HPT (match HPC torque) -> LPT (match Fan+LPC torque)

    Returns a dict of key signals for logging/health/plots.
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
        P0=P0, T0=T0, N1_RPM=N1_RPM, Wc_total=Wc_total, BPR=BPR, eff_mod=eff_mod_fan
    )
    P2 = fan_out["P1_fan"]
    T2 = fan_out["T1_raw"]
    Wc_core = fan_out["Wc_core"]

    # LPC
    lpc_out = lpc.step(
        P2_in=P2, T2_in=T2, N1_RPM=N1_RPM, Wc_core=Wc_core, eff_mod=eff_mod_lpc
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

        # --- NOZZLES (core + bypass) ---
    core_noz = nozzle_calc_isentropic_to_ambient(
        Pt=lpt_out["P_out"],
        Tt=lpt_out["T_out"],
        mdot=comb_out["m_gas"],
        P0=P0,
        A_exit=A_core_nozzle,
    )

    mdot_bypass = fan_out["m_dot_core"] * BPR
    bypass_noz = nozzle_calc_isentropic_to_ambient(
        Pt=fan_out["P1_fan"],
        Tt=fan_out["T1_raw"],
        mdot=mdot_bypass,
        P0=P0,
        A_exit=A_bypass_nozzle,
    )

    thrust_total = core_noz["Thrust"] + bypass_noz["Thrust"]

    # Flatten “signals” for the framework
    signals = {
        # inlet
        "P0": float(P0),
        "T0": float(T0),
        "N1_RPM": float(N1_RPM),
        "N2_RPM": float(N2_RPM),
        "BPR": float(BPR),
        "throttle_cmd": float(throttle_cmd),

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

        "Thrust_core": float(core_noz["Thrust"]),
        "Thrust_bypass": float(bypass_noz["Thrust"]),
        "Thrust": float(thrust_total),
        "Vexit_core": float(core_noz["Ve"]),
        "Vexit_bypass": float(bypass_noz["Ve"]),

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

    return signals