from __future__ import annotations

from src.turbofan.fan_subsystem import FanSubsystem
from src.turbofan.lpc_subsystem import LPCSubsystem
from src.turbofan.hpc_subsystem import HPCSubsystem
from src.turbofan.combustor import combustor_calc
from src.turbofan.hpt_subsystem import HPTSubsystem
from src.turbofan.lpt_subsystem import LPTSubsystem


def main():
    fan = FanSubsystem.from_default_files()
    lpc = LPCSubsystem.from_default_files()
    hpc = HPCSubsystem.from_default_files()
    hpt = HPTSubsystem()
    lpt = LPTSubsystem()

    # Inlet
    P0 = 101325.0
    T0 = 288.15
    N1_RPM = 12000.0
    N2_RPM = 9000.0
    BPR = 5.0

    Wc_total = float(fan.fan_map.flow_vec.mean())

    Eff_Mod_Fan = 0.95
    Eff_Mod_LPC = 0.95
    Eff_Mod_HPC = 0.95

    # ---------------- FAN ----------------
    fan_out = fan.step(
        P0=P0,
        T0=T0,
        N1_RPM=N1_RPM,
        Wc_total=Wc_total,
        BPR=BPR,
        eff_mod=Eff_Mod_Fan,
    )
    P2 = fan_out["P1_fan"]
    T2 = fan_out["T1_raw"]
    Wc_core = fan_out["Wc_core"]

    # ---------------- LPC ----------------
    lpc_out = lpc.step(
        P2_in=P2,
        T2_in=T2,
        N1_RPM=N1_RPM,
        Wc_core=Wc_core,
        eff_mod=Eff_Mod_LPC,
    )
    P3 = lpc_out["P3"]
    T3 = lpc_out["T3"]
    m_core = lpc_out["m_dot_core"]

    # ---------------- HPC ----------------
    hpc_out = hpc.step(
        P1=P3,
        T1=T3,
        N2_RPM=N2_RPM,
        Wc=m_core,
        eff_mod=Eff_Mod_HPC,
    )

    # ---------------- COMBUSTOR ----------------
    # Combustor inlet = HPC outlet
    P3_comb = hpc_out["P2"]
    T3_comb = hpc_out["T2"]
    m_air = hpc_out["m_dot"]

    comb_out = combustor_calc(
        P3=P3_comb,
        T3=T3_comb,
        m_air=m_air,
        throttle_cmd=0.6,
    )

    # ---------------- HPT (BALANCED) ----------------
    hpt_out = hpt.solve_for_balance(
        P4=comb_out["P4"],
        T4=comb_out["T4"],
        m_gas=comb_out["m_gas"],
        N2_RPM=N2_RPM,
        torque_required=hpc_out["Torque_HPC"],
        eta_turb=0.9,
        PR_min=0.1,
        PR_max=0.9,
        tol=1e-2,
        max_iter=60,
    )

    # ---------------- LPT (BALANCED) ----------------
    torque_required_n1 = fan_out["Torque_fan"] + lpc_out["Torque_LPC"]

    lpt_out = lpt.solve_for_balance(
        P_in=hpt_out["P45"],
        T_in=hpt_out["T45"],
        m_gas=comb_out["m_gas"],
        N1_RPM=N1_RPM,
        torque_required=torque_required_n1,
        eta_turb=0.9,
        PR_min=0.1,
        PR_max=0.95,
        tol=1e-2,
        max_iter=60,
    )

    # ---------------- PRINT RESULTS ----------------
    print("\n=== HPC OUT ===")
    for k, v in hpc_out.items():
        print(k, "=", v)

    print("\n=== COMBUSTOR OUT ===")
    for k, v in comb_out.items():
        print(k, "=", v)

    print("\n=== HPT (BALANCED) OUT ===")
    for k, v in hpt_out.items():
        print(k, "=", v)

    print("\nTorque difference:", hpt_out["Torque_HPT"] - hpc_out["Torque_HPC"])
    print("Balanced:", abs(hpt_out["Torque_HPT"] - hpc_out["Torque_HPC"]) < 1.0)

    print("\n=== LPT (BALANCED) OUT ===")
    for k, v in lpt_out.items():
        print(k, "=", v)

    print("\nN1 torque required:", torque_required_n1)
    print("N1 torque difference:", lpt_out["Torque_LPT"] - torque_required_n1)
    print("Balanced:", abs(lpt_out["Torque_LPT"] - torque_required_n1) < 1.0)


if __name__ == "__main__":
    main()