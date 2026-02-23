from __future__ import annotations
from turbofan.fan_subsystem import FanSubsystem
from turbofan.lpc_subsystem import LPCSubsystem
from turbofan.hpc_subsystem import HPCSubsystem

def main():
    fan = FanSubsystem.from_default_files()
    lpc = LPCSubsystem.from_default_files()
    hpc = HPCSubsystem.from_default_files()

    # Inlet
    P0 = 101325.0
    T0 = 288.15
    N1_RPM = 12000.0
    BPR = 5.0

    # Choose a corrected flow inside fan map range
    Wc_total = float(fan.fan_map.flow_vec.mean())

    # PDM modifiers (examples)
    Eff_Mod_Fan = 0.95
    Eff_Mod_LPC = 0.95

    fan_out = fan.step(
        P0=P0, T0=T0, N1_RPM=N1_RPM, Wc_total=Wc_total, BPR=BPR, eff_mod=Eff_Mod_Fan
    )

    # Fan outlet becomes LPC inlet (simplified)
    P2 = fan_out["P1_fan"]
    T2 = fan_out["T1_raw"]
    Wc_core = fan_out["Wc_core"]

    lpc_out = lpc.step(
        P2_in=P2, T2_in=T2, N1_RPM=N1_RPM, Wc_core=Wc_core, eff_mod=Eff_Mod_LPC
    )

    print("\n=== FAN OUT ===")
    for k, v in fan_out.items():
        print(k, "=", v)


    # HPC stage (N2 shaft)
    P3 = lpc_out["P3"]
    T3 = lpc_out["T3"]
    Wc_hpc = lpc_out["m_dot_core"]

    Eff_Mod_HPC = 0.95

    hpc_out = hpc.step(
        P1=P3,
        T1=T3,
        N2_RPM=9000.0,
        Wc=Wc_hpc,
        eff_mod=Eff_Mod_HPC,
    )

    print("\n=== LPC OUT ===")
    for k, v in lpc_out.items():
        print(k, "=", v)

    print("\n=== HPC OUT ===")
    for k, v in hpc_out.items():
        print(k, "=", v)

if __name__ == "__main__":
    main()
