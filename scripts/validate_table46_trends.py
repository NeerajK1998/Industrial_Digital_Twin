from src.turbofan.turbofan_runner import run_turbofan_core_balanced

# Cruise condition (Table 4.3)
P0 = 23841.93
T0 = 218.808
V0 = 231.30

N1_ref = 12000.0
N2_ref = 9000.0
N1_RPM_base = 0.768 * N1_ref
N2_RPM_base = 0.931 * N2_ref


def run_case(name, *, combustor_mode, nozzle_mode, throttle_cmd, N1_RPM, N2_RPM, **kwargs):
    out = run_turbofan_core_balanced(
        throttle_cmd=throttle_cmd,
        P0=P0,
        T0=T0,
        V0=V0,
        N1_RPM=N1_RPM,
        N2_RPM=N2_RPM,
        N1_ref_rpm=N1_ref,
        N2_ref_rpm=N2_ref,
        BPR=9.0,
        combustor_mode=combustor_mode,
        nozzle_mode=nozzle_mode,

        # Phase 1 calibrated nozzle areas for report_simple baseline comparisons
        A_core_nozzle=0.017,
        A_bypass_nozzle=0.061,

        # disable degradation/modifiers unless explicitly set for fault
        #eff_mod_fan=1.0,
        #eff_mod_lpc=1.0,
        #eff_mod_hpc=1.0,
        eta_hpt=0.9,
        eta_lpt=0.9,
        **kwargs
    )

    print(f"\n{name}")
    print(f" Thrust: {out['thrust_total']:.1f}")
    print(f" Fuel  : {out['m_fuel']:.3f}")
    print(f" N1 %  : {out.get('N1_pct', -1):.2f}")
    print(f" N2 %  : {out.get('N2_pct', -1):.2f}")
    print(f" T4    : {out['T4']:.1f}")
    print(f" PR_HPC: {out.get('PR_HPC', -1):.3f}")
    print(f" PR_HPT: {out.get('PR_HPT', -1):.3f}")
    print(f" PR_LPT: {out.get('PR_LPT', -1):.3f}")
    print(f" Td_N2 : {out.get('TorqueDiff_N2', 0.0):.3f}")
    print(f" Td_N1 : {out.get('TorqueDiff_N1', 0.0):.3f}")


# -------------------------
# A) Table 4.6-like trends (steady-state, fixed RPM)
# Use T4_cmd mode to allow fuel to respond via energy balance (fault sensitivity)
# -------------------------
run_case(
    "Baseline (T4_cmd, fixed RPM)",
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
    N1_RPM=N1_RPM_base,
    N2_RPM=N2_RPM_base,
)

run_case(
    "Fan eff -3% (T4_cmd, fixed RPM)",
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
    N1_RPM=N1_RPM_base,
    N2_RPM=N2_RPM_base,
    eff_mod_fan=0.97,
)

run_case(
    "HPC eff -5% (T4_cmd, fixed RPM)",
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
    N1_RPM=N1_RPM_base,
    N2_RPM=N2_RPM_base,
    eff_mod_hpc=0.95,
)

# -------------------------
# B) N1 drag +2 Nm proxy (steady-state constraint)
# We cannot inject shaft drag directly yet (runner has no N1_extra_drag),
# so we emulate the consequence by reducing N1_RPM input slightly.
# -------------------------
run_case(
    "N1 drag proxy: N1_RPM -5% (T4_cmd, fixed N2)",
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
    N1_RPM=0.95 * N1_RPM_base,
    N2_RPM=N2_RPM_base,
)

# -------------------------
# C) Fuel -70% case (fuel_cmd mode)
# In fuel_cmd mode, fuel scales with throttle via mf_max (MATLAB-faithful fuel command).
# 70% reduction => scale to 30% => throttle 0.60 -> 0.18
# -------------------------

run_case(
    "Baseline (fuel_cmd, fixed RPM)",
    combustor_mode="fuel_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
    N1_RPM=N1_RPM_base,
    N2_RPM=N2_RPM_base,
)

run_case(
    "Fuel -70% (fuel_cmd, fixed RPM)",
    combustor_mode="fuel_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.18,
    N1_RPM=N1_RPM_base,
    N2_RPM=N2_RPM_base,
)