from src.turbofan.turbofan_runner import run_turbofan_core_balanced
from src.turbofan.steady_solver import solve_n1_n2_scheduled_pr

# Cruise condition (Table 4.3)
P0 = 23841.93
T0 = 218.808
V0 = 231.30

N1_ref = 12000.0
N2_ref = 9000.0

# Phase 2A: fixed turbine PRs taken from Phase-1 balanced cruise point
PR_HPT_CONST = 0.8470916748046875
PR_LPT_CONST = 0.8982644081115723

N1_RPM_base = 0.768 * N1_ref
N2_RPM_base = 0.931 * N2_ref

# --- Derive PR constants for T4_cmd at the same cruise RPM point ---
_tmp = run_turbofan_core_balanced(
    throttle_cmd=0.60,
    P0=P0, T0=T0, V0=V0,
    N1_RPM=N1_RPM_base,
    N2_RPM=N2_RPM_base,
    N1_ref_rpm=N1_ref,
    N2_ref_rpm=N2_ref,
    BPR=9.0,
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    A_core_nozzle=0.017,
    A_bypass_nozzle=0.061,
    eta_hpt=0.9,
    eta_lpt=0.9,
)

PR_HPT_T4 = float(_tmp["PR_HPT"])
PR_LPT_T4 = float(_tmp["PR_LPT"])

print("\n[T4_cmd] PR constants extracted from balanced run:")
print(" PR_HPT_T4 =", PR_HPT_T4)
print(" PR_LPT_T4 =", PR_LPT_T4)


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
        A_core_nozzle=0.017,
        A_bypass_nozzle=0.061,
        eta_hpt=0.9,
        eta_lpt=0.9,
        **kwargs
    )

    print(f"\n{name}  [fixed RPM]")
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

    return out


def run_case_solved(
    name,
    *,
    combustor_mode,
    nozzle_mode,
    throttle_cmd,
    N1_guess_pct=76.8,
    N2_guess_pct=93.1,
    **kwargs
):
    res = solve_n1_n2_scheduled_pr(
        throttle_cmd=throttle_cmd,
        P0=P0,
        T0=T0,
        V0=V0,
        BPR=9.0,
        N1_ref_rpm=N1_ref,
        N2_ref_rpm=N2_ref,
        combustor_mode=combustor_mode,
        nozzle_mode=nozzle_mode,
        A_core_nozzle=0.017,
        A_bypass_nozzle=0.061,
        eta_hpt=0.9,
        eta_lpt=0.9,
        N1_guess_pct=N1_guess_pct,
        N2_guess_pct=N2_guess_pct,
        **kwargs
    )

    out = res.out

    print(f"\n{name}  [solved RPM]")
    print(f" Solver: {res.success} | {res.message}")
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

    return out


# -------------------------
# A) Table 4.6-like trends (solved RPM)
# -------------------------
base = run_case_solved(
    "Baseline (T4_cmd)",
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
)

run_case_solved(
    "Fan eff -3% (T4_cmd)",
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
    N1_guess_pct=base["N1_pct"],
    N2_guess_pct=base["N2_pct"],
    eff_mod_fan=0.97,
)

run_case_solved(
    "HPC eff -5% (T4_cmd)",
    combustor_mode="T4_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
    N1_guess_pct=base["N1_pct"],
    N2_guess_pct=base["N2_pct"],
    eff_mod_hpc=0.95,
)

# -------------------------
# B) N1 drag proxy (still fixed-RPM hack)
# Keep as fixed-RPM until you implement actual shaft drag torque.
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
# C) Fuel -70% (solved RPM)
# -------------------------
base_fuel = run_case_solved(
    "Baseline (fuel_cmd)",
    combustor_mode="fuel_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.60,
)

run_case_solved(
    "Fuel -70% (fuel_cmd)",
    combustor_mode="fuel_cmd",
    nozzle_mode="report_simple",
    throttle_cmd=0.18,
    N1_guess_pct=base_fuel["N1_pct"],
    N2_guess_pct=base_fuel["N2_pct"],
)