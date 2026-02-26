from src.turbofan.steady_solver import solve_n1_n2_given_pr

# Cruise atmosphere (ISA+0, 35k ft, Mach 0.78)
P0 = 23841.93
T0 = 218.808
V0 = 231.30

N1_ref = 12000.0
N2_ref = 9000.0

# Use PR constants extracted from Phase-1 balanced cruise point
PR_HPT_CONST = 0.8470916748046875
PR_LPT_CONST = 0.8982644081115723

res = solve_n1_n2_given_pr(
    throttle_cmd=0.60,
    P0=P0, T0=T0, V0=V0,
    BPR=9.0,
    N1_ref_rpm=N1_ref,
    N2_ref_rpm=N2_ref,
    PR_HPT=PR_HPT_CONST,
    PR_LPT=PR_LPT_CONST,
    combustor_mode="fuel_cmd",
    nozzle_mode="report_simple",
    A_core_nozzle=0.017,
    A_bypass_nozzle=0.061,
    eff_mod_fan=1.0,
    eff_mod_lpc=1.0,
    eff_mod_hpc=1.0,
    eta_hpt=0.9,
    eta_lpt=0.9,
    N1_guess_pct=76.8,
    N2_guess_pct=93.1,
)

out = res.out
print("\n=== Phase 2A Solved-RPM Cruise (fixed turbine PRs) ===")
print("Solver success:", res.success, "|", res.message)
print(f"Solved N1_RPM: {res.N1_RPM:.2f} ({out['N1_pct']:.2f}%)")
print(f"Solved N2_RPM: {res.N2_RPM:.2f} ({out['N2_pct']:.2f}%)")
print(f"PR_HPT fixed : {PR_HPT_CONST:.6f}")
print(f"PR_LPT fixed : {PR_LPT_CONST:.6f}")
print(f"Thrust_total : {out['thrust_total']:.2f} N")
print(f"m_fuel       : {out['m_fuel']:.6f} kg/s")
print(f"TorqueDiff_N1: {out['TorqueDiff_N1']:.6f}")
print(f"TorqueDiff_N2: {out['TorqueDiff_N2']:.6f}")
print("===============================================\n")

# quick fault poke to prove RPM shifts now happen
res_fault = solve_n1_n2_given_pr(
    throttle_cmd=0.60,
    P0=P0, T0=T0, V0=V0,
    BPR=9.0,
    N1_ref_rpm=N1_ref,
    N2_ref_rpm=N2_ref,
    PR_HPT=PR_HPT_CONST,
    PR_LPT=PR_LPT_CONST,
    combustor_mode="fuel_cmd",
    nozzle_mode="report_simple",
    A_core_nozzle=0.017,
    A_bypass_nozzle=0.061,
    eff_mod_fan=1.0,
    eff_mod_lpc=1.0,
    eff_mod_hpc=0.95,  # HPC efficiency -5%
    eta_hpt=0.9,
    eta_lpt=0.9,
    N1_guess_pct=out["N1_pct"],
    N2_guess_pct=out["N2_pct"],
)

out_f = res_fault.out
print("=== Phase 2A Fault Test (HPC eff -5%) ===")
print("Solver success:", res_fault.success, "|", res_fault.message)
print(f"Solved N1%: {out_f['N1_pct']:.2f}  (baseline {out['N1_pct']:.2f})")
print(f"Solved N2%: {out_f['N2_pct']:.2f}  (baseline {out['N2_pct']:.2f})")
print(f"Thrust_total: {out_f['thrust_total']:.2f} N")
print(f"m_fuel      : {out_f['m_fuel']:.6f} kg/s")
print(f"TorqueDiff_N1: {out_f['TorqueDiff_N1']:.6f}")
print(f"TorqueDiff_N2: {out_f['TorqueDiff_N2']:.6f}")
print("========================================\n")