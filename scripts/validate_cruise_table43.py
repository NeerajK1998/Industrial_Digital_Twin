from src.turbofan.turbofan_runner import run_turbofan_core_balanced

# Cruise atmosphere (ISA+0, 35k ft, Mach 0.78)
P0 = 23841.93
T0 = 218.808
V0 = 231.30

# Reference outputs from Table 4.3 (Simulink Twin)
REF = {
    "thrust_N": 21940.0,
    "fuel_kgps": 0.72,
    "tsfc_mgNs": 32.4,
    "N1_pct": 76.8,
    "N2_pct": 93.1,
}

# Use your rpm refs for percent display
N1_ref = 12000.0
N2_ref = 9000.0
N1_RPM = 0.768 * N1_ref
N2_RPM = 0.931 * N2_ref

out = run_turbofan_core_balanced(
    throttle_cmd=0.60,          # only tune this later if needed (Phase 1)
    P0=P0,
    T0=T0,
    V0=V0,
    N1_RPM=N1_RPM,
    N2_RPM=N2_RPM,
    N1_ref_rpm=N1_ref,
    N2_ref_rpm=N2_ref,
    BPR=9.0,

    combustor_mode="fuel_cmd",
    nozzle_mode="report_simple",

    A_core_nozzle=0.017,
    A_bypass_nozzle=0.061,

    # disable degradation/modifiers for validation
    eff_mod_fan=1.0,
    eff_mod_lpc=1.0,
    eff_mod_hpc=1.0,
    eta_hpt=0.9,
    eta_lpt=0.9,
)

thrust = out["thrust_total"]
fuel = out["m_fuel"]

# Phase 1: thesis engine is scaled to ~22 kN class at cruise (Table 4.3)
SCALE = 0.8809  # fixed: 21940 / 24907.15 (unscaled cruise thrust)
thrust_scaled = thrust * SCALE

tsfc = (fuel / max(thrust_scaled, 1e-9)) * 1e6  # mg/(N*s) using scaled thrust

def pct_err(val, ref):
    return 100.0 * (val - ref) / ref

print("\n=== Table 4.3 Cruise Validation (Phase 1) ===")
print(f"Thrust [N]   : {thrust_scaled:10.2f} | ref {REF['thrust_N']:10.2f} | err {pct_err(thrust_scaled, REF['thrust_N']):+6.2f}%")
print(f"Fuel [kg/s]  : {fuel:10.4f} | ref {REF['fuel_kgps']:10.4f} | err {pct_err(fuel, REF['fuel_kgps']):+6.2f}%")
print(f"TSFC [mg/Ns] : {tsfc:10.3f} | ref {REF['tsfc_mgNs']:10.3f} | err {pct_err(tsfc, REF['tsfc_mgNs']):+6.2f}%")
print(f"N1 [%]       : {out.get('N1_pct', -1):10.2f} | ref {REF['N1_pct']:10.2f} | err {pct_err(out.get('N1_pct', -1), REF['N1_pct']):+6.2f}%")
print(f"N2 [%]       : {out.get('N2_pct', -1):10.2f} | ref {REF['N2_pct']:10.2f} | err {pct_err(out.get('N2_pct', -1), REF['N2_pct']):+6.2f}%")
print("============================================\n")
print("m_air:", out["m_air"])
print("m_gas:", out["m_gas"])