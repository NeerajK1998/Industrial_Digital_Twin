import json
import numpy as np

from src.turbofan.turbofan_runner import run_turbofan_core_balanced

# Cruise environment (same as validation)
P0 = 23842.0
T0 = 218.81
V0 = 230.0

N1_ref = 12000.0
N2_ref = 9000.0

# Fixed RPM reference for schedule extraction
N1_RPM = 0.768 * N1_ref
N2_RPM = 0.931 * N2_ref

# Throttle sweep (adjust resolution if needed)
throttle_points = np.linspace(0.15, 0.90, 10)

schedule = {
    "throttle": [],
    "PR_HPT": [],
    "PR_LPT": []
}

for thr in throttle_points:
    out = run_turbofan_core_balanced(
        throttle_cmd=float(thr),
        P0=P0,
        T0=T0,
        V0=V0,
        N1_RPM=N1_RPM,
        N2_RPM=N2_RPM,
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

    schedule["throttle"].append(float(thr))
    schedule["PR_HPT"].append(float(out["PR_HPT"]))
    schedule["PR_LPT"].append(float(out["PR_LPT"]))

    print(f"Throttle {thr:.2f} -> PR_HPT {out['PR_HPT']:.4f}, PR_LPT {out['PR_LPT']:.4f}")

# Save schedule
with open("src/turbofan/pr_schedule_cruise.json", "w") as f:
    json.dump(schedule, f, indent=4)

print("\nPR schedule saved to src/turbofan/pr_schedule_sls.json")