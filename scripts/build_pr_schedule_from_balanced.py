import json
import numpy as np
from pathlib import Path

from src.turbofan.turbofan_runner import run_turbofan_core_balanced


def main():
    # --- SLS environment ---
    mode = "sls"
    P0 = 101325.0
    T0 = 288.15
    V0 = 0.0

    N1_ref = 12000.0
    N2_ref = 9000.0

    # Fixed RPM reference for schedule extraction (keep faithful to report %)
    N1_RPM = 0.768 * N1_ref
    N2_RPM = 0.931 * N2_ref

    throttle_points = np.linspace(0.15, 0.90, 10)

    schedule = {
        "meta": {
            "mode": mode,
            "P0": float(P0),
            "T0": float(T0),
            "V0": float(V0),
            "N1_ref_rpm": float(N1_ref),
            "N2_ref_rpm": float(N2_ref),
            "N1_pct_fixed": 76.8,
            "N2_pct_fixed": 93.1,
            "BPR": 9.0,
            "combustor_mode": "T4_cmd",
            "nozzle_mode": "report_simple",
            "A_core_nozzle": 0.017,
            "A_bypass_nozzle": 0.061,
            "eta_hpt": 0.9,
            "eta_lpt": 0.9,
        },
        "throttle": [],
        "PR_HPT": [],
        "PR_LPT": [],
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

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "src" / "turbofan" / "pr_schedule_sls.json"

    out_path.write_text(json.dumps(schedule, indent=4))
    print(f"\nPR schedule saved to {out_path}")


if __name__ == "__main__":
    main()