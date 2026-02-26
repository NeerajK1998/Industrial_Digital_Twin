import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Select operating condition
# -----------------------

MODE = "cruise"   # change to "sls" when needed

if MODE == "cruise":
    PR_SCHED = "src/turbofan/pr_schedule_cruise.json"
elif MODE == "sls":
    PR_SCHED = "src/turbofan/pr_schedule_sls.json"
else:
    raise ValueError("Unknown MODE")

from src.turbofan.steady_solver import solve_n1_n2_scheduled_pr

# Cruise environment
if MODE == "cruise":
    P0 = 23842.0      # Pa  (ISA 35,000 ft approx)
    T0 = 218.81       # K   (ISA 35,000 ft approx)
    V0 = 230.0        # m/s (Mach 0.78 at that temp ≈ 230 m/s)
    PR_SCHED = "src/turbofan/pr_schedule_cruise.json"

elif MODE == "sls":
    P0 = 101325.0
    T0 = 288.15
    V0 = 0.0
    PR_SCHED = "src/turbofan/pr_schedule_sls.json"

N1_ref = 12000.0
N2_ref = 9000.0

throttle_range = np.linspace(0.15, 0.75, 20)

results = []

print("\n=== Phase 2C Throttle Sweep (Solved RPM, T4_cmd) ===")

for thr in throttle_range:
    res = solve_n1_n2_scheduled_pr(
        throttle_cmd=float(thr),
        P0=P0,
        T0=T0,
        V0=V0,
        BPR=9.0,
        N1_ref_rpm=N1_ref,
        N2_ref_rpm=N2_ref,
        combustor_mode="T4_cmd",
        nozzle_mode="report_simple",
        pr_schedule_path=PR_SCHED,
    )

    out = res.out

    results.append({
        "throttle": thr,
        "success": res.success,
        "N1_pct": out["N1_pct"],
        "N2_pct": out["N2_pct"],
        "thrust": out["thrust_total"],
        "fuel": out["m_fuel"],
        "PR_HPT": out["PR_HPT"],
        "PR_LPT": out["PR_LPT"],
    })

    print(
        f"Thr {thr:.2f} | "
        f"N1 {out['N1_pct']:.1f}% | "
        f"N2 {out['N2_pct']:.1f}% | "
        f"Thrust {out['thrust_total']:.0f} | "
        f"Fuel {out['m_fuel']:.3f} | "
        f"Solver {res.success}"
    )

df = pd.DataFrame(results)
df.to_csv("outputs/phase2C_sweep.csv", index=False)

print("\nSweep results saved to outputs/phase2C_sweep.csv")

# -----------------------
# Plot trends
# -----------------------

plt.figure()
plt.plot(df["throttle"], df["N1_pct"])
plt.xlabel("Throttle")
plt.ylabel("N1 %")
plt.title("N1 vs Throttle")
plt.grid()
plt.show()

plt.figure()
plt.plot(df["throttle"], df["N2_pct"])
plt.xlabel("Throttle")
plt.ylabel("N2 %")
plt.title("N2 vs Throttle")
plt.grid()
plt.show()

plt.figure()
plt.plot(df["throttle"], df["thrust"])
plt.xlabel("Throttle")
plt.ylabel("Thrust")
plt.title("Thrust vs Throttle")
plt.grid()
plt.show()

plt.figure()
plt.plot(df["throttle"], df["fuel"])
plt.xlabel("Throttle")
plt.ylabel("Fuel Flow")
plt.title("Fuel vs Throttle")
plt.grid()
plt.show()