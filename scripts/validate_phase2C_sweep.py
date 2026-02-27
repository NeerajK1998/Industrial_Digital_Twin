import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.turbofan.steady_solver import solve_n1_n2_scheduled_pr

# -----------------------
# Select operating condition
# -----------------------
MODE = "cruise"   # "cruise" or "sls"

if MODE == "cruise":
    P0, T0, V0 = 23842.0, 218.81, 230.0
    PR_SCHED = "src/turbofan/pr_schedule_cruise.json"
elif MODE == "sls":
    P0, T0, V0 = 101325.0, 288.15, 0.0
    PR_SCHED = "src/turbofan/pr_schedule_sls.json"
else:
    raise ValueError("Unknown MODE")

N1_ref = 12000.0
N2_ref = 9000.0
throttle_range = np.linspace(0.15, 0.75, 20)

print(f"[MODE={MODE}] P0={P0:.1f} Pa, T0={T0:.2f} K, V0={V0:.2f} m/s, PR_SCHED={PR_SCHED}")
print("\n=== Phase 2C Throttle Sweep (Solved RPM, T4_cmd) ===")

results = []
printed_diag = False  # print diagnostics on FIRST SUCCESSFUL point

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
        expected_mode=MODE,
    )

    out = res.out or {}
    success = bool(res.success)

    def safe_float(key: str) -> float:
        """Return float(out[key]) if success and convertible, else NaN."""
        if not success:
            return np.nan
        v = out.get(key, np.nan)
        try:
            return float(v)
        except Exception:
            return np.nan

    # --- one-time diagnostic print (first SUCCESS point) ---
    if success and not printed_diag:
        print("\n--- DIAGNOSTIC: OUT KEYS ---")
        print(sorted(out.keys()))

        print("\n--- DIAGNOSTIC: KEY VALUES (first SUCCESS point) ---")
        for k in [
            "throttle_cmd", "N1_pct", "N2_pct",
            "T4", "FAR",
            "m_air", "m_fuel", "m_gas",
            "Thrust_core", "Thrust_bypass", "thrust_total",
            "P0", "T0", "V0",
            "P5", "T5",
            "PR_HPT", "PR_LPT", "PR_HPT_sched", "PR_LPT_sched",
        ]:
            if k in out:
                print(f"{k:>12s} = {out[k]}")
        printed_diag = True

    row = {
        "mode": MODE,
        "throttle": float(thr),
        "success": success,

        "N1_pct": safe_float("N1_pct"),
        "N2_pct": safe_float("N2_pct"),

        "thrust_raw_N": safe_float("thrust_total"),
        "fuel_raw_kg_s": safe_float("m_fuel"),
        "FAR": safe_float("FAR"),
        "T4_K": safe_float("T4"),
        "m_air_kg_s": safe_float("m_air"),
        "m_gas_kg_s": safe_float("m_gas"),

        "PR_HPT": safe_float("PR_HPT"),
        "PR_LPT": safe_float("PR_LPT"),
    }
    results.append(row)

    # Print using safe_float so formatting never crashes
    n1p = row["N1_pct"]
    n2p = row["N2_pct"]
    thrN = row["thrust_raw_N"]
    fuel = row["fuel_raw_kg_s"]

    print(
        f"Thr {thr:.2f} | "
        f"N1 {n1p:6.1f}% | "
        f"N2 {n2p:6.1f}% | "
        f"Thrust {thrN:8.0f} | "
        f"Fuel {fuel:6.3f} | "
        f"Solver {success}"
    )

# -----------------------
# Save RAW CSV + TSFC (safe)
# -----------------------
df = pd.DataFrame(results)

# TSFC only when thrust is positive and finite
thrust = df["thrust_raw_N"].to_numpy(dtype=float)
fuel = df["fuel_raw_kg_s"].to_numpy(dtype=float)
df["tsfc_raw_mg_Ns"] = np.where(np.isfinite(thrust) & (thrust > 0), (fuel / thrust) * 1e6, np.nan)

os.makedirs("outputs", exist_ok=True)
raw_csv = f"outputs/phase2C_sweep_{MODE}_raw.csv"
df.to_csv(raw_csv, index=False)
print(f"\nSweep results saved to {raw_csv}")

print("\n--- CSV First Row (sanity) ---")
print(df.iloc[0][["mode","throttle","thrust_raw_N","fuel_raw_kg_s","tsfc_raw_mg_Ns","m_air_kg_s","T4_K","FAR"]])

# -----------------------
# Optional: CRUISE scaling (thrust + fuel only)
# -----------------------
df_scaled = None
if MODE == "cruise":
    df_ok = df[df["success"]].copy()

    if len(df_ok) == 0:
        print("\n[WARN] No successful points — skipping scaling.")
    else:
        mid_ok = df_ok.iloc[len(df_ok)//2]

        TARGET_THRUST_N = 21940.0
        scale_factor = TARGET_THRUST_N / float(mid_ok["thrust_raw_N"])

        df_scaled = df.copy()
        df_scaled["thrust_scaled_N"] = df_scaled["thrust_raw_N"] * scale_factor
        df_scaled["fuel_scaled_kg_s"] = df_scaled["fuel_raw_kg_s"] * scale_factor

        thr_s = df_scaled["thrust_scaled_N"].to_numpy(dtype=float)
        fuel_s = df_scaled["fuel_scaled_kg_s"].to_numpy(dtype=float)
        df_scaled["tsfc_scaled_mg_Ns"] = np.where(np.isfinite(thr_s) & (thr_s > 0), (fuel_s / thr_s) * 1e6, np.nan)

        print("\n--- Scaling Summary (cruise only) ---")
        print(f"scale_factor = {scale_factor:.4f} (computed from mid-throttle SUCCESS row)")
        print(f"mid row (raw)   : {mid_ok['thrust_raw_N']} N, {mid_ok['fuel_raw_kg_s']} kg/s, {mid_ok['tsfc_raw_mg_Ns']} mg/Ns")
        print(f"mid row (scaled): {TARGET_THRUST_N} N, {mid_ok['fuel_raw_kg_s']*scale_factor} kg/s, {mid_ok['tsfc_raw_mg_Ns']} mg/Ns")

        scaled_csv = f"outputs/phase2C_sweep_{MODE}_scaled.csv"
        df_scaled.to_csv(scaled_csv, index=False)
        print(f"Saved scaled CSV: {scaled_csv}")

# -----------------------
# Plot helper (OK-only + optional failed markers)
# -----------------------
os.makedirs("outputs/plots", exist_ok=True)

def save_plot_ok(df_all, ycol, xlabel, ylabel, title, filename, df_scaled=None):
    df_ok = df_all[df_all["success"]].copy()
    df_bad = df_all[~df_all["success"]].copy()

    plt.figure()
    plt.plot(df_ok["throttle"], df_ok[ycol])

    # mark failed points (optional, comment out if you dislike markers)
    if len(df_bad) > 0 and ycol in df_bad.columns:
        plt.scatter(df_bad["throttle"], df_bad[ycol])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(f"outputs/plots/{MODE}_{filename}.png", dpi=160, bbox_inches="tight")
    plt.show()

# RAW plots (OK-only)
save_plot_ok(df, "N1_pct", "Throttle", "N1 %", "N1 vs Throttle (raw)", "n1_vs_throttle_raw")
save_plot_ok(df, "N2_pct", "Throttle", "N2 %", "N2 vs Throttle (raw)", "n2_vs_throttle_raw")
save_plot_ok(df, "thrust_raw_N", "Throttle", "Thrust (N)", "Thrust vs Throttle (raw)", "thrust_vs_throttle_raw")
save_plot_ok(df, "fuel_raw_kg_s", "Throttle", "Fuel Flow (kg/s)", "Fuel vs Throttle (raw)", "fuel_vs_throttle_raw")
save_plot_ok(df, "tsfc_raw_mg_Ns", "Throttle", "TSFC (mg/N*s)", "TSFC vs Throttle (raw)", "tsfc_vs_throttle_raw")
save_plot_ok(df, "T4_K", "Throttle", "T4 (K)", "T4 vs Throttle (raw)", "t4_vs_throttle_raw")
save_plot_ok(df, "FAR", "Throttle", "FAR", "FAR vs Throttle (raw)", "far_vs_throttle_raw")

# SCALED plots (cruise only)
if MODE == "cruise" and df_scaled is not None:
    save_plot_ok(df_scaled, "thrust_scaled_N", "Throttle", "Thrust (N)", "Thrust vs Throttle (scaled)", "thrust_vs_throttle_scaled")
    save_plot_ok(df_scaled, "fuel_scaled_kg_s", "Throttle", "Fuel Flow (kg/s)", "Fuel vs Throttle (scaled)", "fuel_vs_throttle_scaled")
    save_plot_ok(df_scaled, "tsfc_scaled_mg_Ns", "Throttle", "TSFC (mg/N*s)", "TSFC vs Throttle (scaled)", "tsfc_vs_throttle_scaled")

# -----------------------
# Mid-throttle summary (SUCCESS-only if possible)
# -----------------------
df_ok = df[df["success"]]
mid = df_ok.iloc[len(df_ok)//2] if len(df_ok) else df.iloc[len(df)//2]

print("\n--- Mid Throttle Summary ---")
print(f"MODE    : {MODE}")
print(f"Throttle: {mid['throttle']:.2f}")
print(f"Thrust  : {mid['thrust_raw_N']:.2f} N (raw)")
print(f"Fuel    : {mid['fuel_raw_kg_s']:.4f} kg/s (raw)")
print(f"TSFC    : {mid['tsfc_raw_mg_Ns']:.2f} mg/(N*s) (raw)")

if MODE == "cruise" and df_scaled is not None:
    df_s_ok = df_scaled[df_scaled["success"]]
    mid_s = df_s_ok.iloc[len(df_s_ok)//2] if len(df_s_ok) else df_scaled.iloc[len(df_scaled)//2]
    print(f"Thrust  : {mid_s['thrust_scaled_N']:.2f} N (scaled)")
    print(f"Fuel    : {mid_s['fuel_scaled_kg_s']:.4f} kg/s (scaled)")
    print(f"TSFC    : {mid_s['tsfc_scaled_mg_Ns']:.2f} mg/(N*s) (scaled)")