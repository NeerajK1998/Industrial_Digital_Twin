import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from src.pdm_loader import load_pdm
from src.health_logic import decide_status, check_health_warnings
from src.logger import append_run_summary
from src.engine_core import simulate_cnc, CNCMission
from src.features import compute_features
from src.condition_monitor import evaluate_condition

st.set_page_config(page_title="Industrial Digital Twin", layout="wide")

DATA_DIR = Path("data")
CONFIG_DIR = Path("configs")
OUTPUTS_DIR = Path("outputs")
RUNSUMMARY = OUTPUTS_DIR / "RunSummary.csv"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def load_runsummary() -> pd.DataFrame:
    if RUNSUMMARY.exists():
        return pd.read_csv(RUNSUMMARY)
    return pd.DataFrame()


# ---------- Sidebar ----------
st.sidebar.header("Configuration")

config_files = sorted(CONFIG_DIR.glob("*.json"))
if not config_files:
    st.sidebar.error("No config files found in ./configs")
    st.stop()

config_choice = st.sidebar.selectbox(
    "Device config",
    options=[p.name for p in config_files],
    index=0
)
config_path = CONFIG_DIR / config_choice
cfg = load_json(config_path)

st.sidebar.subheader("PDM / Lifecycle JSON")
pdm_files = sorted(DATA_DIR.glob("PDM_Data*.json"))
if not pdm_files:
    st.sidebar.error("No PDM JSON files found in ./data")
    st.stop()

pdm_choice = st.sidebar.selectbox(
    "Select PDM JSON",
    options=[p.name for p in pdm_files],
    index=0
)
pdm_path = DATA_DIR / pdm_choice

use_upgrade = st.sidebar.checkbox("Use Upgrade file if present", value=False)
upgrade_file = "PDM_Data_Upgrade.json" if use_upgrade else "___none___.json"

st.sidebar.divider()
st.sidebar.subheader("Quick edit mission")
mission = cfg.get("mission", {})
mission["spindle_rpm_cmd"] = st.sidebar.number_input(
    "Spindle RPM cmd",
    value=float(mission.get("spindle_rpm_cmd", 12000.0)),
    step=500.0
)
mission["feed_cmd"] = st.sidebar.number_input(
    "Feed cmd (mm/min)",
    value=float(mission.get("feed_cmd", 2000.0)),
    step=100.0
)
mission["severity"] = st.sidebar.slider(
    "Severity",
    min_value=0.1,
    max_value=2.0,
    value=float(mission.get("severity", 1.0)),
    step=0.1
)
cfg["mission"] = mission

if st.sidebar.button("Save config changes"):
    save_json(config_path, cfg)
    st.sidebar.success("Saved config.")


# ---------- Main ----------
st.title("Industrial Digital Twin — Streamlit UI (v0)")


# ---------- Live Monitoring (top-level; safe indentation) ----------
st.subheader("Live Monitoring (v0)")
# --- ML Status (baseline) ---
st.markdown("### ML Status (baseline v0)")
try:
    from ml.streamlit_ml import predict_from_live_csv
    model_path = "outputs/ml/baseline_rf.joblib"
    live_csv = "outputs/live_timeseries.csv"
    pred, probs, fi = predict_from_live_csv(model_path, live_csv)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted State", pred)
    with c2:
        st.write("Probabilities")
        st.json(probs)

    if fi:
        st.write("Top feature importances")
        st.table([{"feature": k, "importance": float(v)} for k, v in fi])

except Exception as e:
    st.warning(f"ML not available yet: {e}")
st.divider()


    


live_enable = st.checkbox("Enable Live Mode", value=False)
live_seconds = st.number_input("Live duration (s)", min_value=1, max_value=120, value=10, step=1)
live_dt = st.number_input("Sample dt (s)", min_value=0.1, max_value=2.0, value=0.2, step=0.1)

if live_enable:
    from src.datasources import MockCNCSource
    from src.timeseries_logger import append_timeseries_row

    out_ts = OUTPUTS_DIR / "live_timeseries.csv"

    rpm_cmd = float(cfg.get("mission", {}).get("spindle_rpm_cmd", 12000.0))
    feed_cmd = float(cfg.get("mission", {}).get("feed_cmd", 2000.0))
    severity = float(cfg.get("mission", {}).get("severity", 1.0))

    src = MockCNCSource(
        runtime_s=float(live_seconds),
        dt=float(live_dt),
        rpm_cmd=rpm_cmd,
        feed_cmd=feed_cmd,
        severity=severity,
    )

    st.info("Streaming... (writes outputs/live_timeseries.csv)")
    chart_placeholder = st.empty()
    kpi_placeholder = st.empty()

    t_list, rpm_list, feed_list, vib_list, pwr_list = [], [], [], [], []

    for s in src.stream():
        row = {"t": s.t, **s.signals}
        append_timeseries_row(out_ts, row)

        t_list.append(s.t)
        rpm_list.append(s.signals["spindle_rpm"])
        feed_list.append(s.signals["feed_mm_min"])
        vib_list.append(s.signals["vibration"])
        pwr_list.append(s.signals["power_kw"])

        with kpi_placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Spindle RPM", f"{rpm_list[-1]:.0f}")
            c2.metric("Feed (mm/min)", f"{feed_list[-1]:.0f}")
            c3.metric("Vibration (a.u.)", f"{vib_list[-1]:.3f}")
            c4.metric("Power (kW)", f"{pwr_list[-1]:.2f}")

        df_live = pd.DataFrame({
            "t": t_list,
            "spindle_rpm": rpm_list,
            "feed_mm_min": feed_list,
            "vibration": vib_list,
            "power_kw": pwr_list,
        }).set_index("t")

        chart_placeholder.line_chart(df_live)

        # --- Feature extraction + condition evaluation (live) ---
        if len(vib_list) >= 10:  # wait for enough samples
            feats = compute_features({
                "spindle_rpm": rpm_list,
                "feed_mm_min": feed_list,
                "vibration": vib_list,
                "power_kw": pwr_list,
            })
            cond = evaluate_condition(feats)

            with st.container():
                st.markdown("**Live Condition Monitor**")
                cA, cB, cC, cD = st.columns(4)
                cA.metric("Cond Status", cond["status"])
                cB.metric("Vib RMS", f"{feats.get('vibration_rms', 0):.3f}")
                cC.metric("Vib Peak", f"{feats.get('vibration_peak', 0):.3f}")
                cD.metric("Power Std", f"{feats.get('power_std', 0):.3f}")
                st.write("Alerts:", cond["alerts"])


    st.success("Live stream finished.")


# ---------- Run Control + Latest Results ----------
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Run Control")
    st.write(f"**Device:** {cfg.get('device_name','(unnamed)')}  |  **Type:** {cfg.get('device_type','(unknown)')}")
    run_btn = st.button("▶ Run Framework", type="primary")
    st.caption("This v0 uses the CNC twin model. Later we’ll swap models without changing UI structure.")

with col2:
    st.subheader("Latest Results")
    df = load_runsummary()
    if not df.empty:
        last = df.iloc[-1].to_dict()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("N1 (Spindle RPM)", f"{float(last['N1']):.2f}")
        k2.metric("N2 (Feed)", f"{float(last['N2']):.2f}")
        k3.metric("Thrust (Force)", f"{float(last['Thrust']):.2f}")
        k4.metric("Fuel (Power)", f"{float(last['FuelFlow']):.2f}")
        k5.metric("Status", str(last["Status"]))
        st.write("Warnings:", last.get("HealthWarnings", ""))
    else:
        st.info("No runs yet. Click **Run Framework**.")


# ---------- Run pipeline ----------
if run_btn:
    pdm = load_json(pdm_path)

    ws = load_pdm(
        base_file=pdm_choice,
        upgrade_file=upgrade_file,
        data_dir=str(DATA_DIR),
    )
    ws["DoUpgrade"] = use_upgrade

    m = cfg.get("mission", {})
    mission_obj = CNCMission(
        t_end=float(m.get("t_end", 60.0)),
        dt=float(m.get("dt", 0.1)),
        spindle_rpm_cmd=float(m.get("spindle_rpm_cmd", 12000.0)),
        feed_cmd=float(m.get("feed_cmd", 2000.0)),
        severity=float(m.get("severity", 1.0)),
    )

    outputs = simulate_cnc(ws, mission_obj)

    limits_cfg = cfg.get("limits", {})
    limits = {k: (float(v[0]), float(v[1])) for k, v in limits_cfg.items()}

    status = decide_status(outputs, limits=limits)
    warnings = check_health_warnings(pdm)

    append_run_summary(RUNSUMMARY, outputs, pdm, status, warnings)

    st.success("Run complete and logged.")
    st.rerun()


# ---------- Run Log ----------

st.subheader("Post-run Analysis (from saved time-series)")
ts_path = OUTPUTS_DIR / "live_timeseries.csv"
if ts_path.exists():
    st.caption(f"Using: {ts_path}")
    if st.button("Analyze latest live_timeseries.csv"):
        df_ts = pd.read_csv(ts_path)
        feats = compute_features({
            "spindle_rpm": df_ts["spindle_rpm"].tolist() if "spindle_rpm" in df_ts else [],
            "feed_mm_min": df_ts["feed_mm_min"].tolist() if "feed_mm_min" in df_ts else [],
            "vibration": df_ts["vibration"].tolist() if "vibration" in df_ts else [],
            "power_kw": df_ts["power_kw"].tolist() if "power_kw" in df_ts else [],
        })
        cond = evaluate_condition(feats)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cond Status", cond["status"])
        c2.metric("Vib RMS", f"{feats.get('vibration_rms', 0):.3f}")
        c3.metric("Vib Peak", f"{feats.get('vibration_peak', 0):.3f}")
        c4.metric("Power Mean", f"{feats.get('power_mean', 0):.3f}")
        st.write("Alerts:", cond["alerts"])
        st.json(feats)
else:
    st.info("No outputs/live_timeseries.csv yet. Enable Live Mode once to generate it.")


st.subheader("Run Log")
df = load_runsummary()
if df.empty:
    st.write("No rows yet.")
else:
    st.dataframe(df.tail(50), use_container_width=True)


# ---------- PDM Editor ----------
st.subheader("PDM Editor (add/edit components without code)")
pdm_edit = load_json(pdm_path)

with st.expander("Open PDM Editor", expanded=False):
    comp_names = sorted(list(pdm_edit.keys()))
    colA, colB = st.columns([0.65, 0.35])

    with colA:
        mode = st.radio("Mode", ["Edit existing", "Create new"], horizontal=True)

        if mode == "Edit existing":
            comp = st.selectbox("Select component", options=comp_names, index=0)
        else:
            comp = st.text_input("New component name (e.g., Spindle, X_Axis, CoolantPump)", value="Spindle").strip()

        if not comp:
            st.warning("Enter a component name.")
        else:
            comp_obj = pdm_edit.get(comp, {})
            st.caption("Edit fields below. Use numeric values for numbers.")

            v = st.text_input("Version", value=str(comp_obj.get("Version", "")))
            cycles = st.number_input("CyclesSinceInstall", value=float(comp_obj.get("CyclesSinceInstall", 0)), step=1.0)
            eff = st.number_input("Efficiency_Modifier", value=float(comp_obj.get("Efficiency_Modifier", 1.0)), step=0.01, format="%.4f")
            maxc = st.number_input("MaxCycles (optional)", value=float(comp_obj.get("MaxCycles", 0)) if comp_obj.get("MaxCycles") is not None else 0.0, step=1.0)

            st.divider()
            st.caption("Extra parameters (key-value). Example: temperature_limit=80")
            extras = comp_obj.get("Extra", {})
            if not isinstance(extras, dict):
                extras = {}

            extra_items = list(extras.items())
            show_n = st.number_input("How many extra fields to show", min_value=0, max_value=20, value=min(5, len(extra_items)), step=1)

            # show existing extras
            for i in range(show_n):
                k, old_val = extra_items[i]
                new_val = st.text_input(f"Extra.{k}", value=str(old_val))
                extras[k] = new_val

            new_key = st.text_input("Add new extra key", value="")
            new_val = st.text_input("Add new extra value", value="")
            if st.button("Add extra field"):
                if new_key.strip():
                    extras[new_key.strip()] = new_val
                    st.success("Extra field added (click Save to persist).")
                else:
                    st.warning("Extra key is empty.")

            if st.button("Save PDM changes", type="primary"):
                pdm_edit.setdefault(comp, {})
                pdm_edit[comp]["Version"] = v
                pdm_edit[comp]["CyclesSinceInstall"] = int(cycles)
                pdm_edit[comp]["Efficiency_Modifier"] = float(eff)
                pdm_edit[comp]["MaxCycles"] = None if int(maxc) == 0 else int(maxc)
                pdm_edit[comp]["Extra"] = extras

                save_json(pdm_path, pdm_edit)
                st.success(f"Saved {comp} into {pdm_path.name}")
                st.rerun()

    with colB:
        st.write("**Current PDM file:**")
        st.code(str(pdm_path))
        st.write("**Tip:** Keep turbofan keys too if you still use them in ws mapping.")


# ---------- Component Health Table ----------
st.subheader("Component Health (from PDM JSON)")
pdm_current = load_json(pdm_path)
rows = []
for comp in sorted(pdm_current.keys()):
    part = pdm_current.get(comp, {})
    rows.append({
        "Component": comp,
        "Version": part.get("Version", ""),
        "Cycles": part.get("CyclesSinceInstall", ""),
        "Eff_Mod": part.get("Efficiency_Modifier", ""),
        "MaxCycles": part.get("MaxCycles", ""),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)
