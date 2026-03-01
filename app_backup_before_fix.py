import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.asset_registry import (
    ensure_default_asset_exists,
    list_assets,
    make_asset_id,
    upsert_asset,
    delete_asset,
    AssetRecord,
)

# --- Turbofan ML ---
from ml.predict_live import load_model, predict_from_out
from src.turbofan.turbofan_runner import run_turbofan_core_balanced

# --- Framework / CNC ---
from src.pdm_loader import load_pdm
from src.health_logic import decide_status, check_health_warnings
from src.logger import append_run_summary
from src.engine_core import simulate_cnc, CNCMission
from src.features import compute_features
from src.condition_monitor import evaluate_condition
from src.pdm_registry import load_or_create_pdm, save_pdm, pdm_path_for_asset


# =========================
# CACHED MODEL LOADER
# =========================
@st.cache_resource
def get_turbofan_model_bundle():
    return load_model("artifacts/turbofan_model_v1.joblib")


# =========================
# PATHS / GLOBALS
# =========================
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


# =========================
# UI RENDERERS (so you don't fight indentation)
# =========================
def render_turbofan_ui():
    st.subheader("Turbofan ML (physics → features → model)")

    with st.expander("Run Turbofan ML test", expanded=True):
        c1, c2, c3 = st.columns(3)
        throttle = c1.slider("Throttle", 0.15, 0.95, 0.65, 0.01)
        bpr = c2.slider("BPR", 2.0, 12.0, 8.0, 0.5)
        mode = c3.selectbox("Health Scenario", ["Healthy", "WARNING", "FAULT"], index=0)

        # Presets (match your dataset logic)
        if mode == "Healthy":
            eff_mod_fan, eff_mod_lpc, eff_mod_hpc = 1.0, 1.0, 1.0
            eta_hpt, eta_lpt = 0.92, 0.92
        elif mode == "WARNING":
            eff_mod_fan, eff_mod_lpc, eff_mod_hpc = 0.9, 1.0, 1.0
            eta_hpt, eta_lpt = 0.92, 0.92
        else:  # FAULT
            eff_mod_fan, eff_mod_lpc, eff_mod_hpc = 0.7, 1.0, 1.0
            eta_hpt, eta_lpt = 0.82, 0.82

        if st.button("Run Turbofan + Predict", type="primary"):
            try:
                bundle = get_turbofan_model_bundle()

                out = run_turbofan_core_balanced(
                    throttle_cmd=float(throttle),
                    eff_mod_fan=float(eff_mod_fan),
                    eff_mod_lpc=float(eff_mod_lpc),
                    eff_mod_hpc=float(eff_mod_hpc),
                    eta_hpt=float(eta_hpt),
                    eta_lpt=float(eta_lpt),
                    BPR=float(bpr),
                )

                pred, probs = predict_from_out(out, bundle)

                st.write("**Prediction:**")
                if pred == "OK":
                    st.success("OK")
                elif pred == "WARNING":
                    st.warning("WARNING")
                else:
                    st.error("FAULT")

                st.caption("Probabilities")
                st.json(probs)

                st.caption("Key signals")
                st.write(
                    {
                        "N1_RPM": out.get("N1_RPM"),
                        "N2_RPM": out.get("N2_RPM"),
                        "Thrust": out.get("Thrust"),
                        "m_fuel": out.get("m_fuel"),
                        "T4": out.get("T4"),
                    }
                )

            except Exception as e:
                st.error(f"Turbofan ML failed: {e}")


def render_cnc_ui(cfg: dict, pdm_path: Path, pdm_choice: str, upgrade_file: str, use_upgrade: bool):
    # ---------- Live Monitoring ----------
    st.subheader("Live Monitoring (v0)")

    # --- ML Status (baseline v0) ---
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

    # ---------- Live mode streaming ----------
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

            df_live = pd.DataFrame(
                {
                    "t": t_list,
                    "spindle_rpm": rpm_list,
                    "feed_mm_min": feed_list,
                    "vibration": vib_list,
                    "power_kw": pwr_list,
                }
            ).set_index("t")

            chart_placeholder.line_chart(df_live)

            # --- Feature extraction + condition evaluation (live) ---
            if len(vib_list) >= 10:
                feats = compute_features(
                    {
                        "spindle_rpm": rpm_list,
                        "feed_mm_min": feed_list,
                        "vibration": vib_list,
                        "power_kw": pwr_list,
                    }
                )
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
    if "run_btn" in locals() and run_btn:
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

    # ---------- Post-run analysis ----------
    st.subheader("Post-run Analysis (from saved time-series)")
    ts_path = OUTPUTS_DIR / "live_timeseries.csv"
    if ts_path.exists():
        st.caption(f"Using: {ts_path}")
        if st.button("Analyze latest live_timeseries.csv"):
            df_ts = pd.read_csv(ts_path)
            feats = compute_features(
                {
                    "spindle_rpm": df_ts["spindle_rpm"].tolist() if "spindle_rpm" in df_ts else [],
                    "feed_mm_min": df_ts["feed_mm_min"].tolist() if "feed_mm_min" in df_ts else [],
                    "vibration": df_ts["vibration"].tolist() if "vibration" in df_ts else [],
                    "power_kw": df_ts["power_kw"].tolist() if "power_kw" in df_ts else [],
                }
            )
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

    # ---------- Run Log ----------
    st.subheader("Run Log")
    df = load_runsummary()
    if df.empty:
        st.write("No rows yet.")
    else:
        st.dataframe(df.tail(50), use_container_width=True)

    # ---------- PDM Editor ----------
    st.subheader("Component Builder (per asset)")

pdm_edit = load_or_create_pdm(asset_rec.asset_id)

with st.expander("Add / Edit Components", expanded=True):
    comp_names = sorted(list(pdm_edit.keys()))
    mode = st.radio("Mode", ["Add new", "Edit existing"], horizontal=True)

    if mode == "Edit existing" and comp_names:
        comp = st.selectbox("Select component", options=comp_names)
    elif mode == "Edit existing" and not comp_names:
        st.info("No components yet. Switch to 'Add new'.")
        comp = ""
    else:
        comp = st.text_input("New component name", value="Spindle").strip()

    if comp:
        comp_obj = pdm_edit.get(comp, {})

        c1, c2 = st.columns(2)
        with c1:
            version = st.text_input("Version", value=str(comp_obj.get("Version", "")))
            cycles = st.number_input("CyclesSinceInstall", value=float(comp_obj.get("CyclesSinceInstall", 0)), step=1.0)
            eff = st.number_input("Efficiency_Modifier", value=float(comp_obj.get("Efficiency_Modifier", 1.0)),
                                 step=0.01, format="%.4f")
        with c2:
            maxc = st.number_input("MaxCycles (0 = None)", value=float(comp_obj.get("MaxCycles", 0) or 0), step=1.0)
            notes = st.text_area("Notes", value=str(comp_obj.get("Notes", "")))

        st.caption("Optional limits (for rule-based warnings). Example: vibration_rms 0.0–0.35")
        limits = comp_obj.get("Limits", {})
        if not isinstance(limits, dict):
            limits = {}

        lim_key = st.text_input("Limit signal name", value="")
        lim_lo = st.number_input("Limit low", value=0.0, step=0.01)
        lim_hi = st.number_input("Limit high", value=1.0, step=0.01)

        if st.button("Add/Update limit"):
            if lim_key.strip():
                limits[lim_key.strip()] = [float(lim_lo), float(lim_hi)]
                st.success("Limit updated (click Save Component to persist).")
            else:
                st.warning("Limit signal name is empty.")

        if limits:
            st.write("Current limits:")
            st.json(limits)

        cA, cB, cC = st.columns([1, 1, 2])

        with cA:
            if st.button("Save Component", type="primary"):
                pdm_edit.setdefault(comp, {})
                pdm_edit[comp]["Version"] = version
                pdm_edit[comp]["CyclesSinceInstall"] = int(cycles)
                pdm_edit[comp]["Efficiency_Modifier"] = float(eff)
                pdm_edit[comp]["MaxCycles"] = None if int(maxc) == 0 else int(maxc)
                pdm_edit[comp]["Notes"] = notes
                pdm_edit[comp]["Limits"] = limits

                save_pdm(asset_rec.asset_id, pdm_edit)
                st.success("Saved.")
                st.rerun()

        with cB:
            if st.button("Delete Component"):
                if comp in pdm_edit:
                    pdm_edit.pop(comp, None)
                    save_pdm(asset_rec.asset_id, pdm_edit)
                    st.success("Deleted.")
                    st.rerun()

        with cC:
            st.caption(f"PDM file: `{asset_rec.pdm_json}`")

st.subheader("Component Table")
pdm_current = load_or_create_pdm(asset_rec.asset_id)

rows = []
for c in sorted(pdm_current.keys()):
    o = pdm_current.get(c, {})
    rows.append({
        "Component": c,
        "Version": o.get("Version", ""),
        "Cycles": o.get("CyclesSinceInstall", ""),
        "Eff_Mod": o.get("Efficiency_Modifier", ""),
        "MaxCycles": o.get("MaxCycles", ""),
    })
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No components yet. Add your first component above.")

    # ---------- Component Health Table ----------
    st.subheader("Component Health (from PDM JSON)")
    pdm_current = load_json(pdm_path)
    rows = []
    for comp in sorted(pdm_current.keys()):
        part = pdm_current.get(comp, {})
        rows.append(
            {
                "Component": comp,
                "Version": part.get("Version", ""),
                "Cycles": part.get("CyclesSinceInstall", ""),
                "Eff_Mod": part.get("Efficiency_Modifier", ""),
                "MaxCycles": part.get("MaxCycles", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------- Sidebar ----------
st.sidebar.header("Configuration")

# --- Asset registry selection ---
from src.asset_registry import (
    ensure_default_asset_exists,
    list_assets,
    make_asset_id,
    upsert_asset,
    delete_asset,
    AssetRecord,
)

ensure_default_asset_exists()
assets = list_assets()

if not assets:
    st.sidebar.error("No assets in registry. (data/assets.json)")
    st.stop()

asset_labels = [f"{a.name} ({a.asset_type})" for a in assets]
asset_map = {f"{a.name} ({a.asset_type})": a for a in assets}

selected_label = st.sidebar.selectbox("Select Asset", asset_labels, index=0)
asset_rec = asset_map[selected_label]
asset_type = asset_rec.asset_type  # <- this is the ONLY truth now

st.sidebar.caption(f"Asset ID: `{asset_rec.asset_id}`")
st.sidebar.divider()

# --- Create / Update Asset ---
with st.sidebar.expander("➕ Create / Update Asset", expanded=False):
    new_name = st.text_input("Asset name", value=asset_rec.name)
    new_type = st.selectbox(
        "Asset type",
        ["CNC", "Turbofan"],
        index=0 if asset_rec.asset_type == "CNC" else 1
    )
    new_notes = st.text_area("Notes (optional)", value=asset_rec.notes)

    if st.button("Save / Update Asset"):
        new_id = make_asset_id(new_name)
        new_rec = AssetRecord(
            asset_id=new_id,
            name=new_name.strip(),
            asset_type=new_type,
            device_config=asset_rec.device_config,
            pdm_json=asset_rec.pdm_json,
            notes=new_notes.strip(),
        )
        upsert_asset(new_rec)
        st.sidebar.success("Saved. Reloading…")
        st.rerun()

with st.sidebar.expander("🗑 Delete Asset", expanded=False):
    if st.button("Delete selected asset"):
        ok = delete_asset(asset_rec.asset_id)
        if ok:
            st.sidebar.success("Deleted. Reloading…")
            st.rerun()
        else:
            st.sidebar.warning("Nothing deleted.")

st.sidebar.divider()

# --- Device config picker (always) ---
config_files = sorted(CONFIG_DIR.glob("*.json"))
if not config_files:
    st.sidebar.error("No config files found in ./configs")
    st.stop()

config_names = [p.name for p in config_files]
default_cfg_name = Path(asset_rec.device_config).name if asset_rec.device_config else config_names[0]
cfg_index = config_names.index(default_cfg_name) if default_cfg_name in config_names else 0

config_choice = st.sidebar.selectbox("Device config", options=config_names, index=cfg_index)
config_path = CONFIG_DIR / config_choice
cfg = load_json(config_path)

# Persist into registry ONLY if changed
if asset_rec.device_config != str(config_path):
    asset_rec.device_config = str(config_path)
    upsert_asset(asset_rec)

# --- PDM picker (CNC only for now) ---
pdm_path = None
use_upgrade = False
upgrade_file = "___none___.json"

if asset_type == "CNC":
    st.sidebar.subheader("PDM / Lifecycle JSON")

    pdm_files = sorted(DATA_DIR.glob("PDM_Data*.json"))
    if not pdm_files:
        st.sidebar.error("No PDM JSON files found in ./data")
        st.stop()

    pdm_names = [p.name for p in pdm_files]
    default_pdm_name = Path(asset_rec.pdm_json).name if asset_rec.pdm_json else pdm_names[0]
    pdm_index = pdm_names.index(default_pdm_name) if default_pdm_name in pdm_names else 0

    pdm_choice = st.sidebar.selectbox("Select PDM JSON", options=pdm_names, index=pdm_index)
    pdm_path = DATA_DIR / pdm_choice

    if asset_rec.pdm_json != str(pdm_path):
        asset_rec.pdm_json = str(pdm_path)
        upsert_asset(asset_rec)

    use_upgrade = st.sidebar.checkbox("Use Upgrade file if present", value=False)
    upgrade_file = "PDM_Data_Upgrade.json" if use_upgrade else "___none___.json"

    # Mission controls (CNC only)
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


# =========================
# MAIN
# =========================
st.title("Industrial Digital Twin — Streamlit UI (v0)")

if asset == "Turbofan":
    render_turbofan_ui()
else:
    render_cnc_ui(cfg, pdm_path, pdm_choice, upgrade_file, use_upgrade)