from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------
# Asset Registry (portfolio UI)
# -----------------------------
from src.asset_registry import (
    ensure_default_asset_exists,
    list_assets,
    make_asset_id,
    upsert_asset,
    delete_asset,
    AssetRecord,
)

# -----------------------------
# Turbofan ML
# -----------------------------
# from ml.predict_live import load_model, predict_from_out
import joblib
from ml.predict_live import predict_from_out
from src.turbofan.turbofan_runner import run_turbofan_core_balanced

# -----------------------------
# CNC Framework
# -----------------------------
from src.pdm_loader import load_pdm
from src.health_logic import decide_status, check_health_warnings
from src.logger import append_run_summary
from src.engine_core import simulate_cnc, CNCMission
from src.features import compute_features
from src.condition_monitor import evaluate_condition

# -----------------------------
# Per-asset PDM storage
# -----------------------------
from src.pdm_registry import load_or_create_pdm, save_pdm

# =========================
# Streamlit config + paths
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
# Cached model loader
# =========================
#@st.cache_resource
#def get_turbofan_model_bundle():
#   return load_model("artifacts/turbofan_model_v1.joblib")
@st.cache_resource
def get_turbofan_model_bundle():
    return joblib.load("artifacts/turbofan_model_v1.joblib")


# =========================
# Sidebar: Asset selection
# =========================
st.sidebar.header("Configuration")

ensure_default_asset_exists()
assets = list_assets()
if not assets:
    st.sidebar.error("No assets in registry.")
    st.stop()

asset_labels = [f"{a.name} ({a.asset_type})" for a in assets]
asset_map = {f"{a.name} ({a.asset_type})": a for a in assets}

selected_label = st.sidebar.selectbox("Select Asset", asset_labels, index=0)
asset_rec = asset_map[selected_label]
asset_type = asset_rec.asset_type

st.sidebar.caption(f"Asset ID: `{asset_rec.asset_id}`")
st.sidebar.divider()

# --- Create / Update Asset ---
with st.sidebar.expander("➕ Create / Update Asset", expanded=False):
    new_name = st.text_input("Asset name", value=asset_rec.name)
    new_type = st.selectbox("Asset type", ["CNC", "Turbofan"], index=0 if asset_type == "CNC" else 1)
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

# Persist config choice into registry if changed
if asset_rec.device_config != str(config_path):
    asset_rec.device_config = str(config_path)
    upsert_asset(asset_rec)

# --- CNC-only sidebar controls ---
pdm_choice = None
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

    # Persist pdm choice into registry if changed
    if asset_rec.pdm_json != str(pdm_path):
        asset_rec.pdm_json = str(pdm_path)
        upsert_asset(asset_rec)

    use_upgrade = st.sidebar.checkbox("Use Upgrade file if present", value=False)
    upgrade_file = "PDM_Data_Upgrade.json" if use_upgrade else "___none___.json"

    # Mission controls
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
# UI: Turbofan
# =========================
def render_turbofan_ui():
    st.subheader("Turbofan ML (physics → features → model)")

    with st.expander("Run Turbofan ML test", expanded=True):
        c1, c2, c3 = st.columns(3)
        throttle = c1.slider("Throttle", 0.15, 0.95, 0.65, 0.01)
        bpr = c2.slider("BPR", 2.0, 12.0, 8.0, 0.5)
        mode = c3.selectbox("Health Scenario", ["Healthy", "WARNING", "FAULT"], index=0)

        if mode == "Healthy":
            eff_mod_fan, eff_mod_lpc, eff_mod_hpc = 1.0, 1.0, 1.0
            eta_hpt, eta_lpt = 0.92, 0.92
        elif mode == "WARNING":
            eff_mod_fan, eff_mod_lpc, eff_mod_hpc = 0.9, 1.0, 1.0
            eta_hpt, eta_lpt = 0.92, 0.92
        else:
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

                st.markdown("### Prediction")
                if pred == "OK":
                    st.success("OK")
                elif pred == "WARNING":
                    st.warning("WARNING")
                else:
                    st.error("FAULT")

                st.caption("Probabilities")
                st.json(probs)

                st.caption("Key signals")
                st.json({
                    "N1_RPM": out.get("N1_RPM"),
                    "N2_RPM": out.get("N2_RPM"),
                    "Thrust": out.get("Thrust"),
                    "m_fuel": out.get("m_fuel"),
                    "T4": out.get("T4"),
                })

            except Exception as e:
                st.error(f"Turbofan ML failed: {e}")


# =========================
# UI: CNC
# =========================
def render_cnc_ui():
    st.subheader("Live Monitoring (CNC)")

    # ---- Data source selection (M4) ----
    st.markdown("### Data Source")
    source_mode = st.selectbox("Choose live source", ["Mock stream", "Replay CSV"], index=0)

    if source_mode == "Replay CSV":
        default_csv = str(OUTPUTS_DIR / "live_timeseries.csv")
        replay_path = st.text_input("Replay CSV path", value=default_csv)
        dt = st.number_input("Replay dt (s)", min_value=0.05, max_value=2.0, value=0.2, step=0.05)
    else:
        dt = st.number_input("Mock dt (s)", min_value=0.05, max_value=2.0, value=0.2, step=0.05)

    st.divider()

    # ---- Load CNC model (M2) ----
    st.markdown("### CNC ML (trained model)")
    try:
        from ml.predict_live_cnc import load_model, predict_from_timeseries
        bundle = load_model("artifacts/cnc_model_v1.joblib")
        st.success("CNC model loaded: artifacts/cnc_model_v1.joblib")
    except Exception as e:
        st.warning(f"CNC model not available yet: {e}")
        st.info("Run: python -m ml.build_dataset_cnc  &&  python -m ml.train_cnc_baseline")
        bundle = None

    st.divider()

    # ---- Live streaming / replay and inference loop (M2+M3+M4) ----
    st.markdown("### Live Stream + Inference")
    live_enable = st.checkbox("Enable Live Mode", value=False)
    live_seconds = st.number_input("Duration (s)", min_value=1, max_value=120, value=10, step=1)
    window_n = st.number_input("Inference window samples", min_value=20, max_value=400, value=200, step=10)

    if live_enable:
        from src.timeseries_logger import append_timeseries_row
        from src.insights import severity_score, top_contributors_from_rf, recommended_action

        out_ts = OUTPUTS_DIR / "live_timeseries.csv"

        # Choose source
        if source_mode == "Replay CSV":
            from src.datasources import CSVReplaySource
            src = CSVReplaySource(csv_path=replay_path, dt=float(dt), loop=False)
        else:
            from src.datasources import MockCNCSource
            rpm_cmd = float(cfg.get("mission", {}).get("spindle_rpm_cmd", 12000.0))
            feed_cmd = float(cfg.get("mission", {}).get("feed_cmd", 2000.0))
            severity = float(cfg.get("mission", {}).get("severity", 1.0))
            src = MockCNCSource(
                runtime_s=float(live_seconds),
                dt=float(dt),
                rpm_cmd=rpm_cmd,
                feed_cmd=feed_cmd,
                severity=severity,
            )

        st.info("Streaming... writes outputs/live_timeseries.csv")
        chart_placeholder = st.empty()
        kpi_placeholder = st.empty()
        pred_placeholder = st.empty()
        why_placeholder = st.empty()

        t_list, rpm_list, feed_list, vib_list, pwr_list = [], [], [], [], []

        # Stream samples
        for s in src.stream():
            row = {"t": s.t, **s.signals}
            append_timeseries_row(out_ts, row)

            t_list.append(float(s.t))
            rpm_list.append(float(s.signals.get("spindle_rpm", 0.0)))
            feed_list.append(float(s.signals.get("feed_mm_min", 0.0)))
            vib_list.append(float(s.signals.get("vibration", 0.0)))
            pwr_list.append(float(s.signals.get("power_kw", 0.0)))

            # KPIs
            with kpi_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Spindle RPM", f"{rpm_list[-1]:.0f}")
                c2.metric("Feed (mm/min)", f"{feed_list[-1]:.0f}")
                c3.metric("Vibration (a.u.)", f"{vib_list[-1]:.3f}")
                c4.metric("Power (kW)", f"{pwr_list[-1]:.2f}")

            # chart
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

            # Inference once window is large enough
            if bundle is not None and len(vib_list) >= int(window_n):
                ts = {
                    "spindle_rpm": rpm_list[-int(window_n):],
                    "feed_mm_min": feed_list[-int(window_n):],
                    "vibration": vib_list[-int(window_n):],
                    "power_kw": pwr_list[-int(window_n):],
                }

                pred, probs, feat_row = predict_from_timeseries(ts, bundle)

                # Insights (M3)
                model = bundle["model"]
                top_feats = top_contributors_from_rf(model, feat_row, topk=3)
                sev = severity_score(pred, probs)
                action = recommended_action(pred, top_feats)

                with pred_placeholder.container():
                    st.markdown("### Predicted State")
                    if pred == "OK":
                        st.success(f"OK  (severity {sev}/100)")
                    elif pred == "WARNING":
                        st.warning(f"WARNING  (severity {sev}/100)")
                    else:
                        st.error(f"FAULT  (severity {sev}/100)")
                    st.caption("Probabilities")
                    st.json(probs)

                with why_placeholder.container():
                    st.markdown("### Why / Risk / Action")
                    st.write("**Top contributors:**")
                    st.table([{"feature": f, "score": float(v)} for f, v in top_feats])
                    st.write("**Recommended action:**")
                    st.info(action)

            # Stop condition for mock mode
            if source_mode == "Mock stream" and float(s.t) >= float(live_seconds):
                break

        st.success("Live stream finished.")

    st.divider()

    # ---- Run framework (CNC physics / logic pipeline) ----
    st.markdown("### CNC Run Pipeline (framework)")
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.write(f"**Device:** {cfg.get('device_name','(unnamed)')}  |  **Type:** {cfg.get('device_type','(unknown)')}")
        run_btn = st.button("▶ Run Framework", type="primary")

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
        else:
            st.info("No runs yet. Click **Run Framework**.")

    if run_btn:
        if pdm_path is None or pdm_choice is None:
            st.error("No PDM selected (CNC).")
            return

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

    st.divider()

    # ---- Component Builder (per asset) ----
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
                eff = st.number_input(
                    "Efficiency_Modifier",
                    value=float(comp_obj.get("Efficiency_Modifier", 1.0)),
                    step=0.01,
                    format="%.4f",
                )
            with c2:
                maxc = st.number_input("MaxCycles (0 = None)", value=float(comp_obj.get("MaxCycles", 0) or 0), step=1.0)
                notes = st.text_area("Notes", value=str(comp_obj.get("Notes", "")))

            if st.button("Save Component", type="primary"):
                pdm_edit.setdefault(comp, {})
                pdm_edit[comp]["Version"] = version
                pdm_edit[comp]["CyclesSinceInstall"] = int(cycles)
                pdm_edit[comp]["Efficiency_Modifier"] = float(eff)
                pdm_edit[comp]["MaxCycles"] = None if int(maxc) == 0 else int(maxc)
                pdm_edit[comp]["Notes"] = notes

                save_pdm(asset_rec.asset_id, pdm_edit)
                st.success("Saved.")
                st.rerun()

            if st.button("Delete Component"):
                if comp in pdm_edit:
                    pdm_edit.pop(comp, None)
                    save_pdm(asset_rec.asset_id, pdm_edit)
                    st.success("Deleted.")
                    st.rerun()

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

# =========================
# MAIN
# =========================
st.title("Industrial Digital Twin — Streamlit UI (v0)")

if asset_type == "Turbofan":
    render_turbofan_ui()
else:
    render_cnc_ui()