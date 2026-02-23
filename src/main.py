from __future__ import annotations

import json
from pathlib import Path

from .pdm_loader import load_pdm
from .health_logic import decide_status, check_health_warnings
from .logger import append_run_summary


def main():
    # 1) Load PDM JSON (raw, for warnings + metadata)
    pdm_path = Path("data") / "PDM_Data.json"
    pdm = json.loads(pdm_path.read_text())

    # 2) Load workspace-like variables (degraded efficiencies etc.)
    ws = load_pdm(base_file="PDM_Data.json", upgrade_file="PDM_Data_Upgrade.json", data_dir="data")

    # 3) Select asset model from configs (default: cnc)
    device_cfg_path = Path("configs") / "device_template.json"
    device_cfg = json.loads(device_cfg_path.read_text()) if device_cfg_path.exists() else {}
    asset_type = (device_cfg.get("asset_type") or "cnc").lower()

    # 4) Run simulation (asset-dependent)
    if asset_type == "turbofan":
        # NOTE: import via package path compatible with `python -m src.main`
        from .turbofan.turbofan_runner import run_turbofan_core_balanced

        eff_fan = float(ws.get("Eff_Mod_Fan", 1.0))
        eff_lpc = float(ws.get("Eff_Mod_LPC", 1.0))
        eff_hpc = float(ws.get("Eff_Mod_HPC", 1.0))

        throttle_cmd = float(device_cfg.get("throttle_cmd", 0.6))

        outputs = run_turbofan_core_balanced(
            throttle_cmd=throttle_cmd,
            eff_mod_fan=eff_fan,
            eff_mod_lpc=eff_lpc,
            eff_mod_hpc=eff_hpc,
        )

        # write extra turbofan outputs for UI/debug
        out_path = Path("outputs") / "turbofan_signals.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(outputs, indent=2))

    else:
        # existing CNC twin path (unchanged)
        from .engine_core import simulate_cnc, CNCMission
        outputs = simulate_cnc(ws, CNCMission())

    # 5) Decision + warnings
    status = decide_status(outputs)
    warnings = check_health_warnings(pdm)

    # 6) Normalize outputs for logger schema (logger expects CNC fields)
    log_outputs = outputs.copy()
    if asset_type == "turbofan":
        # Logger expects MATLAB-style keys: N1, N2, Thrust, Fuel
        log_outputs["N1"] = float(outputs.get("N1_RPM", 0.0))
        log_outputs["N2"] = float(outputs.get("N2_RPM", 0.0))

        # Until nozzles are implemented, thrust is unknown -> stub
        log_outputs.setdefault("Thrust", 0.0)

        # FuelFlow column is sourced from outputs["Fuel"] in logger.py
        # Map to combustor fuel flow if available
        if "m_fuel" in outputs:
            log_outputs["Fuel"] = float(outputs["m_fuel"])
        else:
            log_outputs.setdefault("Fuel", 0.0)

    # 7) Log
    append_run_summary("outputs/RunSummary.csv", log_outputs, pdm, status, warnings)

    print("✅ Logged one run")
    print("Asset type:", asset_type)
    print("Status:", status)
    print("Warnings:", warnings)
    print("Workspace vars sample:", {k: ws[k] for k in list(ws)[:5]})
    if asset_type == "turbofan":
        print("Wrote:", "outputs/turbofan_signals.json")


if __name__ == "__main__":
    main()