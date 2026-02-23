import json
from pathlib import Path

COMPONENTS = ["Fan", "LPC", "HPC", "HPT", "LPT"]


def _normalize_eff(x: float) -> float:
    """
    Some JSONs may store efficiency as percent (e.g., 98) instead of fraction (0.98).
    Heuristic: if > 1.5, assume percent.
    """
    if x is None:
        raise ValueError("Efficiency_Modifier is missing")
    x = float(x)
    return x / 100.0 if x > 1.5 else x


def load_pdm(
    base_file="PDM_Data.json",
    upgrade_file="PDM_Data_Upgrade.json",
    data_dir="data",
    degradation_rate=0.0005,
    min_eff=0.80,
):
    """
    Python equivalent of MATLAB loadPDM logic.
    Returns a dictionary of workspace-like variables (Eff_Mod_*, Ver_*, Cycles_*, DoUpgrade).
    """

    base_path = Path(data_dir) / base_file
    if not base_path.exists():
        raise FileNotFoundError(f"Missing baseline PDM file: {base_path}")

    with open(base_path, "r") as f:
        pdm_base = json.load(f)

    upgrade_path = Path(data_dir) / upgrade_file
    if upgrade_path.exists():
        with open(upgrade_path, "r") as f:
            pdm_up = json.load(f)
    else:
        pdm_up = pdm_base

    workspace = {}

    for comp in COMPONENTS:
        raw_eff = _normalize_eff(pdm_base[comp]["Efficiency_Modifier"])
        cycles = int(pdm_base[comp]["CyclesSinceInstall"])

        # degradation rule (same as MATLAB)
        deg_eff = max(min_eff, raw_eff - degradation_rate * cycles)

        # upgrade values
        up_eff = _normalize_eff(pdm_up[comp]["Efficiency_Modifier"])
        up_ver = pdm_up[comp]["Version"]

        workspace[f"Eff_Mod_{comp}"] = float(deg_eff)
        workspace[f"Ver_{comp}"] = pdm_base[comp]["Version"]
        workspace[f"Cycles_{comp}"] = cycles

        workspace[f"Eff_Mod_{comp}_Up"] = float(up_eff)
        workspace[f"Ver_{comp}_Up"] = up_ver

    workspace["DoUpgrade"] = False
    return workspace


if __name__ == "__main__":
    ws = load_pdm()
    for k, v in ws.items():
        print(f"{k}: {v}")
