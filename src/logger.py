from __future__ import annotations
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict

HEADERS = [
    "Timestamp", "N1", "N2", "Thrust", "FuelFlow",
    "FanVer", "FanCycles", "LPCVer", "LPCCycles",
    "HPCVer", "HPCCycles", "HPTVer", "HPTCycles",
    "LPTVer", "LPTCycles",
    "Status", "HealthWarnings",
]


def append_run_summary(
    out_path: str | Path,
    outputs: Dict[str, float],
    pdm: Dict,
    status: str,
    warnings: str,
):
    """
    Append one row to RunSummary.csv with the same schema as MATLAB script.
    outputs must include N1, N2, Thrust, Fuel
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "Timestamp": ts,
        "N1": outputs["N1"],
        "N2": outputs["N2"],
        "Thrust": outputs["Thrust"],
        "FuelFlow": outputs["Fuel"],
        "FanVer": pdm["Fan"]["Version"],
        "FanCycles": pdm["Fan"]["CyclesSinceInstall"],
        "LPCVer": pdm["LPC"]["Version"],
        "LPCCycles": pdm["LPC"]["CyclesSinceInstall"],
        "HPCVer": pdm["HPC"]["Version"],
        "HPCCycles": pdm["HPC"]["CyclesSinceInstall"],
        "HPTVer": pdm["HPT"]["Version"],
        "HPTCycles": pdm["HPT"]["CyclesSinceInstall"],
        "LPTVer": pdm["LPT"]["Version"],
        "LPTCycles": pdm["LPT"]["CyclesSinceInstall"],
        "Status": status,
        "HealthWarnings": warnings,
    }

    write_header = not out_path.exists()

    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    # quick smoke test
    import json
    pdm = json.load(open("data/PDM_Data.json", "r"))
    outputs = {"N1": 25000, "N2": 5000, "Thrust": 300000, "Fuel": 0.05}
    append_run_summary("outputs/RunSummary.csv", outputs, pdm, "FLY", "OK")
    print("Wrote outputs/RunSummary.csv")
