from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features import compute_features
from ml.feature_contract_cnc import CNC_FEATURES_V1, CNC_SCHEMA_VERSION
from ml.degradation_cnc import CNCDegradation, synthesize_timeseries, label_from_degradation, fault_class_from_degradation


def build_dataset(
    out_csv: str = "datasets/cnc_v1.csv",
    n_samples: int = 2000,
    seed: int = 7,
) -> str:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    s = seed

    # parameter sweeps (industrial-ish variety)
    rpm_cmds = [8000, 10000, 12000, 14000]
    feed_cmds = [800, 1200, 2000, 2800]
    severities = [0.6, 1.0, 1.4]

    # degradation levels
    levels_ok = [0.0, 0.05, 0.10]
    levels_warn = [0.20, 0.30, 0.40]
    levels_fault = [0.60, 0.75, 0.90]

    # build samples
    for i in range(n_samples):
        rpm_cmd = rpm_cmds[i % len(rpm_cmds)]
        feed_cmd = feed_cmds[(i // len(rpm_cmds)) % len(feed_cmds)]
        severity = severities[(i // (len(rpm_cmds) * len(feed_cmds))) % len(severities)]

        # choose which fault type
        mode = i % 5  # 0 healthy, 1 bearing, 2 imbalance, 3 tool, 4 mixed

        if mode == 0:
            deg = CNCDegradation(
                bearing_wear=float(levels_ok[i % len(levels_ok)]),
                imbalance=float(levels_ok[(i + 1) % len(levels_ok)]),
                tool_wear=float(levels_ok[(i + 2) % len(levels_ok)]),
                sensor_noise=0.02,
            )
        elif mode == 1:
            x = levels_warn[i % len(levels_warn)] if (i % 3) else levels_fault[i % len(levels_fault)]
            deg = CNCDegradation(bearing_wear=float(x), imbalance=0.0, tool_wear=0.0, sensor_noise=0.02)
        elif mode == 2:
            x = levels_warn[i % len(levels_warn)] if (i % 3) else levels_fault[i % len(levels_fault)]
            deg = CNCDegradation(bearing_wear=0.0, imbalance=float(x), tool_wear=0.0, sensor_noise=0.02)
        elif mode == 3:
            x = levels_warn[i % len(levels_warn)] if (i % 3) else levels_fault[i % len(levels_fault)]
            deg = CNCDegradation(bearing_wear=0.0, imbalance=0.0, tool_wear=float(x), sensor_noise=0.02)
        else:
            deg = CNCDegradation(
                bearing_wear=float(levels_warn[i % len(levels_warn)]),
                imbalance=float(levels_warn[(i + 1) % len(levels_warn)]),
                tool_wear=float(levels_warn[(i + 2) % len(levels_warn)]),
                sensor_noise=0.02,
            )

        ts = synthesize_timeseries(
            n=200,
            dt=0.2,
            rpm_cmd=float(rpm_cmd),
            feed_cmd=float(feed_cmd),
            severity=float(severity),
            deg=deg,
            seed=s,
        )
        s += 1

        feats = compute_features(ts)

        # enforce contract
        row = {k: float(feats.get(k, 0.0)) for k in CNC_FEATURES_V1}
        row["label_state"] = int(label_from_degradation(deg))  # 0 OK / 1 WARNING / 2 FAULT
        row["fault_class"] = int(fault_class_from_degradation(deg))
        row["bearing_wear"] = float(deg.bearing_wear)
        row["imbalance"] = float(deg.imbalance)
        row["tool_wear"] = float(deg.tool_wear)
        row["rpm_cmd"] = float(rpm_cmd)
        row["feed_cmd"] = float(feed_cmd)
        row["severity"] = float(severity)
        row["schema_version"] = CNC_SCHEMA_VERSION
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    path = build_dataset()
    print(f"Wrote: {path}")
