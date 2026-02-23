from __future__ import annotations
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# import your existing pieces
from src.datasources import MockCNCSource
from src.timeseries_logger import append_timeseries_row


@dataclass
class Scenario:
    rpm_cmd: float
    feed_cmd: float
    severity: float
    noise: float
    label: str


def make_label(severity: float) -> str:
    # Simple rule-based labeling to bootstrap ML
    # You can change these thresholds later.
    if severity >= 1.4:
        return "FAULT"
    if severity >= 1.15:
        return "WARNING"
    return "OK"


def extract_features(rows: List[Dict[str, float]]) -> Dict[str, float]:
    # Expect columns: spindle_rpm, feed_mm_min, vibration, power_kw
    vib = np.array([r["vibration"] for r in rows], dtype=float)
    pwr = np.array([r["power_kw"] for r in rows], dtype=float)
    rpm = np.array([r["spindle_rpm"] for r in rows], dtype=float)
    feed = np.array([r["feed_mm_min"] for r in rows], dtype=float)

    def slope(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        t = np.arange(len(x), dtype=float)
        # simple linear regression slope
        A = np.vstack([t, np.ones_like(t)]).T
        m, _b = np.linalg.lstsq(A, x, rcond=None)[0]
        return float(m)

    feats = {
        "vib_mean": float(vib.mean()),
        "vib_std": float(vib.std(ddof=1) if len(vib) > 1 else 0.0),
        "vib_max": float(vib.max()),
        "vib_p95": float(np.percentile(vib, 95)),
        "vib_slope": slope(vib),

        "pwr_mean": float(pwr.mean()),
        "pwr_std": float(pwr.std(ddof=1) if len(pwr) > 1 else 0.0),
        "pwr_max": float(pwr.max()),
        "pwr_p95": float(np.percentile(pwr, 95)),
        "pwr_slope": slope(pwr),

        "rpm_mean": float(rpm.mean()),
        "rpm_std": float(rpm.std(ddof=1) if len(rpm) > 1 else 0.0),

        "feed_mean": float(feed.mean()),
        "feed_std": float(feed.std(ddof=1) if len(feed) > 1 else 0.0),
    }
    return feats


def run_one_scenario(
    out_csv: Path,
    runtime_s: float,
    dt: float,
    sc: Scenario,
) -> Dict[str, float]:
    # generate a timeseries for one run
    src = MockCNCSource(
        runtime_s=runtime_s,
        dt=dt,
        rpm_cmd=sc.rpm_cmd,
        feed_cmd=sc.feed_cmd,
        severity=sc.severity,
        noise=sc.noise,
    )

    rows: List[Dict[str, float]] = []
    for s in src.stream():
        row = {"t": float(s.t), **{k: float(v) for k, v in s.signals.items()}}
        rows.append(row)
        # optional: also log to per-run file if you want later
        # append_timeseries_row(out_csv, row)

    feats = extract_features(rows)
    feats.update({
        "rpm_cmd": float(sc.rpm_cmd),
        "feed_cmd": float(sc.feed_cmd),
        "severity": float(sc.severity),
        "label": sc.label,
    })
    return feats


def main():
    random.seed(7)
    np.random.seed(7)

    out_path = Path("data/ml/dataset.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_s = 6.0
    dt = 0.2
    n_runs = 300

    samples: List[Dict[str, float]] = []

    for _ in range(n_runs):
        rpm_cmd = random.choice([8000, 10000, 12000, 14000])
        feed_cmd = random.choice([800, 1200, 1600, 2000, 2400])

        severity = random.uniform(0.8, 1.6)
        noise = random.uniform(0.0, 0.05)

        label = make_label(severity)
        sc = Scenario(rpm_cmd, feed_cmd, severity, noise, label)

        feats = run_one_scenario(out_csv=Path(""), runtime_s=runtime_s, dt=dt, sc=sc)
        samples.append(feats)

    # Write dataset
    fieldnames = list(samples[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(samples)

    print(f"✅ Wrote {out_path} with {len(samples)} samples and {len(fieldnames)-1} features.")


if __name__ == "__main__":
    main()
