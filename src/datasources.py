from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional
from pathlib import Path
import time
import math
import random
import csv


@dataclass
class Sample:
    t: float
    signals: Dict[str, float]  # e.g., spindle_rpm, feed_mm_min, vibration, power_kw


class DataSource:
    """Abstract-ish interface."""
    def stream(self) -> Iterator[Sample]:
        raise NotImplementedError


@dataclass
class MockCNCSource(DataSource):
    """
    Mock live stream for CNC-like signals.
    Generates time-series samples you can plot + compute features on.
    """
    dt: float = 0.2
    rpm_cmd: float = 12000.0
    feed_cmd: float = 2000.0
    severity: float = 1.0
    noise: float = 0.02
    runtime_s: float = 20.0

    def stream(self) -> Iterator[Sample]:
        t = 0.0
        rpm = 0.0
        feed = 0.0

        # simple first-order response
        tau_rpm = 1.5
        tau_feed = 1.0

        while t <= self.runtime_s:
            # first-order response towards commands
            rpm += (self.rpm_cmd - rpm) * (self.dt / max(tau_rpm, 1e-6))
            feed += (self.feed_cmd - feed) * (self.dt / max(tau_feed, 1e-6))

            # vibration: base + severity + random + a little oscillation
            vib_base = 0.10 + 0.10 * float(self.severity)
            vib = vib_base + 0.05 * math.sin(2 * math.pi * 0.6 * t) + random.gauss(0.0, self.noise)

            # power: proportional to load (rpm, feed, severity)
            pwr = 1.0 + 0.00008 * rpm + 0.0002 * feed + 0.4 * float(self.severity) + random.gauss(0.0, self.noise)

            yield Sample(
                t=t,
                signals={
                    "spindle_rpm": float(rpm),
                    "feed_mm_min": float(feed),
                    "vibration": float(vib),
                    "power_kw": float(pwr),
                },
            )
            t += self.dt
            time.sleep(self.dt * 0.0)  # keep 0.0 for fast demo


@dataclass
class CSVReplaySource(DataSource):
    """
    Replay a previously logged CSV (e.g. outputs/live_timeseries.csv)
    Required columns: t, spindle_rpm, feed_mm_min, vibration, power_kw
    """
    csv_path: str | Path
    dt: float = 0.2
    loop: bool = False
    max_rows: Optional[int] = None

    def stream(self) -> Iterator[Sample]:
        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSVReplaySource: file not found: {path}")

        def _iter_rows():
            with path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                n = 0
                for row in reader:
                    yield row
                    n += 1
                    if self.max_rows is not None and n >= int(self.max_rows):
                        break

        while True:
            for row in _iter_rows():
                # tolerate missing t (compute from index)
                t = float(row.get("t", 0.0))

                signals = {
                    "spindle_rpm": float(row.get("spindle_rpm", 0.0)),
                    "feed_mm_min": float(row.get("feed_mm_min", 0.0)),
                    "vibration": float(row.get("vibration", 0.0)),
                    "power_kw": float(row.get("power_kw", 0.0)),
                }
                yield Sample(t=t, signals=signals)
                time.sleep(self.dt * 0.0)

            if not self.loop:
                break