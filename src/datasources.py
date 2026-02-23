from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, Optional
import time
import math
import random


@dataclass
class Sample:
    t: float
    signals: Dict[str, float]  # e.g., spindle_rpm, feed, vibration, power_kw


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
        tau_feed = 0.8

        n = int(self.runtime_s / self.dt) + 1
        for _ in range(n):
            rpm += (self.rpm_cmd - rpm) * (self.dt / tau_rpm)
            feed += (self.feed_cmd - feed) * (self.dt / tau_feed)

            # vibration proxy rises with severity and feed and some periodicity
            vib = (0.2 * self.severity) + 0.00005 * feed + 0.05 * math.sin(2 * math.pi * 1.2 * t)
            # power proxy rises with rpm and severity (not physical, just plausible)
            power_kw = 0.8 + 0.00008 * rpm * self.severity + 0.2 * random.random()

            # add noise
            rpm_n = rpm * (1 + random.uniform(-self.noise, self.noise))
            feed_n = feed * (1 + random.uniform(-self.noise, self.noise))
            vib_n = vib * (1 + random.uniform(-self.noise, self.noise))
            power_n = power_kw * (1 + random.uniform(-self.noise, self.noise))

            yield Sample(
                t=t,
                signals={
                    "spindle_rpm": rpm_n,
                    "feed_mm_min": feed_n,
                    "vibration": vib_n,
                    "power_kw": power_n,
                },
            )

            if getattr(self,'realtime',False):
                time.sleep(self.dt)
            t += self.dt
