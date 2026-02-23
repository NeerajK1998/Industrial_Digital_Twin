from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from .maps import load_fan_map, FanMap
from .fan import fan_core_calc, GasProps


@dataclass
class FanSubsystem:
    fan_map: FanMap
    gas: GasProps = GasProps()

    @classmethod
    def from_default_files(cls) -> "FanSubsystem":
        fan_map = load_fan_map(Path(__file__).resolve().parents[3] / "Data" / "CFM56_Fan_map.mat")
        return cls(fan_map=fan_map)

    def step(
        self,
        P0: float,
        T0: float,
        N1_RPM: float,
        Wc_total: float,
        BPR: float,
        eff_mod: float = 1.0,
        # Map inputs:
        #   speed_input -> normalized speed (N1/N1_design)
        #   flow_input  -> corrected mass flow Wc_total (kg/s)
        speed_input: float | None = None,
        flow_input: float | None = None,
    ) -> Dict[str, float]:
        """
        One evaluation of the Fan subsystem.

        Returns dict with:
          P1_fan, T1_raw, Torque_fan, Wc_core, m_dot_core, PR_map, eta_map
        """

        if speed_input is None:
            # Map uses normalized speed (see build_CFM56_Fan_map.m)
            N1_design_rpm = 12000.0
            speed_input = float(N1_RPM) / N1_design_rpm
        if flow_input is None:
            flow_input = Wc_total

        PR_map, eta_map = self.fan_map.lookup(speed=float(speed_input), flow=float(flow_input))

        P1_fan, T1_raw, Torque_fan, Wc_core, m_dot_core = fan_core_calc(
            P0=P0,
            T0=T0,
            PR_map=PR_map,
            eta_map=eta_map,
            N1_RPM=N1_RPM,
            Wc_total=Wc_total,
            BPR=BPR,
            eff_mod=eff_mod,
            gas=self.gas,
        )

        return {
            "P1_fan": P1_fan,
            "T1_raw": T1_raw,
            "Torque_fan": Torque_fan,
            "Wc_core": Wc_core,
            "m_dot_core": m_dot_core,
            "PR_map": PR_map,
            "eta_map": eta_map,
        }


if __name__ == "__main__":
    # Quick demo run (numbers are not "validated" yet, but code path is correct)
    fan = FanSubsystem.from_default_files()
    out = fan.step(
        P0=101325.0,
        T0=288.15,
        N1_RPM=12000.0,
        Wc_total=fan.fan_map.flow_vec.mean(),  # keep within map range
        BPR=5.0,
        eff_mod=0.95,
    )
    for k, v in out.items():
        print(k, "=", v)
