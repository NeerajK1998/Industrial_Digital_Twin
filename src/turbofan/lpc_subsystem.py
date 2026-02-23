from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .lpc_maps import load_lpc_map, LPCMap
from .lpc import lpc_core_calc, GasProps


@dataclass
class LPCSubsystem:
    lpc_map: LPCMap
    gas: GasProps = GasProps()

    @classmethod
    def from_default_files(cls) -> "LPCSubsystem":
        # file is in repo root Data/, one level above python_port/
        mat_path = Path(__file__).resolve().parents[3] / "Data" / "CFM56_LPC_map.mat"
        lpc_map = load_lpc_map(mat_path)
        return cls(lpc_map=lpc_map)

    def step(
        self,
        P2_in: float,
        T2_in: float,
        N1_RPM: float,
        Wc_core: float,
        PR_base: float | None = None,
        eta_base: float | None = None,
        eff_mod: float = 1.0,
        # If map uses normalized speed like fan, we can compute it here later.
        speed_input: float | None = None,
        flow_input: float | None = None,
    ) -> Dict[str, float]:
        """
        One evaluation of the LPC subsystem.

        If PR_base/eta_base are not provided, they are looked up from the LPC map.
        Then efficiency modifier is applied: eta = eta_base * eff_mod (bounded in Simulink elsewhere if needed).
        """

        # Map lookup if needed
        if PR_base is None or eta_base is None:
            if speed_input is None:
                # LPC map uses raw RPM (see speed_vec_lpc)
                speed_input = float(N1_RPM)
            if flow_input is None:
                flow_input = float(Wc_core)

            PR_base, eta_base = self.lpc_map.lookup(speed=float(speed_input), flow=float(flow_input))

        PR_LPC = float(PR_base)
        eta_LPC = float(eta_base) * float(eff_mod)

        P3, T3, Torque_LPC, m_dot_core = lpc_core_calc(
            P2_in=P2_in,
            T2_in=T2_in,
            PR_LPC=PR_LPC,
            eta_LPC=eta_LPC,
            N1_RPM=N1_RPM,
            Wc_core=Wc_core,
            gas=self.gas,
        )

        return {
            "P3": P3,
            "T3": T3,
            "Torque_LPC": Torque_LPC,
            "m_dot_core": m_dot_core,
            "PR_LPC": PR_LPC,
            "eta_LPC": eta_LPC,
        }


if __name__ == "__main__":
    lpc = LPCSubsystem.from_default_files()

    # pick safe in-range inputs
    speed_mid = float(lpc.lpc_map.speed_vec.mean())
    flow_mid = float(lpc.lpc_map.flow_vec.mean())

    out = lpc.step(
        P2_in=180000.0,
        T2_in=330.0,
        N1_RPM=12000.0,
        Wc_core=flow_mid,
        speed_input=speed_mid,
        flow_input=flow_mid,
        eff_mod=0.95,
    )
    for k, v in out.items():
        print(k, "=", v)
