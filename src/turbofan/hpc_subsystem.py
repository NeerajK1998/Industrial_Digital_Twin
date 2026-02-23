from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .hpc_maps import load_hpc_map, HPCMap
from .hpc import hpc_core_calc, GasProps


@dataclass
class HPCSubsystem:
    hpc_map: HPCMap
    gas: GasProps = GasProps()

    @classmethod
    def from_default_files(cls) -> "HPCSubsystem":
        # Repo root is 3 levels up from python_port/src/turbofan/
        mat_path = Path(__file__).resolve().parents[3] / "Data" / "CFM56_HPC_map.mat"
        hpc_map = load_hpc_map(mat_path)
        return cls(hpc_map=hpc_map)

    def step(
        self,
        P1: float,
        T1: float,
        N2_RPM: float,
        Wc: float,
        eff_mod: float = 1.0,
        speed_input: float | None = None,
        flow_input: float | None = None,
    ) -> Dict[str, float]:

        # HPC map uses raw RPM
        if speed_input is None:
            speed_input = float(N2_RPM)

        if flow_input is None:
            flow_input = float(Wc)

        PR_base, eta_base = self.hpc_map.lookup(
            speed_rpm=float(speed_input),
            flow_wc=float(flow_input),
        )

        PR = float(PR_base)
        eta = float(eta_base) * float(eff_mod)

        P2, T2, Torque_HPC, m_dot = hpc_core_calc(
            P1=P1,
            T1=T1,
            PR=PR,
            eta=eta,
            N2_RPM=N2_RPM,
            Wc=Wc,
            gas=self.gas,
        )

        return {
            "P2": P2,
            "T2": T2,
            "Torque_HPC": Torque_HPC,
            "m_dot": m_dot,
            "PR_HPC": PR,
            "eta_HPC": eta,
        }


if __name__ == "__main__":
    hpc = HPCSubsystem.from_default_files()

    speed_mid = float(hpc.hpc_map.speed_vec.mean())
    flow_mid  = float(hpc.hpc_map.flow_vec.mean())

    out = hpc.step(
        P1=200000.0,
        T1=400.0,
        N2_RPM=speed_mid,
        Wc=flow_mid,
        eff_mod=0.95,
        speed_input=speed_mid,
        flow_input=flow_mid,
    )

    for k, v in out.items():
        print(k, "=", v)
