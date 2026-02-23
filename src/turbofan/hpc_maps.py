from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator


@dataclass
class HPCMap:
    speed_vec: np.ndarray
    flow_vec: np.ndarray
    PR: np.ndarray
    ETA: np.ndarray
    pr_interp: RegularGridInterpolator
    eta_interp: RegularGridInterpolator

    def lookup(self, speed_rpm: float, flow_wc: float) -> Tuple[float, float]:
        # Clamp to avoid extreme extrapolation
        speed_rpm = float(np.clip(speed_rpm, self.speed_vec.min(), self.speed_vec.max()))
        flow_wc   = float(np.clip(flow_wc,   self.flow_vec.min(),  self.flow_vec.max()))

        # axes order: (speed, flow) because PR shaped (Ns, Nf)
        pt = np.array([speed_rpm, flow_wc])
        PR = float(self.pr_interp(pt)[0])
        ETA = float(self.eta_interp(pt)[0])
        return PR, ETA


def load_hpc_map(mat_path: str | Path) -> HPCMap:
    mat_path = Path(mat_path)
    m = loadmat(mat_path)

    speed_vec = np.array(m["speed_vec_hpc"]).squeeze().astype(float)
    flow_vec  = np.array(m["Wc_hpc_bp"]).squeeze().astype(float)
    PR        = np.array(m["PR_hpc_map"]).astype(float)
    ETA       = np.array(m["eta_hpc_map"]).astype(float)

    # Expect (Ns, Nf)
    if PR.shape != (len(speed_vec), len(flow_vec)):
        if PR.T.shape == (len(speed_vec), len(flow_vec)):
            PR = PR.T
            ETA = ETA.T
        else:
            raise ValueError(f"HPC map shape mismatch PR={PR.shape}, expected {(len(speed_vec), len(flow_vec))}")

    pr_interp = RegularGridInterpolator(
        (speed_vec, flow_vec),
        PR,
        bounds_error=False,
        fill_value=None,
    )
    eta_interp = RegularGridInterpolator(
        (speed_vec, flow_vec),
        ETA,
        bounds_error=False,
        fill_value=None,
    )

    return HPCMap(
        speed_vec=speed_vec,
        flow_vec=flow_vec,
        PR=PR,
        ETA=ETA,
        pr_interp=pr_interp,
        eta_interp=eta_interp,
    )


if __name__ == "__main__":
    m = load_hpc_map("../Data/CFM56_HPC_map.mat")
    PR, ETA = m.lookup(speed_rpm=float(m.speed_vec.mean()), flow_wc=float(m.flow_vec.mean()))
    print("PR:", PR, "ETA:", ETA)
