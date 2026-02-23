from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator


@dataclass
class LPCMap:
    speed_vec: np.ndarray
    flow_vec: np.ndarray
    PR: np.ndarray
    ETA: np.ndarray
    pr_interp: RegularGridInterpolator
    eta_interp: RegularGridInterpolator

    def lookup(self, speed: float, flow: float) -> Tuple[float, float]:
        # Clamp inputs to map range to avoid wild extrapolation
        speed = float(np.clip(speed, self.speed_vec.min(), self.speed_vec.max()))
        flow  = float(np.clip(flow,  self.flow_vec.min(),  self.flow_vec.max()))
        # axes order: (speed, flow) because PR is shaped (Ns, Nf)
        pt = np.array([speed, flow])
        PR = float(self.pr_interp(pt)[0])
        ETA = float(self.eta_interp(pt)[0])
        return PR, ETA


def load_lpc_map(mat_path: str | Path) -> LPCMap:
    mat_path = Path(mat_path)
    m = loadmat(mat_path)

    speed_vec = np.array(m["speed_vec_lpc"]).squeeze().astype(float)
    flow_vec = np.array(m["Wc_lpc_bp"]).squeeze().astype(float)

    PR = np.array(m["PR_lpc_map"]).astype(float)
    ETA = np.array(m["eta_lpc_map"]).astype(float)

    # Expect PR shape = (len(speed_vec), len(flow_vec))
    if PR.shape != (len(speed_vec), len(flow_vec)):
        # try transpose fallback
        if PR.T.shape == (len(speed_vec), len(flow_vec)):
            PR = PR.T
            ETA = ETA.T
        else:
            raise ValueError(
                f"LPC map shape mismatch. PR={PR.shape}, expected {(len(speed_vec), len(flow_vec))}"
            )

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

    return LPCMap(
        speed_vec=speed_vec,
        flow_vec=flow_vec,
        PR=PR,
        ETA=ETA,
        pr_interp=pr_interp,
        eta_interp=eta_interp,
    )


if __name__ == "__main__":
    m = load_lpc_map("../Data/CFM56_LPC_map.mat")
    PR, ETA = m.lookup(speed=float(m.speed_vec.mean()), flow=float(m.flow_vec.mean()))
    print("PR:", PR, "ETA:", ETA)
