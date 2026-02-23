from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator


@dataclass
class FanMap:
    speed_vec: np.ndarray   # shape (Ns,)
    flow_vec: np.ndarray    # shape (Nf,)
    PR: np.ndarray          # shape (Nf, Ns) or (Ns, Nf) depending on file
    ETA: np.ndarray         # same shape as PR
    pr_interp: RegularGridInterpolator
    eta_interp: RegularGridInterpolator

    def lookup(self, speed: float, flow: float) -> Tuple[float, float]:
        """
        Returns (PR_map, eta_map) for given (speed, flow).
        """
        pt = np.array([flow, speed])  # order matches (flow_vec, speed_vec)
        PR = float(self.pr_interp(pt)[0])
        ETA = float(self.eta_interp(pt)[0])
        return PR, ETA


def load_fan_map(mat_path: str | Path) -> FanMap:
    """
    Loads CFM56_Fan_map.mat and builds 2D interpolators.

    Expects variables:
      - speed_vec_fan (1,N) or (N,)
      - flow_vec_fan  (1,N) or (N,)
      - Fan_PR_data   (N,N)
      - Fan_eta_data  (N,N)
    """

    mat_path = Path(mat_path)
    m = loadmat(mat_path)

    speed_vec = np.array(m["speed_vec_fan"]).squeeze().astype(float)
    flow_vec = np.array(m["flow_vec_fan"]).squeeze().astype(float)

    PR = np.array(m["Fan_PR_data"]).astype(float)
    ETA = np.array(m["Fan_eta_data"]).astype(float)

    # Ensure PR/ETA shape matches (len(flow_vec), len(speed_vec))
    # In many compressor maps, rows=flow and cols=speed.
    if PR.shape != (len(flow_vec), len(speed_vec)):
        # try transpose if stored as (Ns, Nf)
        if PR.T.shape == (len(flow_vec), len(speed_vec)):
            PR = PR.T
            ETA = ETA.T
        else:
            raise ValueError(
                f"Map shape mismatch. PR shape={PR.shape}, "
                f"expected {(len(flow_vec), len(speed_vec))} (or transpose)."
            )

    # RegularGridInterpolator expects axes in the same order as array dimensions.
    # Here: PR(flow_index, speed_index)
    pr_interp = RegularGridInterpolator(
        (flow_vec, speed_vec),
        PR,
        bounds_error=False,
        fill_value=None,   # extrapolate
    )
    eta_interp = RegularGridInterpolator(
        (flow_vec, speed_vec),
        ETA,
        bounds_error=False,
        fill_value=None,
    )

    return FanMap(
        speed_vec=speed_vec,
        flow_vec=flow_vec,
        PR=PR,
        ETA=ETA,
        pr_interp=pr_interp,
        eta_interp=eta_interp,
    )


if __name__ == "__main__":
    fan_map = load_fan_map("../Data/CFM56_Fan_map.mat")
    PR, ETA = fan_map.lookup(speed=float(fan_map.speed_vec.mean()), flow=float(fan_map.flow_vec.mean()))
    print("PR:", PR, "ETA:", ETA)
