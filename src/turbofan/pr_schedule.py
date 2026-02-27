import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class PRSchedule:
    def __init__(self, path: str | Path):
        self.path = str(path)

        with open(path, "r") as f:
            data = json.load(f)

        self.meta: Dict[str, Any] = data.get("meta", {})

        self.throttle = np.array(data["throttle"], dtype=float)
        self.PR_HPT = np.array(data["PR_HPT"], dtype=float)
        self.PR_LPT = np.array(data["PR_LPT"], dtype=float)

    def get(self, throttle_cmd: float) -> Tuple[float, float]:
        PR_HPT = float(np.interp(throttle_cmd, self.throttle, self.PR_HPT))
        PR_LPT = float(np.interp(throttle_cmd, self.throttle, self.PR_LPT))
        return PR_HPT, PR_LPT

    def validate(
        self,
        *,
        expected_mode: Optional[str] = None,
        P0: Optional[float] = None,
        T0: Optional[float] = None,
        V0: Optional[float] = None,
        tol_rel: float = 0.03,  # 3% tolerance
    ) -> None:
        """
        Hard guardrail: prevent accidentally using cruise schedule in SLS (and vice-versa).
        Raises ValueError on mismatch.
        """
        if not self.meta:
            # Backward compatible: if no meta, don't block.
            return

        if expected_mode is not None:
            mode = self.meta.get("mode", None)
            if mode is not None and str(mode).lower() != str(expected_mode).lower():
                raise ValueError(
                    f"PR schedule mode mismatch: schedule '{mode}' vs expected '{expected_mode}'. "
                    f"Schedule file: {self.path}"
                )

        def _check(name: str, expected: Optional[float]) -> None:
            if expected is None:
                return
            v = self.meta.get(name, None)
            if v is None:
                return
            v = float(v)
            if abs(v) < 1e-12:
                return
            if abs(float(expected) - v) / abs(v) > tol_rel:
                raise ValueError(
                    f"PR schedule env mismatch for {name}: schedule {v} vs expected {expected} "
                    f"(tol_rel={tol_rel}). Schedule file: {self.path}"
                )

        _check("P0", P0)
        _check("T0", T0)
        _check("V0", V0)