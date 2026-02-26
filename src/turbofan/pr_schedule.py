import json
import numpy as np
from pathlib import Path


class PRSchedule:
    def __init__(self, path: str | Path):
        with open(path, "r") as f:
            data = json.load(f)

        self.throttle = np.array(data["throttle"])
        self.PR_HPT = np.array(data["PR_HPT"])
        self.PR_LPT = np.array(data["PR_LPT"])

    def get(self, throttle_cmd: float):
        PR_HPT = float(np.interp(throttle_cmd, self.throttle, self.PR_HPT))
        PR_LPT = float(np.interp(throttle_cmd, self.throttle, self.PR_LPT))
        return PR_HPT, PR_LPT