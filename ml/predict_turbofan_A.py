from __future__ import annotations

import json
import pandas as pd
from joblib import load

MODEL = "outputs/ml/turbofan_rf_A.joblib"
SIGNALS = "outputs/turbofan_signals.json"

FEATURE_KEYS = [
    "N1_RPM", "N2_RPM",
    "T4", "m_fuel", "FAR", "m_gas",
    "P0", "P2", "P3", "P4", "P5",
    "Thrust", "Thrust_core", "Thrust_bypass",
    "Vexit_core", "Vexit_bypass",
    "TorqueDiff_N1", "TorqueDiff_N2",
]

def main() -> None:
    d = json.load(open(SIGNALS))
    row = {k: float(d.get(k, 0.0) or 0.0) for k in FEATURE_KEYS}
    X = pd.DataFrame([row], columns=FEATURE_KEYS)

    clf = load(MODEL)
    proba = clf.predict_proba(X)[0]
    pred = int(clf.predict(X)[0])

    label = "FLY" if pred == 1 else "GROUND"
    print("Pred:", label)
    print("Probabilities [GROUND, FLY]:", proba.tolist())

if __name__ == "__main__":
    main()
