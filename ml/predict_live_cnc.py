from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from src.features import compute_features
from ml.feature_contract_cnc import CNC_FEATURES_V1


def load_model(path: str = "artifacts/cnc_model_v1.joblib") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CNC model not found: {p}")
    return joblib.load(p)


def predict_from_timeseries(timeseries: dict, bundle: dict) -> tuple[str, dict]:
    model = bundle["model"]
    labels = bundle.get("labels", ["OK", "WARNING", "FAULT"])
    feats = compute_features(timeseries)

    row = {k: float(feats.get(k, 0.0)) for k in CNC_FEATURES_V1}
    X = pd.DataFrame([row], columns=CNC_FEATURES_V1)

    pred_int = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    probs = {labels[i]: float(proba[i]) for i in range(len(labels))}
    return labels[pred_int], probs, row


def predict_from_live_csv(live_csv: str, bundle: dict, window: int = 200) -> tuple[str, dict, dict]:
    df = pd.read_csv(live_csv)
    if len(df) < 5:
        return "OK", {"OK": 1.0, "WARNING": 0.0, "FAULT": 0.0}, {}

    dfw = df.tail(window)
    ts = {
        "spindle_rpm": dfw["spindle_rpm"].astype(float).tolist() if "spindle_rpm" in dfw else [],
        "feed_mm_min": dfw["feed_mm_min"].astype(float).tolist() if "feed_mm_min" in dfw else [],
        "vibration": dfw["vibration"].astype(float).tolist() if "vibration" in dfw else [],
        "power_kw": dfw["power_kw"].astype(float).tolist() if "power_kw" in dfw else [],
    }
    return predict_from_timeseries(ts, bundle)
