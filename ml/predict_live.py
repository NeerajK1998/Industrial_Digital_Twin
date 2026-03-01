from typing import Dict, Any, Tuple
import joblib
import pandas as pd

from ml.feature_contract import FEATURES_V1, SCHEMA_VERSION
from ml.build_dataset_turbofan import add_derived_features  # reuse exact same feature logic


LABEL_MAP = {0: "OK", 1: "WARNING", 2: "FAULT"}


def load_model(path: str = "artifacts/turbofan_model_v1.joblib") -> Dict[str, Any]:
    bundle = joblib.load(path)

    # Safety checks
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Model schema mismatch. Model has {bundle.get('schema_version')} vs expected {SCHEMA_VERSION}"
        )
    if bundle.get("features") != FEATURES_V1:
        raise ValueError("Model features do not match FEATURES_V1 contract.")
    return bundle


def build_feature_row_from_out(out: Dict[str, float]) -> Dict[str, float]:
    """
    Take raw turbofan outputs and ensure derived features exist,
    then return ONLY the FEATURES_V1 keys.
    """
    # Ensure derived features are present
    add_derived_features(out)

    # Ensure schema_version exists (optional, but nice)
    out["schema_version"] = SCHEMA_VERSION

    row = {}
    for k in FEATURES_V1:
        if k not in out:
            raise KeyError(f"Missing required feature key in out: {k}")
        row[k] = float(out[k])
    return row


def predict_from_out(
    out: Dict[str, float],
    bundle: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:

    model = bundle["model"]

    row = build_feature_row_from_out(out)

    # Build DataFrame with correct column names
    X = pd.DataFrame([row], columns=FEATURES_V1)

    pred_int = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    proba_dict = {LABEL_MAP[i]: float(proba[i]) for i in range(len(proba))}
    return LABEL_MAP[pred_int], proba_dict


if __name__ == "__main__":
    # Quick self-test (only if you want)
    print("✅ predict_live module ready.")