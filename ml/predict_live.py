import joblib
import pandas as pd
import numpy as np


def slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x), dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    m, _b = np.linalg.lstsq(A, x, rcond=None)[0]
    return float(m)


def extract_features(df: pd.DataFrame) -> dict:
    vib = df["vibration"].to_numpy(dtype=float)
    pwr = df["power_kw"].to_numpy(dtype=float)
    rpm = df["spindle_rpm"].to_numpy(dtype=float)
    feed = df["feed_mm_min"].to_numpy(dtype=float)

    feats = {
        "vib_mean": float(vib.mean()),
        "vib_std": float(vib.std(ddof=1) if len(vib) > 1 else 0.0),
        "vib_max": float(vib.max()),
        "vib_p95": float(np.percentile(vib, 95)),
        "vib_slope": slope(vib),

        "pwr_mean": float(pwr.mean()),
        "pwr_std": float(pwr.std(ddof=1) if len(pwr) > 1 else 0.0),
        "pwr_max": float(pwr.max()),
        "pwr_p95": float(np.percentile(pwr, 95)),
        "pwr_slope": slope(pwr),

        "rpm_mean": float(rpm.mean()),
        "rpm_std": float(rpm.std(ddof=1) if len(rpm) > 1 else 0.0),

        "feed_mean": float(feed.mean()),
        "feed_std": float(feed.std(ddof=1) if len(feed) > 1 else 0.0),

        # dataset includes these; for live we may not know them -> set 0
        "rpm_cmd": 0.0,
        "feed_cmd": 0.0,
        "severity": 0.0,
    }
    return feats


def main():
    bundle = joblib.load("outputs/ml/baseline_rf.joblib")
    model = bundle["model"]
    cols = bundle["columns"]

    ts = pd.read_csv("outputs/live_timeseries.csv")
    feats = extract_features(ts)
    X = pd.DataFrame([feats])[cols].fillna(0.0)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    print("Prediction:", pred)
    print("Probabilities:", {c: float(p) for c, p in zip(classes, proba)})


if __name__ == "__main__":
    main()

