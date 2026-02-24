from __future__ import annotations

import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET = "outputs/ml/turbofan_dataset_A.csv"
MODEL_OUT = "outputs/ml/turbofan_rf_A.joblib"

FEATURE_KEYS = [
    "N1_RPM", "N2_RPM",
    "T4", "m_fuel", "FAR", "m_gas",
    "P0", "P2", "P3", "P4", "P5",
    "Thrust", "Thrust_core", "Thrust_bypass",
    "Vexit_core", "Vexit_bypass",
    "TorqueDiff_N1", "TorqueDiff_N2",
]

def main() -> None:
    df = pd.read_csv(DATASET)
    X = df[FEATURE_KEYS]
    y = df["label_status"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    Path("outputs/ml").mkdir(parents=True, exist_ok=True)
    dump(clf, MODEL_OUT)
    print(f"Saved: {MODEL_OUT}")

if __name__ == "__main__":
    main()
