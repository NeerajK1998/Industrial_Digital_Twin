from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from ml.feature_contract import FEATURES_V1, SCHEMA_VERSION


def main():
    dataset_path = Path("outputs/ml/turbofan_dataset_A.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # -----------------------------
    # Contract + schema safety checks
    # -----------------------------
    if "schema_version" not in df.columns:
        raise ValueError("Dataset missing 'schema_version' column.")
    found = set(df["schema_version"].astype(str).unique().tolist())
    if found != {SCHEMA_VERSION}:
        raise ValueError(f"Schema mismatch. Found {found}, expected {SCHEMA_VERSION}")

    for col in FEATURES_V1:
        if col not in df.columns:
            raise ValueError(f"Missing feature column in dataset: {col}")

    if "label_state" not in df.columns:
        raise ValueError("Dataset missing target column: label_state")

    # -----------------------------
    # Build X/y using frozen contract
    # -----------------------------
    X = df[FEATURES_V1].copy()
    y = df["label_state"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=7,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=7,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # label_state is numeric: 0=OK, 1=WARNING, 2=FAULT
    labels_int = [0, 1, 2]
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print("Label mapping: 0=OK, 1=WARNING, 2=FAULT")
    print(confusion_matrix(y_test, y_pred, labels=labels_int))

    # -----------------------------
    # Save model artifact (production-friendly)
    # -----------------------------
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "turbofan_model_v1.joblib"
    joblib.dump(
        {
            "model": clf,
            "schema_version": SCHEMA_VERSION,
            "features": FEATURES_V1,
            "target": "label_state",
        },
        out_path
    )
    print(f"\n✅ Saved model to {out_path}")


if __name__ == "__main__":
    main()