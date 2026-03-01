from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from ml.feature_contract_cnc import CNC_FEATURES_V1, CNC_SCHEMA_VERSION


LABELS = ["OK", "WARNING", "FAULT"]  # 0,1,2


def main():
    df = pd.read_csv("datasets/cnc_v1.csv")

    # enforce contract columns
    X = df[CNC_FEATURES_V1].copy()
    y = df["label_state"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=7,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=7,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== CNC Classification report ===")
    
    labels_int = [0, 1, 2]
    print(classification_report(
        y_test, y_pred,
        labels=labels_int,
        target_names=LABELS,
        zero_division=0,
    ))

    print("=== CNC Confusion matrix (rows=true, cols=pred) ===")
    
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": clf,
        "columns": CNC_FEATURES_V1,
        "labels": LABELS,
        "schema_version": CNC_SCHEMA_VERSION,
    }

    out_path = out_dir / "cnc_model_v1.joblib"
    joblib.dump(bundle, out_path)
    print(f"\n✅ Saved CNC model to {out_path}")


if __name__ == "__main__":
    main()
