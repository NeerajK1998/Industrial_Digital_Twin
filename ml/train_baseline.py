from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def main():
    df = pd.read_csv("data/ml/dataset.csv")

    y = df["label"]
    X = df.drop(columns=["label"])

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
    print(classification_report(y_test, y_pred))

    labels = ["OK", "WARNING", "FAULT"]
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    print(labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    out_dir = Path("outputs/ml")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": clf, "columns": list(X.columns)},
        out_dir / "baseline_rf.joblib"
    )
    print(f"\n✅ Saved model to {out_dir / 'baseline_rf.joblib'}")


if __name__ == "__main__":
    main()
