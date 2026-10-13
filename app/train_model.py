"""Reproducible training and evaluation for the synthetic academic dataset."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = (
    "age",
    "gender",
    "blood_pressure",
    "glucose",
    "bmi",
    "heart_rate",
    "cholesterol",
)


def create_synthetic_dataset() -> pd.DataFrame:
    """Return the original 20 hand-authored records; these are not clinical data."""
    return pd.DataFrame(
        {
            "age": [25, 32, 45, 52, 61, 29, 40, 55, 67, 35,
                    48, 58, 23, 70, 44, 38, 63, 50, 31, 57],
            "gender": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
                       1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "blood_pressure": [110, 120, 145, 160, 170, 115, 130, 155, 180, 125,
                               150, 165, 105, 185, 135, 128, 175, 140, 118, 158],
            "glucose": [85, 95, 160, 180, 210, 90, 110, 175, 230, 105,
                        165, 190, 80, 250, 130, 115, 220, 145, 92, 185],
            "bmi": [21.5, 23.2, 28.5, 31.0, 33.5, 22.0, 25.5, 30.2, 35.0, 24.8,
                    29.5, 32.0, 20.5, 36.5, 27.0, 25.0, 34.0, 28.0, 22.5, 31.5],
            "heart_rate": [70, 72, 85, 92, 98, 68, 76, 90, 105, 74,
                           88, 95, 65, 110, 80, 75, 102, 84, 71, 93],
            "cholesterol": [170, 180, 230, 250, 270, 165, 200, 245, 290, 190,
                            235, 260, 160, 310, 215, 195, 280, 225, 175, 255],
            "disease": [0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
                        1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        }
    )


def train_and_evaluate(random_state: int = 42) -> tuple[RandomForestClassifier, dict]:
    """Train on a reproducible stratified split and return held-out metrics."""
    dataset = create_synthetic_dataset()
    features = dataset.loc[:, FEATURE_NAMES]
    target = dataset["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "dataset": "20-record synthetic academic dataset; not clinical data",
        "dataset_rows": len(dataset),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": random_state,
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "accuracy": accuracy_score(y_test, predictions),
        "precision_class_1": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "recall_class_1": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "f1_class_1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Low Risk (0)", "High Risk (1)"],
            output_dict=True,
            zero_division=0,
        ),
        "interpretation": (
            "Descriptive split metrics only. Four held-out synthetic records are "
            "far too few to support reliable clinical performance conclusions."
        ),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving a candidate model; the inference model is never replaced implicitly.",
    )
    args = parser.parse_args()

    model, metrics = train_and_evaluate()
    print(json.dumps(metrics, indent=2))

    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        print(f"Candidate model saved to: {output_path}")


if __name__ == "__main__":
    main()