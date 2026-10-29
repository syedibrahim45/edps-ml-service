"""Train and evaluate an isolated experimental binary diabetes classifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.diabetes_data import validate_dataset
from app.diabetes_schema import BINARY_DIABETES_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)
ARTIFACT_DIR = PROJECT_ROOT / "models" / "diabetes_binary"
SCHEMA_VERSION = "cdc-brfss2015-binary-v1"
RANDOM_SEED = 42
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
SPLIT_CANDIDATES = 200
FEATURE_NAMES = tuple(field.name for field in BINARY_DIABETES_SCHEMA.features)
TARGET_NAME = BINARY_DIABETES_SCHEMA.target_column
CATEGORICAL_FEATURES = tuple(
    field.name for field in BINARY_DIABETES_SCHEMA.features if field.kind == "categorical"
)
NUMERIC_FEATURES = tuple(
    field.name for field in BINARY_DIABETES_SCHEMA.features if field.kind == "numeric"
)


def load_validated_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load exact schema columns without transforming raw feature or target values."""
    dataframe = pd.read_csv(path)
    report = validate_dataset(dataframe, BINARY_DIABETES_SCHEMA)
    if not report.valid:
        raise ValueError(
            "Diabetes dataset validation failed: " + " ".join(report.errors)
        )

    features = dataframe.loc[:, FEATURE_NAMES].copy()
    target = dataframe.loc[:, TARGET_NAME].copy()
    if TARGET_NAME in features.columns or tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("Feature selection violated the configured schema.")
    return features, target


def _groups_from_feature_vectors(features: pd.DataFrame) -> np.ndarray:
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("Feature columns do not match the configured feature order.")
    return (
        features.groupby(
            list(FEATURE_NAMES),
            sort=False,
            dropna=False,
            observed=True,
        )
        .ngroup()
        .to_numpy()
    )


def _best_group_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    groups: np.ndarray,
    holdout_fraction: float,
    *,
    reference_prevalence: float,
    reference_rows: int,
    reference_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=SPLIT_CANDIDATES,
        test_size=holdout_fraction,
        random_state=RANDOM_SEED,
    )
    y = target.to_numpy()
    best: tuple[float, tuple[int, ...], np.ndarray, np.ndarray] | None = None

    for candidate_train, candidate_holdout in splitter.split(features, y, groups):
        holdout_y = y[candidate_holdout]
        train_y = y[candidate_train]
        if np.unique(holdout_y).size < 2 or np.unique(train_y).size < 2:
            continue

        row_fraction = len(candidate_holdout) / reference_rows
        prevalence = float(np.mean(holdout_y == 1.0))
        score = abs(row_fraction - reference_fraction) + abs(
            prevalence - reference_prevalence
        )
        tie_break = tuple(np.sort(candidate_holdout[: min(8, len(candidate_holdout))]))
        if best is None or (score, tie_break) < (best[0], best[1]):
            best = (
                score,
                tie_break,
                np.sort(candidate_train),
                np.sort(candidate_holdout),
            )

    if best is None:
        raise ValueError(
            "Unable to create grouped train/holdout splits containing both classes."
        )
    return best[2], best[3]


def create_grouped_splits(
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, np.ndarray]:
    """Create deterministic 70/15/15 row-targeted splits without feature overlap."""
    if len(features) != len(target) or len(features) == 0:
        raise ValueError("Features and target must have the same non-zero row count.")
    if tuple(features.columns) != FEATURE_NAMES:
        raise ValueError("Feature columns do not match the configured feature order.")
    if TARGET_NAME in features.columns:
        raise ValueError("The target column must never be included in features.")
    if set(pd.unique(target)) != {0.0, 1.0}:
        raise ValueError("The target must contain exactly the configured classes 0.0 and 1.0.")

    groups = _groups_from_feature_vectors(features)
    overall_prevalence = float((target == 1.0).mean())
    remaining, test = _best_group_holdout(
        features,
        target,
        groups,
        TEST_FRACTION,
        reference_prevalence=overall_prevalence,
        reference_rows=len(features),
        reference_fraction=TEST_FRACTION,
    )

    remaining_features = features.iloc[remaining]
    remaining_target = target.iloc[remaining]
    remaining_groups = groups[remaining]
    validation_fraction_of_remaining = VALIDATION_FRACTION / (
        1.0 - TEST_FRACTION
    )
    train_local, validation_local = _best_group_holdout(
        remaining_features,
        remaining_target,
        remaining_groups,
        validation_fraction_of_remaining,
        reference_prevalence=float((remaining_target == 1.0).mean()),
        reference_rows=len(features),
        reference_fraction=VALIDATION_FRACTION,
    )
    train = remaining[train_local]
    validation = remaining[validation_local]

    group_sets = {
        name: set(groups[indices])
        for name, indices in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise RuntimeError("Identical feature vectors crossed split boundaries.")
    return {"train": train, "validation": validation, "test": test}


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode declared categories; pass BMI through without scaling."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )


def build_candidate_models() -> dict[str, Pipeline]:
    """Return bounded, deterministic candidate configurations."""
    candidates: dict[str, Pipeline] = {}
    for class_weight in (None, "balanced"):
        suffix = "unweighted" if class_weight is None else "balanced"
        candidates[f"logistic_regression_{suffix}"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight=class_weight,
                        max_iter=500,
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        candidates[f"random_forest_{suffix}"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=14,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight=class_weight,
                        n_jobs=2,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    return candidates


def _positive_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1.0 not in classes:
        raise ValueError("Candidate model does not expose positive class 1.0.")
    return model.predict_proba(features)[:, classes.index(1.0)]


def select_f1_threshold(target: pd.Series, positive_scores: np.ndarray) -> float:
    """Choose maximum validation F1; break ties by closeness to 0.5 then higher."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(target, positive_scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + np.finfo(float).eps
    )
    best_f1 = float(np.max(f1_values))
    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0, atol=1e-12))
    return float(
        sorted(
            (float(thresholds[index]) for index in best_indices),
            key=lambda threshold: (abs(threshold - 0.5), -threshold),
        )[0]
    )


def calculate_metrics(
    target: pd.Series,
    positive_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate binary classification metrics with class order 0.0, 1.0."""
    actual = np.asarray(target)
    predicted = (positive_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0.0, 1.0]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision_class_1": float(
            precision_score(actual, predicted, pos_label=1.0, zero_division=0)
        ),
        "recall_sensitivity_class_1": float(
            recall_score(actual, predicted, pos_label=1.0, zero_division=0)
        ),
        "specificity_class_0": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1_class_1": float(f1_score(actual, predicted, pos_label=1.0, zero_division=0)),
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc_auc": float(roc_auc_score(actual, positive_scores)),
        "average_precision": float(average_precision_score(actual, positive_scores)),
        "positive_class_prevalence": float(np.mean(actual == 1.0)),
    }


def _class_counts(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts().sort_index()
    return {str(float(label)): int(count) for label, count in counts.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(output_dir: Path = ARTIFACT_DIR) -> tuple[Path, Path]:
    resolved_dir = output_dir.resolve()
    if resolved_dir != ARTIFACT_DIR.resolve():
        raise ValueError("Diabetes artifacts must remain in models/diabetes_binary.")
    return resolved_dir / "selected_model.joblib", resolved_dir / "training_metadata.json"


def train_experiment(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    """Train candidates, select only on validation, then evaluate test once."""
    features, target = load_validated_dataset(dataset_path)
    if tuple(features.columns) != FEATURE_NAMES or TARGET_NAME in features.columns:
        raise ValueError("Feature/target separation does not match the configured schema.")

    full_group_ids = _groups_from_feature_vectors(features)
    duplicate_feature_rows = len(features) - len(np.unique(full_group_ids))
    duplicate_full_rows = int(
        pd.concat(
            [features, target.rename(TARGET_NAME)],
            axis=1,
        ).duplicated(keep="first").sum()
    )
    splits = create_grouped_splits(features, target)
    split_information = {}
    for name, indices in splits.items():
        split_information[name] = {
            "rows": int(len(indices)),
            "class_counts": _class_counts(target.iloc[indices]),
            "positive_prevalence": float((target.iloc[indices] == 1.0).mean()),
        }

    X_train, y_train = features.iloc[splits["train"]], target.iloc[splits["train"]]
    X_validation = features.iloc[splits["validation"]]
    y_validation = target.iloc[splits["validation"]]
    X_test, y_test = features.iloc[splits["test"]], target.iloc[splits["test"]]

    if TARGET_NAME in X_train.columns or set(X_train.columns) != set(FEATURE_NAMES):
        raise ValueError("Target leakage detected in model features.")

    baseline_inference_start = time.perf_counter()
    baseline_score = np.full(len(y_validation), float((y_train == 1.0).mean()))
    baseline_inference_seconds = time.perf_counter() - baseline_inference_start
    baseline_default = calculate_metrics(y_validation, baseline_score, 0.5)
    validation_results: dict[str, dict[str, Any]] = {
        "majority_class_baseline": {
            "model_parameters": {"prediction": 0.0, "selection": "training majority"},
            "training_seconds": 0.0,
            "inference_seconds": baseline_inference_seconds,
            "threshold_selection": "fixed at 0.5",
            "validation_metrics": baseline_default,
            "selected_threshold": 0.5,
            "selected_validation_metrics": baseline_default,
        }
    }
    fitted_models: dict[str, Pipeline] = {}

    for name, model in build_candidate_models().items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scores = _positive_scores(model, X_validation)
        inference_seconds = time.perf_counter() - start

        default_metrics = calculate_metrics(y_validation, scores, 0.5)
        threshold = select_f1_threshold(y_validation, scores)
        selected_metrics = calculate_metrics(y_validation, scores, threshold)
        validation_results[name] = {
            "model_parameters": model.named_steps["classifier"].get_params(
                deep=False
            ),
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "threshold_selection": "maximum class-1 F1 on validation set",
            "validation_metrics_at_0_5": default_metrics,
            "selected_threshold": threshold,
            "selected_validation_metrics": selected_metrics,
        }
        fitted_models[name] = model

    selected_name = max(
        fitted_models,
        key=lambda name: (
            validation_results[name]["selected_validation_metrics"]["f1_class_1"],
            validation_results[name]["selected_validation_metrics"][
                "balanced_accuracy"
            ],
            name,
        ),
    )
    selected_model = fitted_models[selected_name]
    selected_threshold = validation_results[selected_name]["selected_threshold"]

    test_inference_start = time.perf_counter()
    test_scores = _positive_scores(selected_model, X_test)
    test_inference_seconds = time.perf_counter() - test_inference_start
    test_metrics = calculate_metrics(y_test, test_scores, selected_threshold)

    artifact_model_path, metadata_path = _artifact_paths()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, artifact_model_path, compress=3)
    metadata = {
        "experiment": "experimental diabetes binary-status classification",
        "interpretation_limit": (
            "Academic dataset-label experiment only; not a diagnosis, clinical "
            "probability, calibration, or evidence of medical validity."
        ),
        "dataset": {
            "filename": dataset_path.name,
            "rows": int(len(features)),
            "sha256": _sha256(dataset_path),
            "target": TARGET_NAME,
            "target_counts": _class_counts(target),
            "positive_class_prevalence": float((target == 1.0).mean()),
            "duplicate_full_rows": duplicate_full_rows,
            "duplicate_feature_rows": int(duplicate_feature_rows),
        },
        "schema_version": SCHEMA_VERSION,
        "feature_order": list(FEATURE_NAMES),
        "target_classes": [0.0, 1.0],
        "random_seed": RANDOM_SEED,
        "split_strategy": (
            "Deterministic GroupShuffleSplit candidate selection; identical "
            "21-feature vectors are grouped to prevent cross-split overlap. "
            "200 candidates per holdout; closest row fraction and class "
            "prevalence are selected."
        ),
        "splits": split_information,
        "candidates": validation_results,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "selection_rule": (
            "Highest validation class-1 F1 after validation-only threshold "
            "selection; balanced accuracy and model name break ties."
        ),
        "untouched_test_metrics": test_metrics,
        "selected_model_inference_seconds_on_test": test_inference_seconds,
        "artifact": {
            "path": str(artifact_model_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(artifact_model_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    result = train_experiment()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
