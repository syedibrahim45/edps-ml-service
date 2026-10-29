"""Tests for diabetes experiment utilities using synthetic examples only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.diabetes_schema import BINARY_DIABETES_SCHEMA
from app.train_diabetes_model import (
    ARTIFACT_DIR,
    FEATURE_NAMES,
    TARGET_NAME,
    _artifact_paths,
    build_candidate_models,
    calculate_metrics,
    create_grouped_splits,
    load_validated_dataset,
)


def synthetic_frame(rows: int = 80) -> pd.DataFrame:
    """Construct synthetic software-test records, not health data."""
    data = {}
    for field in BINARY_DIABETES_SCHEMA.features:
        values = field.allowed_values
        if field.kind == "numeric":
            data[field.name] = [float(index % 20 + 15) for index in range(rows)]
        else:
            data[field.name] = [
                float(values[(index // 2 + index // 5) % len(values)])
                for index in range(rows)
            ]
    dataframe = pd.DataFrame(data, columns=FEATURE_NAMES)
    dataframe[TARGET_NAME] = [float((index // 4) % 2) for index in range(rows)]
    return dataframe


def test_feature_target_separation_and_feature_order(tmp_path) -> None:
    path = tmp_path / "synthetic.csv"
    synthetic_frame().to_csv(path, index=False)

    features, target = load_validated_dataset(path)

    assert tuple(features.columns) == FEATURE_NAMES
    assert TARGET_NAME not in features.columns
    assert target.name == TARGET_NAME
    assert target.unique().tolist() == [0.0, 1.0]


def test_grouped_split_is_deterministic_and_has_no_feature_overlap() -> None:
    frame = synthetic_frame()
    features = frame.loc[:, FEATURE_NAMES]
    target = frame[TARGET_NAME]

    first = create_grouped_splits(features, target)
    second = create_grouped_splits(features, target)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    groups = features.groupby(list(FEATURE_NAMES), sort=False).ngroup().to_numpy()
    split_groups = {
        name: set(groups[indexes])
        for name, indexes in first.items()
    }
    assert not split_groups["train"] & split_groups["validation"]
    assert not split_groups["train"] & split_groups["test"]
    assert not split_groups["validation"] & split_groups["test"]
    assert set(np.concatenate(list(first.values()))) == set(range(len(frame)))


def test_split_rejects_target_leakage_and_wrong_feature_order() -> None:
    frame = synthetic_frame()
    leaking = frame.loc[:, [*FEATURE_NAMES, TARGET_NAME]]
    with pytest.raises(ValueError, match="feature order"):
        create_grouped_splits(leaking, frame[TARGET_NAME])

    reordered = frame.loc[:, list(reversed(FEATURE_NAMES))]
    with pytest.raises(ValueError, match="feature order"):
        create_grouped_splits(reordered, frame[TARGET_NAME])


def test_preprocessing_pipeline_is_reproducible() -> None:
    frame = synthetic_frame()
    features = frame.loc[:, FEATURE_NAMES]
    target = frame[TARGET_NAME]
    splits = create_grouped_splits(features, target)
    train = features.iloc[splits["train"]]
    train_target = target.iloc[splits["train"]]
    candidate = build_candidate_models()["logistic_regression_unweighted"]
    duplicate = build_candidate_models()["logistic_regression_unweighted"]

    candidate.fit(train, train_target)
    duplicate.fit(train, train_target)

    np.testing.assert_allclose(
        candidate.predict_proba(train),
        duplicate.predict_proba(train),
        rtol=0,
        atol=0,
    )
    assert isinstance(candidate, Pipeline)


def test_calculate_metrics_on_controlled_predictions() -> None:
    target = pd.Series([0.0, 0.0, 1.0, 1.0])
    scores = np.array([0.1, 0.8, 0.3, 0.9])

    result = calculate_metrics(target, scores, threshold=0.5)

    assert result["confusion_matrix_labels_0_1"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
    assert result["precision_class_1"] == 0.5
    assert result["recall_sensitivity_class_1"] == 0.5
    assert result["specificity_class_0"] == 0.5
    assert result["f1_class_1"] == 0.5
    assert result["roc_auc"] == 0.75
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["positive_class_prevalence"] == 0.5


def test_artifact_output_is_isolated_from_generic_models() -> None:
    model_path, metadata_path = _artifact_paths()

    assert model_path.parent == ARTIFACT_DIR
    assert metadata_path.parent == ARTIFACT_DIR
    assert model_path.name == "selected_model.joblib"
    assert metadata_path.name == "training_metadata.json"
    assert model_path.name not in {"disease_model.pkl", "disease_model.joblib"}
    with pytest.raises(ValueError, match="models/diabetes_binary"):
        _artifact_paths(ARTIFACT_DIR.parent)
