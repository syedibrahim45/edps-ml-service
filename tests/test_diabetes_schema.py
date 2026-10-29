"""Tests for the explicit dataset schema using synthetic records only."""

import pandas as pd
import pytest

from app.diabetes_data import validate_dataset
from app.diabetes_schema import (
    BINARY_DIABETES_COLUMNS,
    BINARY_DIABETES_SCHEMA,
)


def make_test_frame() -> pd.DataFrame:
    """Return software-test records, not health or model-training data."""
    data = {}
    for field in BINARY_DIABETES_SCHEMA.features:
        if field.kind == "numeric":
            data[field.name] = [25.0, 30.0]
        else:
            data[field.name] = [
                field.allowed_values[0],
                field.allowed_values[-1],
            ]
    data["Diabetes_binary"] = [0.0, 1.0]
    return pd.DataFrame(data, columns=BINARY_DIABETES_COLUMNS)


def test_schema_matches_confirmed_full_binary_csv_column_order() -> None:
    assert BINARY_DIABETES_COLUMNS == (
        "Diabetes_binary",
        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income",
    )
    assert BINARY_DIABETES_SCHEMA.target_column == "Diabetes_binary"


def test_accepts_valid_binary_schema_with_float_encoded_categories() -> None:
    report = validate_dataset(make_test_frame(), BINARY_DIABETES_SCHEMA)

    assert report.valid
    assert report.row_count == 2
    assert [(item.value, item.count) for item in report.class_distribution] == [
        (0.0, 1),
        (1.0, 1),
    ]
    assert report.invalid_values == {}
    assert report.invalid_target_values == ()


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("HighBP", 0.5),
        ("GenHlth", 6.0),
        ("Age", 14.0),
        ("Education", 0.0),
        ("Income", 9.0),
    ],
)
def test_rejects_out_of_range_categories(column, invalid_value) -> None:
    dataframe = make_test_frame()
    dataframe.loc[0, column] = invalid_value

    report = validate_dataset(dataframe, BINARY_DIABETES_SCHEMA)

    assert not report.valid
    assert column in report.invalid_values


@pytest.mark.parametrize("target_value", [2.0, 0.5, -1.0])
def test_rejects_invalid_binary_target_encodings(target_value) -> None:
    dataframe = make_test_frame()
    dataframe.loc[0, "Diabetes_binary"] = target_value

    report = validate_dataset(dataframe, BINARY_DIABETES_SCHEMA)

    assert not report.valid
    assert report.invalid_target_values


def test_rejects_fractional_ordinal_values_without_coercion() -> None:
    dataframe = make_test_frame()
    dataframe.loc[0, "Age"] = 1.5

    report = validate_dataset(dataframe, BINARY_DIABETES_SCHEMA)

    assert not report.valid
    assert report.invalid_values["Age"][0].value == 1.5


def test_rejects_missing_values_and_malformed_feature_types() -> None:
    dataframe = make_test_frame()
    dataframe.loc[0, "Income"] = pd.NA
    dataframe["BMI"] = dataframe["BMI"].astype(str)

    report = validate_dataset(dataframe, BINARY_DIABETES_SCHEMA)

    assert not report.valid
    assert report.missing_values["Income"] == 1
    assert "BMI" in report.invalid_types


def test_rejects_missing_and_unexpected_columns() -> None:
    dataframe = make_test_frame().drop(columns="Income").assign(extra_field=1.0)

    report = validate_dataset(dataframe, BINARY_DIABETES_SCHEMA)

    assert not report.valid
    assert report.missing_columns == ("Income",)
    assert report.unexpected_columns == ("extra_field",)
