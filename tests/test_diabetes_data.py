"""Tests for schema validation using synthetic software-test records only."""

import pandas as pd
import pytest

from app.diabetes_data import DatasetSchema, FieldSpec, validate_dataset


@pytest.fixture
def test_schema() -> DatasetSchema:
    return DatasetSchema(
        features=(
            FieldSpec(name="test_measurement", kind="numeric"),
            FieldSpec(
                name="test_category",
                kind="categorical",
                allowed_values=("group_a", "group_b"),
            ),
        ),
        target_column="test_target",
        allowed_target_classes=("class_a", "class_b"),
    )


@pytest.fixture
def test_records() -> pd.DataFrame:
    """Synthetic software-test examples; not medical or training data."""
    return pd.DataFrame(
        {
            "test_measurement": [1.25, 2.5, 3.75],
            "test_category": ["group_a", "group_b", "group_a"],
            "test_target": ["class_a", "class_b", "class_a"],
        }
    )


def test_accepts_valid_explicit_schema_and_data(test_schema, test_records) -> None:
    report = validate_dataset(test_records, test_schema)

    assert report.valid
    assert report.row_count == 3
    assert report.missing_columns == ()
    assert report.unexpected_columns == ()
    assert [(item.value, item.count) for item in report.class_distribution] == [
        ("class_a", 2),
        ("class_b", 1),
    ]


def test_reports_missing_required_columns(test_schema, test_records) -> None:
    report = validate_dataset(test_records.drop(columns="test_measurement"), test_schema)

    assert not report.valid
    assert report.missing_columns == ("test_measurement",)
    assert any("Missing required columns" in error for error in report.errors)


def test_rejects_duplicate_feature_column_labels(test_schema) -> None:
    dataframe = pd.DataFrame(
        [[1.0, 2.0, "group_a", "class_a"]],
        columns=[
            "test_measurement",
            "test_measurement",
            "test_category",
            "test_target",
        ],
    )

    report = validate_dataset(dataframe, test_schema)

    assert not report.valid
    assert report.duplicate_columns == ("test_measurement",)
    assert any("Duplicate DataFrame column labels" in error for error in report.errors)
    assert report.invalid_types == {}


def test_rejects_duplicate_target_column_labels(test_schema) -> None:
    dataframe = pd.DataFrame(
        [[1.0, "group_a", "class_a", "class_b"]],
        columns=[
            "test_measurement",
            "test_category",
            "test_target",
            "test_target",
        ],
    )

    report = validate_dataset(dataframe, test_schema)

    assert not report.valid
    assert report.duplicate_columns == ("test_target",)
    assert any("Duplicate DataFrame column labels" in error for error in report.errors)
    assert report.class_distribution == ()


def test_rejects_empty_dataframe_with_expected_columns_and_dtypes(test_schema) -> None:
    dataframe = pd.DataFrame(
        {
            "test_measurement": pd.Series(dtype="float64"),
            "test_category": pd.Series(dtype="string"),
            "test_target": pd.Series(dtype="string"),
        }
    )

    report = validate_dataset(dataframe, test_schema)

    assert not report.valid
    assert report.row_count == 0
    assert report.class_distribution == ()
    assert "Dataset must contain at least one row." in report.errors


def test_rejects_unexpected_columns_by_default(test_schema, test_records) -> None:
    report = validate_dataset(test_records.assign(unreviewed_field=1), test_schema)

    assert not report.valid
    assert report.unexpected_columns == ("unreviewed_field",)
    assert any("allow_extra_columns=True" in error for error in report.errors)


def test_allows_extra_columns_only_when_explicitly_configured(
    test_schema, test_records
) -> None:
    schema = DatasetSchema(
        features=test_schema.features,
        target_column=test_schema.target_column,
        allowed_target_classes=test_schema.allowed_target_classes,
        allow_extra_columns=True,
    )

    report = validate_dataset(test_records.assign(additional_context="test"), schema)

    assert report.valid
    assert report.unexpected_columns == ()


def test_reports_invalid_target_values(test_schema, test_records) -> None:
    test_records.loc[1, "test_target"] = "unapproved_class"

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert [(item.value, item.count) for item in report.invalid_target_values] == [
        ("unapproved_class", 1)
    ]
    assert any("outside the declared classes" in error for error in report.errors)


def test_reports_missing_target_values_without_imputation(
    test_schema, test_records
) -> None:
    test_records.loc[1, "test_target"] = None

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert report.missing_values["test_target"] == 1
    assert any("must be supplied, not imputed" in error for error in report.errors)


def test_rejects_numeric_feature_stored_as_strings(test_schema, test_records) -> None:
    test_records["test_measurement"] = test_records["test_measurement"].astype(str)

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert "test_measurement" in report.invalid_types
    assert "received pandas dtype" in report.invalid_types["test_measurement"]


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_rejects_complex_numeric_features(test_schema, test_records, dtype) -> None:
    test_records["test_measurement"] = pd.Series(
        [1 + 2j, 2 + 3j, 3 + 4j],
        dtype=dtype,
    )

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert "test_measurement" in report.invalid_types
    assert "real numeric values" in report.invalid_types["test_measurement"]


def test_reports_missing_feature_values_without_imputation(
    test_schema, test_records
) -> None:
    test_records.loc[0, "test_measurement"] = float("nan")

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert report.missing_values["test_measurement"] == 1
    assert any("missing value(s)" in error for error in report.errors)


def test_reports_duplicate_rows_without_mutating_frame(test_schema, test_records) -> None:
    original = test_records.copy(deep=True)
    duplicated = pd.concat([test_records, test_records.iloc[[0]]], ignore_index=True)

    report = validate_dataset(duplicated, test_schema)

    assert report.valid
    assert report.duplicate_row_count == 1
    pd.testing.assert_frame_equal(duplicated.iloc[:3], original)


def test_reports_class_counts_in_stable_order(test_schema, test_records) -> None:
    report = validate_dataset(test_records, test_schema)

    assert [(item.value, item.count) for item in report.class_distribution] == [
        ("class_a", 2),
        ("class_b", 1),
    ]


def test_validation_report_is_deterministic(test_schema, test_records) -> None:
    first = validate_dataset(test_records, test_schema)
    second = validate_dataset(test_records, test_schema)

    assert first == second


def test_reports_values_outside_explicit_categorical_spec(
    test_schema, test_records
) -> None:
    test_records.loc[0, "test_category"] = "unapproved_group"

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert [(item.value, item.count) for item in report.invalid_values["test_category"]] == [
        ("unapproved_group", 1)
    ]


def test_accepts_numeric_categorical_codes_when_declared(test_schema, test_records) -> None:
    schema = DatasetSchema(
        features=(
            test_schema.features[0],
            FieldSpec(
                name="test_category",
                kind="categorical",
                allowed_values=(1, 2),
            ),
        ),
        target_column=test_schema.target_column,
        allowed_target_classes=test_schema.allowed_target_classes,
    )
    test_records["test_category"] = [1, 2, 1]

    report = validate_dataset(test_records, schema)

    assert report.valid


@pytest.mark.parametrize(
    "values",
    [
        [["a"], ["b"], ["a"]],
        [{"key": 1}, {"key": 2}, {"key": 1}],
        [{"a", "b"}, {"b", "c"}, {"a", "b"}],
    ],
)
def test_rejects_non_scalar_categorical_values(test_schema, test_records, values) -> None:
    test_records["test_category"] = pd.Series(values, dtype=object)

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert "test_category" in report.invalid_values
    assert any("unsupported non-scalar" in error for error in report.errors)


def test_rejects_mixed_scalar_categories_without_declared_categories(
    test_schema, test_records
) -> None:
    schema = DatasetSchema(
        features=(
            test_schema.features[0],
            FieldSpec(name="test_category", kind="categorical"),
        ),
        target_column=test_schema.target_column,
        allowed_target_classes=test_schema.allowed_target_classes,
    )
    test_records["test_category"] = pd.Series(["group_a", 1, "group_a"], dtype=object)

    report = validate_dataset(test_records, schema)

    assert not report.valid
    assert "test_category" in report.invalid_values
    assert any("mixes scalar types" in error for error in report.errors)


def test_allowed_categorical_values_use_type_aware_comparison(
    test_schema, test_records
) -> None:
    schema = DatasetSchema(
        features=(
            test_schema.features[0],
            FieldSpec(
                name="test_category",
                kind="categorical",
                allowed_values=(1,),
            ),
        ),
        target_column=test_schema.target_column,
        allowed_target_classes=test_schema.allowed_target_classes,
    )
    test_records["test_category"] = pd.Series([1, True, 1], dtype=object)

    report = validate_dataset(test_records, schema)

    assert not report.valid
    assert [(item.value, item.count) for item in report.invalid_values["test_category"]] == [
        (True, 1)
    ]


def test_missing_pandas_categorical_values_are_reported_safely(
    test_schema, test_records
) -> None:
    test_records["test_category"] = pd.Series(
        ["group_a", pd.NA, "group_a"],
        dtype="string",
    )

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert report.missing_values["test_category"] == 1
    assert "test_category" not in report.invalid_values


def test_rejects_non_finite_numeric_values(test_schema, test_records) -> None:
    test_records.loc[0, "test_measurement"] = float("inf")

    report = validate_dataset(test_records, test_schema)

    assert not report.valid
    assert len(report.invalid_values["test_measurement"]) == 1
    assert report.invalid_values["test_measurement"][0].value == float("inf")


@pytest.mark.parametrize("boolean_label", [False, True])
def test_boolean_target_does_not_match_integer_target_class(
    test_records, boolean_label
) -> None:
    schema = DatasetSchema(
        features=(
            FieldSpec(name="test_measurement", kind="numeric"),
            FieldSpec(
                name="test_category",
                kind="categorical",
                allowed_values=("group_a", "group_b"),
            ),
        ),
        target_column="test_target",
        allowed_target_classes=(0, 1),
    )
    test_records["test_target"] = pd.Series([0, boolean_label, 1], dtype=object)

    report = validate_dataset(test_records, schema)

    assert not report.valid
    assert [(item.value, item.count) for item in report.invalid_target_values] == [
        (boolean_label, 1)
    ]
    assert len(report.class_distribution) == 3
    assert [(item.value, item.count) for item in report.class_distribution] == [
        (boolean_label, 1),
        (0, 1),
        (1, 1),
    ]


def test_boolean_and_integer_target_classes_remain_distinct(test_records) -> None:
    schema = DatasetSchema(
        features=(
            FieldSpec(name="test_measurement", kind="numeric"),
            FieldSpec(
                name="test_category",
                kind="categorical",
                allowed_values=("group_a", "group_b"),
            ),
        ),
        target_column="test_target",
        allowed_target_classes=(0, False, 1, True),
    )
    test_records["test_target"] = pd.Series([0, False, 1], dtype=object)

    report = validate_dataset(test_records, schema)

    assert report.valid
    assert [(item.value, item.count) for item in report.class_distribution] == [
        (False, 1),
        (0, 1),
        (1, 1),
    ]


def test_schema_allows_string_target_classes_and_rejects_duplicate_typed_classes() -> None:
    schema = DatasetSchema(
        features=(FieldSpec(name="test_measurement", kind="numeric"),),
        target_column="test_target",
        allowed_target_classes=("negative", "positive"),
    )
    assert schema.allowed_target_classes == ("negative", "positive")

    with pytest.raises(ValueError, match="type-aware comparison"):
        DatasetSchema(
            features=(),
            target_column="test_target",
            allowed_target_classes=(1, 1),
        )
