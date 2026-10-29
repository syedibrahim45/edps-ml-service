"""Configurable validation for approved diabetes dataset files."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


FieldKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    allowed_values: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Field names must be non-empty strings.")
        if self.kind not in ("numeric", "categorical"):
            raise ValueError(f"Unsupported field kind for {self.name!r}: {self.kind!r}.")
        if self.kind == "numeric" and self.allowed_values:
            raise ValueError("allowed_values can only be set for categorical fields.")
        _validate_unique_values(self.allowed_values, f"{self.name} allowed_values")


@dataclass(frozen=True)
class DatasetSchema:
    features: tuple[FieldSpec, ...]
    target_column: str
    allowed_target_classes: tuple[object, ...]
    allow_extra_columns: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.target_column, str) or not self.target_column.strip():
            raise ValueError("The target column name must be a non-empty string.")
        if not self.allowed_target_classes:
            raise ValueError("At least one allowed target class must be specified.")
        _validate_unique_values(self.allowed_target_classes, "allowed_target_classes")

        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique.")
        if self.target_column in names:
            raise ValueError("The target column cannot also be a feature.")


@dataclass(frozen=True)
class ValueCount:
    value: object
    count: int


@dataclass(frozen=True)
class DatasetValidationReport:
    valid: bool
    row_count: int
    missing_columns: tuple[object, ...]
    unexpected_columns: tuple[object, ...]
    duplicate_columns: tuple[object, ...]
    missing_values: dict[str, int]
    invalid_types: dict[str, str]
    invalid_values: dict[str, tuple[ValueCount, ...]]
    invalid_target_values: tuple[ValueCount, ...]
    duplicate_row_count: int
    class_distribution: tuple[ValueCount, ...]
    errors: tuple[str, ...]


def validate_dataset(
    dataframe: pd.DataFrame,
    schema: DatasetSchema,
) -> DatasetValidationReport:
    """Validate a frame against caller-supplied feature and target definitions.

    Categorical values are limited to strings, booleans, integers, and finite
    real numbers. Without declared ``allowed_values``, a feature must use one
    of those scalar types consistently. Validation never imputes, coerces,
    renames, or modifies the supplied frame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be a DatasetSchema.")

    expected_columns = {feature.name for feature in schema.features}
    expected_columns.add(schema.target_column)
    column_labels = list(dataframe.columns)
    duplicate_columns = _duplicate_labels(column_labels)
    missing_columns = tuple(sorted(expected_columns - set(column_labels)))
    unexpected_columns = (
        ()
        if schema.allow_extra_columns
        else tuple(
            sorted(
                set(column_labels) - expected_columns,
                key=_stable_sort_key,
            )
        )
    )

    errors: list[str] = []
    if len(dataframe.index) == 0:
        errors.append("Dataset must contain at least one row.")
    if duplicate_columns:
        errors.append(
            "Duplicate DataFrame column labels are not allowed: "
            + ", ".join(repr(label) for label in duplicate_columns)
            + ". Rename source columns explicitly before validation."
        )
    if missing_columns:
        errors.append(
            "Missing required columns: " + ", ".join(missing_columns) + "."
        )
    if unexpected_columns:
        errors.append(
            "Unexpected columns are not allowed: "
            + ", ".join(repr(label) for label in unexpected_columns)
            + ". Set allow_extra_columns=True only when intentional."
        )

    duplicate_row_count = _duplicate_row_count(dataframe)
    if duplicate_columns:
        return DatasetValidationReport(
            valid=False,
            row_count=len(dataframe),
            missing_columns=missing_columns,
            unexpected_columns=unexpected_columns,
            duplicate_columns=duplicate_columns,
            missing_values={},
            invalid_types={},
            invalid_values={},
            invalid_target_values=(),
            duplicate_row_count=duplicate_row_count,
            class_distribution=(),
            errors=tuple(errors),
        )

    missing_values: dict[str, int] = {}
    invalid_types: dict[str, str] = {}
    invalid_values: dict[str, tuple[ValueCount, ...]] = {}
    invalid_target_values: tuple[ValueCount, ...] = ()
    class_distribution: tuple[ValueCount, ...] = ()

    for field in schema.features:
        if field.name not in dataframe.columns:
            continue
        series = dataframe[field.name]
        missing_count = int(series.isna().sum())
        missing_values[field.name] = missing_count
        if missing_count:
            errors.append(
                f"Feature {field.name!r} has {missing_count} missing value(s); "
                "supply source values or resolve them through an explicit, "
                "documented preprocessing decision."
            )

        if not _has_expected_type(series, field):
            expected_type = (
                "real numeric values"
                if field.kind == "numeric"
                else "supported scalar categories"
            )
            invalid_types[field.name] = (
                f"expected {expected_type}; received pandas dtype {series.dtype!s}"
            )
            errors.append(
                f"Feature {field.name!r} must contain {expected_type}; "
                f"received dtype {series.dtype!s}."
            )
            continue

        observed_values = series.dropna()
        if field.kind == "numeric":
            non_finite = observed_values[
                ~observed_values.map(lambda value: math.isfinite(value))
            ]
            counts = _value_counts(non_finite)
            if counts:
                invalid_values[field.name] = counts
                errors.append(
                    f"Numeric feature {field.name!r} contains non-finite values: "
                    f"{_format_value_counts(counts)}."
                )
            continue

        invalid_categories = [
            value for value in observed_values if _category_key(value) is None
        ]
        if invalid_categories:
            counts = _value_counts(pd.Series(invalid_categories, dtype=object))
            invalid_values[field.name] = counts
            errors.append(
                f"Categorical feature {field.name!r} contains unsupported "
                f"non-scalar or non-finite values: {_format_value_counts(counts)}."
            )
            continue

        observed_keys = [_category_key(value) for value in observed_values]
        if field.allowed_values:
            allowed_keys = {_category_key(value) for value in field.allowed_values}
            disallowed = [
                value
                for value, key in zip(observed_values, observed_keys, strict=True)
                if key not in allowed_keys
            ]
            if disallowed:
                counts = _value_counts(pd.Series(disallowed, dtype=object))
                invalid_values[field.name] = counts
                errors.append(
                    f"Categorical feature {field.name!r} contains values outside "
                    f"the declared categories: {_format_value_counts(counts)}."
                )
        elif len({key[0] for key in observed_keys}) > 1:
            counts = _value_counts(observed_values)
            invalid_values[field.name] = counts
            errors.append(
                f"Categorical feature {field.name!r} mixes scalar types without "
                "an explicit allowed_values declaration. Declare the permitted "
                "categories in the schema or use one scalar type consistently."
            )

    if schema.target_column in dataframe.columns:
        target = dataframe[schema.target_column]
        missing_count = int(target.isna().sum())
        missing_values[schema.target_column] = missing_count
        if missing_count:
            errors.append(
                f"Target {schema.target_column!r} has {missing_count} missing "
                "value(s); target labels must be supplied, not imputed."
            )
        non_missing_target = target.dropna()
        class_distribution = _value_counts(non_missing_target)
        allowed_target_classes = {
            _category_key(value) for value in schema.allowed_target_classes
        }
        invalid_target_values = _value_counts(
            pd.Series(
                [
                    value
                    for value in non_missing_target
                    if _category_key(value) not in allowed_target_classes
                ],
                dtype=object,
            )
        )
        if invalid_target_values:
            errors.append(
                f"Target {schema.target_column!r} contains values outside the "
                f"declared classes: {_format_value_counts(invalid_target_values)}."
            )

    return DatasetValidationReport(
        valid=not errors,
        row_count=len(dataframe),
        missing_columns=missing_columns,
        unexpected_columns=unexpected_columns,
        duplicate_columns=duplicate_columns,
        missing_values=missing_values,
        invalid_types=invalid_types,
        invalid_values=invalid_values,
        invalid_target_values=invalid_target_values,
        duplicate_row_count=duplicate_row_count,
        class_distribution=class_distribution,
        errors=tuple(errors),
    )


def _has_expected_type(series: pd.Series, field: FieldSpec) -> bool:
    dtype = series.dtype
    if field.kind == "numeric":
        return (
            is_numeric_dtype(dtype)
            and not is_bool_dtype(dtype)
            and not is_complex_dtype(dtype)
        )

    return True


def _category_key(value: object) -> tuple[str, object] | None:
    """Return a type-sensitive identity for supported scalar categories."""
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Integral):
        return ("int", int(value))
    if isinstance(value, Real):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return ("float", numeric_value)
        return None
    if isinstance(value, str):
        return ("str", value)
    return None


def _value_counts(series: pd.Series) -> tuple[ValueCount, ...]:
    counts: dict[tuple[str, str], tuple[object, int]] = {}
    for raw_value in series:
        value = raw_value.item() if hasattr(raw_value, "item") else raw_value
        key = _identity_key(value)
        existing = counts.get(key)
        counts[key] = (value, 1 if existing is None else existing[1] + 1)

    return tuple(
        ValueCount(value=value, count=count)
        for _, (value, count) in sorted(counts.items())
    )


def _identity_key(value: object) -> tuple[str, str]:
    category_key = _category_key(value)
    if category_key is not None:
        return category_key[0], repr(category_key[1])
    return f"{type(value).__module__}.{type(value).__qualname__}", _stable_repr(value)


def _stable_repr(value: object) -> str:
    if isinstance(value, dict):
        items = sorted(
            ((_stable_repr(key), _stable_repr(item)) for key, item in value.items())
        )
        return f"dict:{items!r}"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}:{sorted(_stable_repr(item) for item in value)!r}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}:{[_stable_repr(item) for item in value]!r}"
    return repr(value)


def _stable_sort_key(value: object) -> tuple[str, str]:
    return f"{type(value).__module__}.{type(value).__qualname__}", _stable_repr(value)


def _duplicate_row_count(dataframe: pd.DataFrame) -> int:
    row_counts: dict[tuple[tuple[str, str], ...], int] = {}
    for row in dataframe.itertuples(index=False, name=None):
        key = tuple(
            ("missing", "")
            if _is_missing_scalar(value)
            else _identity_key(value)
            for value in row
        )
        row_counts[key] = row_counts.get(key, 0) + 1
    return sum(count - 1 for count in row_counts.values() if count > 1)


def _is_missing_scalar(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    if type(missing).__name__ == "bool_":
        return bool(missing)
    return False


def _duplicate_labels(labels: list[object]) -> tuple[object, ...]:
    duplicate_labels = [
        label
        for index, label in enumerate(labels)
        if label in labels[:index]
    ]
    unique_duplicates = {_stable_sort_key(label): label for label in duplicate_labels}
    return tuple(value for _, value in sorted(unique_duplicates.items()))


def _format_value_counts(counts: tuple[ValueCount, ...]) -> str:
    return ", ".join(f"{item.value!r} ({item.count})" for item in counts)


def _validate_unique_values(values: tuple[object, ...], field_name: str) -> None:
    keys: list[tuple[str, object]] = []
    for value in values:
        key = _category_key(value)
        if key is None:
            raise ValueError(
                f"{field_name} values must be supported scalar categories "
                "(strings, booleans, integers, or finite real numbers)."
            )
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError(
            f"{field_name} values must be unique using type-aware comparison."
        )
