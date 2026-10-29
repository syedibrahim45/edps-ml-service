"""Validate the ignored full binary diabetes CSV and print a data summary."""

import json
from pathlib import Path
import sys

import pandas as pd

from app.diabetes_data import validate_dataset
from app.diabetes_schema import (
    BINARY_DIABETES_COLUMNS,
    BINARY_DIABETES_SCHEMA,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)


def main() -> int:
    if not DATASET_PATH.is_file():
        print(f"Dataset file not found: {DATASET_PATH}", file=sys.stderr)
        return 2

    try:
        dataframe = pd.read_csv(DATASET_PATH)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        print(f"Unable to read dataset CSV: {error}", file=sys.stderr)
        return 2

    report = validate_dataset(dataframe, BINARY_DIABETES_SCHEMA)
    column_order_matches = tuple(dataframe.columns) == BINARY_DIABETES_COLUMNS
    errors = list(report.errors)
    if not column_order_matches:
        errors.append(
            "CSV columns do not match the configured exact column order."
        )

    feature_ranges = {}
    observed_categories = {}
    for field in BINARY_DIABETES_SCHEMA.features:
        series = dataframe.get(field.name)
        if series is None or series.empty or not pd.api.types.is_numeric_dtype(series):
            continue
        feature_ranges[field.name] = {
            "minimum": float(series.min()),
            "maximum": float(series.max()),
        }
        if field.kind == "categorical":
            observed_categories[field.name] = [
                float(value)
                for value in sorted(series.dropna().unique().tolist())
            ]

    result = {
        "dataset_path": str(DATASET_PATH),
        "valid": report.valid and column_order_matches,
        "row_count": report.row_count,
        "column_count": len(dataframe.columns),
        "column_order_matches": column_order_matches,
        "feature_types": {
            name: str(dataframe[name].dtype)
            for name in dataframe.columns
        },
        "target_values": list(BINARY_DIABETES_SCHEMA.allowed_target_classes),
        "class_distribution": [
            {"value": item.value, "count": item.count}
            for item in report.class_distribution
        ],
        "missing_values": report.missing_values,
        "duplicate_row_count": report.duplicate_row_count,
        "feature_ranges": feature_ranges,
        "observed_categories": observed_categories,
        "invalid_feature_values": {
            name: [
                {"value": item.value, "count": item.count}
                for item in counts
            ]
            for name, counts in report.invalid_values.items()
        },
        "invalid_target_values": [
            {"value": item.value, "count": item.count}
            for item in report.invalid_target_values
        ],
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
