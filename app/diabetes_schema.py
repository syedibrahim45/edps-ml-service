"""Explicit schema for the full binary CDC Diabetes Health Indicators file."""

from app.diabetes_data import DatasetSchema, FieldSpec


BINARY_DIABETES_COLUMNS = (
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

_BINARY_FEATURES = (
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
)

_ORDINAL_CATEGORIES = {
    "GenHlth": range(1, 6),
    "MentHlth": range(0, 31),
    "PhysHlth": range(0, 31),
    "Age": range(1, 14),
    "Education": range(1, 7),
    "Income": range(1, 9),
}

# CSV category values are represented as floats (for example 0.0 and 1.0).
# Exact allowlists accept those values without coercion and reject fractions.
BINARY_DIABETES_SCHEMA = DatasetSchema(
    features=(
        *(
            FieldSpec(name=name, kind="categorical", allowed_values=(0.0, 1.0))
            for name in _BINARY_FEATURES
        ),
        FieldSpec(name="BMI", kind="numeric"),
        *(
            FieldSpec(
                name=name,
                kind="categorical",
                allowed_values=tuple(float(value) for value in values),
            )
            for name, values in _ORDINAL_CATEGORIES.items()
        ),
    ),
    target_column="Diabetes_binary",
    allowed_target_classes=(0.0, 1.0),
)

if set(BINARY_DIABETES_COLUMNS) != {
    feature.name for feature in BINARY_DIABETES_SCHEMA.features
} | {BINARY_DIABETES_SCHEMA.target_column}:
    raise RuntimeError("The configured diabetes schema does not match its columns.")
