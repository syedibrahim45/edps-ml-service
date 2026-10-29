"""Isolated loader and inference for the experimental diabetes model."""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.diabetes_schema import BINARY_DIABETES_SCHEMA
from app.schemas import DiabetesPredictionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIABETES_MODEL_DIR = PROJECT_ROOT / "models" / "diabetes_binary"
DIABETES_MODEL_PATH = DIABETES_MODEL_DIR / "selected_model.joblib"
TRAINING_METADATA_PATH = DIABETES_MODEL_DIR / "training_metadata.json"
COMPATIBILITY_PATH = DIABETES_MODEL_DIR / "compatibility.json"
SCHEMA_VERSION = "cdc-brfss2015-binary-v1"
MODEL_IDENTIFIER = "logistic_regression_balanced"
FEATURE_NAMES = tuple(field.name for field in BINARY_DIABETES_SCHEMA.features)
TARGET_CLASSES = (0.0, 1.0)
ARTIFACT_RELATIVE_PATH = "models/diabetes_binary/selected_model.joblib"
PREDICTED_LABELS = {
    0: "No diabetes reported in dataset label",
    1: "Prediabetes or diabetes reported in dataset label",
}
DISCLAIMER = (
    "Experimental screening output for a survey dataset label only; "
    "not a clinical diagnosis or clinically validated risk estimate."
)
logger = logging.getLogger(__name__)


class DiabetesModelUnavailableError(RuntimeError):
    """Raised when the separate diabetes artifact cannot serve predictions."""


class DiabetesModelCompatibilityError(ValueError):
    """Raised when model metadata and the serialized pipeline disagree."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiabetesModelCompatibilityError(
            f"Diabetes {label} is unavailable or invalid."
        ) from error
    if not isinstance(value, dict):
        raise DiabetesModelCompatibilityError(
            f"Diabetes {label} must contain a JSON object."
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DiabetesModelCompatibilityError(
            "Diabetes model artifact is unavailable."
        ) from error
    return digest.hexdigest()


def _validate_metadata(
    metadata: dict[str, Any],
    compatibility: dict[str, Any],
    artifact_path: Path,
) -> float:
    expected_order = list(FEATURE_NAMES)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise DiabetesModelCompatibilityError("Unsupported diabetes schema version.")
    if metadata.get("selected_model") != MODEL_IDENTIFIER:
        raise DiabetesModelCompatibilityError("Unexpected diabetes model identifier.")
    if metadata.get("feature_order") != expected_order:
        raise DiabetesModelCompatibilityError("Diabetes feature order is incompatible.")
    if metadata.get("target_classes") != list(TARGET_CLASSES):
        raise DiabetesModelCompatibilityError("Diabetes target classes are incompatible.")
    dataset_metadata = metadata.get("dataset")
    if not isinstance(dataset_metadata, dict) or dataset_metadata.get(
        "target"
    ) != "Diabetes_binary":
        raise DiabetesModelCompatibilityError("Diabetes target column is incompatible.")

    threshold = metadata.get("selected_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise DiabetesModelCompatibilityError("Diabetes threshold is invalid.")

    recorded_artifact = metadata.get("artifact")
    compatibility_artifact = compatibility.get("artifact")
    expected_relative_path = ARTIFACT_RELATIVE_PATH
    if not isinstance(recorded_artifact, dict) or not isinstance(
        compatibility_artifact, dict
    ):
        raise DiabetesModelCompatibilityError("Diabetes artifact metadata is invalid.")
    for record in (recorded_artifact, compatibility_artifact):
        recorded_path = str(record.get("path", "")).replace("\\", "/")
        if recorded_path != expected_relative_path:
            raise DiabetesModelCompatibilityError("Diabetes artifact path is incompatible.")

    artifact_hash = recorded_artifact.get("sha256")
    if not isinstance(artifact_hash, str) or artifact_hash != compatibility_artifact.get(
        "sha256"
    ):
        raise DiabetesModelCompatibilityError("Diabetes artifact hashes disagree.")

    if compatibility.get("record_type") != "deployment_compatibility_verification":
        raise DiabetesModelCompatibilityError("Diabetes compatibility record is invalid.")
    if compatibility.get("model_identifier") != MODEL_IDENTIFIER:
        raise DiabetesModelCompatibilityError("Diabetes compatibility model ID disagrees.")
    if compatibility.get("schema_version") != SCHEMA_VERSION:
        raise DiabetesModelCompatibilityError(
            "Diabetes compatibility schema version disagrees."
        )
    if compatibility.get("feature_order") != expected_order:
        raise DiabetesModelCompatibilityError(
            "Diabetes compatibility feature order disagrees."
        )
    if compatibility.get("classes") != list(TARGET_CLASSES):
        raise DiabetesModelCompatibilityError(
            "Diabetes compatibility class mapping is incompatible."
        )
    if compatibility.get("selected_classification_threshold") != threshold:
        raise DiabetesModelCompatibilityError(
            "Diabetes compatibility threshold disagrees."
        )
    return float(threshold)


def _load_diabetes_artifacts(
    artifact_path: Path,
    metadata_path: Path,
    compatibility_path: Path,
) -> tuple[Any, float]:
    metadata = _read_json(metadata_path, "training metadata")
    compatibility = _read_json(compatibility_path, "compatibility record")
    threshold = _validate_metadata(metadata, compatibility, artifact_path)

    expected_hash = metadata["artifact"]["sha256"]
    actual_hash = _sha256(artifact_path)
    if actual_hash != expected_hash:
        raise DiabetesModelCompatibilityError("Diabetes artifact hash mismatch.")

    try:
        loaded_model = joblib.load(artifact_path)
    except Exception as error:
        raise DiabetesModelCompatibilityError(
            "Diabetes model artifact could not be loaded."
        ) from error

    if not callable(getattr(loaded_model, "predict_proba", None)):
        raise DiabetesModelCompatibilityError(
            "Diabetes model must support probability predictions."
        )
    if getattr(loaded_model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise DiabetesModelCompatibilityError(
            "Diabetes model must accept exactly 21 features."
        )
    if tuple(getattr(loaded_model, "feature_names_in_", ())) != FEATURE_NAMES:
        raise DiabetesModelCompatibilityError(
            "Diabetes model feature order does not match the API."
        )
    if list(getattr(loaded_model, "classes_", ())) != list(TARGET_CLASSES):
        raise DiabetesModelCompatibilityError(
            "Diabetes model classes do not match the dataset labels."
        )
    return loaded_model, threshold


def load_diabetes_model() -> tuple[Any, float]:
    """Load only the fixed, locally controlled diabetes model artifacts."""
    return _load_diabetes_artifacts(
        DIABETES_MODEL_PATH,
        TRAINING_METADATA_PATH,
        COMPATIBILITY_PATH,
    )


try:
    diabetes_model, diabetes_threshold = load_diabetes_model()
    DIABETES_MODEL_LOAD_ERROR = None
except Exception as error:
    diabetes_model = None
    diabetes_threshold = None
    DIABETES_MODEL_LOAD_ERROR = type(error).__name__
    logger.error(
        "Diabetes prediction model unavailable (%s).",
        DIABETES_MODEL_LOAD_ERROR,
    )


def predict_diabetes(request: DiabetesPredictionRequest) -> dict[str, Any]:
    """Return a dataset-label classification using the metadata threshold."""
    if diabetes_model is None or diabetes_threshold is None:
        raise DiabetesModelUnavailableError("The diabetes model is unavailable.")

    feature_frame = pd.DataFrame(
        [request.model_dump()],
        columns=FEATURE_NAMES,
    )
    if tuple(feature_frame.columns) != FEATURE_NAMES:
        raise DiabetesModelCompatibilityError(
            "Diabetes request feature order is incompatible."
        )
    try:
        probabilities = np.asarray(diabetes_model.predict_proba(feature_frame))
        classes = list(diabetes_model.classes_)
        positive_index = classes.index(1.0)
        score = float(probabilities[0][positive_index])
    except Exception as error:
        logger.error(
            "Diabetes prediction failed (%s).",
            type(error).__name__,
        )
        raise DiabetesModelUnavailableError(
            "The diabetes model could not produce a prediction."
        ) from error

    if not math.isfinite(score) or not 0 <= score <= 1:
        raise DiabetesModelUnavailableError(
            "The diabetes model produced an invalid score."
        )
    predicted_class = int(score >= diabetes_threshold)
    return {
        "predicted_class": predicted_class,
        "predicted_label": PREDICTED_LABELS[predicted_class],
        "positive_class_score": score,
        "threshold": diabetes_threshold,
        "model_id": MODEL_IDENTIFIER,
        "model_version": SCHEMA_VERSION,
        "disclaimer": DISCLAIMER,
    }
