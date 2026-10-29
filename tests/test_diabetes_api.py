"""Diabetes endpoint and isolated model-loader tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import diabetes_model as diabetes_service
from app import main as main_module
from app.diabetes_schema import BINARY_DIABETES_SCHEMA
from app.main import app
from app.schemas import DiabetesPredictionRequest


client = TestClient(app)


def valid_request() -> dict:
    request = {}
    for field in BINARY_DIABETES_SCHEMA.features:
        if field.name == "BMI":
            request[field.name] = 27.5
        else:
            request[field.name] = int(field.allowed_values[0])
    return request


def _write_loader_fixtures(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    artifact = tmp_path / "selected_model.joblib"
    artifact.write_bytes(b"test artifact bytes")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    metadata = {
        "schema_version": diabetes_service.SCHEMA_VERSION,
        "selected_model": diabetes_service.MODEL_IDENTIFIER,
        "feature_order": list(diabetes_service.FEATURE_NAMES),
        "dataset": {"target": "Diabetes_binary"},
        "target_classes": [0.0, 1.0],
        "selected_threshold": 0.6447012298480324,
        "artifact": {
            "path": diabetes_service.ARTIFACT_RELATIVE_PATH,
            "sha256": digest,
        },
    }
    compatibility = {
        "record_type": "deployment_compatibility_verification",
        "model_identifier": diabetes_service.MODEL_IDENTIFIER,
        "schema_version": diabetes_service.SCHEMA_VERSION,
        "feature_order": list(diabetes_service.FEATURE_NAMES),
        "classes": [0.0, 1.0],
        "selected_classification_threshold": metadata["selected_threshold"],
        "artifact": {
            "path": diabetes_service.ARTIFACT_RELATIVE_PATH,
            "sha256": digest,
        },
    }
    metadata_path = tmp_path / "training_metadata.json"
    compatibility_path = tmp_path / "compatibility.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    compatibility_path.write_text(json.dumps(compatibility), encoding="utf-8")
    return artifact, metadata_path, compatibility_path, metadata, compatibility


class CompatibleModel:
    classes_ = np.array([0.0, 1.0])
    n_features_in_ = 21
    feature_names_in_ = np.array(diabetes_service.FEATURE_NAMES)

    def predict_proba(self, features):
        return np.tile([0.25, 0.75], (len(features), 1))


def test_valid_diabetes_request_returns_dataset_label_contract() -> None:
    response = client.post("/predict/diabetes", json=valid_request())

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] in (0, 1)
    assert result["predicted_label"] in {
        "No diabetes reported in dataset label",
        "Prediabetes or diabetes reported in dataset label",
    }
    assert result["predicted_label"] == (
        "Prediabetes or diabetes reported in dataset label"
        if result["predicted_class"] == 1
        else "No diabetes reported in dataset label"
    )
    assert 0 <= result["positive_class_score"] <= 1
    assert result["threshold"] == pytest.approx(0.6447012298480324)
    assert result["model_id"] == "logistic_regression_balanced"
    assert result["model_version"] == "cdc-brfss2015-binary-v1"
    assert "not a clinical diagnosis" in result["disclaimer"]


def test_request_schema_preserves_trained_feature_order() -> None:
    request = DiabetesPredictionRequest.model_validate(valid_request())

    assert tuple(request.model_dump()) == diabetes_service.FEATURE_NAMES


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("HighBP", 2),
        ("HighBP", 0.5),
        ("GenHlth", 0),
        ("GenHlth", 6),
        ("MentHlth", -1),
        ("MentHlth", 31),
        ("PhysHlth", 1.5),
        ("Age", 14),
        ("Education", 7),
        ("Income", 9),
    ],
)
def test_endpoint_rejects_invalid_or_fractional_categories(field, value) -> None:
    response = client.post(
        "/predict/diabetes",
        json={**valid_request(), field: value},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("HighBP", "1"),
        ("HighBP", True),
        ("BMI", "27.5"),
        ("BMI", True),
        ("BMI", float("nan")),
        ("BMI", float("inf")),
        ("BMI", float("-inf")),
    ],
)
def test_endpoint_rejects_strings_booleans_and_non_finite_numbers(field, value) -> None:
    payload = {**valid_request(), field: value}
    if isinstance(value, float) and not np.isfinite(value):
        response = client.post(
            "/predict/diabetes",
            content=json.dumps(payload, allow_nan=True),
            headers={"content-type": "application/json"},
        )
    else:
        response = client.post("/predict/diabetes", json=payload)

    assert response.status_code == 422


def test_endpoint_rejects_missing_and_extra_fields() -> None:
    payload = valid_request()

    missing_response = client.post(
        "/predict/diabetes",
        json={key: value for key, value in payload.items() if key != "BMI"},
    )
    extra_response = client.post(
        "/predict/diabetes",
        json={**payload, "unreviewed": 1},
    )

    assert missing_response.status_code == 422
    assert extra_response.status_code == 422


@pytest.mark.parametrize(
    ("score", "expected_class"),
    [
        (np.nextafter(0.6447012298480324, 0.0), 0),
        (0.6447012298480324, 1),
        (0.8, 1),
    ],
)
def test_score_extraction_and_threshold_boundary(
    monkeypatch, score, expected_class
) -> None:
    class ScoreModel(CompatibleModel):
        def predict_proba(self, features):
            assert tuple(features.columns) == diabetes_service.FEATURE_NAMES
            return np.tile([1.0 - score, score], (len(features), 1))

    monkeypatch.setattr(diabetes_service, "diabetes_model", ScoreModel())
    monkeypatch.setattr(
        diabetes_service,
        "diabetes_threshold",
        0.6447012298480324,
    )

    result = client.post("/predict/diabetes", json=valid_request()).json()

    assert result["positive_class_score"] == pytest.approx(score)
    assert result["predicted_class"] == expected_class


def test_loader_accepts_hash_matched_compatible_artifact(tmp_path, monkeypatch) -> None:
    artifact, metadata, compatibility, _, _ = _write_loader_fixtures(tmp_path)
    fake_model = CompatibleModel()
    monkeypatch.setattr(diabetes_service.joblib, "load", lambda _: fake_model)

    loaded, threshold = diabetes_service._load_diabetes_artifacts(
        artifact,
        metadata,
        compatibility,
    )

    assert loaded is fake_model
    assert threshold == pytest.approx(0.6447012298480324)


def test_loader_rejects_missing_artifact_before_deserialization(
    tmp_path, monkeypatch
) -> None:
    artifact, metadata, compatibility, _, _ = _write_loader_fixtures(tmp_path)
    artifact.unlink()
    called = []
    monkeypatch.setattr(diabetes_service.joblib, "load", lambda _: called.append(True))

    with pytest.raises(diabetes_service.DiabetesModelCompatibilityError):
        diabetes_service._load_diabetes_artifacts(artifact, metadata, compatibility)
    assert not called


def test_loader_rejects_corrupt_artifact_without_exposing_deserialization_error(
    tmp_path, monkeypatch
) -> None:
    artifact, metadata, compatibility, _, _ = _write_loader_fixtures(tmp_path)
    monkeypatch.setattr(
        diabetes_service.joblib,
        "load",
        lambda _: (_ for _ in ()).throw(RuntimeError("private path and detail")),
    )

    with pytest.raises(
        diabetes_service.DiabetesModelCompatibilityError,
        match="could not be loaded",
    ):
        diabetes_service._load_diabetes_artifacts(artifact, metadata, compatibility)


def test_loader_rejects_hash_mismatch_before_deserialization(tmp_path, monkeypatch) -> None:
    artifact, metadata, compatibility, _, _ = _write_loader_fixtures(tmp_path)
    artifact.write_bytes(b"changed artifact")
    called = []
    monkeypatch.setattr(diabetes_service.joblib, "load", lambda _: called.append(True))

    with pytest.raises(diabetes_service.DiabetesModelCompatibilityError, match="hash"):
        diabetes_service._load_diabetes_artifacts(artifact, metadata, compatibility)
    assert not called


@pytest.mark.parametrize(
    ("record_name", "key", "value"),
    [
        ("metadata", "schema_version", "wrong-version"),
        ("metadata", "selected_model", "wrong-model"),
        ("metadata", "feature_order", ["wrong"]),
        ("metadata", "dataset", {"target": "different_target"}),
        ("metadata", "target_classes", [0.0, 2.0]),
        ("metadata", "selected_threshold", 1.2),
        ("compatibility", "classes", [0.0, 2.0]),
        ("compatibility", "feature_order", ["wrong"]),
        ("compatibility", "model_identifier", "wrong-model"),
    ],
)
def test_loader_rejects_incompatible_metadata(
    tmp_path, monkeypatch, record_name, key, value
) -> None:
    artifact, metadata_path, compatibility_path, metadata, compatibility = (
        _write_loader_fixtures(tmp_path)
    )
    record = metadata if record_name == "metadata" else compatibility
    record[key] = value
    path = metadata_path if record_name == "metadata" else compatibility_path
    path.write_text(json.dumps(record), encoding="utf-8")
    called = []
    monkeypatch.setattr(diabetes_service.joblib, "load", lambda _: called.append(True))

    with pytest.raises(diabetes_service.DiabetesModelCompatibilityError):
        diabetes_service._load_diabetes_artifacts(
            artifact,
            metadata_path,
            compatibility_path,
        )
    assert not called


@pytest.mark.parametrize(
    "model",
    [
        object(),
        type("NoProbabilities", (), {"n_features_in_": 21})(),
        type(
            "WrongOrder",
            (),
            {
                "n_features_in_": 21,
                "feature_names_in_": np.array(["wrong"] * 21),
                "classes_": np.array([0.0, 1.0]),
                "predict_proba": lambda self, _: [[0.5, 0.5]],
            },
        )(),
        type(
            "WrongClasses",
            (),
            {
                "n_features_in_": 21,
                "feature_names_in_": np.array(diabetes_service.FEATURE_NAMES),
                "classes_": np.array([0.0, 2.0]),
                "predict_proba": lambda self, _: [[0.5, 0.5]],
            },
        )(),
    ],
)
def test_loader_rejects_incompatible_loaded_pipeline(tmp_path, monkeypatch, model) -> None:
    artifact, metadata, compatibility, _, _ = _write_loader_fixtures(tmp_path)
    monkeypatch.setattr(diabetes_service.joblib, "load", lambda _: model)

    with pytest.raises(diabetes_service.DiabetesModelCompatibilityError):
        diabetes_service._load_diabetes_artifacts(artifact, metadata, compatibility)


def test_unavailable_diabetes_model_returns_503_without_generic_fallback(
    monkeypatch,
) -> None:
    assert main_module.model is not None
    monkeypatch.setattr(diabetes_service, "diabetes_model", None)
    response = client.post("/predict/diabetes", json=valid_request())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Diabetes prediction model is unavailable."
    }
    assert "D:\\" not in response.text
    assert "Traceback" not in response.text


def test_diabetes_health_isolated_from_existing_generic_health(monkeypatch) -> None:
    monkeypatch.setattr(diabetes_service, "diabetes_model", None)

    diabetes_health = client.get("/health/diabetes")
    generic_health = client.get("/health")

    assert diabetes_health.json() == {
        "status": "DEGRADED",
        "model_status": "unavailable",
    }
    assert generic_health.status_code == 200
    assert set(generic_health.json()) == {"status", "service", "model_status"}


def test_existing_generic_and_explainability_routes_remain_available() -> None:
    generic_health = client.get("/health")
    prediction = client.post(
        "/predict",
        json={
            "age": 35,
            "gender": 0,
            "blood_pressure": 125,
            "glucose": 105,
            "bmi": 24.8,
            "heart_rate": 74,
            "cholesterol": 190,
        },
    )
    global_explanation = client.get("/explainability/global")
    individual_explanation = client.post(
        "/explainability/individual",
        json={
            "age": 35,
            "gender": 0,
            "blood_pressure": 125,
            "glucose": 105,
            "bmi": 24.8,
            "heart_rate": 74,
            "cholesterol": 190,
        },
    )
    federated = client.get("/federated/results")

    assert generic_health.status_code == 200
    assert prediction.status_code == 200
    assert set(prediction.json()) == {"prediction", "disease", "probability"}
    assert global_explanation.status_code == 200
    assert individual_explanation.status_code == 200
    assert federated.status_code == 200
