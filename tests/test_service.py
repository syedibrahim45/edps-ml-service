import hashlib

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import main as main_module
from app import model as model_module
from app.federated_simulation import (
    aggregate_client_parameters,
    run_federated_simulation,
)
from app.main import app
from app.model import MODEL_PATH, load_model
from app.schemas import PredictionRequest
from app.train_model import FEATURE_NAMES, create_synthetic_dataset, train_and_evaluate


client = TestClient(app)


def valid_request() -> dict:
    return {
        "age": 35,
        "gender": 0,
        "blood_pressure": 125,
        "glucose": 105,
        "bmi": 24.8,
        "heart_rate": 74,
        "cholesterol": 190,
    }


def test_prediction_request_rejects_missing_extra_and_unsupported_values() -> None:
    payload = valid_request()
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({key: value for key, value in payload.items() if key != "glucose"})
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({**payload, "unexpected": 1})
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({**payload, "gender": 2})
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({**payload, "age": 0})
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate({**payload, "bmi": float("nan")})


def test_model_loads_with_expected_binary_feature_contract_from_any_cwd(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    loaded_model = load_model()

    assert MODEL_PATH.is_absolute()
    assert loaded_model.n_features_in_ == len(FEATURE_NAMES) == 7
    assert loaded_model.classes_.tolist() == [0, 1]
    assert tuple(loaded_model.feature_names_in_) == FEATURE_NAMES


def test_model_loader_rejects_incompatible_artifact(tmp_path) -> None:
    artifact_path = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a classifier"}, artifact_path)

    with pytest.raises(ValueError, match="does not support prediction"):
        load_model(artifact_path)


def test_model_loader_reports_missing_or_corrupt_artifacts(tmp_path) -> None:
    missing_path = tmp_path / "missing.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(missing_path)

    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"not a serialized model")
    with pytest.raises(Exception):
        load_model(corrupt_path)


def test_root_endpoint_is_available() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "EDPS ML Service is running"}


def test_health_and_prediction_response_contract() -> None:
    health_response = client.get("/health")
    prediction_response = client.post("/predict", json=valid_request())

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "UP",
        "service": "EDPS ML Service",
        "model_status": "ready",
    }
    assert prediction_response.status_code == 200
    result = prediction_response.json()
    assert set(result) == {"prediction", "disease", "probability"}
    assert result["prediction"] in (0, 1)
    assert result["disease"] == ("High Risk" if result["prediction"] else "Low Risk")
    assert 0 <= result["probability"] <= 1


def test_identical_prediction_requests_are_reproducible() -> None:
    first = client.post("/predict", json=valid_request())
    second = client.post("/predict", json=valid_request())

    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_unavailable_model_is_reported_without_internal_details(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "model", None)
    monkeypatch.setattr(model_module, "model", None)

    health_response = client.get("/health")
    prediction_response = client.post("/predict", json=valid_request())
    explanation_response = client.get("/explainability/global")
    individual_explanation_response = client.post(
        "/explainability/individual",
        json=valid_request(),
    )

    assert health_response.json()["status"] == "DEGRADED"
    assert health_response.json()["model_status"] == "unavailable"
    assert prediction_response.status_code == 503
    assert explanation_response.status_code == 503
    assert individual_explanation_response.status_code == 503
    assert "Traceback" not in prediction_response.text


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in valid_request().items() if key != "glucose"},
        {**valid_request(), "gender": 2},
        {**valid_request(), "age": 121},
        {**valid_request(), "unexpected": 1},
    ],
)
def test_prediction_endpoint_returns_422_for_invalid_input(payload) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_endpoint_rejects_non_finite_json_number() -> None:
    response = client.post(
        "/predict",
        content=(
            b'{"age":35,"gender":0,"blood_pressure":125,"glucose":105,'
            b'"bmi":NaN,"heart_rate":74,"cholesterol":190}'
        ),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", "35"),
        ("gender", "0"),
        ("blood_pressure", "125"),
        ("age", 121),
        ("blood_pressure", 301),
        ("glucose", 1001),
        ("bmi", 101),
        ("heart_rate", 251),
        ("cholesterol", 1001),
    ],
)
def test_prediction_endpoint_rejects_wrong_types_and_out_of_range_values(
    field, value
) -> None:
    response = client.post("/predict", json={**valid_request(), field: value})

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_global_explainability_is_calculated_and_not_per_prediction() -> None:
    model_module._cached_global_feature_importance.cache_clear()
    response = client.get("/explainability/global")
    repeated_response = client.get("/explainability/global")

    assert response.status_code == 200
    assert repeated_response.status_code == 200
    assert repeated_response.json() == response.json()
    assert model_module._cached_global_feature_importance.cache_info().misses == 1
    result = response.json()
    assert result["scope"] == "global"
    assert result["method"] == "permutation_importance"
    assert result["scoring"] == "neg_log_loss"
    assert result["sample_count"] == 20
    assert "Global permutation importance" in result["description"]
    assert {item["feature"] for item in result["feature_importance"]} == set(FEATURE_NAMES)
    assert "not an explanation for an individual prediction" in result["note"]


def test_individual_explanation_contract_is_separate_from_global_importance() -> None:
    response = client.post("/explainability/individual", json=valid_request())

    assert response.status_code == 200
    result = response.json()
    assert result["scope"] == "individual"
    assert result["method"] == "single_feature_replacement_sensitivity"
    assert set(result["prediction"]) == {"prediction", "disease", "probability"}
    assert [item["feature"] for item in result["feature_effects"]] == list(FEATURE_NAMES)
    assert result["prediction"] == client.post("/predict", json=valid_request()).json()
    assert {
        item["feature"]: item["input_value"] for item in result["feature_effects"]
    } == {name: float(value) for name, value in valid_request().items()}
    assert all(-1 <= item["probability_delta"] <= 1 for item in result["feature_effects"])


def test_training_is_reproducible_and_does_not_replace_inference_model() -> None:
    before = hashlib.sha256(MODEL_PATH.read_bytes()).digest()
    first_model, first_metrics = train_and_evaluate(random_state=42)
    second_model, second_metrics = train_and_evaluate(random_state=42)
    after = hashlib.sha256(MODEL_PATH.read_bytes()).digest()

    features = create_synthetic_dataset().loc[:, FEATURE_NAMES]
    assert first_metrics == second_metrics
    assert np.array_equal(first_model.predict(features), second_model.predict(features))
    assert first_metrics["dataset_rows"] == 20
    assert first_metrics["train_rows"] == 16
    assert first_metrics["test_rows"] == 4
    assert first_metrics["confusion_matrix_labels_0_1"] == [[2, 0], [0, 2]]
    assert first_metrics["class_counts"] == {0: 8, 1: 12}
    assert first_metrics["feature_distribution"]["age"]["minimum"] == 23
    assert before == after


def test_fedavg_uses_sample_weighted_parameter_averaging() -> None:
    weights, intercept = aggregate_client_parameters(
        [
            (1, np.array([1.0, 3.0]), 2.0),
            (3, np.array([5.0, 7.0]), 6.0),
        ]
    )

    assert weights.tolist() == [4.0, 6.0]
    assert intercept == 5.0


def test_federated_simulation_is_reproducible_and_partitions_training_data() -> None:
    first = run_federated_simulation(random_state=42)
    second = run_federated_simulation(random_state=42)

    assert first == second
    assert first["method"] == "FedAvg"
    assert first["model"] == "binary logistic regression"
    assert first["client_count"] == 3
    assert first["client_rows"] == [6, 5, 5]
    assert sum(first["client_rows"]) == first["training_rows"] == 16
    assert first["test_rows"] == 4
    assert len(first["global_coefficients"]) == len(FEATURE_NAMES)
    assert first["simulation_data"] == "synthetic"
    assert len(first["per_round_metrics"]) == first["rounds"]
    assert all(item["global_parameter_l2_norm"] > 0 for item in first["per_round_metrics"])
    assert first["final_evaluation_metrics"]["confusion_matrix_labels_0_1"] == [[0, 2], [0, 2]]


def test_federated_dashboard_endpoint_returns_cached_structured_results(monkeypatch) -> None:
    calls = []
    result = run_federated_simulation(rounds=1)

    def fake_simulation(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(main_module, "run_federated_simulation", fake_simulation)
    main_module._cached_federated_results.cache_clear()
    first = client.get("/federated/results")
    second = client.get("/federated/results")
    main_module._cached_federated_results.cache_clear()

    assert first.status_code == second.status_code == 200
    assert first.json()["simulation_data"] == "synthetic"
    assert first.json()["status"] == "completed"
    assert first.json()["per_round_metrics"] == second.json()["per_round_metrics"]
    assert len(calls) == 1