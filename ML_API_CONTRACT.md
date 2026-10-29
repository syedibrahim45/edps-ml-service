# EDPS ML API Contract

FastAPI is an internal service in the recommended **React → Spring Boot →
FastAPI** architecture. The browser should call Spring Boot, not FastAPI.
Spring Boot is responsible for user-facing authentication, authorization,
request mapping, and translating ML-service errors.

The examples use `http://ml-service:8000` as the FastAPI base URL on the
Spring Boot network. Successful responses are JSON. The `/predict` request
and response contract is unchanged.

## Experimental diabetes screening

`POST /predict/diabetes` is a separate experimental endpoint. It uses the
locally bundled diabetes-specific model and does not fall back to the generic
`/predict` model. Keep FastAPI on the private Spring Boot network; the browser
should call Spring Boot, not this service.

The request requires exactly these 21 feature fields, in the trained pipeline
order:

```json
{
  "HighBP": 1,
  "HighChol": 1,
  "CholCheck": 1,
  "Smoker": 0,
  "Stroke": 0,
  "HeartDiseaseorAttack": 0,
  "PhysActivity": 1,
  "Fruits": 1,
  "Veggies": 1,
  "HvyAlcoholConsump": 0,
  "AnyHealthcare": 1,
  "NoDocbcCost": 0,
  "DiffWalk": 0,
  "Sex": 0,
  "BMI": 27.5,
  "GenHlth": 3,
  "MentHlth": 0,
  "PhysHlth": 0,
  "Age": 9,
  "Education": 4,
  "Income": 5
}
```

All properties are required; unknown properties, strings, booleans, invalid
categories, and fractional categorical values are rejected with HTTP 422.
Binary survey codes accept integer `0` or `1`. `GenHlth` accepts integers 1–5;
`MentHlth` and `PhysHlth` accept 0–30; `Age` is an age bucket from 1–13;
`Education` accepts 1–6; and `Income` accepts 1–8.

`BMI` must be a finite JSON number. Strings, booleans, NaN, infinity, and
malformed values are rejected, with no silent coercion. Although the dataset
observed BMI values from 12 to 98, this is not a clinical validity range and
the endpoint imposes no BMI range restriction.

Example response:

```json
{
  "predicted_class": 0,
  "predicted_label": "No diabetes reported in dataset label",
  "positive_class_score": 0.42,
  "threshold": 0.6447012298480324,
  "model_id": "logistic_regression_balanced",
  "model_version": "cdc-brfss2015-binary-v1",
  "disclaimer": "Experimental screening output for a survey dataset label only; not a clinical diagnosis or clinically validated risk estimate."
}
```

Class 0 means **no diabetes reported in the dataset label**. Class 1 means
**prediabetes or diabetes reported in the dataset label**; it must not be
interpreted as a confirmed diagnosis. `positive_class_score` is the model's
class-1 score, not a clinically calibrated probability. The returned class is
1 when this score is greater than or equal to the threshold. The threshold is
loaded from the trusted training metadata (currently approximately 0.6447),
not from a client request.

`GET /health/diabetes` reports the diabetes model's readiness independently
of `GET /health`. If diabetes metadata or the trusted artifact is missing,
hash-mismatched, or incompatible, diabetes health is `DEGRADED` and
`POST /predict/diabetes` returns HTTP 503 with a generic error. It never uses
the generic model as a substitute. The existing `/health` contract is
unchanged.

The diabetes model is a joblib/scikit-learn artifact and must remain trusted;
never load user-supplied artifacts. Training-time dependency versions were
not recorded. Current-environment compatibility was checked and versions are
recorded in the ignored `models/diabetes_binary/compatibility.json`, but this
does not prove which dependency versions originally created the artifact.
Use the project-pinned dependencies and verify the artifact in the deployment
image. This endpoint is an academic experiment, not a diagnosis or clinical
decision tool.

## Health

`GET /health`

Ready response:

```json
{
  "status": "UP",
  "service": "EDPS ML Service",
  "model_status": "ready"
}
```

If the trusted model artifact is missing, invalid, or incompatible, this
endpoint responds HTTP 200 with `status: "DEGRADED"` and
`model_status: "unavailable"`. Treat this as **not ready** in Spring Boot
and deployment readiness checks. Prediction and explainability routes then
return HTTP 503.

## Prediction

`POST /predict`

Example request:

```json
{
  "age": 35,
  "gender": 0,
  "blood_pressure": 125,
  "glucose": 105,
  "bmi": 24.8,
  "heart_rate": 74,
  "cholesterol": 190
}
```

All seven properties are required. `age` is an integer; `gender` is integer
`0` or `1`; the other five values are finite JSON numbers. Extra properties,
numeric strings, booleans, and values outside these broad guardrails are
rejected with HTTP 422:

| Property | Accepted range |
| --- | ---: |
| `age` | 1–120 |
| `gender` | 0 or 1 |
| `blood_pressure` | 40–300 |
| `glucose` | 20–1000 |
| `bmi` | 5–100 |
| `heart_rate` | 20–250 |
| `cholesterol` | 50–1000 |

These bounds protect the request shape only; they are not clinically
validated thresholds. The training data does not define a real-world meaning
for `gender` 0 versus 1. Spring Boot must keep its agreed mapping consistent.

Observed response for the example above:

```json
{
  "prediction": 0,
  "disease": "Low Risk",
  "probability": 0.0
}
```

`prediction` is 0 or 1, `disease` is the existing `"Low Risk"` or
`"High Risk"` label, and `probability` is the class-1 model probability in
the inclusive range [0, 1] (a fraction, not a percentage). Retain these
three response keys in Spring Boot. The labels and probability do not
represent a diagnosis or clinically calibrated risk. For the same model
artifact and input, inference is deterministic.

## Explainability

### Global

`GET /explainability/global` describes the fitted model across all 20
synthetic records. This is **global feature importance**, not an explanation
for a submitted prediction. The first request calculates permutation
importance; its deterministic result is cached for each service process.
Subsequent dashboard polls reuse that result. The score is the increase in
negative log loss when a feature is shuffled; a positive value indicates
loss increased. This measures sensitivity on this model and dataset, not
causation or clinical relevance.

Observed response:

```json
{
  "scope": "global",
  "method": "permutation_importance",
  "scoring": "neg_log_loss",
  "sample_count": 20,
  "feature_importance": [
    {"feature": "heart_rate", "mean_importance": 0.1216960079, "std_importance": 0.0286903985},
    {"feature": "cholesterol", "mean_importance": 0.1183679063, "std_importance": 0.027756169997},
    {"feature": "age", "mean_importance": 0.0860629332, "std_importance": 0.0198319963},
    {"feature": "blood_pressure", "mean_importance": 0.0806118421, "std_importance": 0.0188862237},
    {"feature": "bmi", "mean_importance": 0.0665829112, "std_importance": 0.0155617630},
    {"feature": "glucose", "mean_importance": 0.0463477424, "std_importance": 0.0109741193},
    {"feature": "gender", "mean_importance": 0.0, "std_importance": 0.0}
  ],
  "description": "Global permutation importance measures the change in negative log loss when one feature is shuffled across the evaluation rows. A larger positive value means shuffling that feature worsened this model's loss on this synthetic dataset; it is not causal or patient-specific.",
  "note": "This is global feature importance calculated on the full synthetic dataset, not an explanation for an individual prediction or an independent validation result."
}
```

The first global calculation took 37.4 seconds in the recorded Windows
environment. Configure a read timeout of at least 60 seconds for the first
request, or warm the endpoint before dashboard traffic. Later calls in that
process use the cache. Each worker has its own cache.

### Individual

`POST /explainability/individual` accepts exactly the same request body and
validation rules as `/predict`. Its `prediction` object corresponds to that
submitted input. Each `feature_effects` item reports the submitted value,
synthetic median/mode reference, probabilities with the input and with the
replacement, and their difference. This is a local single-feature
replacement sensitivity, not an additive attribution, causal effect, SHAP
result, or medical explanation.

Observed response for the example request:

```json
{
  "scope": "individual",
  "method": "single_feature_replacement_sensitivity",
  "prediction": {"prediction": 0, "disease": "Low Risk", "probability": 0.0},
  "reference_population": "per-feature median/mode of the 20 synthetic training rows",
  "feature_effects": [
    {"feature": "age", "input_value": 35.0, "reference_value": 46.5, "probability_with_input": 0.0, "probability_with_reference": 0.18, "probability_delta": -0.18},
    {"feature": "gender", "input_value": 0.0, "reference_value": 1.0, "probability_with_input": 0.0, "probability_with_reference": 0.0, "probability_delta": 0.0},
    {"feature": "blood_pressure", "input_value": 125.0, "reference_value": 142.5, "probability_with_input": 0.0, "probability_with_reference": 0.16, "probability_delta": -0.16},
    {"feature": "glucose", "input_value": 105.0, "reference_value": 152.5, "probability_with_input": 0.0, "probability_with_reference": 0.09, "probability_delta": -0.09},
    {"feature": "bmi", "input_value": 24.8, "reference_value": 28.25, "probability_with_input": 0.0, "probability_with_reference": 0.13, "probability_delta": -0.13},
    {"feature": "heart_rate", "input_value": 74.0, "reference_value": 84.5, "probability_with_input": 0.0, "probability_with_reference": 0.22, "probability_delta": -0.22},
    {"feature": "cholesterol", "input_value": 190.0, "reference_value": 227.5, "probability_with_input": 0.0, "probability_with_reference": 0.22, "probability_delta": -0.22}
  ],
  "note": "Each delta compares the supplied input with a synthetic median/mode replacement for one feature while holding the other inputs fixed. Effects are not additive, causal, clinically meaningful, or a Shapley explanation."
}
```

The dashboard should expose `scope` and `method` (or equivalent labels) so
this input-specific sensitivity is not confused with global importance.

## Experimental federated simulation

`GET /federated/results` returns cached metrics from a deterministic,
single-process FedAvg experiment: three simulated clients, 10 rounds, five
local epochs per round, and sample-count-weighted parameter averaging. This
is a logistic-regression experiment on synthetic data, separate from the
Random Forest used by `/predict`. The clients do not represent hospitals.

The response includes `client_count`, `rounds`, `local_epochs`,
`training_rows`, `test_rows`, `client_rows`, `per_round_metrics`, and
`final_evaluation_metrics`. Each round contains `round`, `accuracy`,
`precision_class_1`, `recall_class_1`, `f1_class_1`,
`confusion_matrix_labels_0_1`, and `global_parameter_l2_norm`.

Observed result summary:

```json
{
  "status": "completed",
  "method": "FedAvg",
  "model": "binary logistic regression",
  "simulation_data": "synthetic",
  "client_count": 3,
  "rounds": 10,
  "local_epochs": 5,
  "training_rows": 16,
  "test_rows": 4,
  "client_rows": [6, 5, 5],
  "first_round": {
    "round": 1,
    "accuracy": 0.5,
    "precision_class_1": 0.5,
    "recall_class_1": 1.0,
    "f1_class_1": 0.6666666667,
    "confusion_matrix_labels_0_1": [[0, 2], [0, 2]],
    "global_parameter_l2_norm": 0.3998111590
  },
  "last_round": {
    "round": 10,
    "accuracy": 0.5,
    "precision_class_1": 0.5,
    "recall_class_1": 1.0,
    "f1_class_1": 0.6666666667,
    "confusion_matrix_labels_0_1": [[0, 2], [0, 2]],
    "global_parameter_l2_norm": 1.4408367556
  },
  "final_evaluation_metrics": {
    "accuracy": 0.5,
    "precision_class_1": 0.5,
    "recall_class_1": 1.0,
    "f1_class_1": 0.6666666667,
    "confusion_matrix_labels_0_1": [[0, 2], [0, 2]]
  }
}
```

`first_round` and `last_round` are explanatory extracts from the actual
`per_round_metrics` array, not additional response properties. The actual
array contains every round. Metrics use only four held-out synthetic rows
and are not performance evidence.

## Errors and Spring Boot handling

| HTTP status | Meaning | Spring Boot action |
| --- | --- | --- |
| 422 | Missing, extra, non-finite, wrong-type, or out-of-range request field | Return a safe client validation error; do not retry |
| 503 | Model unavailable for prediction/explainability | Return service unavailable; alert or mark dependency degraded |
| 5xx | Unexpected service failure | Log context server-side; return a generic error |
| timeout | Service did not answer by the configured client deadline | Cancel and map to 504 or dependency unavailable; avoid unbounded retries |

Configure Spring Boot's connection timeout around 5 seconds and its read
timeout to at least 60 seconds for a cold global-importance calculation.
Warm that endpoint before dashboard traffic. Use a workload-appropriate
shorter read timeout for prediction, set an overall request deadline, and
avoid retrying slow explanation calls automatically. Configure these
timeouts on the Spring Boot HTTP client, not in the React browser.

Spring Boot can send a DTO with matching JSON names and numeric types. The
dashboard should consume Spring Boot's API, never FastAPI's network address.
