# EDPS ML Service Test Report

Execution date: 2026-10-14
Environment: Windows, Python 3.13.1, pytest 9.1.1, scikit-learn 1.9.0
Input data: 20 hand-authored synthetic records; no real patient data
Active artifact: `models/disease_model.pkl` (SHA-256
`1ACA8A37CE0CD4C070D0E9F4442AC698086D61AF699AB4C1B18F2E591A5C0888`)

## Automated tests

Baseline before changes:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Result: **27 passed** in 349.23 seconds.

Full suite after changes:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Result: **28 passed** in 87.90 seconds. Both runs reported 802
NumPy/joblib `DeprecationWarning`s from deserializing the existing model
artifact; neither run had test failures.

Coverage includes health/root and prediction APIs, strict request validation,
missing/extra fields, unsupported values, numeric bounds, non-finite inputs,
missing/corrupt/incompatible artifact handling, HTTP 503 for unavailable
models, probability bounds, deterministic prediction, global importance
cache reuse, individual explanation/input correspondence, no implicit model
replacement, weighted FedAvg, deterministic simulation, and cached
federated dashboard output.

## Active model evaluation

The existing artifact loaded as a `RandomForestClassifier` with the
expected seven-feature order. It was not retrained or replaced. Evaluating
the active artifact on the deterministic stratified split (`random_state=42`)
produced:

| Measurement | Observed value |
| --- | ---: |
| Dataset rows | 20 |
| Class counts | 0: 8, 1: 12 |
| Train / test rows | 16 / 4 |
| Test class counts | 0: 2, 1: 2 |
| Accuracy | 1.0 |
| Class-1 precision | 1.0 |
| Class-1 recall | 1.0 |
| Class-1 F1 | 1.0 |
| Confusion matrix (labels 0, 1) | `[[2, 0], [0, 2]]` |

These are results on only four held-out synthetic records, not a meaningful
estimate of performance on people or a clinical validation.

## API smoke results

Using FastAPI `TestClient` with the sample request in `ML_API_CONTRACT.md`,
the following routes all returned HTTP 200: `/`, `/health`, `/predict`,
`/explainability/global`, `/explainability/individual`, and
`/federated/results`.

Observed prediction:

```json
{"prediction": 0, "disease": "Low Risk", "probability": 0.0}
```

The response contains exactly the existing three prediction fields. Repeating
the identical input returns the same output. The class-1 probability is a
finite value within `[0, 1]`.

Observed global feature importance (sorted descending, permutation
importance with `neg_log_loss`, 20 synthetic rows):

| Feature | Mean importance | Std. deviation |
| --- | ---: | ---: |
| heart_rate | 0.1216960079 | 0.0286903985 |
| cholesterol | 0.1183679063 | 0.0277561690 |
| age | 0.0860629332 | 0.0198319963 |
| blood_pressure | 0.0806118421 | 0.0188862237 |
| bmi | 0.0665829112 | 0.0155617630 |
| glucose | 0.0463477424 | 0.0109741193 |
| gender | 0.0 | 0.0 |

This is a global ranking on the tiny synthetic dataset. It is not
patient-specific, causal, or an independent validation result.

Observed individual explanation for the sample request:

| Feature | Input | Synthetic reference | Input probability | Reference probability | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| age | 35 | 46.5 | 0.0 | 0.18 | -0.18 |
| gender | 0 | 1 | 0.0 | 0.0 | 0.0 |
| blood_pressure | 125 | 142.5 | 0.0 | 0.16 | -0.16 |
| glucose | 105 | 152.5 | 0.0 | 0.09 | -0.09 |
| bmi | 24.8 | 28.25 | 0.0 | 0.13 | -0.13 |
| heart_rate | 74 | 84.5 | 0.0 | 0.22 | -0.22 |
| cholesterol | 190 | 227.5 | 0.0 | 0.22 | -0.22 |

The returned prediction matches `/predict` for the submitted values. These
are single-feature replacement sensitivities, not additive contributions,
causal effects, or SHAP values.

The first global permutation calculation took 37.4 seconds in this Windows
environment. Its result is now cached within each process; the suite verifies
repeated endpoint requests calculate it only once. FedAvg results were
returned in 6.88 seconds on the first API request and are cached per process.

## FedAvg experiment

The API returned `status: "completed"`, method `FedAvg`, model
`binary logistic regression`, `simulation_data: "synthetic"`, 3 simulated
clients, 10 rounds, 5 local epochs, 16 training rows, and 4 test rows.
The deterministic client partition was `[6, 5, 5]`. The response included
metrics for every round and final held-out evaluation:

| Round | Accuracy | Precision | Recall | F1 | Confusion matrix | Parameter L2 norm |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 0.399811 |
| 2 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 0.600783 |
| 3 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 0.742289 |
| 4 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 0.861672 |
| 5 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 0.970552 |
| 6 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 1.072977 |
| 7 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 1.170566 |
| 8 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 1.264110 |
| 9 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 1.354086 |
| 10 | 0.5 | 0.5 | 1.0 | 0.6667 | `[[0, 2], [0, 2]]` | 1.440837 |

Final evaluation metrics were accuracy 0.5, class-1 precision 0.5, recall
1.0, F1 0.6667, and confusion matrix `[[0, 2], [0, 2]]`. These results use
four test rows. The deterministic tests compare complete simulation results
for repeated runs with the same seed. The clients are local simulations,
not hospitals; this experiment does not train or replace the Random Forest.

## Invalid input, missing model, and timeouts

Invalid/missing/extra/wrong-type/out-of-range request data returned JSON HTTP
422. When the model was marked unavailable, `/health` returned
`DEGRADED` / `unavailable`, and prediction plus both explainability routes
returned HTTP 503 without exposing traceback details. The loader tests cover
missing, corrupt, and incompatible artifacts.

FastAPI does not set the calling Spring Boot HTTP client's timeout. Configure
a connection timeout around 5 seconds and at least a 60-second read timeout
for the first global-importance request, or warm that route before use.
See `ML_API_CONTRACT.md` for error translation and timeout guidance.

## GitHub Actions and container validation

The latest remote `Python CI` run on the current base revision
`d08483dcd0e2be933c587ab0daea1b3497b2655a` completed successfully:
[workflow run 38425200276](https://github.com/syedibrahim45/edps-ml-service/actions/runs/38425200276).
The working-tree changes in this report have not been pushed, so that remote
result does not validate these local edits; the full local pytest run above
does.

A Dockerfile and `.dockerignore` were added. A local image build could not be
run because the Docker CLI is not installed in this environment.

## Limitations

All records and labels are synthetic and hand-authored. No real patient
data is included or appropriate for this prototype. The dataset has only
20 rows and is not clinical ground truth. Prediction labels/probabilities,
evaluation metrics, global importance, individual sensitivities, and FedAvg
results are not clinically validated. The FedAvg clients are in-process
simulations, not hospitals; the experiment provides no real distributed
training, privacy, or secure aggregation. Do not use this service for
diagnosis, triage, treatment, or other medical decisions.
