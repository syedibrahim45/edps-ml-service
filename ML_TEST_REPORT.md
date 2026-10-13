# EDPS ML Service Test Report

Execution date: 2026-10-13  
Environment: Windows, Python 3.13.1, pytest 9.1.1, scikit-learn 1.9.0  
Data: 20 hand-authored synthetic records; no real patient data

## Automated Tests

Command executed:

```powershell
venv\Scripts\python.exe -m pytest -v
```

Result: **27 passed** in 78.32 seconds. Pytest reported 802 NumPy/joblib
deprecation warnings from serialized estimator loading; there were no test
failures. Coverage includes root/health/predict responses, missing and
unsupported fields, strict data types, numeric bounds and non-finite values,
unavailable/missing/corrupt/incompatible models, probability bounds, global
and individual explanations, no implicit model replacement, weighted FedAvg,
reproducibility, parameter updates across rounds, and cached dashboard output.

## Prediction Model Evaluation

The active `models/disease_model.pkl` loaded successfully as a binary
`RandomForestClassifier` with the expected input order:
`age`, `gender`, `blood_pressure`, `glucose`, `bmi`, `heart_rate`,
`cholesterol`. It was not replaced. Both this active artifact and the separate
deterministically trained candidate were evaluated on the same stratified
split produced by `random_state=42`:

| Measurement | Observed value |
| --- | ---: |
| Dataset rows | 20 |
| Class counts | 0: 8, 1: 12 |
| Training / test rows | 16 / 4 |
| Test class counts | 0: 2, 1: 2 |
| Accuracy | 1.0 |
| Class-1 precision | 1.0 |
| Class-1 recall | 1.0 |
| Class-1 F1 | 1.0 |
| Confusion matrix (labels 0, 1) | `[[2, 0], [0, 2]]` |

Synthetic feature ranges and means observed from the 20 records:

| Feature | Minimum | Maximum | Mean |
| --- | ---: | ---: | ---: |
| age | 23 | 70 | 46.15 |
| gender | 0 | 1 | 0.55 |
| blood_pressure | 105 | 185 | 143.45 |
| glucose | 80 | 250 | 150.6 |
| bmi | 20.5 | 36.5 | 28.085 |
| heart_rate | 65 | 110 | 84.65 |
| cholesterol | 160 | 310 | 225.0 |

These scores describe only four synthetic test rows and are not useful
estimates of real-world or clinical performance. No external dataset was
downloaded or used because none was reviewed for licensing, provenance, and
appropriate use in this work.

## Federated Simulation

Command executed:

```powershell
venv\Scripts\python.exe -m app.federated_simulation
```

Observed: FedAvg completed with 3 simulated clients, 10 rounds, 5 local epochs
per round, and 16 training rows partitioned as `[6, 5, 5]`; the held-out set
contains 4 synthetic rows. Every round recorded metrics and a positive global
parameter L2 norm, increasing from 0.3998 after round 1 to 1.4408 after round
10. Final metrics: accuracy 0.5, class-1 precision 0.5, recall 1.0, F1 0.6667,
confusion matrix `[[0, 2], [0, 2]]`. Repeating the same seed produced identical
results in pytest. These are simulated clients in one process, not hospitals;
this logistic-regression experiment does not federate the API's Random Forest.

## HTTP Smoke Check

Using FastAPI `TestClient` with the documented seven-field payload, observed
HTTP 200 for `/`, `/health`, `/predict`, `/explainability/global`,
`/explainability/individual`, and `/federated/results`. The prediction response
contained exactly `prediction`, `disease`, and `probability`; the observed
probability was `0.0`, within the documented `[0, 1]` range. Invalid inputs
return JSON HTTP 422 details, and an unavailable prediction model returns a
JSON HTTP 503 response without exposing internal exception text.

## Limitations

The synthetic dataset is tiny, manually authored, and not clinical ground
truth. No model, threshold, probability, feature importance, individual
sensitivity, or federated result is clinically validated. Do not use this
prototype for diagnosis, triage, treatment, or real patient data.