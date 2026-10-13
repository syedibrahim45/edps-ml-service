# EDPS ML Service

This repository contains the machine-learning service for the VTU Phase 2
Early Disease Prediction System project. It is an **academic prototype, not a
clinical diagnostic system**. Do not use its outputs for medical decisions.

## Model and Data

The active inference artifact remains `models/disease_model.pkl`, loaded once
by `app.model`. The separate `models/disease_model.joblib` file is retained but
is not the artifact used by the API. Model paths are resolved from the source
tree, not the process working directory.

`app.train_model.create_synthetic_dataset()` contains the original 20
hand-authored synthetic records (8 label 0, 12 label 1). They are illustrative
values and labels, not observations from patients, a validated clinical
dataset, or ground truth. The labels do not establish disease diagnosis or
medical risk. The dataset is too small and artificial to support reliable
generalization or clinical performance conclusions.

Training is deterministic (`random_state=42` by default), uses a stratified
80/20 split, and reports a confusion matrix, class-1 precision/recall/F1,
accuracy, and a classification report. The training command only evaluates by
default; it does **not** replace the API model. To explicitly write a candidate
artifact, pass `--output models/candidate_disease_model.joblib` and review it
before any deployment change.

Observed output in the current environment (scikit-learn 1.9.0): 16 training
rows and 4 test rows (2 per class); confusion matrix `[[2, 0], [0, 2]]`; accuracy
1.0; class-1 precision 1.0, recall 1.0, and F1 1.0. These are descriptive
results on four synthetic test records, **not reliable performance estimates**.

## Experimental Federated Simulation

Run `python -m app.federated_simulation` to simulate three clients. Each client
trains a binary logistic-regression model locally from the same global
initialization. Ten rounds of local updates are followed by sample-count-
weighted FedAvg: each client's coefficient vector and intercept are multiplied
by its local sample count, summed, and divided by the total client sample
count. Fixed public feature bounds are used for scaling; no global
data-fitted scaler is shared. The simulation uses one process and synthetic
data. It does not implement networked clients, secure aggregation, differential
privacy, or production federated learning. The active Random Forest inference
artifact is not averaged or replaced.

Observed default simulation output: client partition sizes `[6, 5, 5]`;
confusion matrix `[[0, 2], [0, 2]]`; accuracy 0.5; class-1 precision 0.5,
recall 1.0, and F1 0.6667. The four-record test set is too small for a reliable
comparison or performance conclusion.

## Explainability

`GET /explainability/global` calculates global permutation importance using
negative log loss on all 20 synthetic records and returns one measured
probability-sensitivity score per feature. A larger positive score means
shuffling that feature worsened log loss for this fitted model on this dataset;
it is not a causal effect or clinical relevance measure. This is not an
independent validation set and not a per-prediction explanation. No individual
explanation is returned because the service does not calculate one. Importance
values on this tiny synthetic set are exploratory only.

## API Contract

`POST /predict` retains the Spring Boot-facing response keys:

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

All seven fields are required. `gender` is encoded as integer `0` or `1` (the
dataset does not define a real-world gender mapping). Numeric values must be
finite and within these input bounds: age 1-120; blood pressure 40-300; glucose
20-1000; BMI 5-100; heart rate 20-250; cholesterol 50-1000. Unknown fields and
coercions from strings are rejected with HTTP 422. These broad bounds are
input-validation guardrails only, not clinically validated thresholds.

Successful `POST /predict` response:

```json
{
  "prediction": 0,
  "disease": "Low Risk",
  "probability": 0.0
}
```

`prediction` is binary (`0` or `1`), `disease` is the existing human-readable
label (`Low Risk` or `High Risk`), and `probability` is the model's class-1
probability from 0 through 1. These labels and probabilities are prototype
outputs, not diagnoses or calibrated clinical risk.

`GET /health` preserves `status` and `service` and adds `model_status`:

```json
{
  "status": "UP",
  "service": "EDPS ML Service",
  "model_status": "ready"
}
```

If the model cannot load, health reports `DEGRADED` / `unavailable` and
prediction/explainability routes return HTTP 503. The response does not expose
paths, exception text, or configuration values.

## Windows PowerShell Setup

Run these commands from the repository directory:

```powershell
Set-Location D:\Aca-proj\edps-ml-service
py -3.13 -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual-environment executable
directly instead of changing the machine execution policy:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train and print evaluation metrics (does not overwrite the inference model):

```powershell
python -m app.train_model
```

Optionally save a candidate artifact for review:

```powershell
python -m app.train_model --output models/candidate_disease_model.joblib
```

Run the tests:

```powershell
python -m pytest
```

Run the experimental federated simulation:

```powershell
python -m app.federated_simulation
```

Start FastAPI on port 8000:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.