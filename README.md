# EDPS ML Service

This repository contains the machine-learning service for the VTU Phase 2
Early Disease Prediction System project. It is an **academic prototype, not a
clinical diagnostic system**. Do not use its outputs for medical decisions.

## Model and Data

The active inference artifact remains `models/disease_model.pkl`, loaded once
by `app.model`. The separate `models/disease_model.joblib` file is retained but
is not the artifact used by the API. Model paths are resolved from the source
tree, not the process working directory.

The separate experimental diabetes model is loaded by `app.diabetes_model`
and served only by `POST /predict/diabetes`. It does not replace or serve as a
fallback for the generic model. It classifies the dataset-defined binary
survey label (0: no diabetes reported; 1: prediabetes or diabetes reported),
not a clinical diagnosis. Its class-1 score is not a clinically calibrated
probability, and its selected threshold is loaded from the diabetes training
metadata. See [`ML_API_CONTRACT.md`](ML_API_CONTRACT.md) for all 21 required
survey inputs, allowed values, response example, and error behavior.

The diabetes artifact is a trusted joblib/scikit-learn serialization. The
loader checks metadata consistency and the artifact SHA-256 before loading;
never replace it with an uploaded or otherwise untrusted model. Training-time
dependency versions were not recorded. The ignored
`models/diabetes_binary/compatibility.json` records only the versions verified
in the current environment, not claimed training versions. The Docker image
does not package the ignored diabetes artifacts or raw dataset; a deployment
that enables this endpoint must provide the trusted diabetes artifact bundle
as a read-only mount at `models/diabetes_binary/`.

### Diabetes endpoint

`POST /predict/diabetes` requires these 21 survey features:

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

Binary fields accept strict integer codes 0 or 1. `GenHlth` accepts 1–5;
`MentHlth` and `PhysHlth` accept 0–30; age bucket `Age` accepts 1–13;
`Education` accepts 1–6; and `Income` accepts 1–8. All fields are required,
and extra fields, numeric strings, booleans, fractional codes, and non-finite
numbers are rejected with HTTP 422. BMI accepts a finite numeric value without
silent coercion or a configured range limit; the dataset's observed 12–98
range is not a clinical threshold.

The response includes the dataset-defined class and score:

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

Class 0 means no diabetes was reported in the dataset label; class 1 means
prediabetes or diabetes was reported in that label. Class 1 is not a confirmed
diagnosis. `positive_class_score` is not a clinically calibrated probability.
The predicted class is 1 when the score is greater than or equal to the
validation-selected threshold read from trusted metadata. If the diabetes
model is unavailable, the endpoint returns generic HTTP 503 and does not fall
back to the generic EDPS model. `GET /health/diabetes` reports the separate
model's readiness; the existing `GET /health` contract is unchanged. For the
full request/response contract and compatibility caveats, see
[`ML_API_CONTRACT.md`](ML_API_CONTRACT.md).

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

Observed output in the current environment (scikit-learn 1.9.0): the training
routine creates a new candidate forest and evaluates 16 training rows and 4
test rows (2 per class); confusion matrix `[[2, 0], [0, 2]]`; accuracy 1.0;
class-1 precision 1.0, recall 1.0, and F1 1.0. The active `disease_model.pkl`
was separately checked against the same split and produced those same metrics.
These are descriptive results on four synthetic test records, **not reliable
performance estimates**. Full executed results are recorded in
[`ML_TEST_REPORT.md`](ML_TEST_REPORT.md).

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
negative log loss on all 20 synthetic records. Its deterministic result is
cached in-process after the first calculation so dashboard polls do not repeat
the calculation. A larger positive score means shuffling that feature
worsened log loss for this fitted model on this dataset; it is not a causal
effect or clinical relevance measure. This is not an independent validation
set or a per-prediction explanation. Importance values on this tiny synthetic
set are exploratory only.

`POST /explainability/individual` accepts the same seven-field JSON request as
`/predict` and returns the ordinary prediction plus input-specific,
one-feature replacement sensitivity values. Each field compares the input
probability with the probability after replacing just that feature with its
synthetic median/mode while leaving the other inputs fixed. These effects are
not additive, causal, SHAP values, or medical explanations.

`GET /federated/results` returns the default 3-client, 10-round experimental
run, with metrics after every communication round and final held-out metrics.
Its result is cached in-process after the first request, so later dashboard
polls do not retrain the experiment. This is a single-process synthetic
simulation; its clients are not hospitals, and its logistic-regression FedAvg
model is not the Random Forest used by `/predict`.

## Service integration and deployment

Use **React → Spring Boot → FastAPI**. Keep FastAPI on a private network and
let Spring Boot map/validate frontend data, call the ML service, and translate
its HTTP errors; do not expose FastAPI directly to the browser. See
[`ML_API_CONTRACT.md`](ML_API_CONTRACT.md) for request/response shapes and
backend timeout handling and [`DEPLOYMENT.md`](DEPLOYMENT.md) for Docker and
local startup instructions. The included Dockerfile runs as a non-root user,
copies the generic active model and the separate trusted diabetes model files,
and does not include the raw diabetes dataset.

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

The request names and numeric types above are the Spring Boot integration
contract. `gender` accepts only integer `0` or `1`; this service's dataset does
not define which value represents Male or Female. Spring Boot must apply the
same agreed mapping consistently. The `probability` field is a fraction in
`[0, 1]`, not a percentage, and all three existing response keys remain
unchanged.

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

Run the tests (including API, model compatibility, explainability, and FedAvg
checks):

```powershell
python -m pytest -v
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

## Academic and Clinical Limitations

All example data is synthetic and the dataset contains only 20 hand-authored
rows. No publicly licensed external dataset is included: none has been
reviewed and approved for this prototype. Neither the prediction model nor
the simulated experiment has clinical validation, representative population
coverage, calibration evidence, privacy guarantees, or evidence of safe use in
care. Do not submit real patient information or use outputs for diagnosis,
triage, or treatment decisions.