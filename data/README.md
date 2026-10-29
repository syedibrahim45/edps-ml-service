# Diabetes Dataset Validation

The full binary CDC Diabetes Health Indicators CSV is expected at
`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`. Files under
`data/raw/` are excluded from Git; do not stage or commit the dataset.

## Intended workflow

The full binary file has a separate explicit schema in
`app/diabetes_schema.py`; it does not change the generic prediction model or
API. It configures the 21 named input features and `Diabetes_binary` target.
The validation command is:

```powershell
.\venv\Scripts\python.exe -m app.validate_diabetes_dataset
```

Binary/ordinal fields are checked against explicit float-valued category
allowlists. This preserves the CSV's numeric representation: no integer
conversion or coercion occurs, and fractional/unlisted codes fail validation.
BMI is validated as a finite real numeric feature, with its observed minimum
and maximum reported; no clinical BMI threshold is imposed. The target must
be exactly `0.0` or `1.0`. The validator reports the exact header order,
types, missing values, duplicate rows, categories, target counts, feature
ranges, and validation errors. It never imputes, removes rows, rebalances
classes, or modifies the input data.

Do not treat the validator's software-test fixtures as health data.

## Source, license, and citation

Dataset: CDC Diabetes Health Indicators, UCI Machine Learning Repository,
dataset 891, DOI [10.24432/C53919](https://doi.org/10.24432/C53919).
The Kaggle dataset page displays a CC0 license, as confirmed for this
milestone. Retain the dataset citation and source links in academic
documentation even where CC0 does not require attribution:

- [UCI dataset record](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [Kaggle dataset page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- [CDC 2015 BRFSS data and documentation](https://www.cdc.gov/brfss/annual_data/annual_2015.html)

The supplied full binary file is the 2015-named 253,680-row, 22-column
processed file. CDC's raw 2015 BRFSS release has a different record count;
the exact UCI/Kaggle processing and selection relationship is not established
by this validation. This milestone verifies the local file against the
confirmed schema and summary properties, not its provenance or license
independently.

## EDPS fields are not survey substitutes

The current EDPS form contains measured or patient-entered values such as
numeric glucose, blood pressure, cholesterol, and BMI. Survey fields such as
self-reported high blood pressure or high cholesterol are different
measurements with different definitions. They must not be derived from EDPS
clinical measurements without an explicit, source-supported mapping. Missing
survey responses must not be inferred or silently imputed.

## Tests

Run both focused validator test modules with the existing environment:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_diabetes_data.py tests\test_diabetes_schema.py -q
```

The tests use only synthetic in-memory DataFrames created as software-test
examples; they are not medical or model-training data.

## Local dataset validation results

Validated the supplied local file with
`.\venv\Scripts\python.exe -m app.validate_diabetes_dataset`:

| Check | Result |
|---|---:|
| Rows / columns | 253,680 / 22 |
| Exact configured column order | Match |
| Pandas data type for all columns | `float64` |
| `Diabetes_binary` = `0.0` / `1.0` | 218,334 / 35,346 |
| Missing values | 0 |
| Identical duplicate rows beyond first occurrences | 24,206 |
| Invalid categories or target encodings | 0 |
| Validation errors | 0 |

Observed ranges:

- Binary indicators: 0.0–1.0.
- `GenHlth`: 1.0–5.0; `MentHlth` and `PhysHlth`: 0.0–30.0.
- `Age`: 1.0–13.0; `Education`: 1.0–6.0; `Income`: 1.0–8.0.
- `BMI`: 12.0–98.0.

The schema allowlists reject fractional category codes. BMI is checked as
finite numeric data and its observed range is reported; the observed minimum
and maximum are not being asserted as clinical validity thresholds. Duplicate
rows are reported, not removed; the observed class distribution is not
rebalanced.

Focused tests: 42 passed. Complete ML test suite: 70 passed with 802 existing
NumPy/joblib deprecation warnings during model artifact loading.

## Separate experimental model training

The training experiment is isolated in `app/train_diabetes_model.py`. Run it
only after the data validator passes:

```powershell
.\venv\Scripts\python.exe -m app.train_diabetes_model
```

It uses the configured 21 features, groups identical feature vectors before a
deterministic 70/15/15 train/validation/test split, and chooses the split
candidates closest to the requested row fractions and overall positive-class
prevalence. All preprocessing is fitted within each training pipeline.
Logistic Regression and bounded Random Forest candidates compare unweighted
and class-weighted training. Validation selects the model and class-1 F1
threshold; the selected model is evaluated once on the held-out test set.
Artifacts and metadata go only under `models/diabetes_binary/`, which is
Git-ignored. This experimental model is not used by an API and is not for
clinical decisions.

## Future diabetes endpoint BMI input policy

For any future diabetes inference endpoint, the proposed BMI input policy is:

- Accept finite numeric BMI values only.
- Reject strings, booleans, NaN, infinity, and malformed inputs.
- Do not silently coerce input types.
- The observed dataset BMI range is 12–98; this is a dataset observation,
  not a clinical validity threshold.
- Do not impose a new BMI range restriction without separate approval.

This documents an input-validation policy only; no diabetes API endpoint is
implemented by this milestone.

### Experiment run

The fixed-seed grouped split produced:

| Partition | Rows | Class 0 | Class 1 |
|---|---:|---:|---:|
| Train | 177,566 | 152,834 | 24,732 |
| Validation | 38,041 | 32,734 | 5,307 |
| Test | 38,073 | 32,766 | 5,307 |

There are 24,206 exact duplicate full rows and 25,772 repeated feature-vector
rows beyond unique vectors. The split groups identical feature vectors, so
none cross the train/validation/test boundaries. Duplicate records remain in
the data. The test target distribution was used only for split construction
and reporting; candidate selection used validation metrics, and test
predictions were produced only for the selected model.

The selected candidate was weighted Logistic Regression. Its validation
threshold was selected to maximize class-1 F1:

| Candidate | Validation threshold | F1 (class 1) | Balanced accuracy | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.5000 | 0.0000 | 0.5000 | 0.5000 | 0.1395 |
| Logistic Regression, unweighted | 0.2125 | 0.4688 | 0.7307 | 0.8259 | 0.4191 |
| Logistic Regression, balanced | 0.6447 | 0.4690 | 0.7254 | 0.8262 | 0.4157 |
| Random Forest, unweighted | 0.2254 | 0.4647 | 0.7231 | 0.8214 | 0.4273 |
| Random Forest, balanced | 0.6223 | 0.4657 | 0.7212 | 0.8235 | 0.4259 |

At the selected threshold, the untouched test metrics were: accuracy 0.8039,
balanced accuracy 0.7192, precision 0.3737, sensitivity/recall 0.6017,
specificity 0.8367, F1 0.4610, ROC-AUC 0.8252, and average precision 0.4037.
The confusion matrix (true rows, predicted columns; class order 0, 1) was
`[[27414, 5352], [2114, 3193]]`. Positive-class prevalence was 0.1394.

Candidate fit times in seconds were: Logistic Regression unweighted 9.33,
Logistic Regression balanced 29.82, Random Forest unweighted 119.43, and
Random Forest balanced 154.96. Validation inference took 0.25, 0.47, 1.74,
and 0.78 seconds respectively. The selected model's test inference took
0.19 seconds for 38,073 rows. These are local runtime measurements, not
performance guarantees.

Selected model:
`models/diabetes_binary/selected_model.joblib`
(SHA-256 `6d87748f06873074bdda1303925cf37bb906542e030ebd66f15beb1648f4655c`).
The accompanying `training_metadata.json` records the feature schema,
parameters, split counts, validation comparison, threshold, test metrics, and
artifact hash. These experimental dataset-label results are not clinical
calibration, medical validity, or diagnostic accuracy.
