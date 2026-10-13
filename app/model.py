from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.train_model import FEATURE_NAMES, create_synthetic_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "disease_model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """Load and validate the serialized binary classifier."""
    loaded_model = joblib.load(model_path)
    if not callable(getattr(loaded_model, "predict", None)):
        raise ValueError("The serialized model does not support prediction.")
    if not callable(getattr(loaded_model, "predict_proba", None)):
        raise ValueError("The serialized model does not provide probabilities.")
    if set(int(label) for label in loaded_model.classes_) != {0, 1}:
        raise ValueError("The serialized model must contain binary classes 0 and 1.")
    if getattr(loaded_model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise ValueError("The serialized model must accept exactly seven features.")
    stored_feature_names = getattr(loaded_model, "feature_names_in_", None)
    if stored_feature_names is not None and tuple(stored_feature_names) != FEATURE_NAMES:
        raise ValueError("The serialized model feature order does not match the API.")
    return loaded_model


try:
    model = load_model()
    MODEL_LOAD_ERROR = None
except Exception as error:
    model = None
    MODEL_LOAD_ERROR = type(error).__name__


class ModelUnavailableError(RuntimeError):
    """Raised when inference or explanations cannot run without a model."""


def predict_disease(
    age: int,
    gender: int,
    blood_pressure: float,
    glucose: float,
    bmi: float,
    heart_rate: float,
    cholesterol: float
):
    if model is None:
        raise ModelUnavailableError("The prediction model is unavailable.")

    features = pd.DataFrame(
        [[age, gender, blood_pressure, glucose, bmi, heart_rate, cholesterol]],
        columns=FEATURE_NAMES,
    )

    prediction = int(model.predict(features)[0])
    if prediction not in (0, 1):
        raise ValueError("The prediction model returned a non-binary class.")

    positive_class_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(features)[0][positive_class_index])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The prediction model returned an invalid probability.")

    disease = "High Risk" if prediction == 1 else "Low Risk"

    return {
        "prediction": prediction,
        "disease": disease,
        "probability": probability,
    }


def global_feature_importance() -> dict:
    """Calculate global permutation importance on the bundled synthetic data."""
    if model is None:
        raise ModelUnavailableError("The prediction model is unavailable.")

    dataset = create_synthetic_dataset()
    features = dataset.loc[:, FEATURE_NAMES]
    target = dataset["disease"]
    result = permutation_importance(
        model,
        features,
        target,
        n_repeats=20,
        random_state=42,
        scoring="neg_log_loss",
    )
    importances = [
        {
            "feature": name,
            "mean_importance": float(mean),
            "std_importance": float(std),
        }
        for name, mean, std in zip(
            FEATURE_NAMES,
            result.importances_mean,
            result.importances_std,
            strict=True,
        )
    ]
    importances.sort(key=lambda item: item["mean_importance"], reverse=True)
    return {
        "scope": "global",
        "method": "permutation_importance",
        "scoring": "neg_log_loss",
        "sample_count": len(dataset),
        "feature_importance": importances,
        "note": (
            "This is global feature importance calculated on the full synthetic "
            "dataset, not an explanation for an individual prediction or an "
            "independent validation result."
        ),
    }