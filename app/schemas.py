from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    age: int = Field(ge=1, le=120)
    gender: Literal[0, 1]
    blood_pressure: float = Field(ge=40, le=300, allow_inf_nan=False)
    glucose: float = Field(ge=20, le=1000, allow_inf_nan=False)
    bmi: float = Field(ge=5, le=100, allow_inf_nan=False)
    heart_rate: float = Field(ge=20, le=250, allow_inf_nan=False)
    cholesterol: float = Field(ge=50, le=1000, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    disease: Literal["Low Risk", "High Risk"]
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class FeatureImportanceItem(BaseModel):
    feature: str
    mean_importance: float
    std_importance: float


class GlobalFeatureImportanceResponse(BaseModel):
    scope: Literal["global"]
    method: Literal["permutation_importance"]
    scoring: Literal["neg_log_loss"]
    sample_count: int
    feature_importance: list[FeatureImportanceItem]
    note: str