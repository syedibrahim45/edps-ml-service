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
    description: str
    note: str


class IndividualFeatureEffect(BaseModel):
    feature: str
    input_value: float
    reference_value: float
    probability_with_input: float
    probability_with_reference: float
    probability_delta: float


class IndividualExplanationResponse(BaseModel):
    scope: Literal["individual"]
    method: Literal["single_feature_replacement_sensitivity"]
    prediction: PredictionResponse
    reference_population: str
    feature_effects: list[IndividualFeatureEffect]
    note: str


class EvaluationMetrics(BaseModel):
    accuracy: float
    precision_class_1: float
    recall_class_1: float
    f1_class_1: float
    confusion_matrix_labels_0_1: list[list[int]]


class FederatedRoundMetrics(EvaluationMetrics):
    round: int
    global_parameter_l2_norm: float


class FederatedSimulationResponse(BaseModel):
    status: Literal["completed"]
    method: Literal["FedAvg"]
    model: Literal["binary logistic regression"]
    simulation_data: Literal["synthetic"]
    client_count: int
    rounds: int
    local_epochs: int
    aggregation: str
    preprocessing: str
    training_rows: int
    test_rows: int
    client_rows: list[int]
    per_round_metrics: list[FederatedRoundMetrics]
    final_evaluation_metrics: EvaluationMetrics
    interpretation: str