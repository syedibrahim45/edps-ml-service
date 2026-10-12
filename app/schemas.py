from pydantic import BaseModel


class PredictionRequest(BaseModel):
    age: int
    gender: int
    blood_pressure: float
    glucose: float
    bmi: float
    heart_rate: float
    cholesterol: float


class PredictionResponse(BaseModel):
    prediction: int
    disease: str
    probability: float