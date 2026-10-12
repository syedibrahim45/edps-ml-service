from fastapi import FastAPI

from app.schemas import PredictionRequest, PredictionResponse
from app.model import predict_disease


app = FastAPI(
    title="EDPS Machine Learning Service",
    description="Early Disease Prediction System - ML Service",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "EDPS ML Service is running"
    }


@app.get("/health")
def health():
    return {
        "status": "UP",
        "service": "EDPS ML Service"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    result = predict_disease(
        age=request.age,
        gender=request.gender,
        blood_pressure=request.blood_pressure,
        glucose=request.glucose,
        bmi=request.bmi,
        heart_rate=request.heart_rate,
        cholesterol=request.cholesterol
    )

    return result