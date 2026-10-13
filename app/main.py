from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import (
    GlobalFeatureImportanceResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.model import (
    ModelUnavailableError,
    global_feature_importance,
    model,
    predict_disease,
)


app = FastAPI(
    title="EDPS Machine Learning Service",
    description=(
        "Academic prototype for the Early Disease Prediction System. "
        "Not a clinical diagnostic system."
    ),
    version="1.0.0"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": [
                {
                    "type": item["type"],
                    "loc": list(item["loc"]),
                    "msg": item["msg"],
                }
                for item in error.errors()
            ]
        },
    )


@app.get("/")
def root():
    return {
        "message": "EDPS ML Service is running"
    }


@app.get("/health")
def health():
    return {
        "status": "UP" if model is not None else "DEGRADED",
        "service": "EDPS ML Service",
        "model_status": "ready" if model is not None else "unavailable",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        result = predict_disease(
            age=request.age,
            gender=request.gender,
            blood_pressure=request.blood_pressure,
            glucose=request.glucose,
            bmi=request.bmi,
            heart_rate=request.heart_rate,
            cholesterol=request.cholesterol,
        )
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable.",
        ) from error

    return result


@app.get(
    "/explainability/global",
    response_model=GlobalFeatureImportanceResponse,
)
def explain_global_features():
    try:
        return global_feature_importance()
    except ModelUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable.",
        ) from error