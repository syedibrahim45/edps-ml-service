import joblib


MODEL_PATH = "models/disease_model.pkl"


# Load the trained model once when the application starts
model = joblib.load(MODEL_PATH)


def predict_disease(
    age: int,
    gender: int,
    blood_pressure: float,
    glucose: float,
    bmi: float,
    heart_rate: float,
    cholesterol: float
):
    features = [[
        age,
        gender,
        blood_pressure,
        glucose,
        bmi,
        heart_rate,
        cholesterol
    ]]

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        disease = "High Risk"
    else:
        disease = "Low Risk"

    return {
        "prediction": int(prediction),
        "disease": disease,
        "probability": round(float(probability), 2)
    }