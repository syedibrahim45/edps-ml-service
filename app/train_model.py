import os

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Create a sample medical dataset
# --------------------------------------------------

data = {
    "age": [25, 32, 45, 52, 61, 29, 40, 55, 67, 35,
            48, 58, 23, 70, 44, 38, 63, 50, 31, 57],

    "gender": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
               1, 0, 1, 0, 1, 1, 0, 1, 0, 1],

    "blood_pressure": [110, 120, 145, 160, 170, 115, 130, 155, 180, 125,
                       150, 165, 105, 185, 135, 128, 175, 140, 118, 158],

    "glucose": [85, 95, 160, 180, 210, 90, 110, 175, 230, 105,
                165, 190, 80, 250, 130, 115, 220, 145, 92, 185],

    "bmi": [21.5, 23.2, 28.5, 31.0, 33.5, 22.0, 25.5, 30.2, 35.0, 24.8,
            29.5, 32.0, 20.5, 36.5, 27.0, 25.0, 34.0, 28.0, 22.5, 31.5],

    "heart_rate": [70, 72, 85, 92, 98, 68, 76, 90, 105, 74,
                   88, 95, 65, 110, 80, 75, 102, 84, 71, 93],

    "cholesterol": [170, 180, 230, 250, 270, 165, 200, 245, 290, 190,
                    235, 260, 160, 310, 215, 195, 280, 225, 175, 255],

    "disease": [0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
                1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
}


df = pd.DataFrame(data)

print("\nDataset:")
print(df)

print("\nDataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop("disease", axis=1)
y = df["disease"]


# --------------------------------------------------
# 3. Split dataset into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. Create the Machine Learning model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# --------------------------------------------------
# 7. Save the trained model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/disease_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully at:")
print(model_path)