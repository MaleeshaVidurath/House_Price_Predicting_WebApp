from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
#test comment
app = FastAPI()

# Load the trained model
model = joblib.load("house_price_predictor_model.pkl")

# Define the expected feature names
FEATURE_NAMES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN",
    "bedroom_ratio", "household_rooms"
]

@app.post("/predict")
def predict_price(features: dict):
    # Check if all required features are present
    missing_features = [f for f in FEATURE_NAMES if f not in features]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")

    # Convert input data into a NumPy array
    input_data = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)
    return {"predicted_price": prediction[0]}
