import joblib
import numpy as np

model = joblib.load("models/placement_model.pkl")

def predict_placement(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict_proba(features)[0][1]
    return round(prediction * 100, 2)