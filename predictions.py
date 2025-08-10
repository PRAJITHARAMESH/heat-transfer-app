# predictions.py
import joblib
import numpy as np
import os

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# Load once
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError("Run train_model.py first to create model.pkl and scaler.pkl")

_models = joblib.load(MODEL_FILE)   # dict: "MaxTemp","AvgTemp","CenterTemp"
_scaler = joblib.load(SCALER_FILE)

def calculate_efficiency(max_temp):
    base_efficiency = 100
    penalty = max(0, (max_temp - 50) * 2)  # example rule
    return round(max(0, base_efficiency - penalty), 2)

def coolant_suggestion(efficiency):
    if efficiency > 80:
        return "Water - Efficient coolant"
    elif efficiency > 60:
        return "Oil - Moderate coolant"
    elif efficiency > 40:
        return "Special Coolant - Use for high temps"
    else:
        return "Extreme Coolant - Urgently needed"

def metal_recommendation(max_temp):
    if max_temp < 50:
        return "Aluminum - Good heat dissipation"
    elif max_temp < 70:
        return "Copper - Better heat conduction"
    else:
        return "Silver or Copper Alloys - Best for very high heat"

def run_prediction(thermal_cond, source_temp, ambient_temp, block_size):
    # features in the same order used for training
    features = np.array([[thermal_cond, source_temp, ambient_temp, block_size]])
    features_scaled = _scaler.transform(features)

    max_pred = float(_models["MaxTemp"].predict(features_scaled)[0])
    avg_pred = float(_models["AvgTemp"].predict(features_scaled)[0])
    center_pred = float(_models["CenterTemp"].predict(features_scaled)[0])

    efficiency = calculate_efficiency(max_pred)
    coolant = coolant_suggestion(efficiency)
    material = metal_recommendation(max_pred)

    return {
        "pred_max": round(max_pred, 2),
        "pred_avg": round(avg_pred, 2),
        "pred_center": round(center_pred, 2),
        "efficiency": efficiency,
        "coolant": coolant,
        "material": material
    }
