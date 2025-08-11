# check_model.py
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
CSV = "heat_transfer_dataset.csv"

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE) or not os.path.exists(CSV):
    print("Ensure model.pkl, scaler.pkl and heat_transfer_dataset.csv exist.")
    raise SystemExit

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
df = pd.read_csv(CSV)
X = df[["ThermalCond","SourceTemp","AmbientTemp","BlockSize"]]
y = df[["MaxTemp","AvgTemp","CenterTemp"]]
Xs = scaler.transform(X)
y_pred = model.predict(Xs)

print("R2 per output:")
for i, col in enumerate(y.columns):
    print(col, r2_score(y.iloc[:,i], y_pred[:,i]))
print("MSE:", mean_squared_error(y, y_pred))
