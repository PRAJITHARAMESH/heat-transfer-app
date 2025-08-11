# predictions.py
import joblib
import sys
import os

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
    print("Run train_model.py first.")
    sys.exit(1)

import numpy as np
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

if len(sys.argv) != 5:
    print("Usage: python predictions.py <ThermalCond> <SourceTemp> <AmbientTemp> <BlockSize>")
    sys.exit(1)

tc, st, at, bs = map(float, sys.argv[1:])
x = scaler.transform([[tc, st, at, bs]])
pred = model.predict(x)[0]
print("MaxTemp, AvgTemp, CenterTemp:", [round(float(p),2) for p in pred])
