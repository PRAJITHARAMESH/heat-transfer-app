# check_model.py
import joblib
m = joblib.load("model.pkl")
print("type:", type(m))
try:
    print("keys:", list(m.keys()))
except Exception as e:
    print("not a dict:", e)
