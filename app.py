# app.py
from flask import Flask, render_template, request, send_file
import sqlite3
import pandas as pd
import os
import joblib
from datetime import datetime

app = Flask(__name__)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DB_FILE = "predictions.db"

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError("Run train_model.py first to create model.pkl and scaler.pkl")

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

LIMITS = {
    "ThermalCond": (80, 400),
    "SourceTemp": (40, 75),
    "AmbientTemp": (15, 35),
    "BlockSize": (5, 50)
}

def check_limits(inputs):
    names = ["ThermalCond","SourceTemp","AmbientTemp","BlockSize"]
    return [n for n,v in zip(names, inputs) if not (LIMITS[n][0] <= v <= LIMITS[n][1])]

@app.route("/", methods=["GET", "POST"])
def index():
    warning = None
    if request.method == "POST":
        try:
            tc = float(request.form["thermal_cond"])
            st = float(request.form["source_temp"])
            at = float(request.form["ambient_temp"])
            bs = float(request.form["block_size"])
        except Exception:
            warning = "Please enter valid numbers."
            return render_template("index.html", warning=warning)

        out_of_range = check_limits([tc, st, at, bs])
        if out_of_range:
            warning = f"Out of range: {', '.join(out_of_range)}"
            return render_template("index.html", warning=warning)

        Xs = scaler.transform([[tc, st, at, bs]])
        pred = model.predict(Xs)[0]
        max_temp, avg_temp, center_temp = float(pred[0]), float(pred[1]), float(pred[2])

        efficiency = round((st - avg_temp) / st * 100, 2) if st != 0 else None
        eff_text = "Overheating" if (efficiency is not None and efficiency < -100) else (f"{efficiency}%" if efficiency is not None else "N/A")

        if avg_temp < 30:
            status = "Low – No coolant"
        elif 30 <= avg_temp <= 45 and max_temp < 75:
            status = "Medium – Good"
        elif avg_temp > 45 and max_temp < 75:
            status = "Medium-High – Monitor"
        else:
            status = "High – Coolant required"

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO predictions (ThermalCond, SourceTemp, AmbientTemp, BlockSize,
                                     MaxTemp, AvgTemp, CenterTemp, Efficiency, Status, Timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tc, st, at, bs, max_temp, avg_temp, center_temp, efficiency, status, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        result = {
            "MaxTemp": round(max_temp,2),
            "AvgTemp": round(avg_temp,2),
            "CenterTemp": round(center_temp,2),
            "Efficiency": eff_text,
            "Status": status
        }
        return render_template("result.html", result=result)

    return render_template("index.html", warning=warning)

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY Timestamp DESC LIMIT 200", conn)
    conn.close()
    rows = df.to_dict(orient="records")
    return render_template("history.html", history=rows)

@app.route("/download_history")
def download_history():
    csv_file = "predictions_history.csv"
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY Timestamp DESC", conn)
    conn.close()
    if df.empty:
        return "No history found yet!"
    df.to_csv(csv_file, index=False)
    return send_file(csv_file, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
