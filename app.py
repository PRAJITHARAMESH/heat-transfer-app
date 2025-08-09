from flask import Flask, render_template, request, send_file
import sqlite3
import pandas as pd
import os
import joblib

app = Flask(__name__)

# Load ML model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Create DB table if not exists
conn = sqlite3.connect("predictions.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions
             (thermal_cond REAL, source_temp REAL, ambient_temp REAL, block_size REAL,
              pred_max REAL, pred_avg REAL, pred_center REAL)''')
conn.commit()
conn.close()

# ---------- HOME PAGE ----------
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return predict()
    return render_template('index.html')

# ---------- PREDICT ----------
def predict():
    # Form data
    thermal_cond = float(request.form['thermal_cond'])
    source_temp = float(request.form['source_temp'])
    ambient_temp = float(request.form['ambient_temp'])
    block_size = float(request.form['block_size'])

    # Scale & predict
    features = scaler.transform([[thermal_cond, source_temp, ambient_temp, block_size]])
    pred_max, pred_avg, pred_center = model.predict(features)[0]

    # Save to DB
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
              (thermal_cond, source_temp, ambient_temp, block_size,
               pred_max, pred_avg, pred_center))
    conn.commit()
    conn.close()

    return render_template('index.html',
                           prediction_max=round(pred_max, 2),
                           prediction_avg=round(pred_avg, 2),
                           prediction_center=round(pred_center, 2))

# ---------- HISTORY PAGE ----------
@app.route('/history', methods=['GET'])
def history():
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return render_template('history.html', tables=[df.to_html(classes='data', header="true", index=False)])

# ---------- DOWNLOAD CSV ----------
@app.route('/download_history', methods=['GET'])
def download_history():
    csv_file = "predictions_history.csv"

    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    if df.empty:
        return "No history found yet!"

    df.to_csv(csv_file, index=False)
    return send_file(csv_file, as_attachment=True)

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(debug=True)
