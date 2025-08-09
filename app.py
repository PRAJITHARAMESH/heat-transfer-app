from flask import Flask, render_template, request
import joblib
import sqlite3

app = Flask(__name__)

# ----------------------------
# 1. Limits for input validation
# ----------------------------
LIMITS = {
    "ThermalCond": (80, 400),
    "SourceTemp": (40, 75),  # Avoid overheating cases
    "AmbientTemp": (15, 35),
    "BlockSize": (5, 50)
}

def check_limits(inputs):
    names = ["ThermalCond", "SourceTemp", "AmbientTemp", "BlockSize"]
    out_of_range = [n for n, v in zip(names, inputs) if not (LIMITS[n][0] <= v <= LIMITS[n][1])]
    return out_of_range

# ----------------------------
# Load Model & Scaler
# ----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs
    thermal_cond = float(request.form['thermalcond'])
    source_temp = float(request.form['sourcetemp'])
    ambient_temp = float(request.form['ambienttemp'])
    block_size = float(request.form['blocksize'])

    # 1️⃣ Validate input ranges
    out_of_range = check_limits([thermal_cond, source_temp, ambient_temp, block_size])
    if out_of_range:
        return render_template("index.html",
                               message=f"❌ Invalid inputs for: {', '.join(out_of_range)}. Please enter values within allowed range.",
                               result=False)

    # Scale & Predict
    input_scaled = scaler.transform([[thermal_cond, source_temp, ambient_temp, block_size]])
    prediction = model.predict(input_scaled)[0]  # MaxTemp, AvgTemp, CenterTemp
    max_temp, avg_temp, center_temp = prediction

    # 2️⃣ Efficiency Calculation & Status
    efficiency = round((source_temp - avg_temp) / source_temp * 100, 2)
    if efficiency < 0:
        status = "⚠️ Overheating Risk"
    elif efficiency > 100:
        efficiency = 100
        status = "✅ Maximum Efficiency"
    elif avg_temp < 30:
        status = "❄️ Low – No coolant needed"
    elif 30 <= avg_temp <= 45 and max_temp < 75:
        status = "✅ Medium – Good performance"
    elif avg_temp > 45 and max_temp < 75:
        status = "⚠️ Medium High – Acceptable but monitor"
    elif avg_temp > 45 and max_temp >= 75:
        status = "🚨 High – Coolant Required"
    else:
        status = "❓ Uncertain"

    # 3️⃣ Save predictions to DB
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ThermalCond REAL,
            SourceTemp REAL,
            AmbientTemp REAL,
            BlockSize REAL,
            MaxTemp REAL,
            AvgTemp REAL,
            CenterTemp REAL,
            Efficiency REAL,
            Status TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        INSERT INTO predictions (ThermalCond, SourceTemp, AmbientTemp, BlockSize, MaxTemp, AvgTemp, CenterTemp, Efficiency, Status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (thermal_cond, source_temp, ambient_temp, block_size, max_temp, avg_temp, center_temp, efficiency, status))
    conn.commit()
    conn.close()

    return render_template('index.html',
                           result=True,
                           max_temp=round(max_temp, 2),
                           avg_temp=round(avg_temp, 2),
                           center_temp=round(center_temp, 2),
                           efficiency=efficiency,
                           status=status)

# Optional: History Page
@app.route('/history')
def history():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT ThermalCond, SourceTemp, AmbientTemp, BlockSize, MaxTemp, AvgTemp, CenterTemp, Efficiency, Status, Timestamp FROM predictions ORDER BY Timestamp DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)
