from flask import Flask, render_template, request
import joblib
import sqlite3
import datetime

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Parameter limits
LIMITS = {
    "ThermalCond": (80, 400),
    "SourceTemp": (40, 75),
    "AmbientTemp": (15, 35),
    "BlockSize": (5, 50)
}

# -----------------------------
# Database Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thermal_cond REAL,
                    source_temp REAL,
                    ambient_temp REAL,
                    block_size REAL,
                    max_temp REAL,
                    avg_temp REAL,
                    center_temp REAL,
                    efficiency REAL,
                    status TEXT,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Function to check limits
# -----------------------------
def check_limits(inputs):
    names = ["ThermalCond", "SourceTemp", "AmbientTemp", "BlockSize"]
    out = [n for n, v in zip(names, inputs) if not (LIMITS[n][0] <= v <= LIMITS[n][1])]
    return out

# -----------------------------
# Home Route
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read inputs
        inputs = [
            float(request.form['thermalcond']),
            float(request.form['sourcetemp']),
            float(request.form['ambienttemp']),
            float(request.form['blocksize'])
        ]

        # Validate limits
        out_of_range = check_limits(inputs)
        if out_of_range:
            return render_template('index.html', error=f"⚠️ Out of range values: {', '.join(out_of_range)}")

        # Scale inputs
        input_scaled = scaler.transform([inputs])
        prediction = model.predict(input_scaled)[0]  # MaxTemp, AvgTemp, CenterTemp
        max_temp, avg_temp, center_temp = prediction

        # Status logic
        if avg_temp < 30:
            status = "❄️ Low – No coolant needed"
        elif 30 <= avg_temp <= 45 and max_temp < 75:
            status = "✅ Medium – Good performance"
        elif avg_temp > 45 and max_temp < 75:
            status = "⚠️ Medium High – Acceptable but monitor"
        elif avg_temp > 45 and max_temp >= 75:
            status = "🚨 High – Coolant Required"
        else:
            status = "❓ Uncertain"

        # Efficiency calculation
        efficiency = round((inputs[1] - avg_temp) / inputs[1] * 100, 2)
        if efficiency < -100:
            efficiency_display = "Overheating"
        else:
            efficiency_display = f"{efficiency}%"

        # Save to DB
        conn = sqlite3.connect("predictions.db")
        c = conn.cursor()
        c.execute('''INSERT INTO history 
                     (thermal_cond, source_temp, ambient_temp, block_size, max_temp, avg_temp, center_temp, efficiency, status, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (*inputs, max_temp, avg_temp, center_temp, efficiency, status, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        return render_template('index.html',
                               result=True,
                               max_temp=round(max_temp, 2),
                               avg_temp=round(avg_temp, 2),
                               center_temp=round(center_temp, 2),
                               efficiency=efficiency_display,
                               status=status)

    except Exception as e:
        return render_template('index.html', error=f"❌ Error: {str(e)}")

# -----------------------------
# History Route
# -----------------------------
@app.route('/history')
def history():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
