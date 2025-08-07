from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.pkl")
# If you're using scaling:
try:
    scaler = joblib.load("scaler.pkl")
    use_scaler = True
except:
    use_scaler = False

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # Get user input
    thermalcond = float(request.form['thermalcond'])
    sourcetemp = float(request.form['sourcetemp'])
    ambienttemp = float(request.form['ambienttemp'])
    blocksize = float(request.form['blocksize'])

    input_data = np.array([[thermalcond, sourcetemp, ambienttemp, blocksize]])

    # Apply scaler if available
    if use_scaler:
        input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]
    max_temp, avg_temp, center_temp = prediction

    # Recommendation logic
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

    efficiency = round((sourcetemp - avg_temp) / sourcetemp * 100, 2)

    return render_template("index.html",
        result=True,
        max_temp=round(max_temp, 2),
        avg_temp=round(avg_temp, 2),
        center_temp=round(center_temp, 2),
        efficiency=efficiency,
        status=status
    )

if __name__ == '__main__':
    app.run(debug=True)
