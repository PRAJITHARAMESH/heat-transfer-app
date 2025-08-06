from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read input from form
    inputs = [
        float(request.form['thermalcond']),
        float(request.form['sourcetemp']),
        float(request.form['ambienttemp']),
        float(request.form['blocksize'])
    ]

    # Scale and predict
    input_scaled = scaler.transform([inputs])
    prediction = model.predict(input_scaled)[0]

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

    efficiency = round((inputs[1] - avg_temp) / inputs[1] * 100, 2)

    return render_template('index.html',
        result=True,
        max_temp=round(max_temp, 2),
        avg_temp=round(avg_temp, 2),
        center_temp=round(center_temp, 2),
        efficiency=efficiency,
        status=status
    )

if __name__ == '__main__':
    app.run(debug=True)
