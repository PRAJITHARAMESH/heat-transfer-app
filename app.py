from flask import Flask, render_template, request, send_file
import numpy as np
import pickle
import pandas as pd
import io

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Store last prediction for download
last_result = {}

@app.route('/', methods=['GET', 'POST'])
def home():
    global last_result
    prediction = None
    efficiency = None
    coolant_suggestion = None
    material_suggestion = None

    if request.method == 'POST':
        try:
            # Get inputs
            thermal_cond = float(request.form['thermal_cond'])
            source_temp = float(request.form['source_temp'])
            block_size = float(request.form['block_size'])
            ambient_temp = float(request.form['ambient_temp'])

            # Prepare data
            features = np.array([[thermal_cond, source_temp, block_size, ambient_temp]])
            scaled_features = scaler.transform(features)

            # Prediction
            prediction = model.predict(scaled_features)[0]

            # Efficiency % (example formula)
            efficiency = round((source_temp - ambient_temp) / source_temp * 100, 2)

            # Coolant suggestion
            if prediction > 500:
                coolant_suggestion = "High-performance coolant"
            elif prediction > 200:
                coolant_suggestion = "Standard coolant"
            else:
                coolant_suggestion = "No coolant needed"

            # Material suggestion
            if thermal_cond > 150:
                material_suggestion = "Copper or Aluminum"
            else:
                material_suggestion = "Steel or Brass"

            # Store for download
            last_result = {
                "Thermal Conductivity (W/mK)": thermal_cond,
                "Source Temperature (°C)": source_temp,
                "Block Size (mm)": block_size,
                "Ambient Temperature (°C)": ambient_temp,
                "Predicted Heat Transfer Rate": prediction,
                "Efficiency (%)": efficiency,
                "Coolant Suggestion": coolant_suggestion,
                "Material Suggestion": material_suggestion
            }

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        'index.html',
        prediction=prediction,
        efficiency=efficiency,
        coolant=coolant_suggestion,
        material=material_suggestion
    )

@app.route('/download')
def download():
    global last_result
    if not last_result:
        return "No prediction available to download."

    df = pd.DataFrame([last_result])
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, as_attachment=True, download_name="heat_transfer_prediction.csv", mimetype='text/csv')

if __name__ == '__main__':
    app.run(debug=True)
