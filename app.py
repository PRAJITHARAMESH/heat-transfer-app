from flask import Flask, render_template, request, redirect, url_for, send_file
import joblib
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

# Load model and scaler from model.pkl
model_data = joblib.load("model.pkl")
if isinstance(model_data, dict):
    model = model_data["model"]
    scaler = model_data["scaler"]
else:
    model = model_data
    scaler = None

history = []

def calculate_efficiency(max_temp):
    base_efficiency = 100
    penalty = max(0, (max_temp - 50) * 2)
    return round(max(0, base_efficiency - penalty), 2)

def coolant_suggestion(efficiency):
    if efficiency > 80:
        return "Water - Efficient coolant"
    elif efficiency > 60:
        return "Oil - Moderate coolant"
    elif efficiency > 40:
        return "Special Coolant - Use for high temps"
    else:
        return "Extreme Coolant - Urgently needed"

def metal_recommendation(max_temp):
    if max_temp < 50:
        return "Aluminum - Good heat dissipation"
    elif max_temp < 70:
        return "Copper - Better heat conduction"
    else:
        return "Silver or Copper Alloys - Best for very high heat"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            thermal = float(request.form["thermal"])
            source = float(request.form["source"])
            ambient = float(request.form["ambient"])
            block_size = float(request.form["block_size"])

            features = np.array([[thermal, source, ambient, block_size]])
            if scaler:
                features = scaler.transform(features)

            preds = model.predict(features)
            max_temp = round(preds[0][0], 2)
            avg_temp = round(preds[0][1], 2)
            center_temp = round(preds[0][2], 2)

            efficiency = calculate_efficiency(max_temp)
            coolant = coolant_suggestion(efficiency)
            material = metal_recommendation(max_temp)

            status = "Good" if efficiency > 70 else ("Moderate" if efficiency > 40 else "Poor")

            result = {
                "max_temp": max_temp,
                "avg_temp": avg_temp,
                "center_temp": center_temp,
                "efficiency": efficiency,
                "coolant": coolant,
                "material": material,
                "status": status
            }

            # Save to history
            history.append({
                "Thermal Cond": thermal,
                "Source Temp": source,
                "Ambient Temp": ambient,
                "Block Size": block_size,
                "Max Temp": max_temp,
                "Avg Temp": avg_temp,
                "Center Temp": center_temp,
                "Efficiency": efficiency,
                "Coolant": coolant,
                "Material": material,
                "Status": status
            })

            return render_template("result.html", prediction=result)
        except Exception as e:
            return render_template("index.html", error=str(e), history=history)
    else:
        return render_template("index.html", history=history)

@app.route("/history")
def show_history():
    return render_template("history.html", history=history)

@app.route("/download")
def download():
    if not history:
        return redirect(url_for("index"))

    df = pd.DataFrame(history)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype="text/csv",
                     download_name="history.csv",
                     as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
