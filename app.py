from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load models and scaler
try:
    with open("models/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    with open("models/scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

except Exception as e:
    rf_model = None
    xgb_model = None
    scaler_X = None
    print(f"Error loading the model: {e}")

# Thresholds (using values from Colab output)
pcyl_threshold_low = 76907.29506570872
pcyl_threshold_high = 95081.16419429149

rohr_threshold_low = 0.06625760810714898
rohr_threshold_high = 1.9750879969794317

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get input values from the form
            cad = float(request.form["CAD"])
            pinj = float(request.form["Pinj"])
            v = float(request.form["V"])
            padm = float(request.form["Padm"])
            dpdcad = float(request.form["dP/dCAD"])
            dvdCad = float(request.form["dV/dCAD"])

            # Prepare input data
            input_data = np.array([[cad, v, pinj, padm, dpdcad, dvdCad]])
            feature_names = ['CAD', 'V', 'Pinj', 'Padm', 'dP/dCAD', 'dV/dCAD']
            input_df = pd.DataFrame(input_data, columns=feature_names)

            # Scale the data
            input_scaled = scaler_X.transform(input_df)

            # Make predictions
            rohr_pred = rf_model.predict(input_scaled)[0]
            pcyl_pred = xgb_model.predict(input_scaled)[0]

            # Apply thresholds for classification
            rohr_status = "Desired" if rohr_threshold_low <= rohr_pred <= rohr_threshold_high else "Undesired"
            pcyl_status = "Desired" if pcyl_threshold_low <= pcyl_pred <= pcyl_threshold_high else "Undesired"

            # Final classification status
            final_status = "✅ Desired" if rohr_status == "Desired" and pcyl_status == "Desired" else "❌ Undesired"

            return render_template("index.html", rohr=rohr_pred, pcyl=pcyl_pred, status=final_status)

        except Exception as e:
            return f"Prediction error: {e}"

    return render_template("index.html")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
