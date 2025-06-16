from flask import Flask, request, render_template, send_file
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

model = joblib.load("football_match_predictor.pkl")

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    df = pd.read_csv(file)

    # Get model expected features
    expected_features = model.feature_names_in_

    # Check for required columns
    if not set(expected_features).issubset(df.columns):
        return f"Missing columns. Expected at least: {expected_features}"

    # Predict
    predictions = model.predict(df[expected_features])
    label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    df["Prediction"] = [label_map[p] for p in predictions]

    # Save predictions to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"predictions_{timestamp}.csv"
    df.to_csv(output_file, index=False)

    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
