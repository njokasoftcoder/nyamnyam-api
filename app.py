import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for
import joblib
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load trained model and features
model = joblib.load('football_match_predictor.pkl')
with open("model_features.txt", "r") as f:
    required_features = [line.strip() for line in f.readlines()]

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return "No file part in the form"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            # Ensure required features
            missing = [feat for feat in required_features if feat not in df.columns]
            if missing:
                return f"Missing required columns: {missing}"

            input_data = df[required_features]

            # Predict
            predictions = model.predict(input_data)
            probabilities = model.predict_proba(input_data)

            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Predicted Outcome'] = [label_map[p] for p in predictions]
            df['Confidence Score'] = np.max(probabilities, axis=1).round(2)

            # Save results
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            preview_df = df[['Predicted Outcome', 'Confidence Score']].head(50)  # Show first 50 results

            return render_template('results.html',
                                   tables=[preview_df.to_html(classes='data', header=True, index=False)],
                                   download_link=url_for('download_file'))

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

@app.route('/download')
def download_file():
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
    return send_file(output_path, as_attachment=True)
