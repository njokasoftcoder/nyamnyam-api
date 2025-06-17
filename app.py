import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file
import joblib
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the trained model
model = joblib.load('football_match_predictor.pkl')

# Load required feature list
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

            # Check and filter required features
            missing = [feat for feat in required_features if feat not in df.columns]
            if missing:
                return f"Missing required columns: {missing}"

            # Align input dataframe with model features
            input_data = df[required_features]

            # Predict
            prediction = model.predict(input_data)
            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Predicted Outcome'] = [label_map[p] for p in prediction]

            # Save results
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            return send_file(output_path, as_attachment=True)

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
