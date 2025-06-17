""import os
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

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for form
@app.route('/')
def index():
    return render_template('form.html')

# Handle file upload and prediction
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
            # Load CSV
            df = pd.read_csv(filepath)

            # Optional: Add match description column
            if 'TeamHome' in df.columns and 'TeamAway' in df.columns:
                df['Match'] = df['TeamHome'] + " vs " + df['TeamAway']
                df.insert(0, 'Match', df.pop('Match'))

            # Run prediction
            prediction = model.predict(df.drop(columns=['Match'], errors='ignore'))
            prediction_proba = model.predict_proba(df.drop(columns=['Match'], errors='ignore'))

            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map[p] for p in prediction]
            df['Confidence (%)'] = [f"{np.max(proba) * 100:.1f}%" for proba in prediction_proba]

            # Add trap match flag (example logic)
            if 'OddsHome' in df.columns and 'AwayOdds' in df.columns:
                df['TrapAlert'] = np.where(
                    (df['OddsHome'] < df['AwayOdds']) & (df['Prediction'] == 'Away Win'), 'Yes', 'No')

            # Save output
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            # Count results for summary
            result_summary = df['Prediction'].value_counts().to_dict()
            top_confident = df.sort_values(by='Confidence (%)', ascending=False).head(3)[['Match', 'Prediction', 'Confidence (%)']]

            return render_template(
                'results.html',
                tables=[df.to_html(classes='data', index=False)],
                result_summary=result_summary,
                top_confident=top_confident.to_dict(orient='records'),
                download_path=output_path
            )

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
