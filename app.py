import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file
import joblib
from werkzeug.utils import secure_filename
from datetime import datetime

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the trained model
model = joblib.load('match_outcome_model.pkl')  # updated name

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route – render the upload form
@app.route('/')
def index():
    return render_template('form.html')

# Handle CSV upload and run prediction
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
            # Load uploaded data
            df = pd.read_csv(filepath)

            # Run prediction
            prediction = model.predict(df)
            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map[p] for p in prediction]

            # Add Match Number column
            df.insert(0, 'Match No.', range(1, len(df) + 1))

            # Save output CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'predicted_results_{timestamp}.csv'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            df.to_csv(output_path, index=False)

            # Optionally log predictions (can be expanded later)
            log_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_log.csv')
            df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

            # Render results in HTML
            return render_template('results.html', tables=[df.to_html(classes='data', index=False)], filename=output_filename)

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

# Download prediction file
@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
