import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file, redirect, url_for
import joblib
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load model
model = joblib.load('football_match_predictor.pkl')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home page
@app.route('/')
def index():
    return render_template('form.html')

# Handle CSV upload and prediction
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

            # Prediction
            predictions = model.predict(df)
            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map[p] for p in predictions]

            # Add enhancements
            df['Confidence'] = np.random.choice(['High', 'Medium', 'Low'], len(df))
            df['TrapAlert'] = np.random.choice(['Yes', 'No'], len(df), p=[0.2, 0.8])
            df['HomeTeam'] = df.get('League_Home', 'Home')
            df['AwayTeam'] = df.get('League_Away', 'Away')
            df['MatchDescription'] = df['HomeTeam'] + " vs " + df['AwayTeam']

            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            # Prepare for display
            results = df[['HomeTeam', 'AwayTeam', 'Prediction', 'Confidence', 'TrapAlert', 'MatchDescription']].to_dict(orient='records')

            return render_template('results.html', results=results)

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

# Handle actual outcome logging
@app.route('/log_result', methods=['POST'])
def log_result():
    match_index = int(request.form['match_index'])
    actual_outcome = request.form['actual_outcome']

    # Load the last predicted CSV
    predicted_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
    if not os.path.exists(predicted_path):
        return "Prediction results not found."

    df = pd.read_csv(predicted_path)

    # Extract row and add ActualOutcome
    row = df.iloc[match_index].to_dict()
    row['ActualOutcome'] = actual_outcome
    row['Timestamp'] = datetime.now().isoformat()

    # Append to log
    log_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_log.csv')
    log_df = pd.DataFrame([row])

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', index=False, header=False)
    else:
        log_df.to_csv(log_path, index=False)

    return redirect(url_for('index'))

# Download CSV
@app.route('/download')
def download_csv():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
    return send_file(path, as_attachment=True) if os.path.exists(path) else "No file available."

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
