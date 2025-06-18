import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file
import joblib
from werkzeug.utils import secure_filename
from datetime import datetime

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the trained model
model = joblib.load('match_outcome_model.pkl')

# Define expected input columns (update this to match your model exactly)
EXPECTED_COLUMNS = [
    # Include all columns used in training, excluding 'Prediction'
    'OddsHome', 'DrawOdds', 'AwayOdds', 'SofascoreRatingHomeTeam', 'SofascoreRatingAwayTeam',
    'FormHomeTeam', 'FormAwayTeam', 'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'H2H_HomeWins', 'H2H_Draws', 'H2H_AwayWins'
    # Add the rest from your model input features
]

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Sample encoder for Form (extend as needed)
def encode_form(form_str):
    mapping = {'W': 3, 'D': 1, 'L': 0}
    return sum(mapping.get(ch, 0) for ch in str(form_str).upper())

# Sample parser for H2H data
def parse_h2h(h2h_str):
    home_wins = draws = away_wins = 0
    matches = str(h2h_str).split(',')
    for match in matches:
        if '(' in match and ')' in match:
            try:
                teams, score = match.strip().split('(')
                score = score.strip(')')
                home_goals, away_goals = map(int, score.split(':'))
                if home_goals > away_goals:
                    home_wins += 1
                elif home_goals < away_goals:
                    away_wins += 1
                else:
                    draws += 1
            except:
                continue
    return pd.Series([home_wins, draws, away_wins])

# Home route
@app.route('/')
def index():
    return render_template('form.html')

# Upload and predict route
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

            # Preprocessing
            df.fillna(0, inplace=True)

            if 'FormHomeTeam' in df.columns:
                df['FormHomeTeam'] = df['FormHomeTeam'].apply(encode_form)
            if 'FormAwayTeam' in df.columns:
                df['FormAwayTeam'] = df['FormAwayTeam'].apply(encode_form)

            if 'H2H(Latestooldest)' in df.columns:
                df[['H2H_HomeWins', 'H2H_Draws', 'H2H_AwayWins']] = df['H2H(Latestooldest)'].apply(parse_h2h)
                df.drop(columns=['H2H(Latestooldest)'], inplace=True)

            # Convert non-numeric columns safely
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).astype('category').cat.codes

            # Warn about missing expected columns
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing_cols:
                warning_msg = f"⚠️ Warning: Missing columns - {', '.join(missing_cols)}\n\nPrediction may be less accurate."
            else:
                warning_msg = None

            # Reorder and align columns if necessary
            df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

            prediction = model.predict(df)
            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map.get(p, 'Unknown') for p in prediction]
            df.insert(0, 'Match No.', range(1, len(df) + 1))

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'predicted_results_{timestamp}.csv'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            df.to_csv(output_path, index=False)

            # Save prediction log
            log_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_log.csv')
            df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

            return render_template('results.html', tables=[df.to_html(classes='data', index=False)],
                                   filename=output_filename, warning=warning_msg)

        except Exception as e:
            error_log = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_errors.log')
            with open(error_log, 'a') as f:
                f.write(f"[{datetime.now()}] Error: {str(e)}\n")
            return f"❌ Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a .CSV file."

# Download route
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

# Run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
