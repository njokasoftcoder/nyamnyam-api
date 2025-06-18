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

# Load trained model
model = joblib.load('match_outcome_model.pkl')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Expected columns (from your feature list)
EXPECTED_COLUMNS = [
    'OddsHome', 'DrawOdds', 'AwayOdds',
    'SofascoreRatingHomeTeam', 'SofascoreRatingAwayTeam',
    'NumberofmatchesplayedHometeam', 'NumberofmatchesplayedAwayteam',
    'TotalgoalsscoredintheseasonHometeam', 'TotalgoalsscoredintheseasonAwayteam',
    'TotalgoalsconcededHomeTeam', 'TotalgoalsconcededAwayteam',
    'TotalassistsHometeam', 'TotalassistsAwayteam',
    'GoalspergameHometeam', 'GoalspergameAwayteam',
    'Goalconversion(Hometeam)', 'Goalconversion(Awayteam)',
    'ShotsontargetpergameHometeam', 'Shotsontargetpergameawayteam',
    'Bigchancespergamehometeam', 'Bigchancespergamehometeamawayteam',
    'BigchancesmissedpergameHometeam', 'BigchancesmissedpergameAwayteam',
    'Ballpossessionhometeam', 'Ballpossessionawayteam',
    'Accuratepassespergamehometeam', 'Accuratepassespergameawayteam',
    'Accuratelongballspergamehometeam', 'Accuratelongballspergameawayteam',
    'CleansheetsHometeam', 'CleansheetsAwayteam',
    'Goalsconcededpergamehometeam', 'Goalsconcededpergameawayteam',
    'InterceptionspergameHometeam', 'InterceptionspergameAwayteam',
    'Tacklespergamehometeam', 'Tacklespergameawayteam',
    'ClearancespergameHometeam', 'Clearancespergameawayteam',
    'PenaltygoalsconcededHometeam', 'PenaltygoalsconcededAwayteam',
    'Savespergame', 'DuelswonpergameHometeam', 'DuelswonpergameAwayteam',
    'FoulspergameHometeam', 'FoulspergameAwayteam',
    'OffsidespergameHometeam', 'OffsidespergameAwayteam',
    'GoalkickspergameHometeam', 'GoalkickspergameAwayteam',
    'TotalthrowinsHometeam', 'TotalthrowinsAwayteam',
    'TotalyellowcardsawardedHometeam', 'TotalyellowcardsawardedAwayteam',
    'TotalRedcardsawardedHometeam', 'TotalRedcardsawardedAwayteam',
    'FormHomeTeam', 'FormAwayTeam',
    'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'TotalPointsHome', 'TotalPointsAway',
    'TotalshotspergameHometeam', 'TotalshotspergameAwayteam',
    'ShotsontargetpergameHometeam', 'ShotsontargetpergameAwayteam',
    'ShotsofftargetpergameHometeam', 'Shotsofftargetpergame',
    'BlockedshotspergameHometeam', 'BlockedshotspergameAwayteam',
    'CornerspergameHometeam', 'CornerspergameAwayteam',
    'FreekickspergameHometeam', 'FreekickspergameAwayteam',
    'HitwoodworkHometeam', 'HitwoodworkAwayteam',
    'CounterattacksHometeam', 'CounterattacksAwayteam',
    'H2H(Latestooldest)'
]

# Home route
@app.route('/')
def index():
    return render_template('form.html')

# Upload + Predict route
@app.route('/', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return "No file part in the form."

    file = request.files['file']
    if file.filename == '':
        return "No file selected."

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            # Ensure expected columns exist
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            for col in missing_cols:
                df[col] = 0  # default missing columns

            # Keep only expected columns in order
            df_model = df[EXPECTED_COLUMNS].copy()

            # Convert to numeric (any stray strings become NaN)
            df_model = df_model.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Run prediction
            prediction = model.predict(df_model)
            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map.get(p, "Unknown") for p in prediction]
            df.insert(0, 'Match No.', range(1, len(df) + 1))

            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'predicted_results_{timestamp}.csv'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            df.to_csv(output_path, index=False)

            return render_template('results.html', tables=[df.to_html(classes='data', index=False)], filename=output_filename)

        except Exception as e:
            return f"❌ Error during prediction: {str(e)}"

    return "❌ Invalid file format. Please upload a .csv file."

# Route to download result
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
