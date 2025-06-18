from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Columns used in training (must match exactly in order)
feature_columns = [
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
    'Savespergame',
    'DuelswonpergameHometeam', 'DuelswonpergameAwayteam',
    'FoulspergameHometeam', 'FoulspergameAwayteam',
    'OffsidespergameHometeam', 'OffsidespergameAwayteam',
    'GoalkickspergameHometeam', 'GoalkickspergameAwayteam',
    'TotalthrowinsHometeam', 'TotalthrowinsAwayteam',
    'TotalyellowcardsawardedHometeam', 'TotalyellowcardsawardedAwayteam',
    'TotalRedcardsawardedHometeam', 'TotalRedcardsawardedAwayteam',
    'FormHomePoints', 'FormAwayPoints',
    'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'TotalPointsHome', 'TotalPointsAway',
    'TotalshotspergameHometeam', 'TotalshotspergameAwayteam',
    'ShotsofftargetpergameHometeam', 'ShotsofftargetpergameAwayteam',
    'BlockedshotspergameHometeam', 'BlockedshotspergameAwayteam',
    'CornerspergameHometeam', 'CornerspergameAwayteam',
    'FreekickspergameHometeam', 'FreekickspergameAwayteam',
    'HitwoodworkHometeam', 'HitwoodworkAwayteam',
    'CounterattacksHometeam', 'CounterattacksAwayteam'
]

# Helper function to convert form strings like "WDLDW" to points
def transform_form(form_string):
    form_map = {'W': 3, 'D': 1, 'L': 0}
    return sum(form_map.get(char.upper(), 0) for char in form_string)

@app.route("/")
def home():
    return "Nyam Nyam Confidence Fire Prediction is 🔥 live."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON payload
        input_data = request.get_json()
        if not isinstance(input_data, list):
            return jsonify({"error": "Input must be a list of match records."}), 400

        df = pd.DataFrame(input_data)

        # Transform form fields into numeric points if they exist
        if 'FormHomeTeam' in df.columns and 'FormAwayTeam' in df.columns:
            df['FormHomePoints'] = df['FormHomeTeam'].apply(transform_form)
            df['FormAwayPoints'] = df['FormAwayTeam'].apply(transform_form)
            df.drop(['FormHomeTeam', 'FormAwayTeam'], axis=1, inplace=True)

        # Check for missing columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

        # Ensure columns are in correct order
        df = df[feature_columns]

        # Predict
        predictions = model.predict(df)
        decoded_preds = label_encoder.inverse_transform(predictions)

        return jsonify({"predictions": decoded_preds.tolist()})

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
