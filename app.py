from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define all feature columns (must match the model training)
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
    'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'TotalPointsHome', 'TotalPointsAway',
    'TotalshotspergameHometeam', 'TotalshotspergameAwayteam',
    'ShotsofftargetpergameHometeam', 'ShotsofftargetpergameAwayteam',
    'BlockedshotspergameHometeam', 'BlockedshotspergameAwayteam',
    'CornerspergameHometeam', 'CornerspergameAwayteam',
    'FreekickspergameHometeam', 'FreekickspergameAwayteam',
    'HitwoodworkHometeam', 'HitwoodworkAwayteam',
    'CounterattacksHometeam', 'CounterattacksAwayteam',
    'H2H_HomeWins', 'H2H_Draws', 'H2H_Losses'
]

@app.route('/')
def home():
    return "Nyam Nyam Confidence Fire Prediction is 🔥 live."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # Ensure we have a list of records
        if isinstance(input_data, dict):
            input_data = [input_data]

        df = pd.DataFrame(input_data)

        # Fill in any missing columns with zero (assumes numeric-only features)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Keep only the expected feature columns
        df = df[feature_columns]

        # Make predictions
        prediction_encoded = model.predict(df)
        predictions = label_encoder.inverse_transform(prediction_encoded)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
