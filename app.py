from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define all feature columns (MUST match training)
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
    'H2H_HomeWins', 'H2H_Draws', 'H2H_Losses',
    'Home_FormWins', 'Home_FormDraws', 'Home_FormLosses', 'Home_FormScore',
    'Away_FormWins', 'Away_FormDraws', 'Away_FormLosses', 'Away_FormScore'
]

def form_to_features(form_str):
    form_str = str(form_str).upper()
    return {
        'Wins': form_str.count('W'),
        'Draws': form_str.count('D'),
        'Losses': form_str.count('L'),
        'FormScore': form_str.count('W')*3 + form_str.count('D')
    }

def parse_h2h(h2h_str, home_team_name):
    wins = draws = losses = 0
    matches = re.findall(r'(\w+)\s+vs\s+(\w+)\s+\((\d+):(\d+)\)', str(h2h_str))
    for team1, team2, score1, score2 in matches:
        score1, score2 = int(score1), int(score2)
        if team1 == home_team_name:
            if score1 > score2: wins += 1
            elif score1 == score2: draws += 1
            else: losses += 1
        elif team2 == home_team_name:
            if score2 > score1: wins += 1
            elif score1 == score2: draws += 1
            else: losses += 1
    return {
        'H2H_HomeWins': wins,
        'H2H_Draws': draws,
        'H2H_Losses': losses
    }

@app.route('/')
def home():
    return "Nyam Nyam Confidence Fire Prediction is 🔥 live."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        if isinstance(input_data, dict):
            input_data = [input_data]

        df = pd.DataFrame(input_data)

        # Form processing
        if 'FormHomeTeam' in df.columns:
            form_home = df['FormHomeTeam'].apply(form_to_features).apply(pd.Series).add_prefix('Home_Form')
            df = pd.concat([df, form_home], axis=1)

        if 'FormAwayTeam' in df.columns:
            form_away = df['FormAwayTeam'].apply(form_to_features).apply(pd.Series).add_prefix('Away_Form')
            df = pd.concat([df, form_away], axis=1)

        # H2H processing
        if 'H2H(Latestooldest)' in df.columns and 'HomeTeam' in df.columns:
            h2h_counts = df.apply(lambda row: parse_h2h(row['H2H(Latestooldest)'], row['HomeTeam']), axis=1).apply(pd.Series)
            df = pd.concat([df, h2h_counts], axis=1)

        # Fill missing expected columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Keep only expected columns
        df = df[feature_columns]

        prediction_encoded = model.predict(df)
        predictions = label_encoder.inverse_transform(prediction_encoded)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
