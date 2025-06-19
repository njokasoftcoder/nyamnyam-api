from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define all expected feature columns
feature_columns = [
    'OddsHomeTeam', 'DrawOdds', 'OddsAwayTeam',
    'SofascoreRatingHomeTeam', 'SofascoreRatingAwayTeam',
    'NumberofmatchesplayedHomeTeam', 'NumberofmatchesplayedAwayTeam',
    'TotalgoalsscoredintheseasonHomeTeam', 'TotalgoalsscoredintheseasonAwayTeam',
    'TotalgoalsconcededHomeTeam', 'TotalgoalsconcededAwayTeam',
    'TotalassistsHomeTeam', 'TotalassistsAwayTeam',
    'GoalspergameHomeTeam', 'GoalspergameAwayTeam',
    'Goalconversion(HomeTeam)', 'Goalconversion(AwayTeam)',
    'ShotsontargetpergameHomeTeam', 'ShotsontargetpergameawayTeam',
    'BigchancespergamehomeTeam', 'BigchancespergamehomeTeamawayTeam',
    'BigchancesmissedpergameHomeTeam', 'BigchancesmissedpergameAwayTeam',
    'BallpossessionhomeTeam', 'BallpossessionawayTeam',
    'AccuratepassespergamehomeTeam', 'AccuratepassespergameawayTeam',
    'AccuratelongballspergamehomeTeam', 'AccuratelongballspergameawayTeam',
    'CleansheetsHomeTeam', 'CleansheetsAwayTeam',
    'GoalsconcededpergamehomeTeam', 'GoalsconcededpergameawayTeam',
    'InterceptionspergameHomeTeam', 'InterceptionspergameAwayTeam',
    'TacklespergamehomeTeam', 'TacklespergameawayTeam',
    'ClearancespergameHomeTeam', 'ClearancespergameawayTeam',
    'PenaltygoalsconcededHomeTeam', 'PenaltygoalsconcededAwayTeam',
    'Savespergame',
    'DuelswonpergameHomeTeam', 'DuelswonpergameAwayTeam',
    'FoulspergameHomeTeam', 'FoulspergameAwayTeam',
    'OffsidespergameHomeTeam', 'OffsidespergameAwayTeam',
    'GoalkickspergameHomeTeam', 'GoalkickspergameAwayTeam',
    'TotalthrowinsHomeTeam', 'TotalthrowinsAwayTeam',
    'TotalyellowcardsawardedHomeTeam', 'TotalyellowcardsawardedAwayTeam',
    'TotalRedcardsawardedHomeTeam', 'TotalRedcardsawardedAwayTeam',
    'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'TotalPointsHome', 'TotalPointsAway',
    'TotalshotspergameHomeTeam', 'TotalshotspergameAwayTeam',
    'ShotsontargetpergameHomeTeam', 'ShotsontargetpergameAwayTeam',
    'ShotsofftargetpergameHomeTeam', 'ShotsofftargetpergameAwayTeam',
    'BlockedshotspergameHomeTeam', 'BlockedshotspergameAwayTeam',
    'CornerspergameHomeTeam', 'CornerspergameAwayTeam',
    'FreekickspergameHomeTeam', 'FreekickspergameAwayTeam',
    'HitwoodworkHomeTeam', 'HitwoodworkAwayTeam',
    'CounterattacksHomeTeam', 'CounterattacksAwayTeam',
    'H2H_HomeWins', 'H2H_Draws', 'H2H_Losses'
]

def compute_h2h_stats(h2h_string, home_team_name):
    home_wins = draws = losses = 0
    matches = h2h_string.split(',')

    for match in matches:
        match = match.strip()
        score_match = re.search(r'\((\d+):(\d+)\)', match)
        if not score_match:
            continue
        home_goals = int(score_match.group(1))
        away_goals = int(score_match.group(2))

        if f"{home_team_name}" in match.split('vs')[0].strip():
            # home_team was playing at home
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        else:
            # home_team was playing away
            if home_goals < away_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1

    return home_wins, draws, losses

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

        # Check and compute H2H if available
        if 'H2H(Latestooldest)' in df.columns:
            h2h_home_wins = []
            h2h_draws = []
            h2h_losses = []
            for idx, row in df.iterrows():
                home_team = row.get("League_Home", "Home")  # Or another field if appropriate
                h2h_str = row["H2H(Latestooldest)"]
                wins, draws_, losses = compute_h2h_stats(h2h_str, home_team)
                h2h_home_wins.append(wins)
                h2h_draws.append(draws_)
                h2h_losses.append(losses)

            df['H2H_HomeWins'] = h2h_home_wins
            df['H2H_Draws'] = h2h_draws
            df['H2H_Losses'] = h2h_losses
            df.drop(columns=["H2H(Latestooldest)"], inplace=True)

        # Fill missing columns with 0
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        prediction_encoded = model.predict(df)
        predictions = label_encoder.inverse_transform(prediction_encoded)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
