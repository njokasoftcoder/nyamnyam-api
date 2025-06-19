from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Union

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Complete list of all expected feature columns
FEATURE_COLUMNS = [
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
    'FormHomeTeam', 'FormAwayTeam',
    'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'TotalPointsHome', 'TotalPointsAway',
    'TotalshotspergameHometeam', 'TotalshotspergameAwayteam',
    'ShotsontargetpergameHometeam', 'ShotsontargetpergameAwayteam',
    'ShotsofftargetpergameHometeam', 'ShotsofftargetpergameAwayteam',
    'BlockedshotspergameHometeam', 'BlockedshotspergameAwayteam',
    'CornerspergameHometeam', 'CornerspergameAwayteam',
    'FreekickspergameHometeam', 'FreekickspergameAwayteam',
    'HitwoodworkHometeam', 'HitwoodworkAwayteam',
    'CounterattacksHometeam', 'CounterattacksAwayteam',
    'H2H_HomeWins', 'H2H_Draws', 'H2H_Losses'
]

def compute_h2h_stats(h2h_string: str, home_team_name: str) -> tuple:
    """Calculate head-to-head statistics from match history string."""
    home_wins = draws = losses = 0
    if not h2h_string or not isinstance(h2h_string, str):
        return 0, 0, 0

    matches = [m.strip() for m in h2h_string.split(',') if m.strip()]
    
    for match in matches:
        score_match = re.search(r'\((\d+):(\d+)\)', match)
        if not score_match:
            continue
            
        home_goals, away_goals = map(int, score_match.groups())
        teams = [t.strip() for t in match.split('vs')[:2]]
        
        if len(teams) != 2:
            continue
            
        if home_team_name in teams[0]:  # Home team was home in this match
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        else:  # Home team was away in this match
            if away_goals > home_goals:
                home_wins += 1
            elif away_goals == home_goals:
                draws += 1
            else:
                losses += 1
                
    return home_wins, draws, losses

def preprocess_input(data: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """Convert and validate input data into proper feature DataFrame."""
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    
    # Add H2H features if available
    if 'H2H(Latestooldest)' in df.columns:
        df[['H2H_HomeWins', 'H2H_Draws', 'H2H_Losses']] = df.apply(
            lambda row: compute_h2h_stats(
                row['H2H(Latestooldest)'],
                row.get('home_team', 'Home Team')
            ),
            axis=1, result_type='expand'
        )
        df.drop('H2H(Latestooldest)', axis=1, inplace=True)
    
    # Ensure all required columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with 0
    
    return df[FEATURE_COLUMNS]

@app.route('/')
def home():
    return "Nyam Nyam Prediction API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input validation
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No data provided"}), 400
        
        # Preprocess input
        features = preprocess_input(input_data)
        
        # Make prediction
        predictions_encoded = model.predict(features)
        predictions = label_encoder.inverse_transform(predictions_encoded)
        
        # Format response
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "prediction": pred,
                "home_team": input_data[i]['home_team'] if isinstance(input_data, list) else input_data['home_team'],
                "away_team": input_data[i]['away_team'] if isinstance(input_data, list) else input_data['away_team'],
                "confidence": float(np.max(model.predict_proba(features)[i]))  # Add confidence score
            }
            results.append(result)
        
        return jsonify({
            "success": True,
            "results": results if isinstance(input_data, list) else results[0]
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Prediction failed",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
