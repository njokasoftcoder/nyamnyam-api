from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
from typing import Dict, List, Union

app = Flask(__name__)

# Load model and encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Updated to match CSV columns exactly
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

def clean_team_name(name: str) -> str:
    """Standardize team names for H2H comparison"""
    return re.sub(r'[^a-zA-Z]', '', name.lower())

def compute_h2h_stats(h2h_string: str, home_team: str, away_team: str) -> tuple:
    """Improved H2H calculation with team name standardization"""
    home_wins = draws = losses = 0
    if not h2h_string or pd.isna(h2h_string):
        return 0, 0, 0

    clean_home = clean_team_name(home_team)
    clean_away = clean_team_name(away_team)
    
    matches = [m.strip() for m in str(h2h_string).split(',') if m.strip()]
    
    for match in matches:
        # Extract scores
        score_match = re.search(r'\((\d+):(\d+)\)', match)
        if not score_match:
            continue
            
        home_goals, away_goals = map(int, score_match.groups())
        
        # Extract team names
        teams_part = match.split('(')[0].strip()
        teams = [t.strip() for t in teams_part.split('vs')[:2]]
        
        if len(teams) != 2:
            continue
            
        # Determine which team is which
        if clean_team_name(teams[0]) == clean_home:
            # Home team was home in this match
            if home_goals > away_goals:
                home_wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        else:
            # Home team was away in this match
            if away_goals > home_goals:
                home_wins += 1
            elif away_goals == home_goals:
                draws += 1
            else:
                losses += 1
                
    return home_wins, draws, losses

def preprocess_input(data: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """Handle both single and batch predictions with proper column mapping"""
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    
    # Rename columns to match model expectations if needed
    column_mapping = {
        'League_Home': 'home_team',
        'League_Away': 'away_team',
        # Add other mappings as needed
    }
    df = df.rename(columns=column_mapping)
    
    # Calculate H2H stats if available
    if 'H2H(Latestooldest)' in df.columns:
        df[['H2H_HomeWins', 'H2H_Draws', 'H2H_Losses']] = df.apply(
            lambda row: compute_h2h_stats(
                row['H2H(Latestooldest)'],
                row.get('home_team', ''),
                row.get('away_team', '')
            ),
            axis=1, result_type='expand'
        )
        df = df.drop('H2H(Latestooldest)', axis=1)
    
    # Fill missing values
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0  # Or appropriate default
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df[FEATURE_COLUMNS]

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        input_list = input_data if isinstance(input_data, list) else [input_data]
        
        for i, (pred, row) in enumerate(zip(predictions, input_list)):
            results.append({
                "prediction": pred,
                "home_team": row.get('League_Home', row.get('home_team', 'Unknown')),
                "away_team": row.get('League_Away', row.get('away_team', 'Unknown')),
                "confidence": float(np.max(model.predict_proba(features)[i]))
            })
        
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
