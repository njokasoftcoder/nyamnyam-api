from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

# Feature processing reusable function
def process_input(data):
    # Calculate derived features
    home_form_pts = 3*data['home_form'].count('W') + data['home_form'].count('D')
    away_form_pts = 3*data['away_form'].count('W') + data['away_form'].count('D')
    rating_diff = data['home_rating'] - data['away_rating']
    pos_diff = data['home_pos'] - data['away_pos']
    
    # Parse H2H (mock - replace with your actual parser)
    h2h_win_pct = 0.6  # Example: replace with real parsing logic
    avg_home_goals = 1.2
    
    return pd.DataFrame([[
        data['home_rating'], data['away_rating'], rating_diff,
        data['home_pos'], data['away_pos'], pos_diff,
        home_form_pts, away_form_pts,
        data['home_gpg'], data['away_gpg'],
        data['home_gcpg'], data['away_gcpg'],
        data['home_sot'], data['away_sot'],
        h2h_win_pct, avg_home_goals,
        data['home_possession'], data['home_corners'],
        data['home_yellows'], data['away_yellows']
    ]], columns=[
        'Sofascore Rating HomeTeam', 'Sofascore Rating AwayTeam', 'Rating_Diff',
        'League Position HomeTeam', 'League Position AwayTeam', 'Position_Diff',
        'Home_Form_Points', 'Away_Form_Points',
        'Goals per game HomeTeam', 'Goals per game AwayTeam',
        'Goals conceded per game HomeTeam', 'Goals conceded per game AwayTeam',
        'Shots on target per game HomeTeam', 'Shots on target per game AwayTeam',
        'H2H_WinPct', 'H2H_AvgHomeGoals',
        'Ball possession HomeTeam', 'Corners per game HomeTeam',
        'Total yellow cards awarded HomeTeam', 'Total yellow cards awarded AwayTeam'
    ])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = process_input(data)
    proba = model.predict_proba(input_df)[0]
    
    return jsonify({
        'probabilities': {
            'HomeWin': float(proba[0]),
            'Draw': float(proba[1]),
            'AwayWin': float(proba[2])
        },
        'predicted_outcome': ['HomeWin', 'Draw', 'AwayWin'][np.argmax(proba)]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
