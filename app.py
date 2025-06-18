from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Convert form string (e.g., 'WDLDW') to points
def transform_form(form_str):
    if not isinstance(form_str, str):
        return 0
    points = 0
    for c in form_str.upper():
        if c == 'W':
            points += 3
        elif c == 'D':
            points += 1
    return points

# Feature columns expected by the model (MUST match training)
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
    'FormHomePoints', 'FormAwayPoints'
]

@app.route('/')
def home():
    return "Nyam Nyam Confidence Fire Prediction is 🔥 live."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if not isinstance(input_data, list):
            return jsonify({"error": "Input must be a list of match records."}), 400

        df = pd.DataFrame(input_data)
        print("\n📥 Received DataFrame:")
        print(df.head())

        # Convert form fields to numeric points
        if 'FormHomeTeam' in df.columns and 'FormAwayTeam' in df.columns:
            df['FormHomePoints'] = df['FormHomeTeam'].apply(transform_form)
            df['FormAwayPoints'] = df['FormAwayTeam'].apply(transform_form)
            df.drop(['FormHomeTeam', 'FormAwayTeam'], axis=1, inplace=True)

        # Check for required features
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return jsonify({"error": f"Missing required fields: {missing_cols}"}), 400

        df = df[feature_columns]
        print("\n📊 Preprocessed Features:")
        print(df.head())

        predictions = model.predict(df)
        decoded_preds = label_encoder.inverse_transform(predictions)

        print("\n✅ Predictions:")
        print(decoded_preds)

        return jsonify({"predictions": decoded_preds.tolist()})

    except Exception as e:
        print("\n❌ Error occurred during prediction:")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
