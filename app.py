from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# List of expected feature columns (must match training order)
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
    'FormHomeTeam', 'FormAwayTeam',
    'LeaguePositionHomeTeam', 'LeaguePositionAwayTeam',
    'TotalPointsHome', 'TotalPointsAway',
    'TotalshotspergameHometeam', 'TotalshotspergameAwayteam',
    'ShotsofftargetpergameHometeam', 'ShotsofftargetpergameAwayteam',
    'BlockedshotspergameHometeam', 'BlockedshotspergameAwayteam',
    'CornerspergameHometeam', 'CornerspergameAwayteam',
    'FreekickspergameHometeam', 'FreekickspergameAwayteam',
    'HitwoodworkHometeam', 'HitwoodworkAwayteam',
    'CounterattacksHometeam', 'CounterattacksAwayteam',
    'H2H(Latestooldest)'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received data:", data)

        # Handle both list and single match input
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            return jsonify({"error": "Input should be a list or dict"}), 400

        # Validate required features
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing feature columns: {missing}"}), 400

        # Keep only necessary columns in correct order
        df = df[feature_columns]

        # Make prediction
        predictions = model.predict(df)
        predicted_labels = label_encoder.inverse_transform(predictions)

        print("✅ Predictions:", predicted_labels.tolist())
        return jsonify({"predictions": predicted_labels.tolist()})

    except Exception as e:
        print("🔥 Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
