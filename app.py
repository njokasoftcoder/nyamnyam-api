from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import sys

app = Flask(__name__)

# Debugging check
print("✅ Running ULTIMATE CLEAN VERSION OF APP.PY", file=sys.stderr)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model - SIMPLIFIED
try:
    model = joblib.load('match_outcome_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

# ONLY NUMERIC FEATURES - NO TEAM NAMES
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
    'Savespergame', 'DuelswonpergameHometeam', 'DuelswonpergameAwayteam',
    'FoulspergameHometeam', 'FoulspergameAwayteam'
]

@app.route('/predict', methods=['POST'])
def predict():
    """Simplest possible prediction endpoint"""
    try:
        # Get raw JSON data
        data = request.get_json()
        
        # Debug output
        logger.info(f"Received data: {str(data)[:200]}...")
        
        # Convert list to dict if needed
        if isinstance(data, list):
            if not data:
                return jsonify({"error": "Empty list provided"}), 400
            data = data[0]
        
        # Create DataFrame
        try:
            input_df = pd.DataFrame([data])[FEATURE_COLUMNS]
        except KeyError as e:
            missing = [col for col in FEATURE_COLUMNS if col not in data]
            return jsonify({"error": "Missing features", "missing": missing}), 400
        
        # Make prediction
        prediction = model.predict(input_df)
        return jsonify({
            "prediction": str(prediction[0]),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
