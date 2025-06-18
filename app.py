from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import sys
import os

app = Flask(__name__)

# NUCLEAR OPTION - Verify we're running the right file
print(f"🔥 NUCLEAR VERIFICATION - Running from: {os.path.abspath(__file__)}", file=sys.stderr)
print("✅ THIS IS THE CLEAN VERSION WITHOUT ANY TEAM NAME CHECKS", file=sys.stderr)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model - SIMPLIFIED VERSION
try:
    model = joblib.load('match_outcome_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

# ONLY NUMERIC FEATURES - NO TEAM NAMES WHATSOEVER
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
    """Nuclear-proof prediction endpoint"""
    try:
        # Get raw input
        raw_data = request.get_json()
        logger.info(f"Raw input data type: {type(raw_data)}")
        
        # Convert list to dict if needed
        if isinstance(raw_data, list):
            if not raw_data:
                return jsonify({"error": "Empty list provided"}), 400
            raw_data = raw_data[0]
        
        # Validate input type
        if not isinstance(raw_data, dict):
            return jsonify({"error": "Input must be JSON object"}), 400
        
        # DEBUG: Show all received keys
        logger.info(f"Received keys: {list(raw_data.keys())}")
        
        # Create DataFrame with ONLY our features
        try:
            input_df = pd.DataFrame([raw_data])[FEATURE_COLUMNS]
        except KeyError as e:
            missing = [col for col in FEATURE_COLUMNS if col not in raw_data]
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing,
                "required_features": FEATURE_COLUMNS
            }), 400
        
        # Make prediction
        prediction = model.predict(input_df)
        return jsonify({
            "prediction": str(prediction[0]),
            "status": "success",
            "features_used": FEATURE_COLUMNS
        })
        
    except Exception as e:
        logger.error(f"COMPLETE FAILURE: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Prediction failed",
            "exception_type": str(type(e)),
            "exception_details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
