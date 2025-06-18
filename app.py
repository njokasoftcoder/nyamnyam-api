from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
from typing import Dict, Any, Union, List

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and label encoder
try:
    model = joblib.load('match_outcome_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logger.info("Model and label encoder loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise

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
    """Handle prediction requests for match outcomes."""
    try:
        # Get input data
        input_data = request.get_json()
        
        # Handle list input (take first element if it's a list)
        if isinstance(input_data, list):
            if not input_data:
                return jsonify({"error": "Empty list provided"}), 400
            input_data = input_data[0]
            if len(input_data) > 1:
                logger.warning("Received list with multiple items, using first item only")
        
        # Validate input is a dictionary
        if not isinstance(input_data, dict):
            return jsonify({"error": "Input must be a JSON object or list containing one JSON object"}), 400
        
        # Check for missing features
        missing_features = [col for col in FEATURE_COLUMNS if col not in input_data]
        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features,
                "status": "error"
            }), 400
        
        # Prepare DataFrame
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            "prediction": predicted_label,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
