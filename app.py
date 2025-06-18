from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
from typing import Dict, Any, Union

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

def validate_input(input_data: Union[Dict[str, Any], list]) -> bool:
    """Validate that input contains all required features with numeric values."""
    if isinstance(input_data, list):
        if len(input_data) == 0:
            logger.error("Empty list provided as input")
            return False
        input_data = input_data[0]  # Take first element if it's a list
    
    if not isinstance(input_data, dict):
        logger.error(f"Expected dict, got {type(input_data)}")
        return False
    
    if not all(col in input_data for col in FEATURE_COLUMNS):
        missing = [col for col in FEATURE_COLUMNS if col not in input_data]
        logger.error(f"Missing features in input: {missing}")
        return False
    
    if not all(isinstance(input_data[col], (int, float)) for col in FEATURE_COLUMNS):
        invalid = [col for col in FEATURE_COLUMNS if not isinstance(input_data[col], (int, float))]
        logger.error(f"Non-numeric values found for features: {invalid}")
        return False
    
    return True

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict match outcome based on input features.
    
    Accepts either:
    - A single JSON object with all features
    - A list containing one JSON object with all features
    
    Returns JSON with prediction ('Home', 'Draw', or 'Away') or error message.
    """
    try:
        # Get and validate input data
        input_data = request.get_json()
        if input_data is None:
            return jsonify({"error": "No input data provided"}), 400
        
        if not validate_input(input_data):
            return jsonify({"error": "Invalid input data format or values"}), 400
        
        # Handle both list and dict input
        if isinstance(input_data, list):
            input_data = input_data[0]  # Take first element if it's a list
        
        # Prepare DataFrame with correct column order
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        logger.info(f"Prediction successful: {predicted_label}")
        return jsonify({
            "prediction": predicted_label,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
