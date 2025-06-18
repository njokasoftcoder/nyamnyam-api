from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
from typing import Dict, Any

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

# Feature columns must match model training exactly
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

def validate_input(input_data: Dict[str, Any]) -> bool:
    """Validate that input contains all required features with numeric values."""
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
    
    Expects JSON input with all required features as numeric values.
    Returns JSON with prediction ('Home', 'Draw', or 'Away') or error message.
    """
    try:
        # Get and validate input data
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        if not validate_input(input_data):
            return jsonify({"error": "Invalid input data"}), 400
        
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
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
