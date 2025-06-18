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

# Complete feature columns - exactly as used during model training
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

def prepare_input_data(input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Convert input data to a single dictionary format."""
    if input_data is None:
        raise ValueError("No input data provided")
    
    if isinstance(input_data, list):
        if not input_data:
            raise ValueError("Empty list provided")
        if len(input_data) > 1:
            logger.warning("Received list with multiple items, using first item only")
        return input_data[0]
    
    if isinstance(input_data, dict):
        return input_data
    
    raise ValueError(f"Unexpected input type: {type(input_data)}")

def validate_input(input_dict: Dict[str, Any]) -> bool:
    """Validate that input contains all required features with numeric values."""
    if not all(col in input_dict for col in FEATURE_COLUMNS):
        missing = [col for col in FEATURE_COLUMNS if col not in input_dict]
        logger.error(f"Missing features in input: {missing}")
        return False
    
    if not all(isinstance(input_dict[col], (int, float)) for col in FEATURE_COLUMNS):
        invalid = [col for col in FEATURE_COLUMNS if not isinstance(input_dict[col], (int, float))]
        logger.error(f"Non-numeric values found for features: {invalid}")
        return False
    
    return True

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict match outcome based on input features.
    
    Accepts either:
    - A single JSON object: {"feature1": value1, "feature2": value2, ...}
    - A list containing one JSON object: [{"feature1": value1, ...}]
    
    Returns JSON with prediction ('Home', 'Draw', or 'Away') or error message.
    """
    try:
        # Get input data
        raw_data = request.get_json()
        
        # Convert to standard dictionary format
        input_dict = prepare_input_data(raw_data)
        
        # Validate the input
        if not validate_input(input_dict):
            return jsonify({
                "error": "Invalid input data",
                "required_features": FEATURE_COLUMNS,
                "received_features": list(input_dict.keys())
            }), 400
        
        # Prepare DataFrame
        input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            "prediction": predicted_label,
            "status": "success"
        })
        
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
