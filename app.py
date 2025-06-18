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

def prepare_input(raw_data: Union[Dict, List]) -> Dict:
    """Handle both dictionary and list inputs."""
    if raw_data is None:
        raise ValueError("No data provided")
    
    if isinstance(raw_data, list):
        if not raw_data:
            raise ValueError("Empty list provided")
        if len(raw_data) > 1:
            logger.warning("List contains multiple items - using first item only")
        return raw_data[0]
    
    if isinstance(raw_data, dict):
        return raw_data
    
    raise ValueError(f"Expected dict or list, got {type(raw_data)}")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get and prepare input data
        raw_data = request.get_json()
        input_data = prepare_input(raw_data)
        
        # Validate features
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
        
    except ValueError as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400
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
