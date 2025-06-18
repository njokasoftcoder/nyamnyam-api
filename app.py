from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model components
try:
    model = joblib.load('match_outcome_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logger.info("Model components loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model files: {str(e)}")
    raise

# Complete feature set - must match training exactly
REQUIRED_FEATURES = [
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

def validate_input(input_data):
    """Ensure input contains all required features with numeric values."""
    if isinstance(input_data, list):
        if not input_data:
            raise ValueError("Empty list provided")
        input_data = input_data[0]  # Use first item if list
    
    if not isinstance(input_data, dict):
        raise ValueError("Input must be a dictionary or list containing one dictionary")
    
    missing = [f for f in REQUIRED_FEATURES if f not in input_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    non_numeric = [f for f in REQUIRED_FEATURES if not isinstance(input_data.get(f), (int, float))]
    if non_numeric:
        raise ValueError(f"Non-numeric values found for features: {non_numeric}")
    
    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    """Handle match outcome prediction requests."""
    try:
        # Get and validate input
        raw_data = request.get_json()
        if raw_data is None:
            return jsonify({"error": "No data provided"}), 400
        
        input_data = validate_input(raw_data)
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data], columns=REQUIRED_FEATURES)
        
        # Make prediction
        prediction = model.predict(input_df)
        outcome = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            "prediction": outcome,
            "status": "success"
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
