from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# Initialize Flask app
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

# Define feature columns - ONLY NUMERIC FEATURES
FEATURE_COLUMNS = [
    'OddsHome', 'DrawOdds', 'AwayOdds',
    'SofascoreRatingHomeTeam', 'SofascoreRatingAwayTeam',
    # ... include ALL other numeric features from your test data ...
    # BUT REMOVE ALL STRING FEATURES LIKE FormHomeTeam, H2H, etc.
]

@app.route('/predict', methods=['POST'])
def predict():
    """Simplified prediction endpoint"""
    try:
        # Get raw input data
        raw_data = request.get_json()
        
        # Convert list input to dict if needed
        if isinstance(raw_data, list):
            if not raw_data:
                return jsonify({"error": "Empty list provided"}), 400
            raw_data = raw_data[0]
        
        # Validate input is a dictionary
        if not isinstance(raw_data, dict):
            return jsonify({"error": "Input must be JSON object or list with one object"}), 400
        
        # Prepare DataFrame - will automatically select only FEATURE_COLUMNS
        input_df = pd.DataFrame([raw_data])[FEATURE_COLUMNS]
        
        # Make prediction
        prediction = model.predict(input_df)
        outcome = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            "prediction": outcome,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
