from flask import Flask, request, jsonify
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

app = Flask(__name__)

class MobileFootballPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100)
        self.model.classes_ = np.array([0, 1, 2])  # Home, Draw, Away
        self.model.n_classes_ = 3
        self.model.estimators_ = np.empty((100, 3), dtype=object)  # Dummy placeholders

    def predict(self, match_data):
        # Fake predictions for now (replace with actual model logic)
        return {
            "Category": "Outcome",
            "Prediction": "Draw",
            "Compromise Prediction": "Double Chance (Home/Draw)",
            "Score Prediction": "1-1",
            "Goal Market": "Under 2.5",
            "Confidence": "Medium"
        }

predictor = MobileFootballPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    home = data.get("home_team")
    away = data.get("away_team")
    if not home or not away:
        return jsonify({"error": "Missing team names"}), 400
    prediction = predictor.predict(data)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)