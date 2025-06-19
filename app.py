from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import traceback

app = Flask(__name__)

model = joblib.load('match_outcome_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature columns
feature_columns = [...]  # keep as is (your full standardized list)

# H2H parsing helper
def compute_h2h_stats(h2h_str, home_team):
    home_win, draw, away_win = 0, 0, 0
    matches = h2h_str.split(',')
    for match in matches:
        score = re.search(r"\((\d+):(\d+)\)", match)
        if score:
            home_goals, away_goals = map(int, score.groups())
            if home_goals > away_goals:
                home_win += 1
            elif home_goals == away_goals:
                draw += 1
            else:
                away_win += 1
    return home_win, draw, away_win

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if isinstance(input_data, dict):
            input_data = [input_data]

        df = pd.DataFrame(input_data)

        # H2H computation
        if 'H2H(Latestooldest)' in df.columns:
            h2h_home_wins, h2h_draws, h2h_losses = [], [], []
            for idx, row in df.iterrows():
                try:
                    home_team = row["HomeTeam"] if "HomeTeam" in row else "Home"
                    h2h_str = row["H2H(Latestooldest)"]
                    wins, draws_, losses = compute_h2h_stats(h2h_str, home_team)
                    h2h_home_wins.append(wins)
                    h2h_draws.append(draws_)
                    h2h_losses.append(losses)
                except Exception:
                    h2h_home_wins.append(0)
                    h2h_draws.append(0)
                    h2h_losses.append(0)

            df["H2H_HomeWins"] = h2h_home_wins
            df["H2H_Draws"] = h2h_draws
            df["H2H_Losses"] = h2h_losses
            df.drop(columns=["H2H(Latestooldest)"], inplace=True)

        # Fill missing columns
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]

        # Make prediction
        pred_encoded = model.predict(df)
        predictions = label_encoder.inverse_transform(pred_encoded)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        print("🔥 ERROR during prediction:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
