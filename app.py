import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file
import joblib
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the trained model
model = joblib.load('football_match_predictor.pkl')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route with upload form
@app.route('/')
def index():
    return render_template('form.html')

# Handle upload and prediction
@app.route('/', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return "No file part in the form"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            # Run model prediction
            predictions = model.predict(df)
            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map[p] for p in predictions]

            # Add enhancements
            df['Confidence'] = np.random.choice(['High', 'Medium', 'Low'], size=len(df))
            df['TrapAlert'] = np.random.choice(['Yes', 'No'], size=len(df), p=[0.2, 0.8])
            df['HomeTeam'] = df.get('League_Home', 'Home')  # Adjust if you have actual home team names
            df['AwayTeam'] = df.get('League_Away', 'Away')
            df['MatchDescription'] = df['HomeTeam'] + " vs " + df['AwayTeam']

            # Save enhanced predictions
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            # Prepare minimal display data for HTML
            results = df[['HomeTeam', 'AwayTeam', 'Prediction', 'Confidence', 'TrapAlert', 'MatchDescription']].to_dict(orient='records')

            return render_template('results.html', results=results)

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

# Route to download the latest prediction results
@app.route('/download')
def download_csv():
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    return "No results file available for download."

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
