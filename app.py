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

# Route for form page
@app.route('/')
def index():
    return render_template('form.html')

# Route for handling prediction
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
            # Load and predict
            df = pd.read_csv(filepath)
            prediction = model.predict(df)

            # Map numeric predictions to labels
            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map[p] for p in prediction]

            # Save the full results
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            # Show top 50 in table and provide download link
            table_html = df.head(50).to_html(classes='dataframe', index=False)
            return render_template('results.html', tables=table_html, download_link='/download')

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

# Route to allow download of the results
@app.route('/download')
def download_file():
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_results.csv')
    return send_file(output_path, as_attachment=True)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
