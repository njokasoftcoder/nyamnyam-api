import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file, url_for
import joblib
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Load the trained model
model = joblib.load('football_match_predictor.pkl')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Allowed file extensions check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Home route - upload form
@app.route('/')
def index():
    return render_template('form.html')

# Prediction route
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
            # Load and prepare data
            df = pd.read_csv(filepath)
            predictions = model.predict(df)

            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map[p] for p in predictions]

            # Add match numbering
            df.insert(0, 'Match No.', range(1, len(df) + 1))

            # Save to static folder for download
            output_path = os.path.join(app.config['STATIC_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            # Prepare styled table
            def color_prediction(val):
                if val == 'Home Win':
                    return 'background-color: #d4edda; color: #155724;'  # Green
                elif val == 'Draw':
                    return 'background-color: #fff3cd; color: #856404;'  # Yellow
                elif val == 'Away Win':
                    return 'background-color: #f8d7da; color: #721c24;'  # Red
                return ''

            styled_table = df.style.applymap(color_prediction, subset=['Prediction']).hide_index()
            table_html = styled_table.to_html()

            return render_template('results.html', table_html=table_html)

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
