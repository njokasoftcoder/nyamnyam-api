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

# Load model and vectorizer
model = joblib.load('football_match_predictor.pkl')

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('form.html')

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

            # Get prediction probabilities
            probs = model.predict_proba(df)
            preds = model.predict(df)

            label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            df['Prediction'] = [label_map[p] for p in preds]
            df['Confidence'] = [f"{np.max(p) * 100:.1f}%" for p in probs]

            # Add match numbering
            df.insert(0, 'Match No.', range(1, len(df) + 1))

            # Save to static folder for download
            output_path = os.path.join(app.config['STATIC_FOLDER'], 'predicted_results.csv')
            df.to_csv(output_path, index=False)

            # Summary counts
            summary = df['Prediction'].value_counts().to_dict()

            # Top confident picks (optional)
            df['ConfidenceNum'] = [np.max(p) for p in probs]
            top_confident = df.sort_values('ConfidenceNum', ascending=False).head(3)[
                ['Match No.', 'Prediction', 'Confidence']
            ]

            def color_prediction(val):
                if val == 'Home Win':
                    return 'background-color: #d4edda; color: #155724;'
                elif val == 'Draw':
                    return 'background-color: #fff3cd; color: #856404;'
                elif val == 'Away Win':
                    return 'background-color: #f8d7da; color: #721c24;'
                return ''

            styled_table = df.drop(columns=['ConfidenceNum']).style \
                .applymap(color_prediction, subset=['Prediction']) \
                .hide_index()
            table_html = styled_table.to_html()

            return render_template(
                'results.html',
                table_html=table_html,
                summary=summary,
                top_confident=top_confident.to_dict(orient='records')
            )

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return "Invalid file format. Please upload a CSV."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
