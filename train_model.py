import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
data = pd.read_csv("football_data.csv")  # Replace with your actual file name

# Automatically clean percentage-based columns
for col in data.columns:
    if data[col].astype(str).str.contains('%').any():
        data[col] = data[col].astype(str).str.replace('%', '', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce') / 100

# Define target
target = 'Result'  # Ensure this column exists in your CSV

# Drop rows where target is missing
data = data.dropna(subset=[target])

# Encode target labels (e.g. Home Win, Draw, Away Win → 0, 1, 2)
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])

# Save label encoder for later decoding predictions
joblib.dump(label_encoder, 'label_encoder.pkl')

# Encode categorical columns (like League names)
categorical_columns = data.select_dtypes(include=['object']).columns.drop(target, errors='ignore')
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Fill numeric missing values with league-wise mean if available, else column mean
numeric_columns = data.select_dtypes(include='number').columns.drop(target)
for feature in numeric_columns:
    if 'League_Home' in data.columns:
        data[feature] = data.groupby('League_Home')[feature].transform(lambda x: x.fillna(x.mean()))
    data[feature] = data[feature].fillna(data[feature].mean())

# Define features and labels
X = data.drop(columns=[target])
y = data[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model
joblib.dump(model, 'football_model.pkl')
print("Model saved as football_model.pkl")
