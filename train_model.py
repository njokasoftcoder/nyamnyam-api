import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
data = pd.read_csv('sample_match_data.csv')

# Target variable
target = 'Result'

# Drop rows where target is missing
data = data.dropna(subset=[target])

# List of all features (automatically select numeric columns only)
features = data.select_dtypes(include=['number']).columns.tolist()

# Drop the target column from features (just in case it's numeric)
if target in features:
    features.remove(target)

# Fill missing values per league group (if League_Home column exists)
if 'League_Home' in data.columns:
    for feature in features:
        data[feature] = data.groupby('League_Home')[feature].transform(lambda x: x.fillna(x.mean()))
else:
    # Fallback: fill missing values with overall mean
    data[features] = data[features].fillna(data[features].mean())

# Define X and y
X = data[features]
y = data[target]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'match_outcome_model.pkl')
print("\nModel saved as match_outcome_model.pkl")
