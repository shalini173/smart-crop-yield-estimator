# Example: train_model.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your dataset
data = pd.read_csv(r"C:\Users\User\PycharmProjects\Pythonsmart crop yield estimator\data\Crop Yiled with Soil and Weather.csv")
print(data.columns)
# Features and target
X = data.drop("yeild", axis=1)
y = data["yeild"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
import joblib
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "../models/models/crop_yield_model.joblib")
print("Model saved successfully!")

