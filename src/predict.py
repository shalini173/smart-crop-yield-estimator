import os
import joblib
import pandas as pd

# Path to the saved model
model_path = os.path.join(os.path.dirname(__file__), "../models/models/crop_yield_model.joblib")

# Load the trained model
if not os.path.exists(model_path):
    raise Exception(f"❌ Model not found at {model_path}")

model = joblib.load(model_path)
print("✅ Model loaded successfully.\n")

# List of features needed for prediction
features = [
    "N",
    "P",
    "K",
    "Fertilizer"
]

# Ask user for inputs
input_data = {}
for feature in features:
    while True:
        try:
            value = float(input(f"Enter value for {feature}: "))
            input_data[feature] = value
            break
        except ValueError:
            print("❌ Please enter a valid number.")

# Convert to DataFrame
X_new = pd.DataFrame([input_data])

# Predict crop yield
prediction = model.predict(X_new)[0]
print(f"\n📈 Predicted Crop Yield: {prediction:.3f}")
