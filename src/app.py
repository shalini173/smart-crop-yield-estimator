import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Corrected path for your structure
# (src → .. → project root → models → models → file)
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "models", "crop_yield_model.joblib")

# Load the model safely
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")

@app.route('/')
def home():
    return "✅ Smart Crop Yield Estimator is running successfully!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)