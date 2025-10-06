from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Use absolute path to your trained model
model_path = r"C:\Users\User\PycharmProjects\Pythonsmart crop yield estimator\models\models\crop_yield_model.joblib"

try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    raise Exception(f"❌ Model not found at {model_path}\n{e}")

@app.route('/')
def home():
    return "🌾 Smart Crop Yield Estimator API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    required_features = ["N", "P", "K", "Fertilizer"]

    # Check for missing inputs
    for feature in required_features:
        if feature not in data:
            return jsonify({"error": f"Missing feature: {feature}"}), 400

    # Create DataFrame for prediction
    X_new = pd.DataFrame([data])
    prediction = model.predict(X_new)[0]
    return jsonify({"Predicted Crop Yield": round(float(prediction), 3)})

if __name__ == '__main__':
    app.run(debug=True)
