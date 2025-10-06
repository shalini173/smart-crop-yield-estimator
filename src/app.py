from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), r"C:\Users\User\PycharmProjects\Pythonsmart crop yield estimator\models\models\crop_yield_model.joblib")
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Smart Crop Yield Estimator API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data.get('N', 0),
        data.get('P', 0),
        data.get('K', 0)
    ]
    prediction = model.predict([features])[0]
    return jsonify({"predicted_yield": round(prediction, 2)})

if __name__ == '__main__':
    # Dynamic port for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
