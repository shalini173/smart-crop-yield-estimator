import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocess import load_and_merge_data, preprocess_data

MODEL_PATH = "../models/models/crop_yield_model.joblib"

def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise Exception("❌ Model file not found! Train the model first using train_model.py")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded for evaluation.")
    return model


def evaluate_model():
    """Evaluate model performance with visualizations."""

    # Step 1: Load data
    df = load_and_merge_data()
    X, y = preprocess_data(df)

    # Step 2: Load model
    model = load_model()

    # Step 3: Predictions
    y_pred = model.predict(X)

    # Step 4: Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"📊 RMSE (on full dataset): {rmse:.3f}")
    print(f"📈 R² Score: {r2:.3f}")

    # Step 5: Feature importance (only for tree-based models)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = X.columns

        plt.figure(figsize=(8, 5))
        plt.barh(features, importance, color="green")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title("🌱 Feature Importance in Crop Yield Prediction")
        plt.tight_layout()
        plt.show()

    # Step 6: Prediction vs Actual plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.6, color="blue", edgecolor="k")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("⚖️ Predicted vs Actual Crop Yield")
    plt.tight_layout()
    plt.show()

    # Step 7: Error distribution
    errors = y - y_pred
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=20, color="orange", edgecolor="k")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("📉 Error Distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
