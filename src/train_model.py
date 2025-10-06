import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from src.preprocess import load_and_merge_data, preprocess_data

MODEL_PATH = "../models/models/crop_yield_model.joblib"


def train_and_save_model():
    """Train the regression model and save it."""

    # Step 1: Load and preprocess data
    df = load_and_merge_data()
    X, y = preprocess_data(df)

    # Step 2: Train-test split (for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 3: Initialize model
    model = RandomForestRegressor(
        n_estimators=200,  # number of trees
        max_depth=10,  # control tree depth
        random_state=42
    )

    # Step 4: Cross-validation (RMSE)
    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring="neg_root_mean_squared_error")
    print(f"✅ Cross-Validation RMSE: {(-scores).mean():.3f}")

    # Step 5: Train final model
    model.fit(X_train, y_train)

    # Step 6: Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"📊 Test RMSE: {rmse:.3f}")

    # Step 7: Save trained model
    os.makedirs("../models/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
