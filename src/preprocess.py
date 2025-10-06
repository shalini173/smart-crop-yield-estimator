import os
import pandas as pd


def load_and_merge_data():
    # Load only the main dataset
    filepath = os.path.join(os.path.dirname(__file__), "../data/Crop Yiled with Soil and Weather.csv")
    if not os.path.exists(filepath):
        raise Exception(f"❌ Dataset not found: {filepath}")

    df = pd.read_csv(r"C:\Users\User\PycharmProjects\Pythonsmart crop yield estimator\data\Crop Yiled with Soil and Weather.csv")
    df.columns = df.columns.str.strip()  # remove extra spaces
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    return df


def preprocess_data(df):
    # Select features
    features = [
        "N",
        "P",
        "K",
        "Fertilizer"
    ]
    target = "yeild"

    # Check columns
    for col in features + [target]:
        if col not in df.columns:
            raise Exception(f"❌ Missing column in dataset: {col}")

    X = df[features].fillna(0)
    y = df[target].fillna(0)
    return X, y
