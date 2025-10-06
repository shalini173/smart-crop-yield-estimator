import pandas as pd

def load_and_merge_data():
    # Load CSVs
    crop = pd.read_csv("data/crop_dataset_seasonal.csv")
    weather = pd.read_csv("data/weather.csv")

    # Merge datasets (assuming they have 'Year' and 'Region' columns in common)
    df = crop.merge(weather, on=["Year", "Region"])
    df = df.merge(ndvi, on=["Year", "Region"])

    print("✅ Data loaded and merged successfully")
    return df

def preprocess_data(df):
    # Drop missing values
    df = df.dropna()

    # Example feature selection (update based on your dataset)
    features = ["Rainfall", "Temperature", "NDVI", "Soil_N", "Soil_P", "Soil_K"]
    target = "Yield"

    X = df[features]
    y = df[target]

    return X, y
