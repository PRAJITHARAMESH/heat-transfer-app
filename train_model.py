import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
import os

DATA_FILE = "heat_transfer_dataset.csv"
MODEL_FILE = "model.pkl"
RANDOM_STATE = 42

def create_synthetic(num=200, seed=RANDOM_STATE):
    np.random.seed(seed)
    thermal_cond = np.random.uniform(80, 400, num)
    source_temp = np.random.uniform(40, 75, num)
    ambient_temp = np.random.uniform(15, 35, num)
    block_size = np.random.uniform(5, 50, num)
    max_temp = source_temp + (thermal_cond / 500) + np.random.normal(0, 1, num)
    avg_temp = (source_temp + ambient_temp) / 2 + np.random.normal(0, 1, num)
    center_temp = avg_temp - np.random.uniform(0, 5, num)
    df = pd.DataFrame({
        "ThermalCond": thermal_cond,
        "SourceTemp": source_temp,
        "AmbientTemp": ambient_temp,
        "BlockSize": block_size,
        "MaxTemp": max_temp,
        "AvgTemp": avg_temp,
        "CenterTemp": center_temp
    })
    return df

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please add your dataset.")
    return pd.read_csv(path)

def train_and_save():
    # Load & augment data
    data = load_data(DATA_FILE)
    synth = create_synthetic(num=200)
    data = pd.concat([data, synth], ignore_index=True)

    # Features and targets
    X = data[["ThermalCond", "SourceTemp", "AmbientTemp", "BlockSize"]]
    y = data[["MaxTemp", "AvgTemp", "CenterTemp"]]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Multi-output regressor with XGB
    base_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    # Evaluate
    from sklearn.metrics import r2_score, mean_squared_error

    preds = model.predict(X_test)

    print("\nMetrics per target:")
    for i, col in enumerate(y_test.columns):
        r2_val = r2_score(y_test[col], preds[:, i])
        mse_val = mean_squared_error(y_test[col], preds[:, i])
        print(f" {col}: R2 = {r2_val:.3f}, MSE = {mse_val:.3f}")

    # Save model & scaler together as dict
    joblib.dump({"model": model, "scaler": scaler}, MODEL_FILE)
    print(f"\n✅ Model and scaler saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save()
