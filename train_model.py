# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3

CSV_NAME = "heat_transfer_dataset.csv"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DB_FILE = "predictions.db"

def load_or_create_dataset():
    if os.path.exists(CSV_NAME):
        print(f"Loading dataset from {CSV_NAME}")
        df = pd.read_csv(CSV_NAME)
        return df
    print("No CSV found — creating synthetic dataset...")
    np.random.seed(42)
    n = 1000
    thermal = np.random.uniform(80, 400, n)
    source = np.random.uniform(40, 75, n)
    ambient = np.random.uniform(15, 35, n)
    block = np.random.uniform(5, 50, n)

    center = source - np.random.normal(0, 1.5, n)
    avg = ambient + (source - ambient) * (0.2 + 0.6 * (thermal / thermal.max())) - 0.05 * block
    max_temp = np.maximum(center, avg) + np.random.normal(0, 2, n) + (1000.0 / (thermal + 1)) * 0.5

    df = pd.DataFrame({
        "ThermalCond": thermal,
        "SourceTemp": source,
        "AmbientTemp": ambient,
        "BlockSize": block,
        "MaxTemp": max_temp,
        "AvgTemp": avg,
        "CenterTemp": center
    })
    df.to_csv(CSV_NAME, index=False)
    print(f"Synthetic dataset created and saved to {CSV_NAME}")
    return df

def train_and_save(df):
    X = df[["ThermalCond", "SourceTemp", "AmbientTemp", "BlockSize"]]
    y = df[["MaxTemp", "AvgTemp", "CenterTemp"]]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Saved model -> {MODEL_FILE}, scaler -> {SCALER_FILE}")

    from sklearn.metrics import r2_score, mean_squared_error
    y_pred = model.predict(X_test)
    print("R2 (avg):", np.mean([r2_score(y_test.iloc[:,i], y_pred[:,i]) for i in range(3)]))
    print("MSE:", mean_squared_error(y_test, y_pred))

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ThermalCond REAL,
            SourceTemp REAL,
            AmbientTemp REAL,
            BlockSize REAL,
            MaxTemp REAL,
            AvgTemp REAL,
            CenterTemp REAL,
            Efficiency REAL,
            Status TEXT,
            Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"DB initialized: {DB_FILE}")

if __name__ == "__main__":
    df = load_or_create_dataset()
    train_and_save(df)
    init_db()
    print("Done. Run: python app.py")
