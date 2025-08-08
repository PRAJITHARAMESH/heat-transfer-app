import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib

# Step 1: Load your dataset
data = pd.read_csv("heat_transfer_dataset.csv")

# Step 2: Add synthetic data to improve training
np.random.seed(42)
num_synthetic = 200  # number of extra samples

thermal_cond = np.random.uniform(80, 400, num_synthetic)
source_temp = np.random.uniform(40, 75, num_synthetic)
ambient_temp = np.random.uniform(15, 35, num_synthetic)
block_size = np.random.uniform(5, 50, num_synthetic)

# Simulate target values with some realistic relationships + noise
max_temp = source_temp + (thermal_cond / 500) + np.random.normal(0, 1, num_synthetic)
avg_temp = (source_temp + ambient_temp) / 2 + np.random.normal(0, 1, num_synthetic)
center_temp = avg_temp - np.random.uniform(0, 5, num_synthetic)

synthetic_df = pd.DataFrame({
    "ThermalCond": thermal_cond,
    "SourceTemp": source_temp,
    "AmbientTemp": ambient_temp,
    "BlockSize": block_size,
    "MaxTemp": max_temp,
    "AvgTemp": avg_temp,
    "CenterTemp": center_temp
})

# Step 3: Merge real + synthetic
data = pd.concat([data, synthetic_df], ignore_index=True)

# Step 4: Features & Targets
X = data[["ThermalCond", "SourceTemp", "AmbientTemp", "BlockSize"]]
y = data[["MaxTemp", "AvgTemp", "CenterTemp"]]

# Step 5: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Model Training (XGBoost tuned)
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Step 8: Predictions & Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"🔍 R² Score: {r2:.3f}")
print(f"🔍 MSE: {mse:.3f}")

if r2 > 0.80:
    print("✅ Very good model – Ready for deployment")
elif r2 > 0.60:
    print("👍 Good model – Acceptable for project")
elif r2 > 0.40:
    print("⚠️ Weak model – Try improving with tuning or more data")
else:
    print("❌ Poor model – Change algorithm or collect more data")

# Step 9: Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("💾 Model and scaler saved successfully!")
