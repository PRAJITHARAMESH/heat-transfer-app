import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load CSV
data = pd.read_csv("heat_transfer_dataset.csv")  # Place the file in same folder

# Step 2: Split features and targets
X = data[["ThermalCond", "SourceTemp", "AmbientTemp", "BlockSize"]]
y = data[["MaxTemp", "AvgTemp", "CenterTemp"]]

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluation Function with Echo
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

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

# Step 8: Run Evaluation
evaluate_model(y_test, y_pred)
