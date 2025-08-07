import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 🔹 Step 1: Load your dataset
data = pd.read_csv("heat_transfer_dataset.csv")  # Ensure this CSV file is in the same folder

# 🔹 Step 2: Define input features and target outputs
X = data[["ThermalCond", "SourceTemp", "AmbientTemp", "BlockSize"]]
y = data[["MaxTemp", "AvgTemp", "CenterTemp"]]

# 🔹 Step 3: Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 Step 5: Train Random Forest model
model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 🔹 Step 6: Save the model and the scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ model.pkl and scaler.pkl saved successfully in your current folder.")
