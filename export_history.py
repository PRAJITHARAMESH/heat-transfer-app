import sqlite3
import pandas as pd

# Connect DB
conn = sqlite3.connect("predictions.db")

# Read table
df = pd.read_sql_query("SELECT * FROM predictions", conn)

# Export CSV
df.to_csv("predictions_history.csv", index=False)

conn.close()
print("✅ History exported to predictions_history.csv")
