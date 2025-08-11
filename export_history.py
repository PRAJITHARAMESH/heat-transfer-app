# export_history.py
import sqlite3
import pandas as pd

DB = "predictions.db"
OUT = "predictions_history.csv"

conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM predictions ORDER BY Timestamp DESC", conn)
conn.close()

df.to_csv(OUT, index=False)
print(f"✅ Exported {len(df)} rows to {OUT}")
