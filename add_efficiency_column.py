# add_efficiency_column.py
import sqlite3
import pandas as pd
import os

DB = "predictions.db"
if not os.path.exists(DB):
    print("DB not found.")
    raise SystemExit

conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM predictions", conn)

if "Efficiency" not in df.columns:
    df["Efficiency"] = (df["SourceTemp"] - df["AvgTemp"]) / df["SourceTemp"] * 100
    df.to_sql("predictions_temp", conn, index=False, if_exists="replace")
    c = conn.cursor()
    c.execute("DROP TABLE predictions")
    c.execute("ALTER TABLE predictions_temp RENAME TO predictions")
    conn.commit()
    print("Efficiency column added.")
else:
    print("Efficiency column already present.")
conn.close()
