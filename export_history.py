import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_FILE = "predictions.db"   # Database file name
EXPORT_FILE = f"history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"  # CSV with timestamp

# Check if database file exists
if not os.path.exists(DB_FILE):
    print(f"❌ Database file '{DB_FILE}' not found!")
else:
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check if 'predictions' table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        if cursor.fetchone() is None:
            print("❌ Table 'predictions' not found in database.")
        else:
            # Read table data sorted by ID in descending order
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)

            if df.empty:
                print("⚠️ No data found in 'predictions' table.")
            else:
                # Save to CSV
                df.to_csv(EXPORT_FILE, index=False)
                print(f"✅ Exported {len(df)} rows to '{EXPORT_FILE}'")

        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
