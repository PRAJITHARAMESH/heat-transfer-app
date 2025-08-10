import sqlite3

conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

# Add efficiency column if it doesn't exist
try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN efficiency REAL;")
    print("✅ Column 'efficiency' added successfully.")
except Exception as e:
    print("⚠️", e)

conn.commit()
conn.close()
