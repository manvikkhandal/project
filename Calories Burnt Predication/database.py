import sqlite3

# Database connection
database_name = 'Database/Calorie.db'
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# Create table with the specified columns
cursor.execute('''
CREATE TABLE IF NOT EXISTS calorie_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    day TEXT NOT NULL,
    calories REAL NOT NULL,
    gender TEXT NOT NULL,
    age INTEGER NOT NULL,
    height REAL NOT NULL,
    weight REAL NOT NULL,
    duration REAL NOT NULL,
    heart_rate REAL NOT NULL,
    body_temp REAL NOT NULL
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()