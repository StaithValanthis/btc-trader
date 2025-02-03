# test_connection.py
from database import TimescaleDB

try:
    db = TimescaleDB()
    print("Successfully connected to TimescaleDB!")
except Exception as e:
    print(f"Connection failed: {e}")