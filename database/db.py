import sqlite3

DB_PATH = 'face_attendance.db'

def connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    with connect() as conn:
        with open('database/schema.sql') as f:
            conn.executescript(f.read())

