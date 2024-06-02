import sqlite3
import numpy as np

def get_db_connection():
    conn = sqlite3.connect('image_features.db', check_same_thread=False)
    return conn

def create_table():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS features
                 (id INTEGER PRIMARY KEY, img_path TEXT, features BLOB)''')
    conn.commit()
    conn.close()

def store_features(img_path, features):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO features (img_path, features) VALUES (?, ?)", (img_path, features.tobytes()))
    conn.commit()
    conn.close()

def get_all_features():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT img_path, features FROM features")
    rows = c.fetchall()
    conn.close()
    return rows

def clear_database():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM features")
    conn.commit()
    conn.close()
