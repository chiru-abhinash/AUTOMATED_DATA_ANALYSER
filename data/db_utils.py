import sqlite3

def connect_to_db():
    return sqlite3.connect("../database.db")

def execute_query(query, params=()):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def fetch_all(query, params=()):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    conn.close()
    return result


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row  # Allows access to columns by name
    return conn

def close_db_connection(conn):
    conn.close()


