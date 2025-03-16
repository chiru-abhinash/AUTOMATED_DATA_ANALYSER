import sqlite3
import os # Add this import statement

def create_database():
    #conn = sqlite3.connect("../database.db")  # Path to database in the main folder
    #cursor = conn.cursor()
    # Get the present working directory 
    current_dir = os.getcwd() # Create a path to the database in the present working directory 
    db_path = os.path.join(current_dir, "database.db") # Connect to the database 
    conn = sqlite3.connect(db_path) 
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create datasets table with user_id foreign key
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        file_path TEXT NOT NULL,
        user_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # Create insights table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        insight_text TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    """)

    # Create analyses table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        accuracy REAL,
        result_path TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS algorithm_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        algorithm_name TEXT NOT NULL,
        algorithm_description TEXT NOT NULL,
        ml_code TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    """)



    print("Database and tables created successfully.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
