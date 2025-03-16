import sqlite3
import os

def add_dataset(csv_file, description, user_id):
    # Ensure the raw data folder exists
    raw_data_folder = "../data/raw/"
    os.makedirs(raw_data_folder, exist_ok=True)

    # Save CSV file to the raw folder
    file_name = os.path.basename(csv_file)
    file_path = os.path.join(raw_data_folder, file_name)

    # Move the file to the raw data folder (if not already there)
    if not os.path.exists(file_path):
        os.rename(csv_file, file_path)

    # Connect to the database
    conn = sqlite3.connect("../database.db")
    cursor = conn.cursor()

    # Insert dataset metadata into the `datasets` table with user_id
    cursor.execute("""
    INSERT INTO datasets (name, description, file_path, user_id) 
    VALUES (?, ?, ?, ?)
    """, (file_name, description, file_path, user_id))

    conn.commit()
    conn.close()
    print(f"Dataset '{file_name}' added to the database for user ID {user_id}.")

def add_insight(dataset_id, insight_text):
    # Connect to the database
    conn = sqlite3.connect("../database.db")
    cursor = conn.cursor()

    # Insert insight into the `insights` table
    cursor.execute("""
    INSERT INTO insights (dataset_id, insight_text) 
    VALUES (?, ?)
    """, (dataset_id, insight_text))

    conn.commit()
    conn.close()
    print(f"Insight added for dataset ID {dataset_id}.")
