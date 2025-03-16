import sqlite3

def get_user_data(user_id):
    # Connect to the SQLite database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Execute a query to fetch user data
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()

    # Check if data was found and return it
    if user_data:
        print(f"User ID: {user_data[0]}, Username: {user_data[1]}, Email: {user_data[2]}")
    else:
        print("User not found")

    conn.close()

def query_datasets():
    conn = sqlite3.connect("../database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM datasets")
    rows = cursor.fetchall()

    for row in rows:
        print(f"ID: {row[0]}, Name: {row[1]}, Description: {row[2]}, Upload Date: {row[3]}, File Path: {row[4]}")

    conn.close()

def query_insights(dataset_id):
    conn = sqlite3.connect("../database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM insights WHERE dataset_id = ?", (dataset_id,))
    rows = cursor.fetchall()

    for row in rows:
        print(f"Insight ID: {row[0]}, Insight: {row[2]}, Created At: {row[3]}")

    conn.close()

if __name__ == "__main__":
    print("Querying all datasets:")
    query_datasets()
    print("\nQuerying insights for dataset with ID 1:")
    query_insights(1)
