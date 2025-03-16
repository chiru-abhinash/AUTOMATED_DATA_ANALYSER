import sqlite3

def migrate_schema():
    conn = sqlite3.connect("../database.db")
    cursor = conn.cursor()

    # Example: Adding a new column to `datasets` table
    cursor.execute("""
    ALTER TABLE datasets ADD COLUMN new_column TEXT
    """)

    # Commit and close
    conn.commit()
    conn.close()
    print("Schema migrated successfully.")

if __name__ == "__main__":
    migrate_schema()
