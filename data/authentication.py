import sqlite3
import hashlib
import os
from datetime import datetime

# Utility function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to register a new user
def register_user(username, email, password):
    conn = sqlite3.connect("../database.db")
    cursor = conn.cursor()

    # Check if the username or email already exists
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
    if cursor.fetchone():
        print("Username or email already exists.")
        return

    password_hash = hash_password(password)

    # Insert new user into the database
    cursor.execute("""
    INSERT INTO users (username, email, password_hash, created_at)
    VALUES (?, ?, ?, ?)
    """, (username, email, password_hash, datetime.now()))

    conn.commit()
    conn.close()
    print(f"User '{username}' registered successfully.")

# Function to login a user
def login_user(username, password):
    conn = sqlite3.connect("../database.db")
    cursor = conn.cursor()

    # Retrieve user by username
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user and hash_password(password) == user[3]:  # user[3] is the password_hash
        print(f"User '{username}' logged in successfully.")
        return user[0]  # Returning user id
    else:
        print("Invalid username or password.")
        return None



import bcrypt
from data.db_utils import get_db_connection, close_db_connection  # Ensure correct import path

def register_user(username, email, password):
    """
    Registers a new user in the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if the email or username already exists
    cursor.execute("SELECT * FROM users WHERE email = ? OR username = ?", (email, username))
    existing_user = cursor.fetchone()
    if existing_user:
        close_db_connection(conn)
        return False, "Username or email already exists. Please choose a different one."

    # Hash the password
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Insert the new user
    cursor.execute(
        "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
        (username, email, password_hash.decode('utf-8'))
    )
    conn.commit()
    close_db_connection(conn)
    return True, "User registered successfully!"

def login_user(username, password):
    """
    Authenticates a user by checking username and password.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Retrieve the user record
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    close_db_connection(conn)

    if not user:
        return False, "User not found."

    # Check the hashed password
    stored_password_hash = user["password_hash"]
    if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash.encode('utf-8')):
        return True, "Login successful."
    else:
        return False, "Invalid password."




if __name__ == "__main__":
    # Example usage:
    register_user("test_user", "test_user@example.com", "password123")  # Register a new user
    user_id = login_user("test_user", "password123")  # Login with correct credentials
    if user_id:
        print(f"Logged in as user with ID {user_id}")
