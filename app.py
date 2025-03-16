import streamlit as st
from data.authentication import register_user, login_user
from dashboard import show_dashboard
import logging

# --- Logging Setup ---
logging.basicConfig(filename='logs/app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Ensure all required session state variables are initialized
if "username" not in st.session_state:
    st.session_state["username"] = None

if "user_data" not in st.session_state:
    st.session_state["user_data"] = None


def log_error(message):
    logging.error(message)

def log_info(message):
    logging.info(message)

# --- App Setup ---
st.set_page_config(page_title="Automatic Data Analyzer", layout="wide")

# --- User Authentication ---
def user_login():
    st.title("User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        success, message = login_user(username, password)
        if success:
            st.session_state.username = username  # Store the username in session_state
            st.success(f"Welcome, {username}!")
            st.rerun()  # Force rerun to show the dashboard
        else:
            st.error(message)
            log_error(f"Login failed for user: {username}. Reason: {message}")


def user_register():
    st.title("User Registration")
    username = st.text_input("Choose a Username")
    email = st.text_input("Email")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if password != confirm_password:
        st.error("Passwords do not match!")
        return

    if st.button("Register"):
        if username and email and password:
            success, message = register_user(username, email, password)
            if success:
                st.success("Registration successful! You can now login.")
                log_info(f"New user registered: {username}")
            else:
                st.error(message)
                log_error(f"Registration failed for user: {username}. Reason: {message}")
        else:
            st.error("Please fill in all fields.")

# --- Logout Functionality ---

def logout():
    if st.sidebar.button("Logout"):
        st.session_state["username"] = None  # Clear the username
        st.session_state["user_data"] = None  # Clear user data
        st.sidebar.success("Logged out successfully!")
        st.rerun()  # Force re-run to reload the login page


# --- Main Program ---
def main():
    # Check if the user is logged in
    if st.session_state["username"]:
        st.sidebar.title("Dashboard")
        logout()  # Add a logout button
        show_dashboard(st.session_state["username"])  # Show dashboard if logged in
    else:
        st.sidebar.title("Navigation")
        choice = st.sidebar.radio("Choose an Option", ["Login", "Register"])

        if choice == "Login":
            username = user_login()
            if username:
                st.experimental_rerun()  # Force re-run to load the dashboard
        elif choice == "Register":
            user_register()


if __name__ == "__main__":
    main()
