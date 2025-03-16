# import streamlit as st
# import pandas as pd
# from data.query_data import query_datasets, query_insights  # Example functions to query datasets and insights
# from modules.visualizations import generate_visualization
# from modules.ml_algorithms import suggest_ml_algorithms
# from modules.recommendations import generate_insights


# def upload_dataset():
#     st.title("Upload Your Dataset")
#     uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
#     if uploaded_file:
#         # Read the uploaded file based on extension
#         if uploaded_file.name.endswith("csv"):
#             df = pd.read_csv(uploaded_file)
#         elif uploaded_file.name.endswith("xlsx"):
#             df = pd.read_excel(uploaded_file)
        
#         st.write("Dataset Preview:", df.head())
        
#         # Example: Save the uploaded dataset (you would store this in your database)
#         # Here you can write a function to store the dataset in the database or save locally
#         # save_uploaded_data_to_db(df)
        
#         return df
#     return None

# def show_dashboard(username):
#     st.title(f"Welcome to the Automatic Data Analyzer, {username}")

#     # Upload dataset
#     user_data = upload_dataset()
#     if user_data is not None:
#         st.session_state.user_data = user_data  # Store the data in session state for later use

#     # Provide the user with options to interact with the dataset
#     action = st.selectbox(
#         "Choose an action", 
#         ["View Dataset", "Generate Visualizations", "Suggest ML Algorithms", "Generate Insights"]
#     )

#     if action == "View Dataset":
#         if "user_data" in st.session_state:
#             st.write(st.session_state.user_data)
#         else:
#             st.error("Please upload a dataset first.")

#     elif action == "Generate Visualizations":
#         if "user_data" in st.session_state:
#             generate_visualization(st.session_state.user_data)
#         else:
#             st.error("Please upload a dataset first.")

#     elif action == "Suggest ML Algorithms":
#         if "user_data" in st.session_state:
#             suggest_ml_algorithms(st.session_state.user_data)
#         else:
#             st.error("Please upload a dataset first.")

#     elif action == "Generate Insights":
#         if "user_data" in st.session_state:
#             generate_insights(st.session_state.user_data)
#         else:
#             st.error("Please upload a dataset first.")

import streamlit as st
import pandas as pd
import os
from data.query_data import query_datasets, query_insights  # Example functions to query datasets and insights
from modules.visualizations import generate_visualization
from modules.ml_algorithms import suggest_ml_algorithms
from modules.recommendations import generate_insights
from modules.reports import show_reports_page

def load_and_validate_dataset(file):
    """
    Load, validate, and auto-preprocess the dataset.
    - Checks if the dataset is empty.
    - Fills missing values (median for numerical, mode for categorical).
    - Saves the cleaned dataset and provides a download option.
    """
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        
        if df.empty:
            st.error("The dataset is empty. Please upload a valid dataset.")
            return None

        # Auto-preprocessing: Handle missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':  # Categorical data
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:  # Numerical data
                    df[col].fillna(df[col].median(), inplace=True)

        # Save the cleaned dataset
        save_path = os.path.join("data", "preprocessed_data.csv")
        os.makedirs("data", exist_ok=True)
        df.to_csv(save_path, index=False)

        # Provide download option
        st.success("Dataset preprocessed successfully!")
        st.download_button(
            label="Download Preprocessed Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="preprocessed_data.csv",
            mime="text/csv"
        )

        return df

    except Exception as e:
        st.error(f"Error loading the dataset: {str(e)}")
        return None

def upload_dataset():
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = load_and_validate_dataset(uploaded_file)  # Preprocess and return the cleaned dataset
        
        if df is not None:
            st.write("### Preprocessed Dataset Preview")
            st.dataframe(df.head())

        return df
    return None

def show_dashboard(username):
    st.title(f"Welcome to the Automatic Data Analyzer, {username}")

    # Upload dataset
    if "user_data" not in st.session_state:
        st.session_state.user_data = None

    user_data = upload_dataset()
    
    if user_data is not None:
        st.session_state.user_data = user_data  # Store processed dataset in session state

    # Provide the user with options to interact with the dataset
    action = st.selectbox(
        "Choose an action", 
        ["View Dataset", "Generate Visualizations", "Suggest ML Algorithms", "Generate Insights", "Generate Report"]
    )

    if action == "View Dataset":
        if st.session_state.user_data is not None:
            st.write("### Your Dataset")
            st.dataframe(st.session_state.user_data)
        else:
            st.warning("⚠️ Please upload a dataset first.")

    elif action == "Generate Visualizations":
        if st.session_state.user_data is not None:
            generate_visualization(st.session_state.user_data)
        else:
            st.warning("⚠️ Please upload a dataset first.")

    elif action == "Suggest ML Algorithms":
        if st.session_state.user_data is not None:
            suggest_ml_algorithms(st.session_state.user_data)
        else:
            st.warning("⚠️ Please upload a dataset first.")

    elif action == "Generate Insights":
        if st.session_state.user_data is not None:
            generate_insights(st.session_state.user_data)
        else:
            st.warning("⚠️ Please upload a dataset first.")
    elif action == "Generate Report":
        if st.session_state.user_data is not None:
            show_reports_page()
        else:
            st.warning("⚠️ Please upload a dataset first.")
               
