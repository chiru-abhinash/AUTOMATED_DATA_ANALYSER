import pandas as pd
import os
import streamlit as st

def load_and_validate_dataset(file):
    """
    Load, validate, and auto-preprocess the dataset.
    - Checks if the dataset is empty.
    - Fills missing values (median for numerical, mode for categorical).
    - Saves the cleaned dataset and provides a download option.
    """
    try:
        df = pd.read_csv(file)
        
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

# Streamlit UI
st.title("Upload Your Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_and_validate_dataset(uploaded_file)
    if df is not None:
        st.write("### Preprocessed Data Preview")
        st.dataframe(df.head())
