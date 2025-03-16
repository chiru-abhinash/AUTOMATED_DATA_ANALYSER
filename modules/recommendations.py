import numpy as np
import pandas as pd
import streamlit as st

def generate_insights(df):
    """
    Generate dataset insights dynamically with robust error handling.
    """
    st.subheader("🔍 Dataset Insights")

    try:
        if df.empty:
            st.error("⚠️ The dataset is empty. Please upload a valid file.")
            return

        st.write("### 📊 General Dataset Statistics:")
        try:
            st.write(df.describe(include='all'))
        except Exception as e:
            st.error(f"❌ Error generating dataset statistics: {e}")

        # Detect categorical & numerical columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        st.write(f"**🧮 Numerical Columns:** {numeric_cols}")
        st.write(f"**🔠 Categorical Columns:** {categorical_cols}")

        # Handle Missing Values
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.write("### 🚨 Missing Values Report:")
            st.write(missing_values)
            st.write("#### Recommended Actions:")
            for col in df.columns:
                if missing_values[col] > 0:
                    if col in numeric_cols:
                        st.write(f"🔹 {col}: Consider imputing with mean/median.")
                    else:
                        st.write(f"🔹 {col}: Consider filling with the most frequent category or 'Unknown'.")

        # Handle Correlation Matrix (Numerical Only)
        if len(numeric_cols) > 1:
            try:
                st.write("### 🔗 Most Correlated Features:")
                df_encoded = df.copy()

                # Convert categorical data for correlation analysis
                for col in categorical_cols:
                    df_encoded[col] = df_encoded[col].astype('category').cat.codes

                correlation_matrix = df_encoded.corr()
                st.write(correlation_matrix.style.background_gradient(cmap='coolwarm'))
            except Exception as e:
                st.error(f"❌ Error computing correlation: {e}")

        # Detect Outliers
        st.write("### 🚨 Outlier Detection:")
        for col in numeric_cols:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                st.write(f"**{col}:** {outliers.shape[0]} outliers detected.")
            except Exception as e:
                st.error(f"❌ Error detecting outliers in {col}: {e}")

        # Categorical Distribution
        if categorical_cols:
            st.write("### 📊 Categorical Feature Distribution:")
            for col in categorical_cols:
                try:
                    st.write(f"#### {col} Distribution:")
                    st.bar_chart(df[col].value_counts())
                except Exception as e:
                    st.error(f"❌ Error displaying distribution for {col}: {e}")

    except Exception as e:
        st.error(f"❌ Critical Error in Insights Generation: {e}")
