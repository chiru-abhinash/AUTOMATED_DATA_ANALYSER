# import streamlit as st
# import pandas as pd
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()


# # Configure Gemini API
# api_key = "AIzaSyCttibuV1gy1fRUQsGYy7-2clAhSJeUyD0"
# if api_key:
#     genai.configure(api_key=api_key)
# else:
#     st.error("Gemini API key not found. Please check your .env file.")

# # Initialize the Gemini model
# model = genai.GenerativeModel(
#     model_name="gemini-1.5-flash",
#     generation_config={
#         "temperature": 1,
#         "top_p": 0.95,
#         "max_output_tokens": 1000,
#     }
# )

# def suggest_ml_algorithms():
#     """
#     A single function to generate dataset descriptions and suggest machine learning algorithms
#     based on dataset characteristics and constraints.
#     """
#     st.title("Dataset Analysis and Machine Learning Suggestions")

#     # Step 1: Fetch dataset from session state
#     if "dataset" not in st.session_state:
#         st.error("No dataset found in session. Please upload a dataset on the dashboard.")
#         return
#     data = st.session_state["dataset"]

#     # Step 2: Display dataset preview
#     st.write("Preview of your dataset:")
#     st.dataframe(data.head())

#     # Helper function: Generate dataset description
#     def generate_dataset_description(headers, sample_row):
#         prompt = (
#             f"The dataset has the following columns: {', '.join(headers)}. "
#             f"Here is a sample row: {sample_row}. "
#             "Describe the dataset's purpose, potential use cases, and characteristics."
#         )
#         chat_session = model.start_chat(history=[])
#         response = chat_session.send_message(prompt)
#         if response and response.text:
#             return response.text
#         return "Failed to generate a dataset description. Please try again."

#     # Helper function: Recommend ML algorithms
#     def recommend_ml_algorithms(description, data_shape, constraints):
#         prompt = (
#             f"Given the dataset described as: '{description}', with {data_shape[0]} rows and {data_shape[1]} columns, "
#             f"and considering the constraints: {constraints}, recommend suitable machine learning algorithms. "
#             "Explain why these algorithms are suitable and what results they can achieve."
#         )
#         chat_session = model.start_chat(history=[])
#         response = chat_session.send_message(prompt)
#         if response and response.text:
#             return response.text
#         return "Failed to recommend algorithms. Please try again."

#     # Step 3: Generate dataset description
#     if st.button("Describe Dataset"):
#         headers = list(data.columns)
#         sample_row = data.iloc[0].to_dict() if not data.empty else {}
#         description = generate_dataset_description(headers, sample_row)
#         st.subheader("Dataset Description")
#         st.write(description)

#     # Step 4: Validate dataset characteristics and constraints
#     constraints = {}
#     constraints['data_spread'] = "High variability" if data.select_dtypes(include='number').std().mean() > 1 else "Low variability"
#     constraints['memory'] = "Limited" if len(data) > 100000 else "Sufficient"
#     constraints['feasibility'] = "Small" if data.shape[1] < 10 else "Large"

#     st.write("Dataset Constraints:")
#     st.json(constraints)

#     # Step 5: Recommend algorithms based on characteristics
#     dataset_summary = st.text_area(
#         "Describe Your Dataset",
#         placeholder="Provide a brief summary of your dataset (e.g., customer data, sales trends)."
#     )
#     if st.button("Recommend ML Algorithms"):
#         if dataset_summary:
#             recommendations = recommend_ml_algorithms(
#                 dataset_summary, 
#                 data_shape=data.shape, 
#                 constraints=constraints
#             )
#             st.subheader("Recommended Machine Learning Algorithms")
#             st.write(recommendations)
#         else:
#             st.error("Please provide a dataset summary.")

# # Run the function if this file is executed
# if __name__ == "__main__":
#     suggest_ml_algorithms()



import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
api_key = "AIzaSyCttibuV1gy1fRUQsGYy7-2clAhSJeUyD0"  # Replace with your actual API key or use environment variable
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Gemini API key not found. Please check your .env file.")

# Initialize the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "max_output_tokens": 1000,
    }
)

def suggest_ml_algorithms(data):
    """
    Function to handle dataset analysis and ML algorithm suggestions based on dataset characteristics.
    
    Parameters:
        data (pd.DataFrame): The dataset uploaded by the user.
    """
    st.title("Dataset Analysis and Machine Learning Suggestions")

    # Step 1: Display dataset preview
    st.write("### Dataset Preview:")
    st.dataframe(data.head())  # Show the first few rows of the dataset

    # Step 2: Generate dataset description using the Gemini model
    def generate_dataset_description(headers, sample_row):
        """
        Generates a description of the dataset using Gemini's generative model.

        Parameters:
            headers (list): The column headers of the dataset.
            sample_row (dict): A sample row from the dataset.
        
        Returns:
            str: Description of the dataset.
        """
        prompt = (
            f"The dataset has the following columns: {', '.join(headers)}. "
            f"Here is a sample row: {sample_row}. "
            "Describe the dataset's purpose, potential use cases, and characteristics."
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        if response and response.text:
            return response.text
        return "Failed to generate a dataset description. Please try again."

    # Generate dataset description automatically
    headers = list(data.columns)
    sample_row = data.iloc[0].to_dict() if not data.empty else {}
    description = generate_dataset_description(headers, sample_row)

    # Display the generated description
    st.subheader("### Dataset Description:")
    st.write(description)

    # Step 3: Define dataset constraints based on characteristics (e.g., data spread, memory)
    constraints = {
        "data_spread": "High variability" if data.select_dtypes(include='number').std().mean() > 1 else "Low variability",
        "memory": "Limited" if len(data) > 100000 else "Sufficient",
        "feasibility": "Small" if data.shape[1] < 10 else "Large",
    }

    st.write("### Dataset Constraints:")
    st.json(constraints)

    # Step 4: Recommend ML algorithms based on the dataset description and constraints
    def recommend_ml_algorithms(description, data_shape, constraints):
        """
        Recommends suitable machine learning algorithms based on the dataset's description, shape, and constraints.

        Parameters:
            description (str): Description of the dataset.
            data_shape (tuple): The shape of the dataset (number of rows, columns).
            constraints (dict): Characteristics of the dataset.
        
        Returns:
            str: The ML algorithm recommendations.
        """
        prompt = (
            f"Given the dataset described as: '{description}', with {data_shape[0]} rows and {data_shape[1]} columns, "
            f"and considering the constraints: {constraints}, recommend suitable machine learning algorithms. "
            "Explain why these algorithms are suitable and what results they can achieve."
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        if response and response.text:
            return response.text
        return "Failed to recommend algorithms. Please try again."

    # Automatically recommend ML algorithms based on the dataset description and constraints
    recommendations = recommend_ml_algorithms(
        description=description,
        data_shape=data.shape,
        constraints=constraints
    )

    # Display the recommended ML algorithms
    st.subheader("### Recommended Machine Learning Algorithms:")
    st.write(recommendations)
