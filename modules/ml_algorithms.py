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


'''
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

def get_ml_suggestions(data):
    """
    Analyzes the dataset and provides ML algorithm recommendations.

    Parameters:
        data (pd.DataFrame): The dataset uploaded by the user.

    Returns:
        dict: A dictionary containing dataset description, constraints, and recommended ML algorithms.
    """

    # Step 1: Generate dataset description
    headers = list(data.columns)
    sample_row = data.iloc[0].to_dict() if not data.empty else {}

    prompt_desc = (
        f"The dataset has the following columns: {', '.join(headers)}. "
        f"Here is a sample row: {sample_row}. "
        "Describe the dataset's purpose, potential use cases, and characteristics."
    )

    chat_session = model.start_chat(history=[])
    response_desc = chat_session.send_message(prompt_desc)
    dataset_description = response_desc.text if response_desc and response_desc.text else "No description available."

    # Step 2: Define dataset constraints based on characteristics
    constraints = {
        "data_spread": "High variability" if data.select_dtypes(include='number').std().mean() > 1 else "Low variability",
        "memory": "Limited" if len(data) > 100000 else "Sufficient",
        "feasibility": "Small" if data.shape[1] < 10 else "Large",
    }

    # Step 3: Recommend ML algorithms based on dataset analysis
    prompt_ml = (
        f"Given the dataset described as: '{dataset_description}', with {data.shape[0]} rows and {data.shape[1]} columns, "
        f"and considering the constraints: {constraints}, recommend suitable machine learning algorithms. "
        "Explain why these algorithms are suitable and what results they can achieve."
    )

    response_ml = chat_session.send_message(prompt_ml)
    ml_recommendations = response_ml.text if response_ml and response_ml.text else "No recommendations available."

    return {
        "description": dataset_description,
        "constraints": constraints,
        "ml_suggestions": ml_recommendations
    }


'''



import streamlit as st
import pandas as pd
from google import genai

# --------------------------------------------------
# GEMINI CLIENT CONFIGURATION (NEW SDK)
# --------------------------------------------------

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-1.5-flash"  # stable & publicly available


# --------------------------------------------------
# INTERNAL HELPER FUNCTIONS (LOGIC UNCHANGED)
# --------------------------------------------------

def generate_dataset_description(headers, sample_row):
    """
    Generates a description of the dataset using Gemini.
    """
    prompt = (
        f"The dataset has the following columns: {', '.join(headers)}. "
        f"Here is a sample row: {sample_row}. "
        "Describe the dataset's purpose, potential use cases, and characteristics."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Unable to generate dataset description: {str(e)}"


def recommend_ml_algorithms(description, data_shape, constraints):
    """
    Recommends ML algorithms based on dataset description and constraints.
    """
    prompt = (
        f"Given the dataset described as: '{description}', "
        f"with {data_shape[0]} rows and {data_shape[1]} columns, "
        f"and considering the constraints: {constraints}, "
        "recommend suitable machine learning algorithms. "
        "Explain why these algorithms are suitable and what results they can achieve."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Unable to recommend ML algorithms: {str(e)}"


# --------------------------------------------------
# STREAMLIT UI FUNCTION (ORIGINAL FUNCTIONALITY KEPT)
# --------------------------------------------------

def suggest_ml_algorithms(data: pd.DataFrame):
    """
    Handles dataset analysis and ML algorithm suggestions.
    """
    st.title("Dataset Analysis and Machine Learning Suggestions")

    if data.empty:
        st.warning("The uploaded dataset is empty.")
        return

    # Step 1: Dataset preview
    st.write("### Dataset Preview:")
    st.dataframe(data.head())

    # Step 2: Dataset description
    headers = list(data.columns)
    sample_row = data.iloc[0].to_dict()

    description = generate_dataset_description(headers, sample_row)

    st.subheader("### Dataset Description:")
    st.write(description)

    # Step 3: Dataset constraints (UNCHANGED LOGIC)
    constraints = {
        "data_spread": (
            "High variability"
            if not data.select_dtypes(include="number").empty
            and data.select_dtypes(include="number").std().mean() > 1
            else "Low variability"
        ),
        "memory": "Limited" if len(data) > 100000 else "Sufficient",
        "feasibility": "Small" if data.shape[1] < 10 else "Large",
    }

    st.write("### Dataset Constraints:")
    st.json(constraints)

    # Step 4: ML algorithm recommendations
    recommendations = recommend_ml_algorithms(
        description=description,
        data_shape=data.shape,
        constraints=constraints,
    )

    st.subheader("### Recommended Machine Learning Algorithms:")
    st.write(recommendations)


# --------------------------------------------------
# BACKEND / NON-UI FUNCTION (ORIGINAL PURPOSE KEPT)
# --------------------------------------------------

def get_ml_suggestions(data: pd.DataFrame) -> dict:
    """
    Returns dataset description, constraints, and ML suggestions (non-UI usage).
    """
    if data.empty:
        return {
            "description": "Dataset is empty.",
            "constraints": {},
            "ml_suggestions": "No data available.",
        }

    headers = list(data.columns)
    sample_row = data.iloc[0].to_dict()

    description = generate_dataset_description(headers, sample_row)

    constraints = {
        "data_spread": (
            "High variability"
            if not data.select_dtypes(include="number").empty
            and data.select_dtypes(include="number").std().mean() > 1
            else "Low variability"
        ),
        "memory": "Limited" if len(data) > 100000 else "Sufficient",
        "feasibility": "Small" if data.shape[1] < 10 else "Large",
    }

    ml_suggestions = recommend_ml_algorithms(
        description=description,
        data_shape=data.shape,
        constraints=constraints,
    )

    return {
        "description": description,
        "constraints": constraints,
        "ml_suggestions": ml_suggestions,
    }

