# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import streamlit as st
# import openai

# # OpenAI API Key (replace with your key)
# openai.api_key = "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"

# def generate_description_for_visualization(visualization_type, data_columns):
#     """
#     Generate a description for the given visualization using OpenAI's GPT model.
#     """
#     try:
#         prompt = (
#             f"Generate a detailed explanation for a {visualization_type} created using the columns: "
#             f"{', '.join(data_columns)}. Explain what this visualization can reveal about the dataset."
#         )
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=prompt,
#             max_tokens=150,
#             temperature=0.7
#         )
#         return response['choices'][0]['text'].strip()
#     except Exception as e:
#         return f"Error generating description: {e}"

# def generate_visualization(df):
#     """
#     Generate selected visualizations for the given dataset with descriptive explanations.
#     """
#     st.subheader("Data Visualizations")

#     # Select visualizations
#     options = st.multiselect(
#         "Choose visualizations to generate:",
#         ["Pairplot", "Correlation Heatmap", "Histograms", "Boxplots", "Countplots", "Line Plots"]
#     )

#     if "Pairplot" in options and df.select_dtypes(include='number').shape[1] > 1:
#         st.write("Pairplot of the dataset:")
#         pairplot_fig = sns.pairplot(df)
#         st.pyplot(pairplot_fig.fig)
#         desc = generate_description_for_visualization("Pairplot", df.select_dtypes(include='number').columns.tolist())
#         st.markdown(f"**Description:** {desc}")

#     if "Correlation Heatmap" in options and df.select_dtypes(include='number').shape[1] > 1:
#         st.write("Correlation heatmap of the dataset:")
#         correlation_matrix = df.corr()
#         fig, ax = plt.subplots()
#         sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
#         st.pyplot(fig)
#         desc = generate_description_for_visualization("Correlation Heatmap", df.select_dtypes(include='number').columns.tolist())
#         st.markdown(f"**Description:** {desc}")

#     if "Histograms" in options:
#         numeric_columns = df.select_dtypes(include='number').columns
#         if len(numeric_columns) > 0:
#             st.write("Histograms for numeric columns:")
#             for col in numeric_columns:
#                 fig, ax = plt.subplots()
#                 df[col].plot(kind='hist', bins=20, color='orange', alpha=0.7, ax=ax)
#                 ax.set_title(f"Histogram of {col}")
#                 ax.set_xlabel(col)
#                 ax.set_ylabel("Frequency")
#                 st.pyplot(fig)
#                 desc = generate_description_for_visualization("Histogram", [col])
#                 st.markdown(f"**Description:** {desc}")

#     if "Boxplots" in options:
#         numeric_columns = df.select_dtypes(include='number').columns
#         if len(numeric_columns) > 0:
#             st.write("Boxplots for numeric columns:")
#             for col in numeric_columns:
#                 fig, ax = plt.subplots()
#                 sns.boxplot(y=df[col], ax=ax)
#                 ax.set_title(f"Boxplot of {col}")
#                 st.pyplot(fig)
#                 desc = generate_description_for_visualization("Boxplot", [col])
#                 st.markdown(f"**Description:** {desc}")

#     if "Countplots" in options:
#         categorical_columns = df.select_dtypes(include='object').columns
#         if len(categorical_columns) > 0:
#             st.write("Countplots for categorical columns:")
#             for col in categorical_columns:
#                 fig, ax = plt.subplots()
#                 sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
#                 ax.set_title(f"Countplot of {col}")
#                 ax.set_xlabel(col)
#                 ax.set_ylabel("Count")
#                 st.pyplot(fig)
#                 desc = generate_description_for_visualization("Countplot", [col])
#                 st.markdown(f"**Description:** {desc}")

#     if "Line Plots" in options:
#         datetime_columns = df.select_dtypes(include=['datetime64']).columns
#         if len(datetime_columns) > 0:
#             st.write("Line plots for time-series data:")
#             for col in datetime_columns:
#                 fig, ax = plt.subplots()
#                 df.set_index(col).plot(ax=ax)
#                 ax.set_title(f"Line Plot (Index: {col})")
#                 ax.set_ylabel("Values")
#                 st.pyplot(fig)
#                 desc = generate_description_for_visualization("Line Plot", [col])
#                 st.markdown(f"**Description:** {desc}")


'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

# Initialize Hugging Face BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_description_for_visualization(image, visualization_type, col_name, dataset_summary):
    """
    Generate a detailed description for the given image using Hugging Face BLIP model.
    """
    # Convert the matplotlib plot to PIL Image
    image = Image.open(io.BytesIO(image))

    # Customize prompt based on the visualization type
    if visualization_type == "Correlation Heatmap":
        prompt = (
            f"Based on the correlation heatmap created from the dataset with columns: {', '.join(col_name)}, "
            "identify any strong positive or negative correlations between variables. "
            "Mention any pairs that are strongly correlated (near 1 or -1) and suggest what this might mean in terms of the relationship between the variables."
        )
    elif visualization_type == "Histogram":
        prompt = (
            f"For the histogram created using the column '{col_name}', describe the distribution shape. "
            "Is it normal, skewed, bimodal, or uniform? Are there any obvious peaks or outliers? "
            "What does this distribution suggest about the data in terms of trends or potential anomalies?"
        )
    elif visualization_type == "Boxplot":
        prompt = (
            f"For the boxplot created using the column '{col_name}', identify the presence of any outliers. "
            "Explain the spread of the data and whether it is skewed. What do the quartiles suggest about the typical values for this feature?"
        )

    # Preprocess the image and generate description with additional context
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description


def save_figure_as_image(fig):
    """
    Save a matplotlib figure as an image in memory (as a byte stream).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.read()

def generate_visualization(df):
    """
    Generate selected visualizations for the given dataset with descriptive explanations.
    """
    st.subheader("Data Visualizations")

    # Select visualizations
    options = st.multiselect(
        "Choose visualizations to generate:",
        ["Pairplot", "Correlation Heatmap", "Histograms", "Boxplots", "Countplots", "Line Plots"]
    )

    # Generate dataset summary (e.g., column types, basic statistics)
    dataset_summary = (
        f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"Columns include numeric variables such as {', '.join(df.select_dtypes(include='number').columns)}, "
        f"and categorical variables such as {', '.join(df.select_dtypes(include='object').columns)}."
    )

    if "Pairplot" in options and df.select_dtypes(include='number').shape[1] > 1:
        st.write("Pairplot of the dataset:")
        pairplot_fig = sns.pairplot(df)
        image = save_figure_as_image(pairplot_fig.fig)
        st.pyplot(pairplot_fig.fig)
        desc = generate_description_for_visualization(image, "Pairplot", df.select_dtypes(include='number').columns.tolist(), dataset_summary)
        st.markdown(f"**Description:** {desc}")

    if "Correlation Heatmap" in options and df.select_dtypes(include='number').shape[1] > 1:
        st.write("Correlation heatmap of the dataset:")
        correlation_matrix = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        image = save_figure_as_image(fig)
        st.pyplot(fig)
        desc = generate_description_for_visualization(image, "Correlation Heatmap", df.select_dtypes(include='number').columns.tolist(), dataset_summary)
        st.markdown(f"**Description:** {desc}")

    if "Histograms" in options:
        numeric_columns = df.select_dtypes(include='number').columns
        if len(numeric_columns) > 0:
            st.write("Histograms for numeric columns:")
            for col in numeric_columns:
                fig, ax = plt.subplots()
                df[col].plot(kind='hist', bins=20, color='orange', alpha=0.7, ax=ax)
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                image = save_figure_as_image(fig)
                st.pyplot(fig)
                desc = generate_description_for_visualization(image, "Histogram", [col], dataset_summary)
                st.markdown(f"**Description:** {desc}")

    if "Boxplots" in options:
        numeric_columns = df.select_dtypes(include='number').columns
        if len(numeric_columns) > 0:
            st.write("Boxplots for numeric columns:")
            for col in numeric_columns:
                fig, ax = plt.subplots()
                sns.boxplot(y=df[col], ax=ax)
                ax.set_title(f"Boxplot of {col}")
                image = save_figure_as_image(fig)
                st.pyplot(fig)
                desc = generate_description_for_visualization(image, "Boxplot", [col], dataset_summary)
                st.markdown(f"**Description:** {desc}")

    if "Countplots" in options:
        categorical_columns = df.select_dtypes(include='object').columns
        if len(categorical_columns) > 0:
            st.write("Countplots for categorical columns:")
            for col in categorical_columns:
                fig, ax = plt.subplots()
                sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
                ax.set_title(f"Countplot of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                image = save_figure_as_image(fig)
                st.pyplot(fig)
                desc = generate_description_for_visualization(image, "Countplot", [col], dataset_summary)
                st.markdown(f"**Description:** {desc}")

    if "Line Plots" in options:
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            st.write("Line plots for time-series data:")
            for col in datetime_columns:
                fig, ax = plt.subplots()
                df.set_index(col).plot(ax=ax)
                ax.set_title(f"Line Plot (Index: {col})")
                ax.set_ylabel("Values")
                image = save_figure_as_image(fig)
                st.pyplot(fig)
                desc = generate_description_for_visualization(image, "Line Plot", [col], dataset_summary)
                st.markdown(f"**Description:** {desc}")
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

# Initialize Hugging Face BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_description_for_visualization(image, visualization_type, col_name, dataset_summary):
    """
    Generate a detailed description for the given image using Hugging Face BLIP model.
    """
    # Convert the matplotlib plot to PIL Image
    image = Image.open(io.BytesIO(image))

    # Customize prompt based on the visualization type
    prompts = {
        "Correlation Heatmap": f"Analyze the correlation heatmap for dataset columns: {', '.join(col_name)}. Identify strong positive or negative correlations.",
        "Histogram": f"Describe the histogram for '{col_name}'. Mention distribution shape, peaks, and possible anomalies.",
        "Boxplot": f"Interpret the boxplot for '{col_name}'. Identify outliers, quartiles, and data spread."
    }
    
    prompt = prompts.get(visualization_type, "Analyze the provided dataset visualization.")

    # Generate description using BLIP model
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

def save_figure_as_image(fig):
    """
    Save a matplotlib figure as an image in memory.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")  # Avoid excess whitespace
    buf.seek(0)
    return buf.read()

def preprocess_data(df):
    """
    Handle missing values and encode categorical data properly.
    """
    df_clean = df.copy()
    
    # Convert categorical columns to numeric using encoding
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype('category').cat.codes  # Encode categorical values
        
    # Fill missing values
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    
    return df_clean

def generate_visualization(df):
    """
    Generate robust visualizations for the dataset with error handling and optimal UI layout.
    """
    st.subheader("Data Visualizations")

    # Select visualizations
    options = st.multiselect(
        "Choose visualizations to generate:",
        ["Pairplot", "Correlation Heatmap", "Histograms", "Boxplots", "Countplots", "Line Plots"]
    )

    # Preprocess dataset to handle categorical values
    df_clean = preprocess_data(df)

    # Generate dataset summary
    dataset_summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."

    # Pairplot
    if "Pairplot" in options and df_clean.select_dtypes(include='number').shape[1] > 1:
        st.write("Pairplot of the dataset:")
        try:
            pairplot_fig = sns.pairplot(df_clean)
            image = save_figure_as_image(pairplot_fig.fig)
            st.pyplot(pairplot_fig.fig)
            desc = generate_description_for_visualization(image, "Pairplot", df_clean.columns.tolist(), dataset_summary)
            st.markdown(f"**Description:** {desc}")
        except Exception as e:
            st.error(f"Error generating pairplot: {e}")

    # Correlation Heatmap
    if "Correlation Heatmap" in options:
        numeric_df = df_clean.select_dtypes(include=['number'])  # Select only numeric columns
        if numeric_df.shape[1] > 1:
            st.write("Correlation Heatmap:")
            try:
                correlation_matrix = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
                image = save_figure_as_image(fig)
                st.pyplot(fig)
                desc = generate_description_for_visualization(image, "Correlation Heatmap", numeric_df.columns.tolist(), dataset_summary)
                st.markdown(f"**Description:** {desc}")
            except Exception as e:
                st.error(f"Error generating correlation heatmap: {e}")
        else:
            st.warning("Not enough numerical columns for correlation.")

    # Histograms
    if "Histograms" in options:
        numeric_columns = df_clean.select_dtypes(include='number').columns
        if len(numeric_columns) > 0:
            st.write("Histograms:")
            for col in numeric_columns:
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    df[col].plot(kind='hist', bins=20, color='orange', alpha=0.7, ax=ax)
                    ax.set_title(f"Histogram of {col}")
                    ax.set_xlabel(col)
                    image = save_figure_as_image(fig)
                    st.pyplot(fig)
                    desc = generate_description_for_visualization(image, "Histogram", [col], dataset_summary)
                    st.markdown(f"**Description:** {desc}")
                except Exception as e:
                    st.error(f"Error generating histogram for {col}: {e}")

    # Boxplots
    if "Boxplots" in options:
        numeric_columns = df_clean.select_dtypes(include='number').columns
        if len(numeric_columns) > 0:
            st.write("Boxplots:")
            for col in numeric_columns:
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.boxplot(y=df[col], ax=ax)
                    ax.set_title(f"Boxplot of {col}")
                    image = save_figure_as_image(fig)
                    st.pyplot(fig)
                    desc = generate_description_for_visualization(image, "Boxplot", [col], dataset_summary)
                    st.markdown(f"**Description:** {desc}")
                except Exception as e:
                    st.error(f"Error generating boxplot for {col}: {e}")

    # Countplots
    if "Countplots" in options:
        categorical_columns = df.select_dtypes(include='object').columns
        if len(categorical_columns) > 0:
            st.write("Countplots:")
            for col in categorical_columns:
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
                    ax.set_title(f"Countplot of {col}")
                    image = save_figure_as_image(fig)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating countplot for {col}: {e}")

    # Line Plots
    if "Line Plots" in options:
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            st.write("Line Plots for Time-Series Data:")
            for col in datetime_columns:
                try:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    df.set_index(col).plot(ax=ax)
                    ax.set_title(f"Line Plot (Index: {col})")
                    image = save_figure_as_image(fig)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating line plot for {col}: {e}")
