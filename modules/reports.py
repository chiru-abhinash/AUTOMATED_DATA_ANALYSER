import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from modules.ml_algorithms import get_ml_suggestions  # Import ML suggestion function
from modules.recommendations import generate_insights  # Import insights function

# Initialize Hugging Face BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def save_figure_as_image(fig):
    """ Save a matplotlib figure as an image in memory. """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def generate_visualization(df):
    """ Generate visualizations and return as images. """
    figures = {}
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        figures['Correlation Heatmap'] = save_figure_as_image(fig)
        plt.close(fig)
    
    for col in numeric_df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        df[col].plot(kind='hist', bins=20, color='orange', alpha=0.7, ax=ax)
        ax.set_title(f"Histogram - {col}")
        figures[f'Histogram - {col}'] = save_figure_as_image(fig)
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f"Boxplot - {col}")
        figures[f'Boxplot - {col}'] = save_figure_as_image(fig)
        plt.close(fig)
    
    return figures

def generate_pdf_report(df):
    """ Generate and save a PDF report with dataset insights. """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Dataset Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    # Generate ML Suggestions & Insights dynamically
    insights_data = get_ml_suggestions(df)
    dataset_description = insights_data["description"]
    constraints = insights_data["constraints"]
    ml_suggestions = insights_data["ml_suggestions"]

    # Dataset Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Dataset Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}")
    pdf.ln(5)
    
    # Dataset Description
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Dataset Description", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, dataset_description.replace("**", ""))
    pdf.ln(5)
    
    # Dataset Constraints
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Dataset Constraints", ln=True)
    pdf.set_font("Arial", "", 12)
    for key, value in constraints.items():
        pdf.cell(200, 10, f"- {key}: {value}", ln=True)
    pdf.ln(5)
    
    # ML Suggestions Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Suggested Machine Learning Algorithms", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, ml_suggestions.replace("**", "").replace("*", "-"))
    pdf.ln(5)

    # Visualizations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Visualizations", ln=True)
    pdf.ln(5)
    
    figures = generate_visualization(df)
    for title, img_data in figures.items():
        pdf.cell(200, 10, title, ln=True)
        img = Image.open(io.BytesIO(img_data))
        img_path = f"temp_{title}.png"
        img.save(img_path)
        pdf.image(img_path, x=10, w=180)
        pdf.ln(5)
    
    # Save the PDF
    pdf_path = "dataset_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

def show_reports_page():
    """ Streamlit UI for generating and displaying reports. """
    st.title("📊 Report Generation")
    
    if "user_data" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.user_data
    
    if st.button("Generate Report"):
        with st.spinner("Generating report, please wait..."):
            report_path = generate_pdf_report(df)
            st.success("Report generated successfully!")
            with open(report_path, "rb") as file:
                st.download_button("📥 Download Report", file, file_name="Dataset_Report.pdf", mime="application/pdf")



'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize Hugging Face BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_description_for_visualization(image, visualization_type, col_name, dataset_summary):
    """
    Generate a detailed description for the given image using Hugging Face BLIP model.
    """
    image = Image.open(io.BytesIO(image))
    
    prompts = {
        "Correlation Heatmap": f"Analyze the correlation heatmap for dataset columns: {', '.join(col_name)}.",
        "Histogram": f"Describe the histogram for '{col_name}'.",
        "Boxplot": f"Interpret the boxplot for '{col_name}'.",
    }
    
    prompt = prompts.get(visualization_type, "Analyze the provided dataset visualization.")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    
    return description

def save_figure_as_image(fig):
    """ Save a matplotlib figure as an image in memory. """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def generate_visualization(df):
    """ Generate visualizations and return as images. """
    figures = {}
    
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        figures['Correlation Heatmap'] = save_figure_as_image(fig)
        plt.close(fig)
    
    for col in numeric_df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        df[col].plot(kind='hist', bins=20, color='orange', alpha=0.7, ax=ax)
        figures[f'Histogram - {col}'] = save_figure_as_image(fig)
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(y=df[col], ax=ax)
        figures[f'Boxplot - {col}'] = save_figure_as_image(fig)
        plt.close(fig)
    
    return figures

def generate_pdf_report(df):
    """ Generate and save a PDF report with dataset insights. """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Dataset Analysis Report", ln=True, align="C")
    pdf.ln(10)
    
    # Fetch stored ML suggestions and insights
    ml_suggestions = st.session_state.get("ml_suggestions", [])
    insights = st.session_state.get("insights", [])
    
    # Dataset summary
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}", ln=True)
    pdf.ln(5)
    
    # ML Suggestions
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Suggested ML Algorithms", ln=True)
    pdf.set_font("Arial", "", 12)
    if ml_suggestions:
        for algo in ml_suggestions:
            pdf.cell(200, 10, f"- {algo}", ln=True)
    else:
        pdf.cell(200, 10, "No suggestions available", ln=True)
    pdf.ln(5)
    
    # Insights
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Generated Insights", ln=True)
    pdf.set_font("Arial", "", 12)
    if insights:
        for insight in insights:
            pdf.multi_cell(0, 10, f"- {insight}")
    else:
        pdf.cell(200, 10, "No insights available", ln=True)
    pdf.ln(5)
    
    # Visualizations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Visualizations", ln=True)
    pdf.ln(5)
    
    figures = generate_visualization(df)
    for title, img_data in figures.items():
        pdf.cell(200, 10, title, ln=True)
        img = Image.open(io.BytesIO(img_data))
        img_path = f"temp_{title}.png"
        img.save(img_path)
        pdf.image(img_path, x=10, w=180)
        pdf.ln(5)
    
    # Save the PDF
    pdf_path = "dataset_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

def show_reports_page():
    """ Streamlit UI for generating and displaying reports. """
    st.title("📊 Report Generation")
    
    if "user_data" not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
    
    df = st.session_state.user_data
    
    if st.button("Generate Report"):
        with st.spinner("Generating report, please wait..."):
            report_path = generate_pdf_report(df)
            st.success("Report generated successfully!")
            with open(report_path, "rb") as file:
                st.download_button("📥 Download Report", file, file_name="Dataset_Report.pdf", mime="application/pdf")


'''
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from fpdf import FPDF
# import os
# import io

# from modules.ml_algorithms import suggest_ml_algorithms
# from modules.recommendations import generate_insights

# # ---------------------- Data Preprocessing ----------------------
# def preprocess_data(df):
#     """
#     Handle missing values and encode categorical data properly.
#     """
#     df_clean = df.copy()

#     # Convert categorical columns to numeric using encoding
#     for col in df_clean.select_dtypes(include=['object']).columns:
#         df_clean[col] = df_clean[col].astype('category').cat.codes  # Encode categorical values

#     # Fill missing values
#     df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

#     return df_clean

# # ---------------------- Visualization Generator ----------------------
# def generate_visualization(df, viz_type, return_fig=False):
#     """
#     Generate different types of visualizations based on user selection.
#     """
#     df_clean = preprocess_data(df)

#     try:
#         if viz_type == "Pairplot":
#             pairplot_fig = sns.pairplot(df_clean)
#             return pairplot_fig.fig if return_fig else st.pyplot(pairplot_fig.fig)

#         elif viz_type == "Correlation Heatmap":
#             numeric_df = df_clean.select_dtypes(include=['number'])
#             if numeric_df.shape[1] > 1:
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
#                 return fig if return_fig else st.pyplot(fig)
#             else:
#                 st.warning("Not enough numerical columns for correlation.")

#         elif viz_type == "Histograms":
#             numeric_columns = df_clean.select_dtypes(include='number').columns
#             if numeric_columns.any():
#                 fig, axes = plt.subplots(nrows=len(numeric_columns), figsize=(8, 5 * len(numeric_columns)))
#                 for ax, col in zip(axes, numeric_columns):
#                     df_clean[col].hist(ax=ax, bins=20, color='orange', alpha=0.7)
#                     ax.set_title(f"Histogram of {col}")
#                 return fig if return_fig else st.pyplot(fig)
        
#         elif viz_type == "Boxplots":
#             numeric_columns = df_clean.select_dtypes(include='number').columns
#             if numeric_columns.any():
#                 fig, ax = plt.subplots(figsize=(8, 5))
#                 sns.boxplot(data=df_clean[numeric_columns], ax=ax)
#                 ax.set_title("Boxplot of Numerical Features")
#                 return fig if return_fig else st.pyplot(fig)

#         elif viz_type == "Countplots":
#             categorical_columns = df_clean.select_dtypes(include='object').columns
#             if categorical_columns.any():
#                 fig, ax = plt.subplots(figsize=(8, 5))
#                 for col in categorical_columns:
#                     sns.countplot(x=df[col], ax=ax, order=df[col].value_counts().index)
#                     ax.set_title(f"Countplot of {col}")
#                 return fig if return_fig else st.pyplot(fig)

#     except Exception as e:
#         st.error(f"Error generating {viz_type}: {e}")

#     return None

# # ---------------------- PDF Report Generator ----------------------
# def generate_pdf_report(df):
#     """
#     Generates a PDF report containing dataset visualizations, suggested ML algorithms, and insights.
#     """
#     if df is None or df.empty:
#         return None  # Prevent generating a report for an empty dataset

#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.add_page()
#     pdf.set_font("Arial", style='B', size=16)
#     pdf.cell(200, 10, "Dataset Analysis Report", ln=True, align='C')
#     pdf.ln(10)

#     # Dataset Overview
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}", ln=True)
#     pdf.ln(5)

#     # Suggested ML Algorithms
#     pdf.set_font("Arial", style='B', size=14)
#     pdf.cell(200, 10, "Suggested ML Algorithms", ln=True)
#     pdf.set_font("Arial", size=12)

#     # Ensure ml_suggestions is not None
#     ml_suggestions = suggest_ml_algorithms(df) or []

#     if ml_suggestions:
#         for algo in ml_suggestions:
#             pdf.cell(200, 10, f"- {algo}", ln=True)
#     else:
#         pdf.cell(200, 10, "No suggestions available", ln=True)

#     pdf.ln(5)

#     # Insights
#     pdf.set_font("Arial", style='B', size=14)
#     pdf.cell(200, 10, "Generated Insights", ln=True)
#     pdf.set_font("Arial", size=12)
#     insights = generate_insights(df) or []

#     if insights:
#         for insight in insights:
#             pdf.multi_cell(0, 10, f"- {insight}")
#     else:
#         pdf.cell(200, 10, "No insights available", ln=True)

#     pdf.ln(5)

#     # Visualizations
#     pdf.set_font("Arial", style='B', size=14)
#     pdf.cell(200, 10, "Visualizations", ln=True)
#     pdf.set_font("Arial", size=12)
#     os.makedirs("temp_reports", exist_ok=True)

#     visualizations = ["Pairplot", "Correlation Heatmap", "Histograms", "Boxplots", "Countplots"]
#     for viz in visualizations:
#         fig = generate_visualization(df, viz_type=viz, return_fig=True)
#         if fig:
#             image_path = f"temp_reports/{viz}.png"
#             fig.savefig(image_path)
#             pdf.image(image_path, x=10, w=180)
#             pdf.ln(5)

#     report_path = "temp_reports/Dataset_Report.pdf"
#     pdf.output(report_path)
#     return report_path

# # ---------------------- Streamlit Page ----------------------
# def show_reports_page():
#     st.subheader("Generate Report")
#     if "user_data" not in st.session_state or st.session_state.user_data is None:
#         st.warning("Please upload a dataset first.")
#         return

#     if st.button("Generate Report PDF"):
#         report_path = generate_pdf_report(st.session_state.user_data)
#         if report_path:
#             with open(report_path, "rb") as f:
#                 st.download_button("Download Report PDF", f, file_name="Dataset_Report.pdf", mime="application/pdf")
#         else:
#             st.error("Failed to generate report. Ensure dataset is valid.")






# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from fpdf import FPDF
# import os
# import io
# from PIL import Image
# from modules.ml_algorithms import suggest_ml_algorithms
# from modules.recommendations import generate_insights

# def generate_visualization(df, viz_type):
#     """
#     Generate a visualization and return the image buffer for inclusion in the PDF.
#     """
#     fig, ax = plt.subplots(figsize=(6, 4))
    
#     if viz_type == "Correlation Heatmap":
#         numeric_df = df.select_dtypes(include=['number'])
#         if numeric_df.shape[1] > 1:
#             sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
#         else:
#             return None
#     elif viz_type == "Histogram":
#         for col in df.select_dtypes(include='number').columns:
#             df[col].plot(kind='hist', bins=20, alpha=0.7, ax=ax)
#             ax.set_title(f"Histogram of {col}")
#             break  # Show one example histogram
#     elif viz_type == "Boxplot":
#         for col in df.select_dtypes(include='number').columns:
#             sns.boxplot(y=df[col], ax=ax)
#             ax.set_title(f"Boxplot of {col}")
#             break  # Show one example boxplot
#     else:
#         return None
    
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight')
#     plt.close(fig)
#     buf.seek(0)
#     return buf

# def generate_pdf_report(df):
#     """
#     Generates a PDF report containing dataset visualizations, suggested ML algorithms, and insights.
#     """
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=10)
#     pdf.add_page()
#     pdf.set_font("Arial", style='B', size=16)
#     pdf.cell(200, 10, "Dataset Analysis Report", ln=True, align='C')
#     pdf.ln(10)
    
#     # Dataset Overview
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, f"Total Rows: {df.shape[0]} | Total Columns: {df.shape[1]}", ln=True)
#     pdf.ln(5)
    
#     # Suggested ML Algorithms
#     pdf.set_font("Arial", style='B', size=14)
#     pdf.cell(200, 10, "Suggested ML Algorithms", ln=True)
#     pdf.set_font("Arial", size=12)
#     ml_suggestions = suggest_ml_algorithms(df) or ["No suggestions available"]
#     for algo in ml_suggestions:
#         pdf.cell(200, 10, f"- {algo}", ln=True)
#     pdf.ln(5)
    
#     # Insights
#     pdf.set_font("Arial", style='B', size=14)
#     pdf.cell(200, 10, "Generated Insights", ln=True)
#     pdf.set_font("Arial", size=12)
#     insights = generate_insights(df) or ["No insights available"]
#     for insight in insights:
#         pdf.multi_cell(0, 10, f"- {insight}")
#     pdf.ln(5)
    
#     # Visualizations
#     pdf.set_font("Arial", style='B', size=14)
#     pdf.cell(200, 10, "Visualizations", ln=True)
#     pdf.ln(5)
#     os.makedirs("temp_reports", exist_ok=True)
    
#     visualizations = ["Correlation Heatmap", "Histogram", "Boxplot"]
#     for viz in visualizations:
#         img_buf = generate_visualization(df, viz)
#         if img_buf:
#             image_path = f"temp_reports/{viz}.png"
#             with open(image_path, "wb") as f:
#                 f.write(img_buf.read())
#             pdf.image(image_path, x=10, w=180)
#             pdf.ln(5)
    
#     report_path = "temp_reports/Dataset_Report.pdf"
#     pdf.output(report_path)
#     return report_path

# def show_reports_page():
#     st.subheader("Generate Report")
#     if "user_data" not in st.session_state or st.session_state.user_data is None:
#         st.warning("Please upload a dataset first.")
#         return
    
#     if st.button("Generate Report PDF"):
#         with st.spinner("Generating report..."):
#             report_path = generate_pdf_report(st.session_state.user_data)
#             with open(report_path, "rb") as f:
#                 st.download_button("Download Report PDF", f, file_name="Dataset_Report.pdf", mime="application/pdf")
