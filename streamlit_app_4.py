import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64

# Load the model and tokenizer
checkpoint = "/workspaces/blank-app/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function for file preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(filepath):
    try:
        pipe_sum = pipeline(
            'summarization',
            model=base_model,
            tokenizer=tokenizer,
            max_length=500,
            min_length=50
        )

        # Preprocess the file
        input_text = file_preprocessing(filepath)
        st.info(f"Extracted text for summarization: {input_text[:500]}...")  # Debug: Show part of the input text

        # Ensure there's enough text for summarization
        if len(input_text) == 0:
            st.error("The PDF file seems empty or contains unsupported content.")
            return "No summary generated due to empty input."

        # Run the summarization model
        result = pipe_sum(input_text)
        st.info(f"Model raw output: {result}")  # Debug: Show the raw result from the model

        # Extract summary from result
        summary = result[0]['summary_text']  # Corrected key name
        return summary

    except Exception as e:
        st.error(f"An error occurred during summarization: {str(e)}")
        return "No summary generated due to error."

@st.cache_data
# Function to display the PDF
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app
st.title("Document Summarization App")


uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    # Create temporary file to save uploaded file
    filepath = "temp_" + uploaded_file.name
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Split into two columns for display
    col1, col2 = st.columns(2)

    # Display PDF in the first column
    with col1:
        st.info("Uploaded PDF")
        displayPDF(filepath)

    # Summarize the PDF in the second column
    with col2:
        st.info("Generating Summary...")
        summary = llm_pipeline(filepath)
        st.success("Summarization Complete")
        st.write(summary)
