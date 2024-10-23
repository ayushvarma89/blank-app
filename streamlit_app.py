import streamlit as st
from gtts import gTTS
from PyPDF2 import PdfReader
from pydub import AudioSegment
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK resources
nltk.download('punkt')

# Function to convert PDF to speech
def pdf_to_speech(pdf_file):
    # Read the PDF file
    pdf_reader = PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)

    # Extract text from all pages
    full_text = ""
    for page_num in range(total_pages):
        page = pdf_reader.pages[page_num]
        full_text += page.extract_text() + " "

    return full_text  # Return the extracted text

def text_to_audio(text):
    # Convert extracted text to speech using gTTS
    tts = gTTS(text=text, lang='en')
    audio_file = "output.mp3"
    tts.save(audio_file)

    return audio_file  # Return the generated audio file path

def summarize_text(text, num_sentences=3):
    """Summarize the given text by extracting the first few sentences."""
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    return ' '.join(sentences[:num_sentences])  # Return the first `num_sentences` sentences

# Streamlit App
st.title("PDF to Speech Converter")

# Upload PDF file
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    # Convert PDF to text
    pdf_text = pdf_to_speech(uploaded_file)

    # Display PDF summary
    summary = summarize_text(pdf_text, num_sentences=3)
    st.markdown(f"### Summary of PDF Content:\n\n{summary}")

    # Convert text to audio
    audio_file = text_to_audio(pdf_text)

    # Audio player
    st.audio(audio_file, format='audio/mp3')
