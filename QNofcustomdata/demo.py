import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# Configure the Google Generative AI API with your API key
api_key = "AIzaSyBK6Cdw1szVPjWK-57uDtOTR1nNm6Y6v38"

# Create a GoogleGenerativeAI LLM instance
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Streamlit specific code
import streamlit as st

# Streamlit app
st.title("Text Generation with Google Generative AI")

# Text input for question
question = st.text_input("Enter your question:")

# Button to generate text
if st.button("Generate Text"):
    generated_text = llm.invoke(question)
    st.text_area("Generated Text:", generated_text)
