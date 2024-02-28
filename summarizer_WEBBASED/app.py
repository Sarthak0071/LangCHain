import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
import google.generativeai as genai

# Load environment variables
load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to summarize web content
def summarize_web_content(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    llm = GoogleGenerativeAI(model="gemini-pro")
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.run(docs)
    return summary

# Streamlit UI
st.title("Web Page Summarizer")

url = st.text_input("Enter the URL of the web page you want to summarize:")
if st.button("Summarize"):
    if url:
        summary = summarize_web_content(url)
        st.header("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter a URL.")
