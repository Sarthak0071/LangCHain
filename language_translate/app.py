


from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("Chat with Google Generative AI")

# Function to interact with the model
def mod(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    system_response = model([
        SystemMessage(content="Perform Language translate only in hindi"),
        HumanMessage(content=text),
    ])
    return system_response

# Input for user message content
user_message = st.text_area("Enter your message here:",height=200)

if st.button("Send"):
    # Call the mod function with user input
    system_response = mod(user_message)

    # Display system response if it exists
    if system_response:
        st.write("System Response:")
        st.write(system_response.content)
    else:
        st.write("No response from the system.")

