


from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_youtube(url):
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=["en", "hi","ne"],
        translation="en",
    )
    docs = loader.load()
    text = f"{docs}"
    return text 

def model_summarize(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    message = model(
        [
            SystemMessage(content="Summarize in around 200 words"),
            HumanMessage(content=text),
        ]
    )
    return message

# Streamlit code
def main():
    st.title("YouTube Channel Summarizer(Hindi,Nepali,English)")
    youtube_link = st.text_input("Enter YouTube Channel Link:")
    if st.button("Summarize"):
        if youtube_link:
            st.info("Please wait, summarizing the content...")
            try:
                youtube_text = load_youtube(youtube_link)
                summary = model_summarize(youtube_text)
                st.success(summary.content)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()



