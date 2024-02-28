
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_answer(question, uploaded_file):
    if uploaded_file is None:
        st.error("Please upload a document first.")
        return ""

    llm = GoogleGenerativeAI(model="gemini-pro")

    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = Docx2txtLoader(uploaded_file.name)
    docs = loader.load()

    os.remove(uploaded_file.name)  # Remove the uploaded file after loading

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

st.title("Document Question Answering System")
uploaded_file = st.file_uploader("Upload Document", type=['docx'])
question = st.text_area("Ask your question here:")
st.info("Document uploaded successfully.")
if st.button("Get Answer"):
    answer = get_answer(question, uploaded_file)
    st.write("Answer:", answer)
