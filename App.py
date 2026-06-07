import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain.chains import RetrievalQA

# ----------------------------------------
# Streamlit Page Config
# ----------------------------------------

st.set_page_config(
    page_title="Multi Source Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi Source Chatbot")
st.write("Upload PDF, TXT, DOCX files and chat with your documents.")

# ----------------------------------------
# Sidebar
# ----------------------------------------

with st.sidebar:
    st.header("Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

# ----------------------------------------
# Session State
# ----------------------------------------

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------------
# Process Documents
# ----------------------------------------

def load_documents(files):
    documents = []

    for file in files:

        suffix = "." + file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        if suffix == ".pdf":
            loader = PyPDFLoader(temp_path)

        elif suffix == ".txt":
            loader = TextLoader(temp_path)

        elif suffix == ".docx":
            loader = Docx2txtLoader(temp_path)

        else:
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents

# ----------------------------------------
# Create Vector Store
# ----------------------------------------

def create_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore

# ----------------------------------------
# Build QA Chain
# ----------------------------------------

def create_qa_chain(vectorstore, api_key):

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain

# ----------------------------------------
# Process Button
# ----------------------------------------

if st.sidebar.button("Process Documents"):

    if not uploaded_files:
        st.error("Please upload documents.")
    elif not groq_api_key:
        st.error("Please enter Groq API Key.")
    else:

        with st.spinner("Processing Documents..."):

            docs = load_documents(uploaded_files)

            vectorstore = create_vector_store(docs)

            qa_chain = create_qa_chain(
                vectorstore,
                groq_api_key
            )

            st.session_state.qa_chain = qa_chain

        st.success("Documents processed successfully!")

# ----------------------------------------
# Chat Interface
# ----------------------------------------

query = st.chat_input("Ask a question...")

if query:

    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    if st.session_state.qa_chain:

        with st.spinner("Generating response..."):

            result = st.session_state.qa_chain.invoke(
                {"query": query}
            )

            answer = result["result"]

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

    else:

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": "Please upload and process documents first."
            }
        )

# ----------------------------------------
# Display Chat
# ----------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])
