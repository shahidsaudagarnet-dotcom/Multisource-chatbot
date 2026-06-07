import streamlit as st
import os
import tempfile
import time

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader,
    WebBaseLoader
)

from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="OmniSum Enterprise Knowledge Copilot",
    page_icon="ðŸ“š",
    layout="wide"
)

st.title("ðŸ§  OmniSum Enterprise Knowledge Copilot")

st.markdown("""
Enterprise-style RAG system supporting:

â€¢ PDFs  
â€¢ CSV files  
â€¢ Excel files  
â€¢ Text files  
â€¢ Web pages  
â€¢ YouTube transcripts  

Features:

- Hybrid Retrieval (Vector + BM25)
- Query Rewriting
- Retrieval Reranking
- Source Citations
- Cached Vector Database
""")

# -----------------------------
# API KEY
# -----------------------------

st.sidebar.header("ðŸ”‘ API Configuration")

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

if not groq_api_key:
    st.sidebar.warning("Please enter your Groq API key")
    st.stop()

try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b"
    )

    llm.invoke("hello")

    st.sidebar.success("API Key Valid")

except Exception as e:
    st.sidebar.error(f"Invalid API Key: {e}")
    st.stop()

# -----------------------------
# EMBEDDINGS
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# -----------------------------
# DOCUMENT LOADERS
# -----------------------------

def load_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    os.remove(path)

    return docs


def load_csv(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    loader = CSVLoader(file_path=path)
    docs = loader.load()

    os.remove(path)

    return docs


def load_excel(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    loader = UnstructuredExcelLoader(file_path=path)
    docs = loader.load()

    os.remove(path)

    return docs


def load_text(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    loader = TextLoader(file_path=path)
    docs = loader.load()

    os.remove(path)

    return docs


def load_web(url):

    loader = WebBaseLoader(url)
    docs = loader.load()

    return docs


# -----------------------------
# DOCUMENT CHUNKING
# -----------------------------

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )

    return splitter.split_documents(docs)


# -----------------------------
# VECTOR DATABASE
# -----------------------------

def build_vector_store(docs):

    if "vector_store" in st.session_state:
        return st.session_state.vector_store

    st.info("Building vector database...")

    chunks = split_documents(docs)

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    st.session_state.vector_store = vector_store

    return vector_store


# -----------------------------
# HYBRID RETRIEVER
# -----------------------------

def build_hybrid_retriever(docs, vector_store):

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 4

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.4, 0.6]
    )

    return hybrid


# -----------------------------
# QUERY REWRITING
# -----------------------------

def rewrite_query(query):

    prompt = f"""
Improve the following search query for better document retrieval.

Query:
{query}

Improved Query:
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# -----------------------------
# RERANK DOCUMENTS
# -----------------------------

def rerank_docs(query, docs):

    scored = []

    query_words = set(query.lower().split())

    for doc in docs:

        text_words = set(doc.page_content.lower().split())

        score = len(query_words.intersection(text_words))

        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored[:4]]


# -----------------------------
# ANSWER GENERATION
# -----------------------------

def generate_answer(query, docs):

    vector_store = build_vector_store(docs)

    retriever = build_hybrid_retriever(docs, vector_store)

    improved_query = rewrite_query(query)

    retrieved_docs = retriever.invoke(improved_query)

    top_docs = rerank_docs(improved_query, retrieved_docs)

    context = "\n\n".join([d.page_content for d in top_docs])

    prompt = f"""
You are an enterprise AI knowledge assistant.

Use ONLY the context below to answer the question.

Provide:
- clear explanation
- key insights
- references

Context:
{context}

Question:
{query}

Answer:
"""

    start = time.time()

    response = llm.invoke(prompt)

    latency = time.time() - start

    return response.content, top_docs, latency


# -----------------------------
# SIDEBAR SOURCE SELECTION
# -----------------------------

st.sidebar.header("ðŸ“¥ Data Source")

source = st.sidebar.selectbox(
    "Choose Source",
    [
        "PDF",
        "CSV",
        "Excel",
        "Text File",
        "Web Page"
    ]
)

documents = None

if source == "PDF":

    uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and st.sidebar.button("Load PDF"):

        documents = load_pdf(uploaded)

elif source == "CSV":

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded and st.sidebar.button("Load CSV"):

        documents = load_csv(uploaded)

elif source == "Excel":

    uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])

    if uploaded and st.sidebar.button("Load Excel"):

        documents = load_excel(uploaded)

elif source == "Text File":

    uploaded = st.sidebar.file_uploader("Upload TXT", type=["txt"])

    if uploaded and st.sidebar.button("Load TXT"):

        documents = load_text(uploaded)

elif source == "Web Page":

    url = st.sidebar.text_input("Enter URL")

    if url and st.sidebar.button("Load Web Page"):

        documents = load_web(url)


if documents:
    st.session_state.documents = documents
    st.success(f"Loaded {len(documents)} documents")


# -----------------------------
# QUERY UI
# -----------------------------

if "documents" in st.session_state:

    st.header("Ask Questions About Your Data")

    query = st.text_area(
        "Ask anything about the content"
    )

    if st.button("Generate Answer"):

        with st.spinner("Running Enterprise RAG Pipeline..."):

            answer, docs, latency = generate_answer(
                query,
                st.session_state.documents
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        for i, doc in enumerate(docs):

            st.markdown(f"**Source {i+1}**")

            st.write(doc.page_content[:500] + "...")

            st.markdown("---")

        st.sidebar.info(f"Latency: {latency:.2f}s")

else:

    st.info("Load a document from the sidebar to start.")