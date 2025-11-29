import os
import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from pathlib import Path
import tempfile

# --------------------------
# STREAMLIT UI SETUP
# --------------------------
st.set_page_config(page_title="Knowledge Base Agent — Rooman (FREE Groq Model)", layout="wide")

st.title("📚 Knowledge Base Agent — Rooman (FREE Groq Model)")
st.markdown("Ask questions about the company docs. Upload additional .txt files to extend the knowledge base.")

# --------------------------
# GROQ API KEY INPUT
# --------------------------
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

client = None
if groq_api_key:
    client = Groq(api_key=groq_api_key)

# --------------------------
# LOAD DOCUMENTS
# --------------------------
DATA_DIR = Path("docs")
uploaded_files = st.file_uploader("Upload .txt files (optional)", type=["txt"], accept_multiple_files=True)

docs = []

# Load local docs
if DATA_DIR.exists():
    for p in DATA_DIR.glob("*.txt"):
        loader = TextLoader(str(p), encoding="utf8")
        docs.extend(loader.load())

# Load uploaded docs
for uf in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uf.read())
        temp_path = tmp.name
    loader = TextLoader(temp_path, encoding="utf8")
    docs.extend(loader.load())

# Error if no docs
if len(docs) == 0:
    st.error("No documents found. Please add files to the docs folder or upload some.")
    st.stop()

# --------------------------
# SPLIT INTO CHUNKS
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = splitter.split_documents(docs)

# --------------------------
# FREE EMBEDDINGS (MiniLM)
# --------------------------
@st.cache_resource
def build_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(texts, embeddings)
    return store

vectorstore = build_store(texts)

# --------------------------
# ANSWER GENERATION FUNCTION
# --------------------------
def answer_with_context(question, k=4):
    # Retrieve docs
    try:
        docs = vectorstore.similarity_search(question, k)
    except:
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)[:k]

    # Build context
    context_parts = []
    for i, d in enumerate(docs):
        context_parts.append(f"--- Document {i+1} ---\n{d.page_content}\n")
    context = "\n".join(context_parts) if context_parts else "No relevant context found."

    # If Groq key not provided
    if not client:
        return f"CONTEXT FOUND (But no Groq key):\n\n{context}"

    # Build prompt for Groq
    prompt = f"""
    You are a helpful AI assistant. Use ONLY the provided context to answer.
    If answer is not in the context, say "I don't know based on the documents."

    CONTEXT:
    {context}

    QUESTION: {question}
    """

    # Generate answer using Groq
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )

        # ⭐ CORRECTED GROQ RESPONSE ACCESS
        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {e}"

# --------------------------
# CHAT UI
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your question here:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        answer = answer_with_context(query)
    st.session_state.history.append((query, answer))

# Display conversation
for q, a in reversed(st.session_state.history):
    st.write(f"**User:** {q}")
    st.write(f"**AI:** {a}")
    st.markdown("---")
