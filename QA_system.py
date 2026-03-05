import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import tempfile
import hashlib
import os

st.set_page_config(page_title="PDF QA with Hugging Face", layout="wide")
st.title("📄 PDF Question Answering System (Optimized with Progress)")

# ---------- Helper Functions ----------

def hash_file(file_path):
    """Compute hash of PDF for caching"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

@st.cache_resource(show_spinner=False)
def get_vectorstore_with_progress(chunks, use_gpu=True):
    """Create FAISS vectorstore with progress bar"""
    device = "cuda" if use_gpu else "cpu"
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    vectors = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        vec = embeddings_model.embed_query(chunk)
        vectors.append(vec)
        progress_bar.progress((i + 1) / len(chunks))
    # Build FAISS index from vectors and chunks
    vectorstore = FAISS.from_texts(chunks, embeddings_model)
    return vectorstore

# ---------- Streamlit App ----------

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Check Hugging Face token
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("❌ Please set the environment variable HUGGINGFACEHUB_API_TOKEN")
        st.stop()

    st.success("PDF Uploaded Successfully ✅")

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    pdf_hash = hash_file(pdf_path)

    # Extract text
    with st.spinner("Extracting text from PDF..."):
        full_text = extract_text_from_pdf(pdf_path)

    # Split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=50  # Reduced overlap for faster processing
    )
    chunks = text_splitter.split_text(full_text)
    st.write(f"PDF split into {len(chunks)} chunks.")

    # Build vectorstore with progress
    st.info("Generating embeddings (progress will be shown)...")
    vectorstore = get_vectorstore_with_progress(chunks, use_gpu=False)  # Set use_gpu=True if GPU available

    # User query
    query = st.text_input("Ask a question from the PDF:")

    if query:
        with st.spinner("Searching for answers..."):
            docs = vectorstore.similarity_search(query, k=3)
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-small",
                model_kwargs={"temperature": 0, "max_length": 256}
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=query)

        st.subheader("📝 Answer")
        st.write(answer)