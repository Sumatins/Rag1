import os

import streamlit as st

from services.vector_db import VectorDB
from services.rag import RAGService
from loaders.pdf_loader import PDFLoader
from config import DB_DIRECTORY

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("📄 PDF RAG")

    st.markdown("---")

    st.write("### Settings")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.success("Vector Database Loaded")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

# ---------------------------------------------------
# Load Database
# ---------------------------------------------------

@st.cache_resource
def load_chain():
    db_path = os.path.abspath(DB_DIRECTORY)

    if not os.path.exists(db_path):
        pdf_path = os.path.abspath(os.path.join("data", "Sumati_N_Sannaragikoppa_Resume (1).pdf"))
        if os.path.exists(pdf_path):
            docs = PDFLoader.load_pdf(pdf_path)
            if docs:
                VectorDB.create(docs)
            else:
                raise ValueError("The PDF file could not be read.")
        else:
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    db = VectorDB.load()
    chain = RAGService.create_chain(db)
    return chain


chain = load_chain()

# ---------------------------------------------------
# Title
# ---------------------------------------------------

st.title("📄 PDF Question Answering")

st.caption("Ask questions about your PDF")

# ---------------------------------------------------
# Display Chat History
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------------------------------------------
# User Input
# ---------------------------------------------------

question = st.chat_input("Ask a question about your PDF...")

if question:

    # Show User Question

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Assistant

    with st.chat_message("assistant"):

        with st.spinner("Searching document..."):

            try:

                response = chain.invoke(
                    {
                        "input": question
                    }
                )

                answer = response.get("answer", "")

                if not answer or answer.strip() == "":
                    answer = "I couldn't find that information in the uploaded document."

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")

                print(e)