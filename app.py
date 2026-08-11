import os
import shutil
import time

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from config import (
    GROQ_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 RAG Chatbot")

st.write(
    "RAG system with configurable chunking strategy, "
    "chunk size, and chunk overlap."
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "processed" not in st.session_state:
    st.session_state.processed = False


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model=MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0,
    )


embedding_model = load_embedding()

llm = load_llm()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")


    # ========================================================
    # CHUNKING STRATEGY
    # ========================================================

    st.subheader("Chunking Strategy")

    chunking_strategy = st.selectbox(
        "Select chunking strategy",
        options=[
            "recursive",
            "character",
        ],
        format_func=lambda x: x.title(),
    )


    # ========================================================
    # CHUNK SIZE
    # ========================================================

    st.subheader("Chunk Size")

    chunk_size = st.slider(
        "Select chunk size",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100,
        help=(
            "Controls the maximum number of characters "
            "in each chunk."
        ),
    )


    # ========================================================
    # CHUNK OVERLAP
    # ========================================================

    st.subheader("Chunk Overlap")

    chunk_overlap = st.slider(
        "Select chunk overlap",
        min_value=0,
        max_value=1000,
        value=200,
        step=50,
        help=(
            "Controls how many characters from the "
            "previous chunk are repeated in the next chunk."
        ),
    )


    # ========================================================
    # VALIDATE OVERLAP
    # ========================================================

    if chunk_overlap >= chunk_size:

        st.error(
            "Chunk overlap must be smaller than chunk size."
        )

    else:

        st.success(
            "Chunking configuration is valid."
        )


    # ========================================================
    # SHOW CURRENT SETTINGS
    # ========================================================

    st.divider()

    st.write("### Current Settings")

    st.write(
        f"**Strategy:** {chunking_strategy.title()}"
    )

    st.write(
        f"**Chunk Size:** {chunk_size}"
    )

    st.write(
        f"**Chunk Overlap:** {chunk_overlap}"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 Document Processing")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    if st.button(
        "🚀 Process PDF",
        use_container_width=True,
    ):

        if chunk_overlap >= chunk_size:

            st.error(
                "Chunk overlap must be smaller than chunk size."
            )

            st.stop()


        # ----------------------------------------------------
        # Create temporary PDF
        # ----------------------------------------------------

        temp_dir = "temp_uploads"

        os.makedirs(
            temp_dir,
            exist_ok=True,
        )

        pdf_path = os.path.join(
            temp_dir,
            uploaded_file.name,
        )


        with open(
            pdf_path,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getbuffer()
            )


        # ----------------------------------------------------
        # Start processing timer
        # ----------------------------------------------------

        start_time = time.perf_counter()


        with st.spinner(
            "Loading and splitting PDF..."
        ):

            # Load PDF
            loader = PyPDFLoader(
                pdf_path
            )

            documents = loader.load()


            # ------------------------------------------------
            # Create splitter
            # ------------------------------------------------

            if chunking_strategy == "recursive":

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=[
                        "\n\n",
                        "\n",
                        ". ",
                        " ",
                        "",
                    ],
                )

            else:

                splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                )


            # ------------------------------------------------
            # Split documents
            # ------------------------------------------------

            chunks = splitter.split_documents(
                documents
            )


        # ----------------------------------------------------
        # Create new vector database
        # ----------------------------------------------------

        with st.spinner(
            "Creating vector database..."
        ):

            # Remove old database from session
            st.session_state.vector_db = None


            # Create unique collection name
            collection_name = (
                "rag_"
                + chunking_strategy
                + "_"
                + str(chunk_size)
                + "_"
                + str(chunk_overlap)
            )


            # Create Chroma database
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                collection_name=collection_name,
            )


            st.session_state.vector_db = vector_db

            st.session_state.chunks = chunks

            st.session_state.processed = True


        # ----------------------------------------------------
        # Stop processing timer
        # ----------------------------------------------------

        processing_time = (
            time.perf_counter() - start_time
        )


        # ----------------------------------------------------
        # Display processing results
        # ----------------------------------------------------

        st.success(
            "PDF processed successfully!"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Chunks Created",
                len(chunks),
            )


        with col2:

            st.metric(
                "Chunk Size",
                chunk_size,
            )


        with col3:

            st.metric(
                "Chunk Overlap",
                chunk_overlap,
            )


        with col4:

            st.metric(
                "Processing Time",
                f"{processing_time:.2f}s",
            )


        st.info(
            f"Strategy: **{chunking_strategy.title()}**"
        )


        # ----------------------------------------------------
        # Preview chunks
        # ----------------------------------------------------

        with st.expander(
            "🔍 Preview Generated Chunks"
        ):

            preview_count = min(
                5,
                len(chunks),
            )


            for i in range(
                preview_count
            ):

                st.write(
                    f"### Chunk {i + 1}"
                )

                st.write(
                    chunks[i].page_content
                )

                st.write("---")


# ============================================================
# QUESTION ANSWERING
# ============================================================

if st.session_state.processed:

    st.divider()

    st.header("💬 Ask Questions")


    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    if question:

        # ----------------------------------------------------
        # Display question
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.write(question)


        # ----------------------------------------------------
        # Start response timer
        # ----------------------------------------------------

        start_time = time.perf_counter()


        try:

            # =================================================
            # RETRIEVAL
            # =================================================

            documents = (
                st.session_state.vector_db
                .similarity_search(
                    question,
                    k=5,
                )
            )


            # =================================================
            # CREATE CONTEXT
            # =================================================

            context = "\n\n".join(
                document.page_content
                for document in documents
            )


            # =================================================
            # PROMPT
            # =================================================

            prompt = ChatPromptTemplate.from_template(
                """
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.

If the answer is not available in the context,
say that you could not find the answer.

Context:
{context}

Question:
{question}

Answer:
"""
            )


            formatted_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": question,
                }
            )


            # =================================================
            # LLM
            # =================================================

            response = llm.invoke(
                formatted_prompt
            )


            answer = response.content


            # =================================================
            # RESPONSE TIME
            # =================================================

            response_time = (
                time.perf_counter()
                - start_time
            )


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    "### Answer"
                )

                st.write(
                    answer
                )


                st.divider()


                # ------------------------------------------------
                # Performance
                # ------------------------------------------------

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.success(
                        f"⏱️ Response Time\n\n"
                        f"{response_time:.4f} seconds"
                    )


                with col2:

                    st.info(
                        f"📚 Retrieved Chunks\n\n"
                        f"{len(documents)}"
                    )


                with col3:

                    st.info(
                        f"🧩 Chunk Strategy\n\n"
                        f"{chunking_strategy.title()}"
                    )


                # ------------------------------------------------
                # Configuration
                # ------------------------------------------------

                st.write(
                    f"**Chunk Size:** {chunk_size}"
                )

                st.write(
                    f"**Chunk Overlap:** {chunk_overlap}"
                )

                st.write(
                    f"**Embedding:** `{EMBEDDING_MODEL}`"
                )

                st.write(
                    f"**Reasoning LLM:** `{MODEL_NAME}`"
                )


                # ------------------------------------------------
                # Retrieved chunks
                # ------------------------------------------------

                with st.expander(
                    "📖 View Retrieved Chunks"
                ):

                    for i, document in enumerate(
                        documents,
                        start=1,
                    ):

                        st.write(
                            f"### Retrieved Chunk {i}"
                        )

                        st.write(
                            document.page_content
                        )

                        st.write("---")


        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )