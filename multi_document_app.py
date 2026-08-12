import streamlit as st
from dotenv import load_dotenv

from multi_document_rag import MultiDocumentRAG


load_dotenv()


st.set_page_config(
    page_title="Multi Document RAG Chatbot",
    page_icon="📚",
    layout="wide",
)


st.title(
    "📚 Different Document Types"
)

st.markdown(
    """
This RAG system supports:

- PDF
- TXT
- DOCX
- CSV
- PPTX

Upload one or multiple documents and ask
questions across all of them.
"""
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = MultiDocumentRAG()


if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


rag = st.session_state.rag


# ---------------------------------------------------------
# File uploader
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload documents",
    type=[
        "pdf",
        "txt",
        "docx",
        "csv",
        "pptx",
    ],
    accept_multiple_files=True,
)


# ---------------------------------------------------------
# Index documents
# ---------------------------------------------------------

if uploaded_files:

    if st.button(
        "📥 Process Documents",
        type="primary",
    ):

        with st.spinner(
            "Reading and indexing documents..."
        ):

            try:

                rag.chunks = []
                rag.embeddings = None

                total_chunks = 0

                file_names = []

                for uploaded_file in (
                    uploaded_files
                ):

                    file_bytes = (
                        uploaded_file.getvalue()
                    )

                    count = rag.add_document(
                        file_bytes,
                        uploaded_file.name,
                    )

                    total_chunks += count

                    file_names.append(
                        uploaded_file.name
                    )

                rag.build_index()

                st.session_state.indexed_files = (
                    file_names
                )

                st.success(
                    f"Successfully indexed "
                    f"{len(file_names)} document(s) "
                    f"with {total_chunks} chunks."
                )

            except Exception as e:

                st.error(
                    f"Document processing failed: {e}"
                )


# ---------------------------------------------------------
# Show indexed files
# ---------------------------------------------------------

if st.session_state.indexed_files:

    st.subheader(
        "Indexed Documents"
    )

    for filename in (
        st.session_state.indexed_files
    ):

        extension = (
            filename
            .split(".")[-1]
            .upper()
        )

        st.write(
            f"✅ {filename} ({extension})"
        )


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

if rag.embeddings is not None:

    st.divider()

    st.subheader(
        "Ask a Question"
    )

    question = st.text_input(
        "Question",
        placeholder=(
            "Example: Which document mentions "
            "the highest revenue?"
        ),
    )

    ask = st.button(
        "🔍 Ask Question",
        type="primary",
    )

    if ask:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()

        with st.spinner(
            "Searching documents..."
        ):

            try:

                results = rag.retrieve(
                    question,
                    top_k=4,
                )

            except Exception as e:

                st.error(
                    f"Retrieval failed: {e}"
                )

                st.stop()

        st.subheader(
            "Retrieved Sources"
        )

        for chunk, score in results:

            with st.expander(
                (
                    f"{chunk.document_name} | "
                    f"{chunk.document_type} | "
                    f"{chunk.source}"
                )
            ):

                st.write(
                    f"Similarity: "
                    f"{score:.3f}"
                )

                st.write(
                    chunk.text
                )

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = rag.answer(
                    question,
                    results,
                )

            except Exception as e:

                st.error(
                    f"Answer generation failed: {e}"
                )

                st.stop()

        st.subheader(
            "Answer"
        )

        st.markdown(answer)