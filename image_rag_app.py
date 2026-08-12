import streamlit as st
from dotenv import load_dotenv

from image_rag import create_image_rag


load_dotenv()


st.set_page_config(
    page_title="Image RAG Chatbot",
    page_icon="🖼️",
    layout="wide",
)


st.title("🖼️Image RAG Chatbot")

st.write(
    """
Upload a PDF containing images, tables,
charts or diagrams and ask questions
about the visual content.
"""
)


if "rag" not in st.session_state:
    st.session_state.rag = None


if "filename" not in st.session_state:
    st.session_state.filename = None


uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    if (
        st.session_state.filename
        != uploaded_file.name
    ):

        with st.spinner(
            "Analyzing PDF pages..."
        ):

            try:

                rag = create_image_rag()

                # Analyze only first 10 pages.
                # This avoids exceeding API limits.
                rag.build_index(
                    pdf_bytes,
                    max_pages=10,
                )

                st.session_state.rag = rag

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.success(
                    f"Successfully analyzed "
                    f"{len(rag.pages)} page(s)."
                )

            except Exception as e:

                st.error(
                    f"PDF processing failed: {e}"
                )

                st.stop()


if st.session_state.rag:

    rag = st.session_state.rag

    st.divider()

    st.subheader(
        "Ask about the PDF"
    )

    question = st.text_input(
        "Question",
        placeholder=(
            "Example: What is the highest "
            "value in the chart?"
        ),
    )

    if st.button(
        "🔍 Ask",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()

        with st.spinner(
            "Finding relevant pages..."
        ):

            pages = rag.retrieve(
                question,
                top_k=2,
            )

        st.subheader(
            "Relevant Pages"
        )

        columns = st.columns(
            len(pages)
        )

        for column, page in zip(
            columns,
            pages,
        ):

            with column:

                st.image(
                    page.image_bytes,
                    caption=(
                        f"Page {page.page_number}"
                    ),
                    use_container_width=True,
                )

        with st.spinner(
            "Generating visual answer..."
        ):

            answer = rag.answer(
                question,
                pages,
            )

        st.subheader(
            "Answer"
        )

        st.markdown(answer)