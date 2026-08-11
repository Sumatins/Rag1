import streamlit as st
import PyPDF2

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="PDF Q&A - Top K Retrieval",
    page_icon="📚"
)

st.title("📚 PDF Question Answering")

st.write("Upload PDFs and select how many relevant results to retrieve.")

# -----------------------------------
# PDF Upload
# -----------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------------
# TOP-K SLIDER
# -----------------------------------
top_k = st.slider(
    "Select Top-K",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

st.write(f"Selected Top-K: **{top_k}**")

# -----------------------------------
# Read Uploaded PDFs
# -----------------------------------
documents = []

if uploaded_files:

    for uploaded_file in uploaded_files:

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        pdf_text = ""

        for page_number, page in enumerate(pdf_reader.pages, start=1):

            text = page.extract_text()

            if text:
                pdf_text += text + "\n"

        documents.append({
            "filename": uploaded_file.name,
            "text": pdf_text
        })

    st.success(
        f"{len(documents)} PDF(s) uploaded successfully."
    )

# -----------------------------------
# Question
# -----------------------------------
question = st.text_input(
    "Ask a question about your PDFs:"
)

# -----------------------------------
# Retrieval
# -----------------------------------
if question and documents:

    question_words = question.lower().split()

    results = []

    # Calculate relevance score for every PDF
    for document in documents:

        document_text = document["text"].lower()

        score = sum(
            1 for word in question_words
            if word in document_text
        )

        results.append({
            "filename": document["filename"],
            "text": document["text"],
            "score": score
        })

    # Sort by relevance score
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # -----------------------------------
    # Apply TOP-K
    # -----------------------------------
    top_results = results[:top_k]

    # -----------------------------------
    # Answer
    # -----------------------------------
    st.subheader("💡 Answer")

    if top_results and top_results[0]["score"] > 0:

        answer = top_results[0]["text"][:2000]

        st.write(answer)

        # -----------------------------------
        # Source Citations
        # -----------------------------------
        st.subheader("📚 Source Citations")

        for result in top_results:

            if result["score"] > 0:

                st.write(
                    f"📄 **{result['filename']}** "
                    f"(Relevance Score: {result['score']})"
                )

    else:

        st.warning(
            "No relevant information was found in the uploaded PDFs."
        )

# -----------------------------------
# Instructions
# -----------------------------------
else:

    st.info(
        "Upload PDFs and enter a question to retrieve relevant results."
    )

