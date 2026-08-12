import re
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Reranking",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄Reranking")

st.write(
    "Retriever gets Top 10 documents, "
    "then the Reranker selects the best Top 3 documents."
)


# =========================================================
# SAMPLE DOCUMENTS
# =========================================================

documents = [
    {
        "id": 1,
        "text": "CropCare 360 is an agricultural platform that helps farmers monitor crop health and improve crop productivity."
    },
    {
        "id": 2,
        "text": "Crop diseases can be detected using image analysis and machine learning techniques."
    },
    {
        "id": 3,
        "text": "CropCare 360 helps farmers monitor crop health, detect diseases, and receive agricultural recommendations."
    },
    {
        "id": 4,
        "text": "Weather conditions such as rainfall and temperature can affect crop growth."
    },
    {
        "id": 5,
        "text": "Artificial intelligence can help farmers identify plant diseases from crop images."
    },
    {
        "id": 6,
        "text": "Agricultural recommendations can help farmers improve crop productivity."
    },
    {
        "id": 7,
        "text": "Crop health monitoring helps farmers identify problems at an early stage."
    },
    {
        "id": 8,
        "text": "Machine learning models can analyze agricultural data and provide useful predictions."
    },
    {
        "id": 9,
        "text": "CropCare 360 combines crop monitoring, disease detection, and agricultural recommendations."
    },
    {
        "id": 10,
        "text": "Farmers can monitor crops using digital agriculture platforms."
    },
    {
        "id": 11,
        "text": "Soil moisture and nutrient levels are important for healthy crop growth."
    },
    {
        "id": 12,
        "text": "Agricultural technology helps farmers make better decisions about crop management."
    }
]


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


# =========================================================
# RETRIEVER
# =========================================================
# Retrieves TOP 10
# =========================================================

def retrieve(
    query,
    documents,
    top_k=10
):

    query_words = tokenize(query)

    results = []

    for document in documents:

        document_words = tokenize(
            document["text"]
        )

        score = len(
            query_words.intersection(
                document_words
            )
        )

        results.append({
            "id": document["id"],
            "text": document["text"],
            "retrieval_score": score
        })

    results.sort(
        key=lambda x: x["retrieval_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# RERANKER
# =========================================================
# Takes Top 10 and selects Top 3
# =========================================================

def rerank(
    query,
    retrieved_documents,
    top_k=3
):

    query_words = tokenize(query)

    reranked_documents = []

    for document in retrieved_documents:

        document_words = tokenize(
            document["text"]
        )

        # Number of matching query words
        keyword_score = len(
            query_words.intersection(
                document_words
            )
        )

        # Give additional importance to
        # exact phrase matches
        phrase_bonus = 0

        query_lower = query.lower()
        document_lower = document[
            "text"
        ].lower()

        if query_lower in document_lower:
            phrase_bonus = 5

        # Final reranking score
        rerank_score = (
            keyword_score
            + phrase_bonus
        )

        reranked_documents.append({
            "id": document["id"],
            "text": document["text"],
            "retrieval_score": document[
                "retrieval_score"
            ],
            "rerank_score": rerank_score
        })

    reranked_documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_documents[:top_k]


# =========================================================
# LLM
# =========================================================

def generate_answer(
    query,
    top_documents
):

    if not top_documents:

        return (
            "I could not find relevant "
            "information."
        )

    answer_parts = []

    for document in top_documents:

        answer_parts.append(
            document["text"]
        )

    return " ".join(answer_parts)


# =========================================================
# STREAMLIT SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ Retrieval Configuration"
)

retriever_top_k = st.sidebar.number_input(
    "Retriever Top-K",
    min_value=10,
    max_value=10,
    value=10
)

reranker_top_k = st.sidebar.number_input(
    "Reranker Top-K",
    min_value=3,
    max_value=3,
    value=3
)


# =========================================================
# QUESTION
# =========================================================

query = st.text_input(
    "Enter your question:",
    placeholder=(
        "Example: What is CropCare 360 used for?"
    )
)


# =========================================================
# RUN BUTTON
# =========================================================

if st.button(
    "🚀 Run Reranking",
    type="primary"
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # =================================================
        # STEP 1
        # RETRIEVER → TOP 10
        # =================================================

        retrieved_documents = retrieve(
            query,
            documents,
            top_k=10
        )

        st.subheader(
            "1️⃣ Retriever → Top 10"
        )

        st.info(
            f"Retriever returned "
            f"{len(retrieved_documents)} documents."
        )

        for rank, document in enumerate(
            retrieved_documents,
            start=1
        ):

            with st.expander(
                f"Rank {rank} — "
                f"Document {document['id']}"
            ):

                st.write(
                    f"**Retrieval Score:** "
                    f"{document['retrieval_score']}"
                )

                st.write(
                    document["text"]
                )


        # =================================================
        # STEP 2
        # RERANKER → TOP 3
        # =================================================

        reranked_documents = rerank(
            query,
            retrieved_documents,
            top_k=3
        )

        st.subheader(
            "2️⃣ Reranker → Top 3"
        )

        st.success(
            "Reranking completed successfully!"
        )

        for rank, document in enumerate(
            reranked_documents,
            start=1
        ):

            st.write(
                f"### 🏆 Rank {rank} — "
                f"Document {document['id']}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Retrieval Score",
                    document[
                        "retrieval_score"
                    ]
                )

            with col2:

                st.metric(
                    "Rerank Score",
                    document[
                        "rerank_score"
                    ]
                )

            st.write(
                document["text"]
            )

            st.divider()


        # =================================================
        # STEP 3
        # LLM
        # =================================================

        answer = generate_answer(
            query,
            reranked_documents
        )

        st.subheader(
            "3️⃣ LLM → Final Answer"
        )

        st.success(
            answer
        )


        # =================================================
        # FINAL PIPELINE
        # =================================================

        st.subheader(
            "📌 Final RAG Pipeline"
        )

        st.code(
            """
Question
   ↓
Retriever
   ↓
Top 10
   ↓
Reranker
   ↓
Top 3
   ↓
LLM
   ↓
Final Answer
            """
        )