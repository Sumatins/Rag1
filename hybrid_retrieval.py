import re
import math
import streamlit as st
from collections import Counter


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hybrid Retrieval",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎Hybrid Retrieval")

st.write(
    "Hybrid Retrieval combines Keyword Search and Vector Search."
)


# =========================================================
# SAMPLE DOCUMENTS
# =========================================================

documents = [
    {
        "id": 1,
        "text": "CropCare 360 is an agricultural platform that helps farmers monitor crops and improve crop productivity."
    },
    {
        "id": 2,
        "text": "Crop diseases can be detected using image analysis and machine learning techniques."
    },
    {
        "id": 3,
        "text": "Farmers can use CropCare 360 to monitor crop health, detect diseases, and receive agricultural recommendations."
    },
    {
        "id": 4,
        "text": "Weather conditions such as rainfall and temperature can affect crop growth and agricultural productivity."
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
    }
]


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# =========================================================
# KEYWORD SEARCH
# =========================================================

def keyword_search(
    query,
    documents,
    top_k
):

    query_words = tokenize(query)

    results = []

    for document in documents:

        document_words = tokenize(
            document["text"]
        )

        score = sum(
            1
            for word in query_words
            if word in document_words
        )

        results.append({
            "id": document["id"],
            "text": document["text"],
            "keyword_score": score
        })

    results.sort(
        key=lambda x: x["keyword_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# VECTOR SEARCH
# =========================================================

def create_vectors(documents):

    vocabulary = set()

    for document in documents:

        vocabulary.update(
            tokenize(document["text"])
        )

    vocabulary = sorted(vocabulary)

    vectors = []

    for document in documents:

        words = Counter(
            tokenize(document["text"])
        )

        vector = [
            words[word]
            for word in vocabulary
        ]

        vectors.append(vector)

    return vocabulary, vectors


def cosine_similarity(
    vector1,
    vector2
):

    dot_product = sum(
        a * b
        for a, b in zip(
            vector1,
            vector2
        )
    )

    magnitude1 = math.sqrt(
        sum(
            a * a
            for a in vector1
        )
    )

    magnitude2 = math.sqrt(
        sum(
            b * b
            for b in vector2
        )
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (
        magnitude1 * magnitude2
    )


def vector_search(
    query,
    documents,
    top_k
):

    vocabulary, vectors = create_vectors(
        documents
    )

    query_words = Counter(
        tokenize(query)
    )

    query_vector = [
        query_words[word]
        for word in vocabulary
    ]

    results = []

    for document, vector in zip(
        documents,
        vectors
    ):

        score = cosine_similarity(
            query_vector,
            vector
        )

        results.append({
            "id": document["id"],
            "text": document["text"],
            "vector_score": score
        })

    results.sort(
        key=lambda x: x["vector_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(
    query,
    documents,
    top_k,
    keyword_weight,
    vector_weight
):

    keyword_results = keyword_search(
        query,
        documents,
        len(documents)
    )

    vector_results = vector_search(
        query,
        documents,
        len(documents)
    )

    keyword_scores = {
        result["id"]:
        result["keyword_score"]
        for result in keyword_results
    }

    vector_scores = {
        result["id"]:
        result["vector_score"]
        for result in vector_results
    }

    max_keyword = max(
        keyword_scores.values()
    ) if keyword_scores else 1

    results = []

    for document in documents:

        keyword_score = keyword_scores.get(
            document["id"],
            0
        )

        vector_score = vector_scores.get(
            document["id"],
            0
        )

        normalized_keyword = (
            keyword_score / max_keyword
            if max_keyword > 0
            else 0
        )

        hybrid_score = (
            keyword_weight * normalized_keyword
            + vector_weight * vector_score
        )

        results.append({
            "id": document["id"],
            "text": document["text"],
            "keyword_score": keyword_score,
            "vector_score": vector_score,
            "hybrid_score": hybrid_score
        })

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# STREAMLIT CONTROLS
# =========================================================

st.sidebar.header("⚙️ Retrieval Settings")

top_k = st.sidebar.slider(
    "Top-K Results",
    min_value=1,
    max_value=10,
    value=3
)

keyword_weight = st.sidebar.slider(
    "Keyword Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

vector_weight = 1.0 - keyword_weight

st.sidebar.write(
    f"Vector Weight: {vector_weight:.1f}"
)


# =========================================================
# QUERY
# =========================================================

query = st.text_input(
    "Enter your question:",
    placeholder="Example: What is CropCare 360 used for?"
)


# =========================================================
# SEARCH BUTTON
# =========================================================

if st.button(
    "🔎 Run Hybrid Retrieval",
    type="primary"
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # -------------------------------------------------
        # Keyword Search
        # -------------------------------------------------

        keyword_results = keyword_search(
            query,
            documents,
            top_k
        )

        st.subheader(
            "🔤 Keyword Search Results"
        )

        for result in keyword_results:

            st.write(
                f"**Document {result['id']}**"
            )

            st.write(
                f"Keyword Score: "
                f"{result['keyword_score']}"
            )

            st.write(
                result["text"]
            )

            st.divider()


        # -------------------------------------------------
        # Vector Search
        # -------------------------------------------------

        vector_results = vector_search(
            query,
            documents,
            top_k
        )

        st.subheader(
            "🧠 Vector Search Results"
        )

        for result in vector_results:

            st.write(
                f"**Document {result['id']}**"
            )

            st.write(
                f"Vector Score: "
                f"{result['vector_score']:.4f}"
            )

            st.write(
                result["text"]
            )

            st.divider()


        # -------------------------------------------------
        # Hybrid Retrieval
        # -------------------------------------------------

        hybrid_results = hybrid_search(
            query,
            documents,
            top_k,
            keyword_weight,
            vector_weight
        )

        st.subheader(
            "🚀 Hybrid Retrieval Results"
        )

        st.success(
            "Keyword Search + Vector Search combined successfully!"
        )

        for rank, result in enumerate(
            hybrid_results,
            start=1
        ):

            st.write(
                f"### Rank {rank} — "
                f"Document {result['id']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Keyword Score",
                    result["keyword_score"]
                )

            with col2:
                st.metric(
                    "Vector Score",
                    f"{result['vector_score']:.4f}"
                )

            with col3:
                st.metric(
                    "Hybrid Score",
                    f"{result['hybrid_score']:.4f}"
                )

            st.write(
                result["text"]
            )

            st.divider()


        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        st.subheader(
            "📌 Hybrid Retrieval Pipeline"
        )

        st.code(
            """
Question
   ↓
Keyword Search
   +
Vector Search
   ↓
Combine Scores
   ↓
Hybrid Ranking
   ↓
Top-K Results
            """
        )