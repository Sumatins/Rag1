import re
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Retrieval Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊Retrieval Evaluation")

st.write(
    "Evaluate the retrieval system using 10 questions "
    "with expected answers."
)


# =========================================================
# KNOWLEDGE BASE
# =========================================================

documents = [
    {
        "id": "doc1",
        "text": "CropCare 360 is an agricultural platform that helps farmers monitor crop health."
    },
    {
        "id": "doc2",
        "text": "CropCare 360 can help detect crop diseases using image analysis."
    },
    {
        "id": "doc3",
        "text": "Farmers can receive agricultural recommendations through CropCare 360."
    },
    {
        "id": "doc4",
        "text": "CropCare 360 helps farmers improve crop productivity."
    },
    {
        "id": "doc5",
        "text": "Crop diseases can affect crop productivity and farmer income."
    },
    {
        "id": "doc6",
        "text": "Weather conditions such as rainfall and temperature affect crop growth."
    },
    {
        "id": "doc7",
        "text": "Artificial intelligence can identify plant diseases from crop images."
    },
    {
        "id": "doc8",
        "text": "Crop health monitoring helps farmers identify problems early."
    },
    {
        "id": "doc9",
        "text": "Digital agriculture platforms help farmers make better crop management decisions."
    },
    {
        "id": "doc10",
        "text": "Soil moisture and nutrient levels are important for healthy crop growth."
    }
]


# =========================================================
# 10 QUESTIONS
# =========================================================

questions = [
    {
        "question": "What is CropCare 360?",
        "expected_answer": "CropCare 360 is an agricultural platform.",
        "relevant_docs": ["doc1"]
    },
    {
        "question": "How does CropCare 360 help farmers monitor crops?",
        "expected_answer": "CropCare 360 helps farmers monitor crop health.",
        "relevant_docs": ["doc1", "doc8"]
    },
    {
        "question": "Can CropCare 360 detect crop diseases?",
        "expected_answer": "CropCare 360 can help detect crop diseases using image analysis.",
        "relevant_docs": ["doc2"]
    },
    {
        "question": "What recommendations can farmers receive?",
        "expected_answer": "Farmers can receive agricultural recommendations.",
        "relevant_docs": ["doc3"]
    },
    {
        "question": "How does CropCare 360 improve productivity?",
        "expected_answer": "CropCare 360 helps farmers improve crop productivity.",
        "relevant_docs": ["doc4"]
    },
    {
        "question": "What can affect crop productivity?",
        "expected_answer": "Crop diseases can affect crop productivity.",
        "relevant_docs": ["doc5"]
    },
    {
        "question": "How does weather affect crops?",
        "expected_answer": "Rainfall and temperature can affect crop growth.",
        "relevant_docs": ["doc6"]
    },
    {
        "question": "How can artificial intelligence identify plant diseases?",
        "expected_answer": "Artificial intelligence can identify plant diseases from crop images.",
        "relevant_docs": ["doc7"]
    },
    {
        "question": "Why is crop health monitoring useful?",
        "expected_answer": "Crop health monitoring helps farmers identify problems early.",
        "relevant_docs": ["doc8"]
    },
    {
        "question": "What factors are important for healthy crop growth?",
        "expected_answer": "Soil moisture and nutrient levels are important for healthy crop growth.",
        "relevant_docs": ["doc10"]
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

def retrieve(question, top_k=3):

    query_words = tokenize(question)

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
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# ANSWER GENERATION
# =========================================================

def generate_answer(retrieved_documents):

    if not retrieved_documents:
        return "No answer found."

    return retrieved_documents[0]["text"]


# =========================================================
# ANSWER CORRECTNESS
# =========================================================

def calculate_correctness(
    generated_answer,
    expected_answer
):

    generated_words = tokenize(
        generated_answer
    )

    expected_words = tokenize(
        expected_answer
    )

    if not expected_words:
        return 0

    common_words = (
        generated_words.intersection(
            expected_words
        )
    )

    return (
        len(common_words)
        / len(expected_words)
    )


# =========================================================
# RUN EVALUATION BUTTON
# =========================================================

if st.button(
    "🚀 Run Retrieval Evaluation",
    type="primary"
):

    retrieval_hits = 0

    total_retrieved = 0
    total_relevant = 0
    total_relevant_retrieved = 0

    correctness_scores = []

    results_table = []

    # =====================================================
    # EVALUATE 10 QUESTIONS
    # =====================================================

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        expected_answer = item[
            "expected_answer"
        ]

        relevant_docs = set(
            item["relevant_docs"]
        )

        retrieved = retrieve(
            question,
            top_k=3
        )

        retrieved_ids = [
            result["id"]
            for result in retrieved
        ]

        retrieved_set = set(
            retrieved_ids
        )

        relevant_retrieved = (
            retrieved_set.intersection(
                relevant_docs
            )
        )

        # Retrieval accuracy
        if relevant_retrieved:
            retrieval_hits += 1

        # Precision
        precision = (
            len(relevant_retrieved)
            / len(retrieved_set)
            if retrieved_set
            else 0
        )

        # Recall
        recall = (
            len(relevant_retrieved)
            / len(relevant_docs)
            if relevant_docs
            else 0
        )

        # Answer
        generated_answer = generate_answer(
            retrieved
        )

        # Correctness
        correctness = calculate_correctness(
            generated_answer,
            expected_answer
        )

        correctness_scores.append(
            correctness
        )

        total_retrieved += len(
            retrieved_set
        )

        total_relevant += len(
            relevant_docs
        )

        total_relevant_retrieved += len(
            relevant_retrieved
        )

        results_table.append({
            "Question": number,
            "Precision": round(
                precision, 2
            ),
            "Recall": round(
                recall, 2
            ),
            "Answer Correctness": round(
                correctness, 2
            ),
            "Retrieved Documents": ", ".join(
                retrieved_ids
            )
        })


    # =====================================================
    # FINAL METRICS
    # =====================================================

    retrieval_accuracy = (
        retrieval_hits
        / len(questions)
    )

    overall_precision = (
        total_relevant_retrieved
        / total_retrieved
        if total_retrieved
        else 0
    )

    overall_recall = (
        total_relevant_retrieved
        / total_relevant
        if total_relevant
        else 0
    )

    answer_correctness = (
        sum(correctness_scores)
        / len(correctness_scores)
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.success(
        "Evaluation completed successfully!"
    )

    st.subheader(
        "📈 Overall Evaluation Metrics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Retrieval Accuracy",
            f"{retrieval_accuracy:.2%}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{overall_precision:.2%}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{overall_recall:.2%}"
        )

    with col4:
        st.metric(
            "Answer Correctness",
            f"{answer_correctness:.2%}"
        )


    # =====================================================
    # DETAILED RESULTS
    # =====================================================

    st.subheader(
        "📋 Question-by-Question Results"
    )

    st.dataframe(
        results_table,
        use_container_width=True
    )


    # =====================================================
    # EXPECTED VS GENERATED ANSWERS
    # =====================================================

    st.subheader(
        "📝 Expected vs Generated Answers"
    )

    for number, item in enumerate(
        questions,
        start=1
    ):

        retrieved = retrieve(
            item["question"],
            top_k=3
        )

        generated = generate_answer(
            retrieved
        )

        with st.expander(
            f"Question {number}: {item['question']}"
        ):

            st.write(
                "**Expected Answer:**"
            )

            st.info(
                item["expected_answer"]
            )

            st.write(
                "**Generated Answer:**"
            )

            st.success(
                generated
            )
