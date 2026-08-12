import time

from embeddings.embedding import get_embedding
from llm.llm import load_llm

from config import EMBEDDING_MODELS, REASONING_MODELS


TEST_QUESTION = """
What is Retrieval Augmented Generation (RAG)?
Explain how RAG works in a simple way.
"""


# ============================================================
# EMBEDDING MODEL TEST
# ============================================================

def test_embeddings():

    print("\n" + "=" * 80)
    print("EMBEDDING MODEL BENCHMARK")
    print("=" * 80)

    results = []

    for model_name in EMBEDDING_MODELS:

        print("\n" + "-" * 80)
        print(f"Embedding Model: {model_name}")
        print("-" * 80)

        try:

            # ------------------------------------------------
            # Measure model loading time
            # ------------------------------------------------

            load_start = time.perf_counter()

            embedding = get_embedding(model_name)

            load_time = time.perf_counter() - load_start

            print(f"Model loading time : {load_time:.4f} seconds")

            # ------------------------------------------------
            # Measure actual embedding generation time
            # ------------------------------------------------

            embed_start = time.perf_counter()

            vector = embedding.embed_query(TEST_QUESTION)

            embed_time = time.perf_counter() - embed_start

            print(f"Embedding time     : {embed_time:.4f} seconds")

            print(f"Vector dimensions  : {len(vector)}")

            total_time = load_time + embed_time

            print(f"TOTAL TIME         : {total_time:.4f} seconds")

            results.append({
                "model": model_name,
                "load_time": load_time,
                "embed_time": embed_time,
                "total_time": total_time,
                "status": "Success"
            })

        except Exception as e:

            print(f"ERROR: {e}")

            results.append({
                "model": model_name,
                "load_time": None,
                "embed_time": None,
                "total_time": None,
                "status": "Failed"
            })

    return results


# ============================================================
# REASONING LLM TEST
# ============================================================

def test_reasoning_models():

    print("\n\n" + "=" * 80)
    print("REASONING LLM BENCHMARK")
    print("=" * 80)

    results = []

    for model_name in REASONING_MODELS:

        print("\n" + "-" * 80)
        print(f"Reasoning LLM: {model_name}")
        print("-" * 80)

        try:

            # ------------------------------------------------
            # Load LLM
            # ------------------------------------------------

            load_start = time.perf_counter()

            llm = load_llm(model_name)

            load_time = time.perf_counter() - load_start

            print(f"LLM loading time   : {load_time:.4f} seconds")

            # ------------------------------------------------
            # Measure API response time
            # ------------------------------------------------

            response_start = time.perf_counter()

            response = llm.invoke(TEST_QUESTION)

            response_time = time.perf_counter() - response_start

            print(f"Response time      : {response_time:.4f} seconds")

            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            answer = response.content

            print("\nGenerated Answer:")
            print(answer)

            # ------------------------------------------------
            # Count response information
            # ------------------------------------------------

            characters = len(answer)
            words = len(answer.split())

            print("\nResponse Statistics:")
            print(f"Characters         : {characters}")
            print(f"Words              : {words}")
            print(f"TOTAL TIME         : {load_time + response_time:.4f} seconds")

            results.append({
                "model": model_name,
                "load_time": load_time,
                "response_time": response_time,
                "total_time": load_time + response_time,
                "words": words,
                "status": "Success"
            })

        except Exception as e:

            print(f"ERROR: {e}")

            results.append({
                "model": model_name,
                "load_time": None,
                "response_time": None,
                "total_time": None,
                "words": 0,
                "status": "Failed"
            })

    return results


# ============================================================
# FINAL RESULTS
# ============================================================

def print_final_results(embedding_results, llm_results):

    print("\n\n")
    print("=" * 80)
    print("FINAL LEVEL 1 - TASK 1 RESULTS")
    print("=" * 80)

    # --------------------------------------------------------
    # Embedding results
    # --------------------------------------------------------

    print("\n")
    print("EMBEDDING MODEL COMPARISON")
    print("-" * 80)

    print(
        f"{'MODEL':<50}"
        f"{'LOAD':>10}"
        f"{'EMBED':>10}"
        f"{'TOTAL':>10}"
    )

    print("-" * 80)

    for result in embedding_results:

        if result["status"] == "Success":

            print(
                f"{result['model']:<50}"
                f"{result['load_time']:>10.4f}"
                f"{result['embed_time']:>10.4f}"
                f"{result['total_time']:>10.4f}"
            )

        else:

            print(
                f"{result['model']:<50}"
                f"{'FAILED':>10}"
            )

    # --------------------------------------------------------
    # Reasoning results
    # --------------------------------------------------------

    print("\n")
    print("REASONING LLM COMPARISON")
    print("-" * 80)

    print(
        f"{'MODEL':<40}"
        f"{'LOAD':>10}"
        f"{'RESPONSE':>12}"
        f"{'TOTAL':>10}"
    )

    print("-" * 80)

    for result in llm_results:

        if result["status"] == "Success":

            print(
                f"{result['model']:<40}"
                f"{result['load_time']:>10.4f}"
                f"{result['response_time']:>12.4f}"
                f"{result['total_time']:>10.4f}"
            )

        else:

            print(
                f"{result['model']:<40}"
                f"{'FAILED':>10}"
            )

    # --------------------------------------------------------
    # Fastest embedding
    # --------------------------------------------------------

    successful_embeddings = [
        r for r in embedding_results
        if r["status"] == "Success"
    ]

    if successful_embeddings:

        fastest_embedding = min(
            successful_embeddings,
            key=lambda x: x["embed_time"]
        )

        print("\n")
        print("=" * 80)
        print("FASTEST EMBEDDING MODEL")
        print("=" * 80)

        print(f"Model : {fastest_embedding['model']}")
        print(
            f"Embedding response time : "
            f"{fastest_embedding['embed_time']:.4f} seconds"
        )

    # --------------------------------------------------------
    # Fastest reasoning LLM
    # --------------------------------------------------------

    successful_llms = [
        r for r in llm_results
        if r["status"] == "Success"
    ]

    if successful_llms:

        fastest_llm = min(
            successful_llms,
            key=lambda x: x["response_time"]
        )

        print("\n")
        print("=" * 80)
        print("FASTEST REASONING LLM")
        print("=" * 80)

        print(f"Model : {fastest_llm['model']}")
        print(
            f"Response time : "
            f"{fastest_llm['response_time']:.4f} seconds"
        )

    # --------------------------------------------------------
    # Recommended combination
    # --------------------------------------------------------

    if successful_embeddings and successful_llms:

        fastest_embedding = min(
            successful_embeddings,
            key=lambda x: x["embed_time"]
        )

        fastest_llm = min(
            successful_llms,
            key=lambda x: x["response_time"]
        )

        print("\n")
        print("=" * 80)
        print("RECOMMENDED FAST COMBINATION")
        print("=" * 80)

        print(
            f"Embedding Model : "
            f"{fastest_embedding['model']}"
        )

        print(
            f"Reasoning LLM   : "
            f"{fastest_llm['model']}"
        )

        print(
            f"\nEmbedding Time  : "
            f"{fastest_embedding['embed_time']:.4f} seconds"
        )

        print(
            f"LLM Response Time: "
            f"{fastest_llm['response_time']:.4f} seconds"
        )

        print("\nRecommendation:")
        print(
            "Use the fastest embedding model + "
            "fastest reasoning LLM for the low-latency RAG configuration."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    embedding_results = test_embeddings()

    llm_results = test_reasoning_models()

    print_final_results(
        embedding_results,
        llm_results
    )