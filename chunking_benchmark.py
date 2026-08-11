import time

from loaders.pdf_loader import load_and_split_pdf


# ============================================================
# CHANGE THIS TO YOUR PDF
# ============================================================

PDF_PATH = "data/Sumati_N_Sannaragikoppa_Resume (1).pdf"


# ============================================================
# CONFIGURATIONS TO TEST
# ============================================================

CONFIGURATIONS = [

    {
        "strategy": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },

    {
        "strategy": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },

    {
        "strategy": "recursive",
        "chunk_size": 1500,
        "chunk_overlap": 200,
    },

    {
        "strategy": "recursive",
        "chunk_size": 2000,
        "chunk_overlap": 200,
    },

    {
        "strategy": "character",
        "chunk_size": 1000,
        "chunk_overlap": 100,
    },
]


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark():

    print("\n")
    print("=" * 90)
    print("LEVEL 1 - TASK 2")
    print("CHUNKING STRATEGY BENCHMARK")
    print("=" * 90)


    results = []


    for config in CONFIGURATIONS:

        print("\n" + "-" * 90)

        print(
            f"Strategy    : {config['strategy']}"
        )

        print(
            f"Chunk size  : {config['chunk_size']}"
        )

        print(
            f"Chunk overlap: {config['chunk_overlap']}"
        )


        # ----------------------------------------------------
        # Start timer
        # ----------------------------------------------------

        start_time = time.perf_counter()


        try:

            chunks = load_and_split_pdf(
                PDF_PATH,
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                strategy=config["strategy"],
            )


            elapsed = time.perf_counter() - start_time


            # ------------------------------------------------
            # Calculate statistics
            # ------------------------------------------------

            number_of_chunks = len(chunks)

            if number_of_chunks > 0:

                total_characters = sum(
                    len(chunk.page_content)
                    for chunk in chunks
                )

                average_chunk_size = (
                    total_characters /
                    number_of_chunks
                )

            else:

                total_characters = 0

                average_chunk_size = 0


            print(
                f"Number of chunks : {number_of_chunks}"
            )

            print(
                f"Average chunk size: "
                f"{average_chunk_size:.2f} characters"
            )

            print(
                f"Processing time  : "
                f"{elapsed:.4f} seconds"
            )


            results.append(
                {
                    "strategy": config["strategy"],
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "chunks": number_of_chunks,
                    "average_size": average_chunk_size,
                    "time": elapsed,
                }
            )


        except Exception as e:

            print(
                f"ERROR: {e}"
            )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n\n")

    print("=" * 100)
    print("FINAL CHUNKING COMPARISON")
    print("=" * 100)


    print(
        f"{'Strategy':<15}"
        f"{'Size':>10}"
        f"{'Overlap':>12}"
        f"{'Chunks':>12}"
        f"{'Avg Size':>15}"
        f"{'Time':>15}"
    )


    print("-" * 100)


    for result in results:

        print(
            f"{result['strategy']:<15}"
            f"{result['chunk_size']:>10}"
            f"{result['chunk_overlap']:>12}"
            f"{result['chunks']:>12}"
            f"{result['average_size']:>15.2f}"
            f"{result['time']:>15.4f}"
        )


    print("\n")
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)


    if results:

        fastest = min(
            results,
            key=lambda x: x["time"]
        )


        print(
            f"Fastest configuration:"
        )

        print(
            f"Strategy     : "
            f"{fastest['strategy']}"
        )

        print(
            f"Chunk size   : "
            f"{fastest['chunk_size']}"
        )

        print(
            f"Chunk overlap: "
            f"{fastest['chunk_overlap']}"
        )

        print(
            f"Chunks       : "
            f"{fastest['chunks']}"
        )

        print(
            f"Processing time: "
            f"{fastest['time']:.4f} seconds"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_benchmark()
