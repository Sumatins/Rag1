from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass

import numpy as np
import pymupdf
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

VISION_MODEL = "qwen/qwen3.6-27b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keep these small to stay safely below the Groq TPM limit.
PAGE_DPI = 130
TOP_K = 2

# Maximum output tokens for each vision request.
PAGE_ANALYSIS_TOKENS = 500
ANSWER_TOKENS = 800

# Retry settings for rate limiting.
MAX_RETRIES = 4
RETRY_SECONDS = 8


@dataclass
class PageData:
    page_number: int
    image_bytes: bytes
    description: str


class ImageRAG:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.pages = []
        self.embeddings = None

    # ---------------------------------------------------------
    # Convert image to base64 data URL
    # ---------------------------------------------------------

    def image_to_data_url(
        self,
        image_bytes: bytes,
    ) -> str:

        encoded = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        return (
            "data:image/png;base64,"
            + encoded
        )

    # ---------------------------------------------------------
    # Groq vision call with automatic retry
    # ---------------------------------------------------------

    def ask_vision(
        self,
        prompt: str,
        images: list[bytes],
        max_tokens: int,
    ) -> str:

        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]

        for image in images:

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.image_to_data_url(
                            image
                        )
                    },
                }
            )

        for attempt in range(MAX_RETRIES):

            try:

                response = (
                    self.client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a precise "
                                    "PDF visual analysis "
                                    "assistant. Only use "
                                    "information visible "
                                    "in the supplied image."
                                ),
                            },
                            {
                                "role": "user",
                                "content": content,
                            },
                        ],
                        temperature=0,
                        max_completion_tokens=max_tokens,
                    )
                )

                return (
                    response
                    .choices[0]
                    .message
                    .content
                    .strip()
                )

            except Exception as e:

                error_text = str(e)

                if (
                    "429" in error_text
                    or "rate_limit" in error_text
                    or "Rate limit" in error_text
                ):

                    if attempt < MAX_RETRIES - 1:

                        wait_time = (
                            RETRY_SECONDS
                            * (attempt + 1)
                        )

                        time.sleep(
                            wait_time
                        )

                        continue

                raise

        raise RuntimeError(
            "Vision request failed after retries."
        )

    # ---------------------------------------------------------
    # Render PDF pages as images
    # ---------------------------------------------------------

    def extract_pages(
        self,
        pdf_bytes: bytes,
    ):

        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

        pages = []

        try:

            for index, page in enumerate(
                document
            ):

                pixmap = page.get_pixmap(
                    dpi=PAGE_DPI,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )

                image_bytes = pixmap.tobytes(
                    "png"
                )

                pages.append(
                    (
                        index + 1,
                        image_bytes,
                    )
                )

        finally:

            document.close()

        return pages

    # ---------------------------------------------------------
    # Analyze ONE page
    # ---------------------------------------------------------

    def describe_page(
        self,
        page_number: int,
        image_bytes: bytes,
    ):

        prompt = f"""
Analyze PDF page {page_number}.

Return concise factual notes only.

Identify:

- title
- main image/figure
- table
- chart/graph
- diagram
- important labels
- important numbers
- units
- relationships

Do not guess.

Keep the answer under 300 words.
"""

        return self.ask_vision(
            prompt=prompt,
            images=[image_bytes],
            max_tokens=PAGE_ANALYSIS_TOKENS,
        )

    # ---------------------------------------------------------
    # Build visual index
    # ---------------------------------------------------------

    def build_index(
        self,
        pdf_bytes: bytes,
        max_pages: int = 10,
    ):

        extracted_pages = self.extract_pages(
            pdf_bytes
        )

        # Limit pages for a demo / free API tier.
        extracted_pages = extracted_pages[
            :max_pages
        ]

        self.pages = []

        for page_number, image_bytes in (
            extracted_pages
        ):

            description = self.describe_page(
                page_number,
                image_bytes,
            )

            self.pages.append(
                PageData(
                    page_number=page_number,
                    image_bytes=image_bytes,
                    description=description,
                )
            )

            # Small pause between vision requests.
            time.sleep(1)

        texts = [
            (
                f"Page {page.page_number}\n"
                f"{page.description}"
            )
            for page in self.pages
        ]

        self.embeddings = (
            self.embedder.encode(
                texts,
                normalize_embeddings=True,
            )
        )

    # ---------------------------------------------------------
    # Retrieve relevant pages
    # ---------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
    ):

        if (
            not self.pages
            or self.embeddings is None
        ):

            raise ValueError(
                "The PDF has not been indexed."
            )

        query_embedding = (
            self.embedder.encode(
                [question],
                normalize_embeddings=True,
            )
        )

        scores = cosine_similarity(
            query_embedding,
            self.embeddings,
        )[0]

        indices = np.argsort(
            scores
        )[::-1][:top_k]

        return [
            self.pages[index]
            for index in indices
        ]

    # ---------------------------------------------------------
    # Final answer
    # ---------------------------------------------------------

    def answer(
        self,
        question: str,
        retrieved_pages: list[PageData],
    ):

        source_information = "\n\n".join(
            [
                (
                    f"PAGE {page.page_number}\n"
                    f"{page.description}"
                )
                for page in retrieved_pages
            ]
        )

        prompt = f"""
Answer the question using only the
supplied PDF page images.

QUESTION:
{question}

PAGE NOTES:
{source_information}

Rules:

- Inspect the actual images.
- Give the exact visible value when possible.
- For tables preserve row/column meaning.
- For charts identify axes and values.
- For diagrams explain connections.
- Do not invent information.
- If the answer is not visible, say so.

Keep the answer concise.

End with:
Sources: Page X, Page Y
"""

        images = [
            page.image_bytes
            for page in retrieved_pages
        ]

        return self.ask_vision(
            prompt=prompt,
            images=images,
            max_tokens=ANSWER_TOKENS,
        )


def create_image_rag():

    return ImageRAG()