from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


def get_embedding(model_name: str | None = None):

    return HuggingFaceEmbeddings(
        model_name=model_name or EMBEDDING_MODEL
    )

