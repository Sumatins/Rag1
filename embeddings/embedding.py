from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


def get_embedding(model_name=None):
    model = model_name or EMBEDDING_MODEL

    return HuggingFaceEmbeddings(
        model_name=model
    )
