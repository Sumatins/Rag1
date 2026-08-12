from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME


def load_llm(model_name: str | None = None):

    return ChatGroq(
        model=model_name or MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
