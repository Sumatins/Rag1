from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME


def load_llm(model_name=None):
    model = model_name or MODEL_NAME

    return ChatGroq(
        model=model,
        api_key=GROQ_API_KEY,
        temperature=0
    )