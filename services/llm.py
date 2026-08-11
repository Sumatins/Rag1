from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME


class LLM:

    @staticmethod
    def load():
        return ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0
        )
