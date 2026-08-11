from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Current/default reasoning model
MODEL_NAME = "llama-3.3-70b-versatile"

# Current/default embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DB_DIRECTORY = "chroma_db"

# Models to compare for Level 1 - Task 1
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

REASONING_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]