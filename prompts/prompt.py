from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful AI Assistant.

Answer ONLY using the context below.

If the answer is not found, reply:

"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{input}

Answer:
"""
)