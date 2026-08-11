from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from prompts.prompt import RAG_PROMPT
from services.llm import LLM


class RAGService:

    @staticmethod
    def create_chain(vector_db):
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        document_chain = create_stuff_documents_chain(
            llm=LLM.load(),
            prompt=RAG_PROMPT
        )

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        return retrieval_chain
