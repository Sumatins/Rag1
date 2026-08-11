from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.chains import create_retrieval_chain

from prompts.prompt import RAG_PROMPT

from llm.llm import load_llm


def get_rag_chain(vector_db):

    retriever = vector_db.as_retriever(
        search_kwargs={"k":3}
    )

    document_chain = create_stuff_documents_chain(
        llm=load_llm(),
        prompt=RAG_PROMPT
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain