import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DB_DIRECTORY
from embeddings.embedding import get_embedding
from services.embeddings import EmbeddingModel


class VectorDB:

    @staticmethod
    def create(documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding(),
            persist_directory=os.path.abspath(DB_DIRECTORY)
        )

        return vector_db

    @staticmethod
    def load():
        embeddings = EmbeddingModel.load_embeddings()

        db = Chroma(
            persist_directory=os.path.abspath(DB_DIRECTORY),
            embedding_function=embeddings
        )

        return db
