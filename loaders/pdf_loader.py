from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


def load_and_split_pdf(
    file_path,
    chunk_size=1000,
    chunk_overlap=200,
    strategy="recursive",
):
    """
    Load a PDF and split it according to the selected
    chunking strategy, chunk size, and overlap.
    """

    # Load PDF
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    # Select chunking strategy
    if strategy == "recursive":

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    elif strategy == "character":

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    else:

        raise ValueError(
            f"Unknown chunking strategy: {strategy}"
        )

    # Split documents
    chunks = splitter.split_documents(documents)

    return chunks
