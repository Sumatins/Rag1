import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_multiple_pdfs(
    pdf_paths,
    chunk_size=1000,
    chunk_overlap=200,
):
    """
    Load multiple PDF files and split them into chunks.

    Parameters:
        pdf_paths: List of PDF file paths
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """

    all_documents = []

    # --------------------------------------------------------
    # Load every PDF
    # --------------------------------------------------------

    for pdf_path in pdf_paths:

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        # Add PDF name to metadata
        for document in documents:

            document.metadata["source"] = os.path.basename(
                pdf_path
            )

        all_documents.extend(documents)


    # --------------------------------------------------------
    # Chunk all documents
    # --------------------------------------------------------

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


    chunks = splitter.split_documents(
        all_documents
    )


    # --------------------------------------------------------
    # Make sure source metadata is preserved
    # --------------------------------------------------------

    for chunk in chunks:

        if "source" not in chunk.metadata:

            chunk.metadata["source"] = "Unknown"


    return chunks
