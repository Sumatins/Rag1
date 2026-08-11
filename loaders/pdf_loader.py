from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:

    @staticmethod
    def load_pdf(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
