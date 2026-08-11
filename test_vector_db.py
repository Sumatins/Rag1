from loaders.pdf_loader import PDFLoader
from services.vector_db import VectorDB

docs = PDFLoader.load_pdf(r"data\Sumati_N_Sannaragikoppa_Resume (1).pdf")

db = VectorDB.create(docs)

print("Database Created Successfully")