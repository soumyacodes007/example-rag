import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_url: str):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        os.unlink(tmp_path)
        return pages
    except Exception as e:
        print("Error loading PDF:", e)
        return []
