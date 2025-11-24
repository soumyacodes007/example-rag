# embeddings.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vectorstore(splits):
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    return vectorstore
