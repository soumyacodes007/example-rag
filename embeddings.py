# embeddings.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vectorstore(splits):
    # 1. Remove empty chunks
    splits = [d for d in splits if d.page_content and d.page_content.strip()]
    if not splits:
        raise ValueError("No non-empty document chunks found. Check your PDF/text loader.")

    # 2. Embedding model
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 3. Sanity check
    test_vector = embeddings.embed_query("test")
    if not test_vector:
        raise ValueError("Embedding model returned an empty vector. Check model name or installation.")

    # 4. Vectorstore
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
