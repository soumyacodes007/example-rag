# splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    splits = splitter.split_documents(docs)

    # Remove any empty text
    splits = [d for d in splits if d.page_content and d.page_content.strip()]

    if not splits:
        raise ValueError("Text splitter produced no valid chunks.")

    return splits
