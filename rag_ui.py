# rag_ui.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag(llm, retriever):
    template = """Answer based only on this context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def start_ui(rag_chain):
    print("RAG system ready. Type your question (or 'exit')\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        print("\nAI:", rag_chain.invoke(q), "\n")
