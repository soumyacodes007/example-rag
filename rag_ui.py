import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from loader import load_pdf
from splitter import split_docs
from embeddings import create_vectorstore
from llm import get_llm

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def build_rag(llm, retriever):
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the provided context to answer the question."),
        ("user", "Question: {question}\n\nContext:\n{context}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


@st.cache_resource
def init_chain():
    PDF_URL = os.getenv("DATA_URL")
    if not PDF_URL:
        st.error("DATA_URL missing in .env file.")
        st.stop()

    data = load_pdf(PDF_URL)
    splits = split_docs(data)
    vectorstore = create_vectorstore(splits)
    retriever = vectorstore.as_retriever()
    llm = get_llm(GROQ_API_KEY)
    return build_rag(llm, retriever)
