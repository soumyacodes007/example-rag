import streamlit as st
import os
from dotenv import load_dotenv

from loader import load_data
from splitter import split_docs
from embeddings import create_vectorstore
from llm import get_llm
from rag_ui import build_rag

# Load environment variables
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in .env file.")
    st.stop()



@st.cache_resource
def init_chain():
    """Initialize and cache the RAG chain (vectorstore + LLM + prompt)."""
    data = load_data()
    splits = split_docs(data)
    vectorstore = create_vectorstore(splits)
    retriever = vectorstore.as_retriever()
    llm = get_llm(GROQ_API_KEY)
    rag_chain = build_rag(llm, retriever)
    return rag_chain


def main():
    st.set_page_config(page_title="RAG Chat", page_icon="", layout="centered")
    
    st.title(" RAG Chat Assistant")
    st.caption("Ask questions about the course content")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize RAG chain
    if "rag_chain" not in st.session_state:
        with st.spinner("Loading RAG system..."):
            st.session_state.rag_chain = init_chain()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the course..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
