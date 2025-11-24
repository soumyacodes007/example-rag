# llm.py
import os
from langchain_groq import ChatGroq

def get_llm(api_key):
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    temperature = float(os.getenv("TEMPERATURE", "0"))
    
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key
    )
