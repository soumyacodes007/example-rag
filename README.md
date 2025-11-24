# RAG Chat Assistant

A simple RAG (Retrieval-Augmented Generation) chatbot for educational demonstrations.

## Features
- Chat interface for asking questions about course content
- Uses Groq LLM for fast responses
- Vector search with ChromaDB and HuggingFace embeddings
- Simple and clean UI built with Streamlit

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
```

Edit `.env` and set your `GROQ_API_KEY`:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment

### Environment Variables

The app uses the following environment variables (see `.env.example`):

- `GROQ_API_KEY` (required) - Your Groq API key
- `GROQ_MODEL` (optional) - Model to use (default: openai/gpt-oss-20b)
- `EMBEDDING_MODEL` (optional) - Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
- `DATA_URL` (optional) - URL to load data from
- `TEMPERATURE` (optional) - LLM temperature (default: 0)

### Deploy to Streamlit Cloud (Free)

1. Push this code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file: `streamlit_app.py`
7. Add your secrets in Advanced settings > Secrets:
```toml
GROQ_API_KEY = "your_api_key_here"
```
8. Click "Deploy"

### Deploy to Other Platforms

For platforms like Heroku, Railway, or Render, set environment variables in their dashboard:
```bash
GROQ_API_KEY=your_api_key_here
```

## Project Structure
```
├── streamlit_app.py    # Main chat UI
├── app.py             # CLI version
├── rag_ui.py          # RAG chain builder
├── llm.py             # LLM configuration
├── embeddings.py      # Vector store setup
├── loader.py          # Document loader
├── splitter.py        # Text splitter
└── requirements.txt   # Dependencies
```

## For Students

This is a minimal RAG implementation showing:
1. Document loading from web
2. Text splitting into chunks
3. Creating embeddings and vector store
4. Building a retrieval chain
5. Chat interface for Q&A

Perfect for learning the basics of RAG systems!
