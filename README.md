# QueryChat: Conversational RAG with LangChain & LangGraph

QueryChat is a Python-based conversational Retrieval-Augmented Generation (RAG) system that leverages LangChain, LangGraph, and Sentence Transformers to provide intelligent, context-aware answers from your document collection. It supports multi-turn chat, query rewriting, decomposition, and document retrieval with vector search.

## Project Structure
```
main/
  assistant.py         # LLM interface and structured output
  embedding.py         # Embedding model wrapper (Sentence Transformers)
  generator.py         # Answer generation node (LangGraph)
  main.py              # CLI entry point for chat
  prompts.py           # Prompt templates
  rag_model.py         # Main RAG graph (rewriter, retriever, generator)
  rag_pipeline.py      # Document ingestion and vectorstore feeding
  retriever.py         # Query rephrasing, decomposition, retrieval
  test_gen.py          # Test data for generator
  vectorstore.py       # ChromaDB vector store wrapper
models/
  all-MiniLM-L6-v2.pt/ # Sentence Transformer model files
vector_db/             # ChromaDB persistent storage
requirements.txt       # Python dependencies
.env                   # Environment variables
```

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Sentence Transformer model** (if not present in `models/`)
4. **Configure environment**: Edit `.env` for API keys, chunk size, etc.
5. **Ingest documents**:
   - Place your PDFs/Word docs in the `documents/` folder.
   - Run the ingestion pipeline (see below).

## Usage
### Ingest Documents
```bash
python -m main.rag_pipeline
```

### Start Chat
```bash
python -m main.main
```

### Example Conversation
```
Welcome to the RAG Model Demo!, Type exit to quit the demo.
AI: I am a RAG model that can rephrase, decompose and retrieve documents based on your query. How can I assist you today?
Human: What are agentic design patterns?
AI: [Comprehensive answer based on your documents]
```

## Environment Variables (`.env`)
- `MODEL_NAME`, `BASE_URL`, `API_KEY`: For LLM API (if used)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Document chunking parameters
- `PERSIST_DIRECTORY`: ChromaDB storage directory

## Extending
- Add new prompt templates in `main/prompts.py`
- Add new document loaders or chunkers in `main/rag_pipeline.py`
- Add new graph nodes in `main/rag_model.py`

## License
MIT License

## Acknowledgements
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

