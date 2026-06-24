import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Paths
PDF_DIR = "data/pdfs"
VECTORSTORE_DIR = "data/vectorstore/chroma_db"

# Chunking
chunk_size = int(os.getenv("CHUNK_SIZE")) if os.getenv("CHUNK_SIZE") else 600
chunk_overlap = int(os.getenv("CHUNK_OVERLAP")) if os.getenv("CHUNK_OVERLAP") else 100

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") if os.getenv("EMBEDDING_MODEL") else "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB
collection_name = os.getenv("COLLECTION_NAME") if os.getenv("COLLECTION_NAME") else "ds_textbooks"

#GEMINI
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model   = "llama-3.1-8b-instant"  # or the latest version

# RETRIEVAL
top_k = 5  # how many chunks to retrieve from the vector store for each query