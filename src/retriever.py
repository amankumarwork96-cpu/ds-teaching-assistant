import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import chromadb
from sentence_transformers import SentenceTransformer
from config import VECTORSTORE_DIR, EMBEDDING_MODEL, collection_name, top_k

# LOAD MODEL AND CHROMADB ONCE, REUSE ACROSS QUERIES
_model = None
_collection = None

def _load_resources():
    """
    Loads the embedding model and ChromaDB collection into meomory.
    Uses global variabales so we don't reload on every query.
    """
    global _model, _collection

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBEDDING_MODEL)

    if _collection is None:
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        _collection = client.get_collection(collection_name)


def retrieve(query: str) -> list[dict]:
    """
    Takes a question string and returns the top_k most relevant chunks.
    Each result includes the text, source filename, and page number.
    """
    _load_resources()

    # embed the qery using the same model during ingestion
    query_vector = _model.encode([query]).tolist()

    # search ChromaDB for nearest chunks
    results = _collection.query(
        query_embeddings = query_vector,
        n_results = top_k,
        include = ["documents", "metadatas", "distances"]
    )

    # format results into clean list of dicts
    chunks = []
    for i in range(len(results["documents"][0])):
       chunks.append({
           "text" : results["documents"][0][i],
           "source" : results["metadatas"][0][i]["source"],
           "page" : results["metadatas"][0][i]["page"],
           "distance" : round(results["distances"][0][i], 4)
       })
    
    return chunks


# Quick test
if __name__ == "__main__":
    query = "What is the difference between supervised and unsupervised learning?"
    results = retrieve(query)

    print(f"\nQuery: {query}\n")
    for i, chunk in enumerate(results):
        print(f"--- Result{i+1} ---")
        print(f"Source : {chunk['source']}, Page {chunk['page']}")
        print(f"Distance: {chunk['distance']}")
        print(f"Text : {chunk['text'][:200]}...")
        print()
