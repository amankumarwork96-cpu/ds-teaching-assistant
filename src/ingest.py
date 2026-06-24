import fitz  # PyMuPDF
import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import PDF_DIR, chunk_size, chunk_overlap, VECTORSTORE_DIR, EMBEDDING_MODEL, collection_name


def load_pdfs(PDF_DIR: str) -> list[dict]:
    """
    Reads every PDF in the folder and returns a list of pages.
    Each page is a dict with the text content and metadata.
    """
    all_pages = []

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        doc = fitz.open(pdf_path)

        print(f"Loading: {filename} ({len(doc)} pages)")

        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # skips empty pages
            if len(text.strip()) == 50:
                continue

            all_pages.append({
                "text": text,
                "source": filename,
                "page": page_num + 1
            })
        doc.close()
    print(f"\nTotal pages loaded: {len(all_pages)}")
    return all_pages



def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Splits each page's text into overlapping chunks,
    Preserves the source and page metadata for citations.
    """
    all_chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if len(chunk_text.strip()) < 50:
                start = end - chunk_overlap  # move back for overlap
                continue

            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"]
            })
            
            chunk_id += 1
            start = end - chunk_overlap  # move back for overlap

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def store_in_chromadb(chunks: list[dict]) -> None:
    print("\nLoading embedding_model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")

    collection = client.create_collection(collection_name)
    print(f"Created collection: {collection_name}")

    BATCH_SIZE = 100
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids = ids,
            documents = texts,
            embeddings = embeddings,
            metadatas = metadatas 
        )
        print(f"Stored chunks {i+1} to {min(i+BATCH_SIZE, total)} of {total}")
    
    print(f"\nDone. {total} chunks stored in ChromaDB.")

# RUN PIPELINE
if __name__ == "__main__":
    pages = load_pdfs(PDF_DIR)
    chunks = chunk_pages(pages)
    store_in_chromadb(chunks)