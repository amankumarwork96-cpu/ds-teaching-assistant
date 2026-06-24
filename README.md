# 📚 DS Teaching Assistant (RAG-Based)

An AI-powered teaching assistant that answers Data Science questions using your own textbooks — built with RAG (Retrieval-Augmented Generation).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3-orange)

---

## 🧠 How It Works
PDFs → Text Extraction → Chunking → Embeddings → ChromaDB
Query → Embed Query → Similarity Search → Top-K Chunks → LLM → Answer
1. PDFs are loaded, split into overlapping chunks, and embedded using `sentence-transformers`
2. Embeddings are stored in ChromaDB (local vector database)
3. When a student asks a question, it's embedded and matched against stored chunks
4. Top 5 most relevant chunks are retrieved and passed to Groq (Llama 3.1)
5. The LLM generates a grounded answer with source citations

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | PyMuPDF (fitz) |
| Chunking | Custom sliding window |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (persistent, local) |
| LLM | Llama 3.1 8B via Groq API |
| UI | Streamlit |

---

## 📂 Project Structure
ds-teaching-assistant/

│

├── data/

│   └── pdfs/              ← Add your PDFs here (not included)

│

├── src/

│   ├── config.py          ← All settings in one place

│   ├── ingest.py          ← PDF loading, chunking, ChromaDB storage

│   ├── retriever.py       ← Semantic search

│   ├── chain.py           ← Prompt builder + LLM call

│   └── utils.py           ← Helper functions

│

├── app.py                 ← Streamlit UI

├── requirements.txt

├── .env                   ← API keys (not pushed to GitHub)

└── README.md

---

## 🚀 Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/amankumarwork96-cpu/ds-teaching-assistant.git
cd ds-teaching-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the project root:
GROQ_API_KEY=your_groq_api_key_here

Get a free key at [console.groq.com](https://console.groq.com)

**4. Add your PDFs**

Place your Data Science PDFs inside `data/pdfs/`

**5. Run ingestion (one time only)**
```bash
python src/ingest.py
```

**6. Launch the app**
```bash
streamlit run app.py
```

---

## 📖 Adding Your Own PDFs

This project does not include PDFs due to copyright restrictions.
Add any Data Science PDFs to `data/pdfs/` and run `python src/ingest.py` to rebuild the knowledge base.

Recommended free textbooks:
- [Introduction to Data Science - Irizarry](https://rafalab.dfci.harvard.edu/dsbook/)
- [Python Data Science Handbook - VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Think Stats - Allen Downey](https://greenteapress.com/wp/think-stats-2e/)

---

## 💡 Key Concepts Demonstrated

- **RAG pipeline** — end to end from PDF to grounded answer
- **Vector similarity search** — semantic retrieval using cosine distance
- **Prompt engineering** — structured context injection with citations
- **Modular code** — each step is a separate testable module
- **Local inference** — ChromaDB runs entirely on disk, no cloud dependency

---

## 👤 Author

**Aman Kumar**
BBA Graduate | Aspiring Data Analyst
[GitHub](https://github.com/amankumarwork96-cpu) | [LinkedIn](www.linkedin.com/in/aman-kumar-7656a0300)