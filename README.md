# RAG POC — Quick Start Guide

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline using **FastAPI**, **Inngest**, **LlamaIndex**, **Qdrant**, **Groq (Llama 3.3)**, and **Streamlit**.

## Tech Stack
| Component | Technology |
|-----------|-----------|
| API Server | FastAPI |
| Orchestration| Inngest (Workflows, Observability, Proxy Polling) |
| Vector DB | Qdrant (via Docker) |
| Chunking | LlamaIndex SentenceSplitter |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers - Local) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Frontend | Streamlit (Dark Glassmorphism UI) |

---

## Prerequisites
1. **Python 3.10+**
2. **Docker Desktop** (for Qdrant)
3. **Node.js** (for Inngest Dev Server)
4. **Groq API Key** ([Get it here](https://console.groq.com))

---

## Setup

### 1. Environment
Copy the template and add your credentials:
```bash
cp .env.template .env
# Edit .env and paste your GROQ_API_KEY
```

### 2. Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Qdrant (Docker)
```bash
docker-compose up -d
# Dashboard: http://localhost:6333/dashboard
```

---

## Running Locally

Open **3 terminals** in the project folder with the venv activated:

**Terminal 1 — FastAPI Backend**
```bash
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Inngest Dev Server**
```bash
# Point to your local FastAPI endpoint
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
# Dashboard: http://localhost:8288
```

**Terminal 3 — Streamlit UI**
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
# UI: http://localhost:8501
```

---

## Observability & Inspection

### Qdrant Dashboard
Visualize your stored vectors and metadata at **[http://localhost:6333/dashboard](http://localhost:6333/dashboard)**.

### Database Inspection Tool
Run the included utility to see exactly what's inside your local Qdrant:
```bash
source .venv/bin/activate
python inspect_db.py
```

---

## Architecture Flow

1. **Ingestion**: `Streamlit` → `FastAPI` → `Inngest Event` → `LlamaIndex` (Chunk) → `Local Model` (Embed) → `Qdrant` (Store).
2. **Querying**: `Streamlit` → `FastAPI` → `Inngest Event` → `Search Qdrant` → `Context` + `Question` → `Groq LLM` → `Answer`.

---

## Project Structure
```
Experiment1/
├── main.py            # FastAPI app + Inngest workflows
├── customtypes.py     # Pydantic type models
├── vector_db.py       # Qdrant client (QuadrantStorage)
├── data_loader.py     # PDF chunking + Local embeddings
├── streamlit_app.py   # Streamlit frontend
├── inspect_db.py      # CLI utility to see database points
├── docker-compose.yml # Qdrant service configuration
├── requirements.txt   # Python dependencies
└── .env               # Secrets (not committed)
```
