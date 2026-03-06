# RAG POC — Quick Start Guide

A production-like RAG pipeline using **FastAPI**, **Inngest**, **LlamaIndex**, **Qdrant**, **OpenAI GPT-4o-mini**, and **Streamlit**.

## Prerequisites
| Tool | Purpose |
|------|---------|
| Python 3.10+ | Runtime |
| Docker Desktop | Qdrant vector DB |
| Node.js (v18+) | Inngest dev server |
| Groq API Key | LLM inference (llama-3.3-70b-versatile) |

---

## Setup

### 1. Environment
```bash
cp .env.template .env
# Edit .env and paste your OPENAI_API_KEY
```

### 2. Python virtual environment
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

**Terminal 1 — FastAPI backend**
```bash
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Inngest dev server**
```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
# Dashboard: http://localhost:8288
```

**Terminal 3 — Streamlit UI**
```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8501
# UI: http://localhost:8501
```

---

## Usage

1. Open `http://localhost:8501` in your browser.
2. Upload a PDF → click **Start Ingestion** and watch the Inngest dashboard trace the steps.
3. Once ingestion is complete, type a question → click **Get Answer**.
4. The answer (grounded in the PDF context) will appear within seconds.

---

## Architecture

```
Streamlit UI
    │
    │  POST  /api/inngest  (send events)
    ▼
FastAPI + Inngest Server
    ├── rag/ingest_pdf  ─────► LlamaIndex (chunk) → OpenAI (embed) → Qdrant (store)
    └── rag/query_pdf_ai ────► OpenAI (embed query) → Qdrant (search) → GPT-4o-mini (answer)
```

## Production Toggle
Set `INNGEST_PRODUCTION=true` in `.env` and configure `INNGEST_EVENT_KEY` to switch to Inngest's managed cloud.

---

## Project Structure
```
Experiment1/
├── main.py            # FastAPI app + Inngest workflows
├── customtypes.py     # Pydantic type models
├── vector_db.py       # Qdrant client (QuadrantStorage)
├── data_loader.py     # PDF chunking + OpenAI embedding
├── streamlit_app.py   # Streamlit frontend
├── docker-compose.yml # Qdrant service
├── requirements.txt   # Python dependencies
└── .env               # API keys (not committed)
```
