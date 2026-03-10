"""
main.py - FastAPI application with full Inngest observability.

Every API endpoint is instrumented:
  - Middleware fires `api/request.received` and `api/request.completed` for every request.
  - Errors (4xx/5xx) fire `api/request.errored`.
  - Dedicated REST endpoints trigger Inngest RAG workflows.
  - An Inngest monitor function logs all API activity to the dashboard.

Stack: FastAPI · Inngest · Qdrant · Groq (llama-3.3-70b-versatile) · sentence-transformers
"""

import os
import time
import uuid
from typing import Any

import inngest
import inngest.fast_api
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from customtypes import RagUpsertResult, RagSearchResult
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QuadrantStorage

load_dotenv()

# ─── Clients ──────────────────────────────────────────────────────────────────
# Lazy-loaded so the module can be imported without the key set (e.g. for tests).
_groq_client: Groq | None = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client

inngest_client = inngest.Inngest(
    app_id="rag-poc",
    is_production=os.getenv("INNGEST_PRODUCTION", "false").lower() == "true",
)

# ─── Qdrant Storage (lazy-loaded) ──────────────────────────────────────────────────────────
_db: QuadrantStorage | None = None

def get_db() -> QuadrantStorage:
    global _db
    if _db is None:
        _db = QuadrantStorage()
    return _db

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG POC API",
    description="Production-grade RAG pipeline with full Inngest observability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBSERVABILITY MIDDLEWARE
# Fires Inngest events for RAG requests: received + completed/errored
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MONITORED_PATHS = {"/api/ingest", "/api/query"}


@app.middleware("http")
async def inngest_observability_middleware(request: Request, call_next):
    """Emit Inngest events for inbound API requests."""
    path = request.url.path

    # Only monitor actual RAG endpoints to prevent public internet bot noise
    if path not in MONITORED_PATHS:
        return await call_next(request)

    async def safe_send_event(event: inngest.Event):
        try:
            await inngest_client.send(event)
        except Exception as e:
            # Observability should never crash the main application
            print(f"[telemetry-warning] Failed to send Inngest event {event.name}: {e}")

    request_id = str(uuid.uuid4())
    start_ts = time.time()

    # Fire: request received
    await safe_send_event(
        inngest.Event(
            name="api/request.received",
            data={
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "query": str(request.query_params),
                "client_host": request.client.host if request.client else "unknown",
                "timestamp": start_ts,
            },
        )
    )

    # Process the request
    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start_ts) * 1000, 2)

        # Fire: request completed
        await safe_send_event(
            inngest.Event(
                name="api/request.completed",
                data={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "success": response.status_code < 400,
                },
            )
        )

        # Fire: error event for 4xx / 5xx responses
        if response.status_code >= 400:
            await safe_send_event(
                inngest.Event(
                    name="api/request.errored",
                    data={
                        "request_id": request_id,
                        "method": request.method,
                        "path": path,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                    },
                )
            )

        return response

    except Exception as exc:
        duration_ms = round((time.time() - start_ts) * 1000, 2)
        await safe_send_event(
            inngest.Event(
                name="api/request.errored",
                data={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": 500,
                    "error": str(exc),
                    "duration_ms": duration_ms,
                },
            )
        )
        raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REST ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IngestRequest(BaseModel):
    filename: str
    file_b64: str  # Base64 encoded PDF file content

class QueryRequest(BaseModel):
    question: str


@app.get("/api/health", tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.
    Returns server status and Qdrant connectivity.
    """
    try:
        collections = get_db().client.get_collections()
        qdrant_status = "ok"
        collection_names = [c.name for c in collections.collections]
    except Exception as e:
        qdrant_status = f"error: {e}"
        collection_names = []

    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "collections": collection_names,
        "app": "rag-poc",
        "version": "1.0.0",
    }


@app.get("/api/collections", tags=["Monitoring"])
async def list_collections():
    """
    List all Qdrant collections with point counts.
    Useful for monitoring ingested document volumes.
    """
    try:
        collections = get_db().client.get_collections().collections
        details = []
        for col in collections:
            info = get_db().client.get_collection(col.name)
            # qdrant-client v1.17+ uses points_count, not vectors_count
            count = getattr(info, "points_count", getattr(info, "vectors_count", 0))
            details.append({
                "name": col.name,
                "vectors_count": count,
                "status": str(info.status),
            })
        return {"collections": details}
    except Exception as e:
        print(f"[api] Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/inngest/runs/{event_id}", tags=["Monitoring"])
async def get_inngest_runs(event_id: str):
    """
    Proxy endpoint to poll for Inngest run status from a local dev server.
    This allows Streamlit Cloud (via ngrok) to check run status without 
    needing direct access to the local Inngest port.
    """
    import requests
    dev_server_url = os.getenv("INNGEST_DEV_URL", "http://127.0.0.1:8288")
    try:
        r = requests.get(f"{dev_server_url}/v1/events/{event_id}/runs", timeout=5)
        return r.json()
    except Exception as e:
        return {"data": [], "error": str(e)}


@app.post("/api/ingest", tags=["RAG"])
async def ingest_pdf(body: IngestRequest):
    """
    Trigger asynchronous PDF ingestion via Inngest.
    Returns the Inngest event ID to track progress on the dashboard.
    """
    event_ids = await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "filename": body.filename,
                "file_b64": body.file_b64,
            },
        )
    )
    return {
        "status": "queued",
        "event_id": event_ids[0] if event_ids else None,
        "message": f"Ingestion started for '{body.filename}'. Track on Inngest dashboard.",
        "dashboard": "http://localhost:8288",
    }


@app.post("/api/query", tags=["RAG"])
async def query_pdf(body: QueryRequest):
    """
    Trigger asynchronous RAG query via Inngest.
    Returns the Inngest event ID to poll for the answer.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    event_ids = await inngest_client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": body.question.strip()},
        )
    )
    return {
        "status": "queued",
        "event_id": event_ids[0] if event_ids else None,
        "message": "Query submitted. Poll the Inngest dashboard for the answer.",
        "dashboard": "http://localhost:8288",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INNGEST FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── Monitor: Log all API activity ────────────────────────────────────────────
@inngest_client.create_function(
    fn_id="api-monitor",
    trigger=inngest.TriggerEvent(event="api/request.completed"),
)
async def api_monitor(ctx: inngest.Context, **kwargs) -> dict:
    """
    Receives every completed API request event.
    In production you'd write to a metrics store (Datadog, Prometheus, etc.).
    Here it provides a searchable trace in the Inngest dashboard.
    """
    event = getattr(ctx, "event", kwargs.get("event"))
    data = event.data
    log_entry = {
        "request_id": data.get("request_id"),
        "method": data.get("method"),
        "path": data.get("path"),
        "status_code": data.get("status_code"),
        "duration_ms": data.get("duration_ms"),
        "success": data.get("success", True),
    }
    print(f"[api-monitor] {log_entry}")
    return log_entry


# ─── Monitor: Alert on errors ─────────────────────────────────────────────────
@inngest_client.create_function(
    fn_id="api-error-alert",
    trigger=inngest.TriggerEvent(event="api/request.errored"),
    # Retry up to 3 times with exponential backoff
    retries=3,
)
async def api_error_alert(ctx: inngest.Context, **kwargs) -> dict:
    """
    Triggered on any 4xx/5xx response.
    In production: send a Slack/PagerDuty alert here.
    """
    step = getattr(ctx, "step", kwargs.get("step"))
    event = getattr(ctx, "event", kwargs.get("event"))
    data = event.data

    async def log_error():
        error_summary = {
            "request_id": data.get("request_id"),
            "method": data.get("method"),
            "path": data.get("path"),
            "status_code": data.get("status_code"),
            "error": data.get("error", "HTTP error"),
            "duration_ms": data.get("duration_ms"),
            "severity": "critical" if data.get("status_code", 0) >= 500 else "warning",
        }
        print(f"[api-error-alert] 🚨 ERROR: {error_summary}")
        # TODO (production): POST to Slack webhook / PagerDuty / etc.
        return error_summary

    return await step.run("log-and-alert", log_error)


# ─── Workflow 1: Ingest PDF ────────────────────────────────────────────────────
@inngest_client.create_function(
    fn_id="rag-ingest-pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context, **kwargs) -> dict[str, Any]:
    step = getattr(ctx, "step", kwargs.get("step"))
    event = getattr(ctx, "event", kwargs.get("event"))
    filename: str = event.data["filename"]
    file_b64: str = event.data["file_b64"]
    print(f"[inngest] Starting ingestion for: {filename}")

    # Step 1: Write base64 to a local temporary file for LlamaIndex to process
    async def write_temp_file() -> str:
        import base64
        import tempfile
        import os
        
        file_bytes = base64.b64decode(file_b64)
        fd, path = tempfile.mkstemp(suffix=".pdf")
        with os.fdopen(fd, 'wb') as f:
            f.write(file_bytes)
        return path

    local_path = await step.run("write-temp-file", write_temp_file)

    # Step 2: Load and chunk the PDF
    async def chunk_pdf() -> list[dict]:
        # Pass the original filename to preserve it in the metadata
        raw_chunks = load_and_chunk_pdf(local_path, original_filename=filename)
        print(f"[inngest] Chunked '{filename}' into {len(raw_chunks)} pieces.")
        # Inngest step outputs MUST be JSON serializable, so convert Pydantic to dicts
        return [c.model_dump() for c in raw_chunks]

    chunks = await step.run("load-and-chunk", chunk_pdf)

    # Step 3: Embed and upsert into Qdrant
    async def embed_and_upsert() -> dict:
        if not chunks:
            return RagUpsertResult(
                source=filename,
                chunks_upserted=0,
                status="success",
            ).model_dump()

        texts = [c["text"] for c in chunks]
        vectors = embed_text(texts)
        payloads = [
            {"text": c["text"], "source": c["source"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ]
        count = get_db().upsert(vectors=vectors, payloads=payloads)
        return RagUpsertResult(
            source=chunks[0]["source"],
            chunks_upserted=count,
            status="success",
        ).model_dump()

    return await step.run("embed-and-upsert", embed_and_upsert)


# ─── Workflow 2: Query PDF with AI (Groq) ─────────────────────────────────────
@inngest_client.create_function(
    fn_id="rag-query-pdf-ai",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
    throttle=inngest.Throttle(
        limit=10,
        period=60 * 1000,  # 1 minute in milliseconds
    ),
)
async def rag_query_pdf_ai(ctx: inngest.Context, **kwargs) -> dict[str, Any]:
    step = getattr(ctx, "step", kwargs.get("step"))
    event = getattr(ctx, "event", kwargs.get("event"))
    question: str = event.data["question"]

    # Step 1: Embed the question and search Qdrant
    async def search_context() -> list[dict]:
        query_vector = embed_text([question])[0]
        results: list[RagSearchResult] = get_db().search(query_vector=query_vector, top_k=5)
        return [r.model_dump() for r in results]

    search_results = await step.run("search-context", search_context)

    # Step 2: Call Groq LLM with retrieved context
    async def generate_answer() -> dict:
        context_text = "\n\n".join(
            f"[Source: {r['source']} | Score: {r['score']:.3f}]\n{r['text']}"
            for r in search_results
        )
        completion = get_groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question based ONLY on "
                        "the provided context. If the context does not contain the answer, say so clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return {
            "answer": completion.choices[0].message.content,
            "sources": [r["source"] for r in search_results],
            "question": question,
            "model": "llama-3.3-70b-versatile",
        }

    answer_result = await step.run("generate-answer", generate_answer)

    # Step 3: Call Groq LLM as a Judge to evaluate hallucination
    async def evaluate_hallucination() -> dict:
        context_text = "\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in search_results
        )
        answer = answer_result["answer"]
        
        eval_prompt = f"""
You are a strict Hallucination Grader for a Retrieval-Augmented Generation (RAG) system.
Your job is to act like a Lynx evaluation model and determine if the generated answer is faithful to the provided context.

Context:
{context_text}

Generated Answer:
{answer}

Instructions:
1. Carefully check if ANY factual claim made in the generated answer goes beyond what is explicitly stated in the context.
2. If the answer states that the information is not in the context, this is NOT a hallucination (it is faithful).
3. Provide a brief reasoning explaining your verdict.
4. Output exactly ONE line at the very end formatted as "Verdict: [True/False]". True means it CONTAINS hallucinations (unsupported facts). False means it is FAITHFUL.

Reasoning and Verdict:
"""
        completion = get_groq_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        
        response_text = completion.choices[0].message.content.strip()
        last_line = response_text.split("\n")[-1].strip().lower()
        is_hallucinated = "true" in last_line
        
        return {
            "is_hallucinated": is_hallucinated,
            "reasoning": response_text
        }

    eval_result = await step.run("evaluate-hallucination", evaluate_hallucination)

    answer_result["evaluation"] = eval_result
    return answer_result


# ─── Mount Inngest serve route ────────────────────────────────────────────────
inngest.fast_api.serve(
    app,
    inngest_client,
    [api_monitor, api_error_alert, rag_ingest_pdf, rag_query_pdf_ai],
)
