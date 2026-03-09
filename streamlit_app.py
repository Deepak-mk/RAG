"""
streamlit_app.py - Streamlit frontend for the RAG POC.

Supports two deployment modes:
  LOCAL  (dev):  Calls FastAPI backend at localhost:8000 which talks to local Qdrant + Inngest
  CLOUD  (prod): Calls Groq + Qdrant Cloud directly (no FastAPI needed) for Streamlit Cloud deployment

Mode is determined by the presence of QDRANT_URL in environment / Streamlit secrets.
"""

import os
import time
import tempfile
import json
from typing import Optional

import requests
import streamlit as st

# ─── Detect deployment mode ───────────────────────────────────────────────────
def _get_secret(key: str) -> Optional[str]:
    """Read from Streamlit secrets first, then env vars."""
    try:
        return st.secrets.get(key) or os.getenv(key)
    except Exception:
        return os.getenv(key)

GROQ_API_KEY = _get_secret("GROQ_API_KEY")
QDRANT_URL   = _get_secret("QDRANT_URL")
QDRANT_API_KEY = _get_secret("QDRANT_API_KEY")

# We force IS_CLOUD to False so Streamlit ALWAYS routes through the FastAPI backend.
# This ensures that Inngest observability is always triggered, exactly like the tutorial video.
IS_CLOUD = False
BACKEND_URL    = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
INNGEST_DEV_URL = os.getenv("INNGEST_DEV_URL", "http://127.0.0.1:8288")
POLL_INTERVAL_SEC = 2
MAX_POLL_ATTEMPTS = 60

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG POC — PDF Intelligence",
    page_icon="📄",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #e0e0e0; }
  .main-header {
    text-align: center; padding: 2rem 0 0.5rem; font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .sub-header { text-align:center; color:#94a3b8; margin-bottom:1.5rem; font-size:0.95rem; }
  .card {
    background: rgba(255,255,255,0.06); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12); border-radius: 16px;
    padding: 1.6rem; margin-bottom: 1.2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  .status-badge {
    display:inline-block; padding:4px 12px; border-radius:999px;
    font-size:0.75rem; font-weight:600; letter-spacing:0.05em; text-transform:uppercase;
  }
  .badge-running { background:#1e3a5f; color:#60a5fa; border:1px solid #3b82f6; }
  .badge-success { background:#14382a; color:#34d399; border:1px solid #10b981; }
  .badge-error   { background:#3b1515; color:#f87171; border:1px solid #ef4444; }
  .badge-pending { background:#2d2a14; color:#fbbf24; border:1px solid #f59e0b; }
  .metric-box {
    background: rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:10px; padding:0.8rem 1rem; margin:0.4rem 0;
  }
  .metric-label { color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; }
  .metric-value { color:#e2e8f0; font-size:1.1rem; font-weight:600; margin-top:2px; }
  .answer-box {
    background: rgba(103,232,249,0.06); border-left:4px solid #67e8f9;
    border-radius:8px; padding:1rem 1.4rem; font-size:1rem; line-height:1.7;
  }
  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #a78bfa, #67e8f9); color:#0f0c29;
    font-weight:700; border:none; border-radius:10px; padding:0.55rem 1.6rem;
    transition: opacity 0.2s, transform 0.1s;
  }
  div[data-testid="stButton"] > button:hover { opacity:0.88; transform:scale(1.02); }

  .login-title {
    text-align:center; font-size:1.9rem; font-weight:700;
    background:linear-gradient(90deg,#a78bfa,#67e8f9);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0.3rem;
  }
  .login-sub { text-align:center; color:#94a3b8; font-size:0.9rem; margin-bottom:2rem; }
</style>
""", unsafe_allow_html=True)


# ─── Auth helpers ─────────────────────────────────────────────────────────────
APP_USERNAME = _get_secret("APP_USERNAME") or os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = _get_secret("APP_PASSWORD") or os.getenv("APP_PASSWORD", "admin123")


def show_login() -> None:
    """Render the login page and handle credential verification."""
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown('<div class="login-title">📄 PDF Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Sign in to continue</div>', unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🔐 Sign In", use_container_width=True)

        if submitted:
            if username == APP_USERNAME and password == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("\u274c Invalid username or password. Please try again.")


# ─── Auth gate ────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated", False):
    show_login()
    st.stop()  # Nothing below renders until the user logs in


# ─── Helpers ──────────────────────────────────────────────────────────────────
def badge(label: str, cls: str) -> str:
    return f'<span class="status-badge {cls}">{label}</span>'

def metric_box(label: str, value: str) -> str:
    return (f'<div class="metric-box">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>')


# ─────────────────────────────────────────────────────────────────────────────
# CLOUD MODE: Direct calls to Groq + Qdrant Cloud (no FastAPI needed)
# ─────────────────────────────────────────────────────────────────────────────
def cloud_ingest(file_bytes: bytes, filename: str) -> dict:
    """Save PDF to temp file and run the full RAG ingestion pipeline directly."""
    from data_loader import load_and_chunk_pdf, embed_text
    from vector_db import QuadrantStorage

    db = QuadrantStorage(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    chunks = load_and_chunk_pdf(tmp_path)
    texts = [c.text for c in chunks]
    vectors = embed_text(texts)
    payloads = [{"text": c.text, "source": c.source, "chunk_index": c.chunk_index} for c in chunks]
    count = db.upsert(vectors=vectors, payloads=payloads)
    return {"source": filename, "chunks_upserted": count, "status": "success"}


def cloud_query(question: str) -> dict:
    """Embed question → search Qdrant Cloud → call Groq for answer."""
    from groq import Groq
    from data_loader import embed_text
    from vector_db import QuadrantStorage

    db = QuadrantStorage(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    query_vector = embed_text([question])[0]
    results = db.search(query_vector=query_vector, top_k=5)

    context_text = "\n\n".join(
        f"[Source: {r.source} | Score: {r.score:.3f}]\n{r.text}" for r in results
    )
    groq_client = Groq(api_key=GROQ_API_KEY)
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",
             "content": "Answer based ONLY on the provided context. If the context doesn't contain the answer, say so."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return {
        "answer": completion.choices[0].message.content,
        "sources": [r.source for r in results],
        "question": question,
        "model": "llama-3.3-70b-versatile",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MODE: Calls FastAPI → Inngest workflows
# ─────────────────────────────────────────────────────────────────────────────
def get_health() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=3)
        return r.json() if r.status_code == 200 else {"status": "error"}
    except Exception:
        return {"status": "unreachable"}

def get_collections() -> list:
    try:
        r = requests.get(f"{BACKEND_URL}/api/collections", timeout=3)
        return r.json().get("collections", []) if r.status_code == 200 else []
    except Exception:
        return []

def local_send_ingest(filename: str, file_b64: str) -> str | None:
    try:
        r = requests.post(f"{BACKEND_URL}/api/ingest", json={"filename": filename, "file_b64": file_b64}, timeout=60)
        r.raise_for_status()
        return r.json().get("event_id")
    except Exception as e:
        st.error(f"Failed to trigger ingestion: {e}")
        return None

def local_send_query(question: str) -> str | None:
    try:
        r = requests.post(f"{BACKEND_URL}/api/query", json={"question": question}, timeout=10)
        r.raise_for_status()
        return r.json().get("event_id")
    except Exception as e:
        st.error(f"Failed to submit query: {e}")
        return None

def poll_run_result(event_id: str) -> dict | None:
    for _ in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL_SEC)
        try:
            resp = requests.get(f"{INNGEST_DEV_URL}/v1/events/{event_id}/runs", timeout=5)
            if resp.status_code != 200:
                continue
            runs = resp.json().get("data", [])
            if not runs:
                continue
            run = runs[0]
            status = run.get("status", "")
            if status == "Completed":
                return {"status": "success", "data": json.loads(run.get("output", "{}"))}
            elif status in ("Failed", "Cancelled"):
                return {"status": "error", "message": run.get("output", "Unknown error")}
        except Exception:
            pass
    return {"status": "timeout", "message": "Run timed out. Check the Inngest dashboard."}


# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 PDF Intelligence — RAG POC</div>', unsafe_allow_html=True)
mode_label = "☁️ Cloud Mode (Qdrant Cloud + Groq)" if IS_CLOUD else "🖥️ Local Mode (FastAPI + Inngest + Docker)"
st.markdown(f'<div class="sub-header">Powered by Qdrant · Groq llama-3.3-70b · LlamaIndex · sentence-transformers &nbsp;|&nbsp; <b>{mode_label}</b></div>', unsafe_allow_html=True)


# ━━━━ SYSTEM STATUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.expander("🖥️ System Status", expanded=True):
    if st.button("🔄 Refresh Status", key="refresh_status"):
        st.rerun()

    s1, s2, s3, s4 = st.columns(4)

    if IS_CLOUD:
        with s1:
            st.markdown(metric_box("Mode", "☁️ Cloud"), unsafe_allow_html=True)
        with s2:
            qdrant_ok = bool(QDRANT_URL)
            st.markdown(metric_box("Qdrant Cloud", "🟢 Configured" if qdrant_ok else "🔴 Missing URL"), unsafe_allow_html=True)
        with s3:
            groq_ok = bool(GROQ_API_KEY)
            st.markdown(metric_box("Groq", "🟢 Key Set" if groq_ok else "🔴 Key Missing"), unsafe_allow_html=True)
        with s4:
            st.markdown(metric_box("LLM Model", "🤖 llama-3.3-70b"), unsafe_allow_html=True)
    else:
        health = get_health()
        collections = get_collections()
        with s1:
            api_ok = health.get("status") == "ok"
            st.markdown(metric_box("FastAPI Backend", "🟢 Online" if api_ok else "🔴 Offline"), unsafe_allow_html=True)
        with s2:
            qdrant_ok = health.get("qdrant") == "ok"
            st.markdown(metric_box("Qdrant (Docker)", "🟢 Online" if qdrant_ok else "🔴 Offline"), unsafe_allow_html=True)
        with s3:
            total_vectors = sum(c.get("vectors_count") or 0 for c in collections)
            st.markdown(metric_box("Vectors Indexed", f"{total_vectors:,}"), unsafe_allow_html=True)
        with s4:
            st.markdown(
                metric_box("Inngest Dashboard", f"<a href='{INNGEST_DEV_URL}' target='_blank' style='color:#67e8f9'>Open ↗</a>"),
                unsafe_allow_html=True,
            )
        if collections:
            st.markdown("**Collections:**")
            for col in collections:
                st.markdown(f"&nbsp;&nbsp;`{col['name']}` — **{col.get('vectors_count', 0):,}** vectors")

st.markdown("---")

# ━━━━ MAIN PANELS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT: Ingestion ───────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Ingest a PDF")
    if IS_CLOUD:
        st.caption("Chunks, embeds, and stores PDF directly to Qdrant Cloud.")
    else:
        st.caption("Triggers the `rag/ingest_pdf` Inngest workflow → chunk → embed → Qdrant.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        st.write(f"**Selected:** `{uploaded_file.name}` ({uploaded_file.size:,} bytes)")
        if st.button("🚀 Start Ingestion", key="ingest_btn"):
            if IS_CLOUD:
                with st.spinner("Ingesting PDF directly to Qdrant Cloud…"):
                    try:
                        result_data = cloud_ingest(uploaded_file.read(), uploaded_file.name)
                        st.markdown(badge("✅ Complete", "badge-success"), unsafe_allow_html=True)
                        st.success(f"**{result_data['chunks_upserted']} chunks** upserted from `{result_data['source']}`")
                        st.session_state["pdf_ingested"] = True
                    except Exception as e:
                        st.markdown(badge("Error", "badge-error"), unsafe_allow_html=True)
                        st.error(f"Ingestion failed: {e}")
            else:
                import base64
                file_bytes = uploaded_file.read()
                file_b64 = base64.b64encode(file_bytes).decode("utf-8")
                
                with st.spinner("Sending event to Inngest…"):
                    event_id = local_send_ingest(uploaded_file.name, file_b64)
                if event_id:
                    st.markdown(badge("Ingestion Running", "badge-running"), unsafe_allow_html=True)
                    st.info(f"📡 Event ID: `{event_id}`  |  [View on Dashboard]({INNGEST_DEV_URL})")
                    progress_bar = st.progress(0, text="Processing…")
                    with st.spinner("Waiting for workflow to complete…"):
                        result = poll_run_result(event_id)
                    progress_bar.progress(100)
                    if result and result["status"] == "success":
                        data = result["data"]
                        st.markdown(badge("✅ Complete", "badge-success"), unsafe_allow_html=True)
                        st.success(f"**{data.get('chunks_upserted', '?')} chunks** upserted from `{data.get('source', '?')}`")
                    elif result and result["status"] == "error":
                        st.markdown(badge("Error", "badge-error"), unsafe_allow_html=True)
                        st.error(f"Ingestion failed: {result.get('message')}")
                    else:
                        st.markdown(badge("Timeout", "badge-pending"), unsafe_allow_html=True)
                        st.warning("Still running — check the Inngest dashboard.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT: Query ──────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Ask a Question")
    if IS_CLOUD:
        st.caption("Searches Qdrant Cloud + calls Groq llama-3.3-70b directly.")
    else:
        st.caption("Triggers the `rag/query_pdf_ai` Inngest workflow → search → Groq LLM.")

    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the key findings in this document?",
        height=120,
        label_visibility="collapsed",
    )

    if st.button("💡 Get Answer", key="query_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            if IS_CLOUD:
                with st.spinner("Searching Qdrant Cloud + calling Groq…"):
                    try:
                        result_data = cloud_query(question.strip())
                        st.markdown(badge("✅ Complete", "badge-success"), unsafe_allow_html=True)
                        st.markdown("**Answer:**")
                        st.markdown(f'<div class="answer-box">{result_data["answer"]}</div>', unsafe_allow_html=True)
                        if result_data.get("sources"):
                            unique_sources = list(dict.fromkeys(result_data["sources"]))
                            st.markdown("**Sources:** " + " · ".join(f"`{s}`" for s in unique_sources))
                        st.caption(f"🤖 Model: `{result_data.get('model', 'llama-3.3-70b-versatile')}`")
                    except Exception as e:
                        st.markdown(badge("Error", "badge-error"), unsafe_allow_html=True)
                        st.error(f"Query failed: {e}")
            else:
                with st.spinner("Sending query event to Inngest…"):
                    event_id = local_send_query(question.strip())
                if event_id:
                    st.markdown(badge("Querying", "badge-running"), unsafe_allow_html=True)
                    st.info(f"📡 Event ID: `{event_id}`  |  [View on Dashboard]({INNGEST_DEV_URL})")
                    with st.spinner("Waiting for Groq LLM response…"):
                        result = poll_run_result(event_id)
                    if result and result["status"] == "success":
                        data = result["data"]
                        st.markdown(badge("✅ Complete", "badge-success"), unsafe_allow_html=True)
                        st.markdown("**Answer:**")
                        st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)
                        if data.get("sources"):
                            unique_sources = list(dict.fromkeys(data["sources"]))
                            st.markdown("**Sources:** " + " · ".join(f"`{s}`" for s in unique_sources))
                        st.caption(f"🤖 Model: `{data.get('model', 'llama-3.3-70b-versatile')}`")
                    elif result and result["status"] == "error":
                        st.markdown(badge("Error", "badge-error"), unsafe_allow_html=True)
                        st.error(f"Query failed: {result.get('message')}")
                    else:
                        st.markdown(badge("Timeout", "badge-pending"), unsafe_allow_html=True)
                        st.warning("Still running — check the Inngest dashboard.")
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns([10, 1])
with col_f1:
    st.caption(
        "🛠 **Stack:** LlamaIndex · Qdrant · Groq llama-3.3-70b · sentence-transformers · Streamlit"
        + ("" if IS_CLOUD else f"  |  **FastAPI + Inngest:** [{BACKEND_URL}/docs]({BACKEND_URL}/docs)")
    )
with col_f2:
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.rerun()
