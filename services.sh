#!/bin/bash

# Configuration
VENV_PATH=".venv/bin/activate"
FASTAPI_PORT=8000
INNGREST_PORT=8288
STREAMLIT_PORT=8501

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

FASTAPI_LOG="$LOG_DIR/fastapi.log"
INNGREST_LOG="$LOG_DIR/inngest.log"
STREAMLIT_LOG="$LOG_DIR/streamlit.log"

usage() {
    echo "Usage: ./services.sh {start|stop|status|logs}"
    exit 1
}

start() {
    echo "🚀 Starting RAG services in the background..."

    # 1. FastAPI
    if ! pgrep -f "uvicorn main:app" > /dev/null; then
        echo "Starting FastAPI on port $FASTAPI_PORT..."
        source "$VENV_PATH"
        nohup uvicorn main:app --host 0.0.0.0 --port $FASTAPI_PORT --reload > "$FASTAPI_LOG" 2>&1 &
    else
        echo "FastAPI is already running."
    fi

    # 2. Inngest
    if ! pgrep -f "inngest-cli dev" > /dev/null; then
        echo "Starting Inngest Dev Server..."
        # Use npx to run inngest-cli
        nohup npx inngest-cli@latest dev --no-discovery -u "http://localhost:$FASTAPI_PORT/api/inngest" > "$INNGREST_LOG" 2>&1 &
    else
        echo "Inngest is already running."
    fi

    # 3. Streamlit
    if ! pgrep -f "streamlit run streamlit_app.py" > /dev/null; then
        echo "Starting Streamlit on port $STREAMLIT_PORT..."
        source "$VENV_PATH"
        export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
        export STREAMLIT_SERVER_HEADLESS=true
        export BACKEND_URL="http://127.0.0.1:$FASTAPI_PORT"
        nohup streamlit run streamlit_app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 > "$STREAMLIT_LOG" 2>&1 &
    else
        echo "Streamlit is already running."
    fi

    echo "✅ Services started. Use './services.sh status' to check."
}

stop() {
    echo "🛑 Stopping RAG services..."
    pkill -f "uvicorn main:app"
    pkill -f "inngest-cli dev"
    pkill -f "streamlit run streamlit_app.py"
    echo "✅ Services stopped."
}

status() {
    echo "📊 Service Status:"
    
    # FastAPI
    if pgrep -f "uvicorn main:app" > /dev/null; then
        echo "  🟢 FastAPI: Running on port $FASTAPI_PORT"
    else
        echo "  🔴 FastAPI: STOPPED"
    fi

    # Inngest
    if pgrep -f "inngest-cli dev" > /dev/null; then
        echo "  🟢 Inngest: Running on port $INNGREST_PORT"
    else
        echo "  🔴 Inngest: STOPPED"
    fi

    # Streamlit
    if pgrep -f "streamlit run streamlit_app.py" > /dev/null; then
        echo "  🟢 Streamlit: Running on port $STREAMLIT_PORT"
    else
        echo "  🔴 Streamlit: STOPPED"
    fi

    # Ngrok
    if pgrep -f "ngrok http" > /dev/null; then
        echo "  🟢 Ngrok: Tunnel Active"
    else
        echo "  ⚪ Ngrok: Not running"
    fi
}

logs() {
    echo "📜 Tailing logs (Ctrl+C to exit)..."
    tail -f "$FASTAPI_LOG" "$INNGREST_LOG" "$STREAMLIT_LOG"
}

case "$1" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    logs) logs ;;
    *) usage ;;
esac
