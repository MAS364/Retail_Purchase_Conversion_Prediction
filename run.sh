#!/usr/bin/env bash
# run.sh — Start both the API and the Streamlit dashboard
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
#
# Prerequisites:
#   pip install -r requirements.txt
#   pip install -r requirements-ui.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🛒 Purchase Predictor — Starting services"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start FastAPI in background
echo "  → Starting FastAPI on port 8080..."
uvicorn app:app --host 0.0.0.0 --port 8080 &
API_PID=$!

# Wait for API to be ready
echo "  → Waiting for API..."
for i in $(seq 1 15); do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "  ✅ API is ready"
        break
    fi
    sleep 1
done

# Start Streamlit
echo "  → Starting Streamlit on port 8501..."
echo ""
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false &
ST_PID=$!

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 Services running:"
echo "     API:       http://localhost:8080"
echo "     Swagger:   http://localhost:8080/docs"
echo "     Dashboard: http://localhost:8501"
echo ""
echo "  Press Ctrl+C to stop both."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Trap Ctrl+C to kill both
cleanup() {
    echo ""
    echo "  Shutting down..."
    kill $API_PID $ST_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for either to exit
wait
