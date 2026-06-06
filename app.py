"""
app.py — FastAPI production API for retail purchase conversion prediction.

Endpoints
---------
GET   /              → interactive API docs redirect
GET   /health        → JSON health check
GET   /models        → list available models
POST  /predict       → single session prediction (23 features)
POST  /predict/batch → batch predictions (JSON, up to 1000)
POST  /predict/csv   → upload raw or session-level CSV → preprocessing + prediction (all backend)

Auto-generated docs at /docs (Swagger) and /redoc (ReDoc).
"""

import io
import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Allow imports from src/ ────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schemas import (
    SessionInput,
    PredictionResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelsResponse,
    CsvPredictionResponse,
    CsvPredictionRow,
)
from utils import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from feature_engineering import build_session_features

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Columns that indicate raw event-level data
RAW_EVENT_MARKERS = {"session_id", "user_action", "event_index", "time_spent_sec"}

# Integer fields that pandas may produce as float after aggregation
INT_FIELDS = [
    "session_length", "interaction_count", "view_count", "click_count",
    "wishlist_count", "add_to_cart_count", "has_cart_action",
    "has_wishlist_action", "hour", "day_of_week", "month", "is_weekend",
]
FLOAT_FIELDS = [
    "price", "total_time_spent", "avg_time_per_interaction",
    "cart_to_view_ratio", "click_to_view_ratio",
]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("purchase_predictor")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "mlp")

MODEL_MAP = {
    "decision_tree": "decision_tree_pipeline.joblib",
    "naive_bayes":   "naive_bayes_pipeline.joblib",
    "svm":           "svm_pipeline.joblib",
    "mlp":           "mlp_pipeline.joblib",
}

# ── Model storage ──────────────────────────────────────────────────────────────
pipelines: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    for name, filename in MODEL_MAP.items():
        path = MODEL_DIR / filename
        if path.exists():
            pipelines[name] = joblib.load(path)
            logger.info("Loaded model: %s", name)
        else:
            logger.warning("Model file not found, skipping: %s", path)
    if not pipelines:
        logger.error("No models loaded! Ensure models/ contains .joblib files.")
    yield
    pipelines.clear()
    logger.info("Models unloaded.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Purchase Predictor API",
    description=(
        "E-commerce session conversion prediction service.\n\n"
        "**Three prediction modes:**\n"
        "- `POST /predict` — single session (23 features as JSON)\n"
        "- `POST /predict/batch` — batch JSON (up to 1000 sessions)\n"
        "- `POST /predict/csv` — upload a CSV file (raw events OR session-level). "
        "All preprocessing + feature engineering happens server-side."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request logging ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "%s %s → %s (%.1f ms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_pipeline(model_name: str):
    """Retrieve a pipeline by name or raise 400."""
    if model_name not in pipelines:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Choose from: {list(pipelines.keys())}",
        )
    return pipelines[model_name]


def _run_prediction(data: dict, model_name: str) -> dict:
    pipeline = _get_pipeline(model_name)
    df = pd.DataFrame([data])
    prediction = int(pipeline.predict(df)[0])
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = round(float(pipeline.predict_proba(df)[0][1]), 4)
    return {
        "prediction": prediction,
        "label": "purchase" if prediction == 1 else "no_purchase",
        "probability": probability,
        "model": model_name,
    }


def _is_raw_event_data(df: pd.DataFrame) -> bool:
    """Detect if a DataFrame is raw event-level data."""
    return bool(RAW_EVENT_MARKERS.intersection(df.columns))


def _preprocess_to_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw event-level data → session-level features.
    Also handles type casting to match model expectations.
    """
    session_df = build_session_features(df)

    missing = [f for f in ALL_FEATURES if f not in session_df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"After feature engineering, missing columns: {missing}",
        )

    result = session_df[ALL_FEATURES].copy()

    # Cast types — pandas aggregation produces float64 for int columns
    for col in INT_FIELDS:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
    for col in FLOAT_FIELDS:
        if col in result.columns:
            result[col] = result[col].fillna(0.0).astype(float)
    for col in CATEGORICAL_FEATURES:
        if col in result.columns:
            result[col] = result[col].fillna("unknown").astype(str)

    return result


def _run_batch_prediction(feature_df: pd.DataFrame, model_name: str) -> list[dict]:
    """Run predictions on a session-level DataFrame."""
    pipeline = _get_pipeline(model_name)

    predictions = pipeline.predict(feature_df).tolist()
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(feature_df)[:, 1].tolist()

    results = []
    for i, pred in enumerate(predictions):
        prob = round(probabilities[i], 4) if probabilities is not None else None
        results.append({
            "index": i,
            "prediction": int(pred),
            "label": "purchase" if pred == 1 else "no_purchase",
            "probability": prob,
            "model": model_name,
        })
    return results


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check for Cloud Run / load balancers."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(pipelines.keys()),
        default_model=DEFAULT_MODEL,
    )


@app.get("/models", response_model=ModelsResponse, tags=["System"])
async def list_models():
    """List all available models."""
    return ModelsResponse(
        available_models=list(pipelines.keys()),
        default=DEFAULT_MODEL,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    session: SessionInput,
    model: str = Query(default=DEFAULT_MODEL, description="Model to use"),
):
    """Single prediction — 23 session features as JSON body."""
    result = _run_prediction(session.model_dump(), model)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_batch(body: BatchRequest):
    """Batch prediction — JSON array of up to 1000 sessions."""
    results = []
    for i, session in enumerate(body.sessions):
        try:
            result = _run_prediction(session.model_dump(), body.model or DEFAULT_MODEL)
            results.append(PredictionResponse(index=i, **result))
        except HTTPException:
            results.append(PredictionResponse(
                index=i, prediction=-1, label="error",
                probability=None, model=body.model or DEFAULT_MODEL,
            ))
    return BatchResponse(predictions=results, count=len(results))


@app.post("/predict/csv", response_model=CsvPredictionResponse, tags=["Predictions"])
async def predict_csv(
    file: UploadFile = File(..., description="CSV file (raw events or session-level features)"),
    model: str = Query(default=DEFAULT_MODEL, description="Model to use"),
):
    """
    Upload a CSV file for batch prediction.

    **Accepts two formats — auto-detected:**
    - **Raw event-level data** (columns: session_id, user_action, event_index, etc.)
      → Runs the full feature engineering pipeline server-side, then predicts.
    - **Session-level features** (23 model columns: price, view_count, category, etc.)
      → Predicts directly.

    All preprocessing, type casting, and feature conversion happens in the backend.
    The client just uploads a file.
    """
    # ── Read CSV ───────────────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    logger.info("CSV uploaded: %s — %d rows, %d columns", file.filename, len(df), len(df.columns))

    # ── Detect format and preprocess ───────────────────────────────────────
    raw_data = _is_raw_event_data(df)
    raw_event_count = len(df) if raw_data else 0

    if raw_data:
        logger.info("Detected raw event-level data (%d events). Running feature engineering…", len(df))
        feature_df = _preprocess_to_sessions(df)
        logger.info("Engineered %d sessions from %d events.", len(feature_df), raw_event_count)
    else:
        # Validate that session-level columns are present
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"CSV is missing {len(missing)} required feature columns: "
                    f"{missing[:10]}{'…' if len(missing) > 10 else ''}. "
                    f"Upload either raw event-level data (with session_id, user_action) "
                    f"or session-level data with all 23 model features."
                ),
            )
        feature_df = df[ALL_FEATURES].copy()
        # Apply same type casting
        for col in INT_FIELDS:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(0).astype(int)
        for col in FLOAT_FIELDS:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(0.0).astype(float)
        for col in CATEGORICAL_FEATURES:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna("unknown").astype(str)

    # ── Run predictions ────────────────────────────────────────────────────
    t0 = time.time()
    results = _run_batch_prediction(feature_df, model)
    latency_ms = round((time.time() - t0) * 1000)

    total = len(results)
    purchases = sum(1 for r in results if r["prediction"] == 1)

    logger.info(
        "CSV prediction complete — %d sessions, %d purchases (%.1f%%), %dms",
        total, purchases, purchases / max(total, 1) * 100, latency_ms,
    )

    return CsvPredictionResponse(
        predictions=[CsvPredictionRow(**r) for r in results],
        count=total,
        purchases=purchases,
        conversion_rate=round(purchases / max(total, 1), 4),
        data_format="raw_events" if raw_data else "session_features",
        raw_event_count=raw_event_count,
        session_count=total,
        latency_ms=latency_ms,
    )


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
