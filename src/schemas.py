"""
schemas.py — Pydantic models for request validation and response serialization.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SessionInput(BaseModel):
    """All 23 features required for a single session prediction."""

    # Numerical features
    price: float = Field(..., ge=0, description="Product price", examples=[150.0])
    total_time_spent: float = Field(..., ge=0, description="Total session time (seconds)", examples=[120])
    session_length: int = Field(..., ge=1, description="Number of pages/events in session", examples=[6])
    interaction_count: int = Field(..., ge=0, description="Total interactions", examples=[6])
    view_count: int = Field(..., ge=0, examples=[3])
    click_count: int = Field(..., ge=0, examples=[2])
    wishlist_count: int = Field(..., ge=0, examples=[0])
    add_to_cart_count: int = Field(..., ge=0, examples=[1])
    avg_time_per_interaction: float = Field(..., ge=0, examples=[20.0])
    cart_to_view_ratio: float = Field(..., ge=0, examples=[0.33])
    click_to_view_ratio: float = Field(..., ge=0, examples=[0.67])
    has_cart_action: int = Field(..., ge=0, le=1, examples=[1])
    has_wishlist_action: int = Field(..., ge=0, le=1, examples=[0])
    hour: int = Field(..., ge=0, le=23, examples=[14])
    day_of_week: int = Field(..., ge=0, le=6, examples=[2])
    month: int = Field(..., ge=1, le=12, examples=[3])
    is_weekend: int = Field(..., ge=0, le=1, examples=[0])

    # Categorical features
    category: str = Field(..., examples=["electronics"])
    brand: str = Field(..., examples=["apple"])
    channel: str = Field(..., examples=["web"])
    device_type: str = Field(..., examples=["desktop"])
    region: str = Field(..., examples=["uk"])
    traffic_source: str = Field(..., examples=["organic"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "price": 150.0, "total_time_spent": 120,
                    "session_length": 6, "interaction_count": 6,
                    "view_count": 3, "click_count": 2,
                    "wishlist_count": 0, "add_to_cart_count": 1,
                    "avg_time_per_interaction": 20.0,
                    "cart_to_view_ratio": 0.33, "click_to_view_ratio": 0.67,
                    "has_cart_action": 1, "has_wishlist_action": 0,
                    "hour": 14, "day_of_week": 2, "month": 3, "is_weekend": 0,
                    "category": "electronics", "brand": "apple",
                    "channel": "web", "device_type": "desktop",
                    "region": "uk", "traffic_source": "organic",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    index: Optional[int] = None
    prediction: int
    label: str
    probability: Optional[float] = None
    model: str


class BatchRequest(BaseModel):
    sessions: list[SessionInput] = Field(..., max_length=1000)
    model: Optional[str] = Field(default=None)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    default_model: str


class ModelsResponse(BaseModel):
    available_models: list[str]
    default: str


# ── CSV upload response ────────────────────────────────────────────────────────

class CsvPredictionRow(BaseModel):
    index: int
    prediction: int
    label: str
    probability: Optional[float] = None
    model: str


class CsvPredictionResponse(BaseModel):
    """Response from the CSV upload prediction endpoint."""
    predictions: list[CsvPredictionRow]
    count: int
    purchases: int
    conversion_rate: float
    data_format: str = Field(description="'raw_events' or 'session_features'")
    raw_event_count: int = Field(description="Number of raw events (0 if session-level)")
    session_count: int = Field(description="Number of sessions predicted")
    latency_ms: int
