"""
FastAPI Backend Application
Production RESTful API serving multimodal inference, weather connectors,
geo-fenced treatment lookups, agronomist review workflows, and operational drift monitoring.
"""

import sys
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Setup path imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "04_Weather"))
sys.path.insert(0, str(REPO_ROOT / "05_Treatment"))
sys.path.insert(0, str(REPO_ROOT / "api"))

from config import config
from inference import InferencePipeline
from weather_connector import WeatherConnector
from treatment_engine import TreatmentEngine
from feedback import FeedbackManager
from retraining import DriftMonitor
from dashboard import dashboard_router

app = FastAPI(
    title="Crop Disease Diagnosis & Agronomic Advisory ML System",
    version="1.0.0",
    description="Production multimodal AI API for 20-crop disease diagnosis, Grad-CAM attribution, and geo-fenced treatment."
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Dashboard UI
app.include_router(dashboard_router)

# Lazy singletons
_pipeline: Optional[InferencePipeline] = None
_weather_connector: Optional[WeatherConnector] = None
_treatment_engine: Optional[TreatmentEngine] = None
_feedback_manager: Optional[FeedbackManager] = None
_drift_monitor: Optional[DriftMonitor] = None

def get_pipeline() -> InferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline()
    return _pipeline

def get_weather_connector() -> WeatherConnector:
    global _weather_connector
    if _weather_connector is None:
        _weather_connector = WeatherConnector()
    return _weather_connector

def get_treatment_engine() -> TreatmentEngine:
    global _treatment_engine
    if _treatment_engine is None:
        _treatment_engine = TreatmentEngine()
    return _treatment_engine

def get_feedback_manager() -> FeedbackManager:
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = FeedbackManager()
    return _feedback_manager

def get_drift_monitor() -> DriftMonitor:
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor

# -------------------------------------------------------------
# Pydantic Schemas for Request / Response Validation
# -------------------------------------------------------------
class SurveyRequest(BaseModel):
    root_condition: str = Field(default="unknown", description="normal, discolored, rotting, unknown")
    stem_health: str = Field(default="unknown", description="healthy, weak, lesions_spots, wilting, unknown")
    recent_chemical_spray: str = Field(default="unknown", description="no, yes_pesticide, yes_fungicide, yes_fertilizer, unknown")
    crop_growth_stage: Optional[str] = Field(default="vegetative", description="seedling, vegetative, flowering, fruiting, maturity")
    notes: Optional[str] = None

class FeedbackSubmission(BaseModel):
    prediction_id: str
    rating: int = Field(ge=1, le=5, description="Farmer satisfaction rating (1-5)")
    user_comment: Optional[str] = None
    crop_observed: Optional[str] = None

class AgronomistReviewRequest(BaseModel):
    prediction_id: str
    verdict: str = Field(description="correct, incorrect, unknown, new_disease, needs_more_data")
    verified_crop: Optional[str] = None
    verified_disease: Optional[str] = None
    agronomist_id: str = "certified_agronomist"
    notes: Optional[str] = None

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Crop Disease Diagnosis ML System",
        "supported_crops": 20,
        "status": "healthy",
        "model_version": config.MODEL_VERSION,
        "dashboard_url": "/dashboard",
        "documentation_url": "/docs"
    }

@app.get("/health")
def health():
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "model_version": config.MODEL_VERSION,
        "is_trained_weights_loaded": pipeline.is_trained,
        "backbone": config.BACKBONE,
        "supported_classes": pipeline.num_classes,
        "ood_threshold": config.OOD_CONFIDENCE_THRESHOLD
    }

@app.post("/predict")
async def predict_crop_disease(
    image: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    root_condition: str = Form("unknown"),
    stem_health: str = Form("unknown"),
    recent_chemical_spray: str = Form("unknown"),
    growth_stage: str = Form("vegetative"),
    region: Optional[str] = Form(None)
):
    """
    Primary multimodal diagnosis endpoint.
    Uploads a crop leaf image alongside GPS and 3-question symptom survey.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a recognized image format."
        )

    suffix = Path(image.filename or "temp.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await image.read()
        tmp.write(contents)
        temp_img_path = Path(tmp.name)

    try:
        pipeline = get_pipeline()
        survey = {
            "root_condition": root_condition,
            "stem_health": stem_health,
            "recent_chemical_spray": recent_chemical_spray,
            "crop_growth_stage": growth_stage
        }

        result = pipeline.diagnose(
            image_input=temp_img_path,
            latitude=latitude,
            longitude=longitude,
            survey_data=survey,
            growth_stage=growth_stage,
            region=region
        )
        return JSONResponse(status_code=200, content=result)
    finally:
        if temp_img_path.exists():
            try:
                temp_img_path.unlink()
            except Exception:
                pass

@app.post("/survey")
def validate_survey(survey: SurveyRequest):
    """Validates and echoes standardized 3-question survey response."""
    return {
        "valid": True,
        "received_data": survey.model_dump(),
        "status": "conforms_to_schema"
    }

@app.get("/weather")
def get_weather(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0)
):
    """Retrieves normalized weather context for GPS coordinates."""
    connector = get_weather_connector()
    return connector.get_weather(latitude, longitude)

@app.get("/treatment")
def get_treatment(
    crop: str = Query(...),
    disease: str = Query(...),
    growth_stage: str = Query("vegetative"),
    latitude: Optional[float] = Query(None, ge=-90.0, le=90.0),
    longitude: Optional[float] = Query(None, ge=-180.0, le=180.0),
    region: Optional[str] = Query(None)
):
    """Queries verified geo-fenced agronomic treatment protocol."""
    engine = get_treatment_engine()
    return engine.lookup(
        crop=crop,
        disease=disease,
        growth_stage=growth_stage,
        latitude=latitude,
        longitude=longitude,
        region=region
    )

@app.get("/review-queue")
def get_review_queue(limit: int = Query(50, ge=1, le=200)):
    """Retrieves uncertain cases currently awaiting agronomist review."""
    feedback = get_feedback_manager()
    return feedback.get_pending_review_queue(limit=limit)

@app.post("/review")
def submit_agronomist_review(review: AgronomistReviewRequest):
    """Agronomist submits ground-truth verdict on an uncertain prediction."""
    feedback = get_feedback_manager()
    try:
        res = feedback.submit_agronomist_review(
            prediction_id=review.prediction_id,
            verdict=review.verdict,
            verified_crop=review.verified_crop,
            verified_disease=review.verified_disease,
            agronomist_id=review.agronomist_id,
            notes=review.notes
        )
        return {"status": "review_recorded", "data": res}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/feedback")
def submit_user_feedback(feedback_data: FeedbackSubmission):
    """Farmers submit subjective satisfaction feedback on diagnostic accuracy."""
    fb_manager = get_feedback_manager()
    fb_file = fb_manager.feedback_dir / "user_ratings.jsonl"
    with open(fb_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_data.model_dump()) + "\n")
    return {"status": "feedback_saved"}

@app.get("/drift/stats")
def get_drift_stats(window: int = Query(100, ge=10, le=1000)):
    """Computes statistical drift, OOD rejection rate, and agronomist correction metrics."""
    monitor = get_drift_monitor()
    return monitor.analyze_drift(window_size=window)

