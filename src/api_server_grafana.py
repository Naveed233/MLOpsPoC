#!/usr/bin/env python3
"""
api_server_grafana.py
FastAPI inference server with Prometheus metrics for Grafana monitoring

Metrics exposed:
- prediction_requests_total: Counter of total predictions
- prediction_latency_seconds: Histogram of prediction latency
- prediction_price_yen: Histogram of predicted prices
- model_load_time_seconds: Gauge of model load time
- predictions_by_prefecture: Counter by prefecture
- predictions_by_property_type: Counter by property type
"""

import os
import json
import time
from datetime import datetime
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

###################
# PYDANTIC MODELS
###################

class PropertyType(str, Enum):
    MANSION = "マンション"
    HOUSE = "戸建て"
    LAND = "土地"

class StructureType(str, Enum):
    RC = "RC"
    SRC = "SRC"
    WOOD = "木造"
    STEEL = "鉄骨造"
    OTHER = "その他"

class PredictRequest(BaseModel):
    external_id: Optional[str] = Field(None, description="Client ID for matching actuals")
    prefecture: str = Field(..., description="例: 東京都, 千葉県")
    city_ward: str = Field(..., description="例: 渋谷区, 千葉市中央区")
    property_type: PropertyType
    building_structure: StructureType
    building_age_years: float = Field(..., ge=0)
    effective_area_m2: float = Field(..., gt=0)
    coverage_ratio: Optional[float] = Field(None, ge=0, le=100)
    floor_area_ratio: Optional[float] = Field(None, ge=0, le=1000)

class PredictResponse(BaseModel):
    prediction_yen: float
    model_tag: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_tag: str
    timestamp: str

###################
# PROMETHEUS METRICS
###################

# Counters
prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['status', 'prefecture', 'property_type']
)

# Histograms
prediction_latency_seconds = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)

prediction_price_yen = Histogram(
    'prediction_price_yen',
    'Predicted property prices in yen',
    buckets=[10_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000, 150_000_000, 200_000_000, 500_000_000]
)

# Gauges
model_load_time_seconds = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model'
)

model_info = Gauge(
    'model_info',
    'Model metadata',
    ['tag', 'saved_at']
)

###################
# FASTAPI APP
###################

app = FastAPI(
    title="Real Estate Price Prediction API (Grafana-enabled)",
    description="Production ML API with Prometheus metrics",
    version="2.0.0"
)

# Global state
model_pipeline = None
model_metadata = {}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_DIR = os.path.join(BASE_DIR, "models", "production")
PRED_LOG = os.path.join(BASE_DIR, "data", "predictions", "predictions.jsonl")

def load_production_model():
    """Load production model and metadata"""
    global model_pipeline, model_metadata
    
    start_time = time.time()
    
    model_path = os.path.join(PROD_DIR, "model.joblib")
    metadata_path = os.path.join(PROD_DIR, "metadata.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_pipeline = joblib.load(model_path)
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
    
    load_time = time.time() - start_time
    model_load_time_seconds.set(load_time)
    
    # Set model info
    model_info.labels(
        tag=model_metadata.get('tag', 'unknown'),
        saved_at=model_metadata.get('saved_at_utc', 'unknown')
    ).set(1)
    
    return model_pipeline, model_metadata

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_production_model()
        print(f"✅ Model loaded: {model_metadata.get('tag', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_pipeline is not None else "unhealthy",
        model_loaded=model_pipeline is not None,
        model_tag=model_metadata.get("tag", "unknown"),
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict_price", response_model=PredictResponse)
async def predict_price(request: PredictRequest):
    """
    Predict property price with Prometheus metrics tracking
    """
    if model_pipeline is None:
        prediction_requests_total.labels(status='error', prefecture='unknown', property_type='unknown').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Prepare features with engineering
        import pandas as pd
        
        features = {
            "prefecture": request.prefecture,
            "city_ward": request.city_ward,
            "property_type": request.property_type.value,
            "building_structure": request.building_structure.value,
            "building_age_years": request.building_age_years,
            "effective_area_m2": request.effective_area_m2,
            "coverage_ratio": request.coverage_ratio if request.coverage_ratio is not None else 60.0,
            "floor_area_ratio": request.floor_area_ratio if request.floor_area_ratio is not None else 200.0,
        }
        
        # Create input DataFrame
        X = pd.DataFrame([features])
        
        # Add engineered features
        X["is_tokyo"] = (X["prefecture"] == "東京都").astype(int)
        X["age_bucket"] = pd.cut(
            X["building_age_years"],
            bins=[-np.inf, 5, 15, 30, np.inf],
            labels=["0-5", "6-15", "16-30", "30+"]
        ).astype(str)
        
        # Make prediction
        prediction = model_pipeline.predict(X)[0]
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Update Prometheus metrics
        prediction_requests_total.labels(
            status='success',
            prefecture=request.prefecture,
            property_type=request.property_type.value
        ).inc()
        
        prediction_latency_seconds.observe(latency)
        prediction_price_yen.observe(prediction)
        
        # Log prediction
        log_prediction(request, prediction, latency)
        
        return PredictResponse(
            prediction_yen=float(prediction),
            model_tag=model_metadata.get("tag", "unknown"),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        prediction_requests_total.labels(
            status='error',
            prefecture=request.prefecture,
            property_type=request.property_type.value
        ).inc()
        
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def log_prediction(request: PredictRequest, prediction: float, latency_ms: float):
    """Log prediction to JSONL file"""
    os.makedirs(os.path.dirname(PRED_LOG), exist_ok=True)
    
    log_entry = {
        "ts_utc": datetime.utcnow().isoformat(),
        "request": request.dict(),
        "prediction_yen": prediction,
        "model_tag": model_metadata.get("tag", "unknown"),
        "latency_ms": round(latency_ms * 1000, 2)
    }
    
    if request.external_id:
        log_entry["external_id"] = request.external_id
    
    with open(PRED_LOG, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

