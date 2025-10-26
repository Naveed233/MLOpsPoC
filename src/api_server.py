#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_server.py
- Loads the current production model (models/production/model.joblib)
- Exposes POST /predict_price
- Validates and engineers minimal features to match the training pipeline
- Logs each prediction to data/predictions/predictions.jsonl
Run:
  uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import os
import json
import time
import joblib
import threading
from datetime import datetime
from typing import Optional, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD_MODEL_PATH = os.path.join(BASE_DIR, "models", "production", "model.joblib")
PROD_META_PATH = os.path.join(BASE_DIR, "models", "production", "metadata.json")
PRED_LOG_DIR = os.path.join(BASE_DIR, "data", "predictions")
os.makedirs(PRED_LOG_DIR, exist_ok=True)

# Optional API key (set API_KEY env var to enforce)
API_KEY_VALUE = os.environ.get("API_KEY", "").strip()

# Feature sets MUST match the training pipeline
CATEGORICAL_FEATURES = [
    "prefecture",
    "city_ward",
    "property_type",
    "building_structure",
    "age_bucket",
]
NUMERIC_FEATURES = [
    "effective_area_m2",
    "building_age_years",
    "coverage_ratio",
    "floor_area_ratio",
    "is_tokyo",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ---------- Pydantic schemas ----------
PropertyType = Literal["マンション", "戸建て", "土地"]
StructureType = Literal["RC", "SRC", "木造", "鉄骨"]


class PredictRequest(BaseModel):
    external_id: Optional[str] = Field(
        None, description="Client-provided identifier used later to match actual sale price"
    )
    prefecture: str = Field(..., description="例: 東京都, 千葉県")
    city_ward: str = Field(..., description="例: 渋谷区, 千葉市中央区")
    property_type: PropertyType
    building_structure: StructureType
    building_age_years: float = Field(..., ge=0)
    effective_area_m2: float = Field(
        ..., gt=0, description="マンション=延床面積, 戸建て/土地=土地面積の近似でも可"
    )
    coverage_ratio: Optional[float] = Field(None, ge=0, le=100)
    floor_area_ratio: Optional[float] = Field(None, ge=0, le=1000)


class PredictResponse(BaseModel):
    predicted_price_yen: float
    model_tag: str
    model_saved_at_utc: str
    latency_ms: int


# ---------- Load model ----------
_model_lock = threading.Lock()
_model = None
_model_meta = None


def _load_model():
    global _model, _model_meta
    if not os.path.exists(PROD_MODEL_PATH):
        raise RuntimeError("Production model not found. Train a model and promote it first.")
    _model = joblib.load(PROD_MODEL_PATH)
    if os.path.exists(PROD_META_PATH):
        with open(PROD_META_PATH, "r", encoding="utf-8") as f:
            _model_meta = json.load(f)
    else:
        _model_meta = {"tag": "unknown", "saved_at_utc": "unknown"}


def _ensure_model_loaded():
    if _model is None:
        with _model_lock:
            if _model is None:
                _load_model()


# ---------- FastAPI app ----------
app = FastAPI(title="Price Estimation API", version="0.1.0")


@app.on_event("startup")
def startup_event():
    _ensure_model_loaded()


@app.get("/health")
def health():
    try:
        _ensure_model_loaded()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/version")
def version():
    _ensure_model_loaded()
    return {
        "model_tag": _model_meta.get("tag", "unknown"),
        "saved_at_utc": _model_meta.get("saved_at_utc", "unknown"),
    }


def _age_bucket(age: float) -> str:
    if age is None or np.isnan(age):
        return "20+"
    if age <= 5:
        return "0-5"
    if age <= 10:
        return "5-10"
    if age <= 20:
        return "10-20"
    return "20+"


def _to_dataframe(req: PredictRequest) -> pd.DataFrame:
    # engineer derived features exactly like in training
    is_tokyo = 1 if ("東京" in req.prefecture) else 0
    age_bucket = _age_bucket(req.building_age_years)
    df = pd.DataFrame(
        [
            {
                "prefecture": req.prefecture,
                "city_ward": req.city_ward,
                "property_type": req.property_type,
                "building_structure": req.building_structure,
                "age_bucket": age_bucket,
                "effective_area_m2": req.effective_area_m2,
                "building_age_years": req.building_age_years,
                "coverage_ratio": req.coverage_ratio if req.coverage_ratio is not None else 60.0,
                "floor_area_ratio": (
                    req.floor_area_ratio if req.floor_area_ratio is not None else 200.0
                ),
                "is_tokyo": is_tokyo,
            }
        ]
    )
    # Ensure column order and presence
    for col in ALL_FEATURES:
        if col not in df.columns:
            # fill any missing numeric with 0, categorical with '不明'
            df[col] = 0 if col in NUMERIC_FEATURES else "不明"
    return df[ALL_FEATURES]


def _auth_guard(x_api_key: Optional[str]) -> None:
    if not API_KEY_VALUE:
        return
    if not x_api_key or x_api_key != API_KEY_VALUE:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _log_prediction(request_dict: dict, pred: float, latency_ms: int):
    log_path = os.path.join(PRED_LOG_DIR, "predictions.jsonl")
    rec = {
        "ts_utc": datetime.utcnow().isoformat(),
        "request": request_dict,
        "prediction_yen": float(pred),
        "model_tag": _model_meta.get("tag"),
        "latency_ms": latency_ms,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@app.post("/predict_price", response_model=PredictResponse)
def predict_price(req: PredictRequest, x_api_key: Optional[str] = Header(default=None)):
    _auth_guard(x_api_key)
    _ensure_model_loaded()

    t0 = time.time()
    try:
        X = _to_dataframe(req)
        yhat = _model.predict(X)
        pred = float(yhat[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")
    latency_ms = int((time.time() - t0) * 1000)

    # async-safe file append
    _log_prediction(req.dict(), pred, latency_ms)

    return PredictResponse(
        predicted_price_yen=pred,
        model_tag=_model_meta.get("tag", "unknown"),
        model_saved_at_utc=_model_meta.get("saved_at_utc", "unknown"),
        latency_ms=latency_ms,
    )
