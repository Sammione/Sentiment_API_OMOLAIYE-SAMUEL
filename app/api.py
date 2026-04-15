from __future__ import annotations

import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictRequest, BatchPredictRequest,
    PredictResponse, BatchPredictResponse,
    Prediction, HealthResponse
)
from .predict import load_model, predict_one, predict_batch, ModelBundle
from .config import settings

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-safe singleton for model loading
class ModelSingleton:
    """Thread-safe singleton for loading and caching the model bundle."""
    
    _instance: Optional["ModelSingleton"] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> "ModelSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        self._bundle: Optional[ModelBundle] = None
        self._load_lock = threading.Lock()
        self._initialized = True
    
    def get_bundle(self) -> ModelBundle:
        """Get the model bundle, loading it if necessary (thread-safe)."""
        if self._bundle is None:
            with self._load_lock:
                if self._bundle is None:
                    self._bundle = load_model(settings.default_model_type)
        return self._bundle

# Global singleton instance
_model_singleton = ModelSingleton()

def get_bundle() -> ModelBundle:
    """Get the cached model bundle (thread-safe)."""
    return _model_singleton.get_bundle()

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        bundle = get_bundle()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model=bundle.model_type,
            model_version=bundle.version,
            details={"model_type_env": settings.default_model_type},
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model="unknown",
            model_version="unknown",
            details={"error": str(e)},
        )

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        bundle = get_bundle()
        label, score, cleaned = predict_one(bundle, req.text)
        pred = Prediction(label=label, score=score, model=bundle.model_type, model_version=bundle.version)
        return PredictResponse(prediction=pred, cleaned_text=cleaned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(req: BatchPredictRequest):
    try:
        bundle = get_bundle()
        preds = []
        for label, score, _ in predict_batch(bundle, req.texts):
            preds.append(Prediction(label=label, score=score, model=bundle.model_type, model_version=bundle.version))
        return BatchPredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
