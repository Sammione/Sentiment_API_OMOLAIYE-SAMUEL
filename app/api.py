from __future__ import annotations

from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictRequest, BatchPredictRequest,
    PredictResponse, BatchPredictResponse,
    Prediction, HealthResponse
)
from .predict import load_model, predict_one, predict_batch
from .config import settings

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def get_bundle():
    return load_model(settings.default_model_type)

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
