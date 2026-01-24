from __future__ import annotations

from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Dict

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify.")

class BatchPredictRequest(BaseModel):
    texts: conlist(str, min_length=1) = Field(..., description="List of texts to classify.")

class Prediction(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    model: str
    model_version: str

class PredictResponse(BaseModel):
    prediction: Prediction
    cleaned_text: Optional[str] = None

class BatchPredictResponse(BaseModel):
    predictions: List[Prediction]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
    model_version: str
    details: Dict[str, str] = {}
