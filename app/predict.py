from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, List

import joblib
import numpy as np

from .preprocess import clean_text
from .config import settings

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

@dataclass
class ModelBundle:
    model_type: str
    model: Any
    tokenizer: Any = None
    version: str = "1.0.0"

def _baseline_paths() -> Tuple[Path, Path]:
    return settings.model_dir / "baseline.joblib", settings.reports_dir / "baseline_meta.json"

def _transformer_paths() -> Tuple[Path, Path]:
    return settings.model_dir / "distilbert", settings.reports_dir / "transformer_meta.json"

def load_model(model_type: str = "best") -> ModelBundle:
    """Load a trained model from disk.

    model_type:
      - baseline: scikit-learn pipeline
      - transformer: HF model folder
      - best: auto-pick best_model.json if available, else transformer if exists, else baseline
    
    Raises:
        FileNotFoundError: If the requested model files are not found.
        RuntimeError: If transformer dependencies are not available.
        ValueError: If an unknown model type is provided.
    """
    if model_type == "best":
        best_path = settings.reports_dir / "best_model.json"
        if best_path.exists():
            import json
            d = json.loads(best_path.read_text(encoding="utf-8"))
            model_type = d.get("model_type", "transformer")
        else:
            # fallback preference: transformer > baseline
            if (settings.model_dir / "distilbert").exists():
                model_type = "transformer"
            else:
                model_type = "baseline"

    if model_type == "baseline":
        model_path, meta_path = _baseline_paths()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found at {model_path}. "
                "Please train the model first using: python -m app.train --model baseline"
            )
        model = joblib.load(model_path)
        version = "baseline-1"
        return ModelBundle(model_type="baseline", model=model, version=version)

    if model_type == "transformer":
        if torch is None:
            raise RuntimeError("Transformer dependencies not available. Install requirements.txt.")
        model_dir, meta_path = _transformer_paths()
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Transformer model not found at {model_dir}. "
                "Please train the model first using: python -m app.train --model transformer"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        version = "distilbert-1"
        return ModelBundle(model_type="transformer", model=model, tokenizer=tokenizer, version=version)

    raise ValueError(f"Unknown model_type: {model_type}")

def predict_one(bundle: ModelBundle, text: str) -> Tuple[str, float, str]:
    cleaned = clean_text(text)
    if bundle.model_type == "baseline":
        proba = bundle.model.predict_proba([cleaned])[0]
        classes = list(bundle.model.classes_)
        idx = int(np.argmax(proba))
        return classes[idx], float(proba[idx]), cleaned

    # transformer
    import torch
    inputs = bundle.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=settings.transformer_max_length)
    with torch.no_grad():
        outputs = bundle.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    label = bundle.model.config.id2label.get(idx, str(idx))
    return str(label), float(probs[idx]), cleaned

def predict_batch(bundle: ModelBundle, texts: List[str]) -> List[Tuple[str, float, str]]:
    return [predict_one(bundle, t) for t in texts]
