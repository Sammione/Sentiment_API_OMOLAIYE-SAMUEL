from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import settings

# Use labels from config for consistency
LABELS = list(settings.labels)

def build_baseline_model() -> Pipeline:
    """Strong baseline: TF-IDF + Logistic Regression (balanced)"""
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=settings.baseline_ngram_range,
            min_df=settings.baseline_min_df,
            max_df=settings.baseline_max_df,
            sublinear_tf=settings.baseline_sublinear_tf
        )),
        ("clf", LogisticRegression(
            max_iter=settings.baseline_max_iter,
            class_weight=settings.baseline_class_weight,
            n_jobs=settings.baseline_n_jobs
        ))
    ])

def top_tokens_per_class(model: Pipeline, top_k: int = 15) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """Interpretability: top positive/negative tokens per class for the linear model."""
    vec: TfidfVectorizer = model.named_steps["tfidf"]
    clf: LogisticRegression = model.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_  # shape (n_classes, n_features)
    out: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for i, label in enumerate(clf.classes_):
        w = coefs[i]
        top_pos_idx = np.argsort(w)[-top_k:][::-1]
        top_neg_idx = np.argsort(w)[:top_k]
        out[str(label)] = {
            "top_positive": [(feature_names[j], float(w[j])) for j in top_pos_idx],
            "top_negative": [(feature_names[j], float(w[j])) for j in top_neg_idx],
        }
    return out
