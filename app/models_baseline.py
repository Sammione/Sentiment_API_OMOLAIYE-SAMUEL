from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABELS = ["negative", "neutral", "positive"]  # fixed order for reports

def build_baseline_model() -> Pipeline:
    # Strong baseline: TF-IDF + Logistic Regression (balanced)
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None
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
