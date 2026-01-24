from __future__ import annotations

import re
from dataclasses import dataclass
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")
NON_TEXT_RE = re.compile(r"[^0-9A-Za-z\s\.,!?'\"-]")

ALLOWED_LABELS = {"positive", "negative", "neutral"}

def normalize_label(raw: str) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("positive", "pos"):
        return "positive"
    if s in ("negative", "neg"):
        return "negative"
    if s in ("neutral", "neu"):
        return "neutral"
    # drop anything else (e.g., "irrelevant")
    return None

def clean_text(text: str) -> str:
    t = str(text)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    # keep hashtag words but remove the '#'
    t = t.replace("#", "")
    t = NON_TEXT_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

@dataclass
class Dataset:
    X: pd.Series
    y: pd.Series

def load_dataset(csv_path: str) -> Dataset:
    """Load the Twitter training CSV.

    Expected columns (no header):
      0: id
      1: entity/topic
      2: sentiment label (Positive/Negative/Neutral/Irrelevant)
      3: text
    """
    df = pd.read_csv(csv_path, header=None, names=["id","entity","label","text"], dtype={"id":"int64"}, encoding_errors="ignore")
    df = df.dropna(subset=["text","label"])
    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()
    return Dataset(X=df["clean_text"], y=df["label"])
