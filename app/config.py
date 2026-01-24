from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    data_path: Path = PROJECT_ROOT / "data" / "twitter_training.csv"
    model_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    default_model_type: str = os.getenv("MODEL_TYPE", "best")  # best | baseline | transformer
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

settings = Settings()
