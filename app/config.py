from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    # Data paths
    data_path: Path = PROJECT_ROOT / "data" / "twitter_training.csv"
    model_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    
    # Model configuration
    default_model_type: str = os.getenv("MODEL_TYPE", "best")  # best | baseline | transformer
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    
    # Label configuration
    labels: tuple = ("negative", "neutral", "positive")
    
    # Baseline model hyperparameters
    baseline_ngram_range: tuple = (1, 2)
    baseline_min_df: int = 2
    baseline_max_df: float = 0.95
    baseline_sublinear_tf: bool = True
    baseline_max_iter: int = 2000
    baseline_class_weight: str = "balanced"
    baseline_n_jobs: int = None
    
    # Transformer model hyperparameters
    transformer_model_name: str = "distilbert-base-uncased"
    transformer_epochs: int = 2
    transformer_batch_size: int = 16
    transformer_lr: float = 2e-5
    transformer_max_length: int = 128
    transformer_logging_steps: int = 50
    
    # Evaluation and visualization
    plot_figsize: tuple = (6, 5)
    plot_fraction: float = 0.046
    plot_pad: float = 0.04
    
    # MLflow configuration
    mlflow_tracking_uri: str = "file:./reports/mlruns"
    mlflow_hf_checkpoints_dir: str = "reports/hf_checkpoints"
    
    # API configuration
    default_api_url: str = "http://localhost:8000"

settings = Settings()
