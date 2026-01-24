from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

@dataclass
class EvalReport:
    accuracy: float
    f1_macro: float
    labels: List[str]
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Dict[str, float]]

def evaluate(y_true, y_pred, labels: List[str]) -> EvalReport:
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return EvalReport(
        accuracy=acc,
        f1_macro=f1m,
        labels=labels,
        confusion_matrix=cm.tolist(),
        classification_report={k: {kk: float(vv) for kk, vv in v.items()} for k, v in rep.items() if isinstance(v, dict)},
    )

def save_report(report: EvalReport, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
