from __future__ import annotations

from typing import List

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["negative", "neutral", "positive"]

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

def train_distilbert(
    X_train: List[str],
    y_train: List[str],
    X_val: List[str],
    y_val: List[str],
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
):
    label2id = {lab: i for i, lab in enumerate(LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = HFDataset.from_dict({"text": X_train, "label": [label2id[y] for y in y_train]}).map(tokenize, batched=True)
    val_ds = HFDataset.from_dict({"text": X_val, "label": [label2id[y] for y in y_val]}).map(tokenize, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="reports/hf_checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    return model, tokenizer, {"labels": LABELS, "model_name": model_name}
