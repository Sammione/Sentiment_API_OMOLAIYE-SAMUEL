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

from .config import settings

# Use labels from config for consistency
LABELS = list(settings.labels)

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
    model_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    seed: int | None = None,
):
    # Use config defaults if not provided
    model_name = model_name or settings.transformer_model_name
    epochs = epochs if epochs is not None else settings.transformer_epochs
    batch_size = batch_size if batch_size is not None else settings.transformer_batch_size
    lr = lr if lr is not None else settings.transformer_lr
    seed = seed if seed is not None else settings.random_seed
    
    label2id = {lab: i for i, lab in enumerate(LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=settings.transformer_max_length)

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
        output_dir=settings.mlflow_hf_checkpoints_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=settings.transformer_logging_steps,
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
