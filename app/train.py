from __future__ import annotations

import argparse
import json
from typing import Dict
import shutil

import joblib
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split

from .config import settings
from .preprocess import load_dataset
from .evaluate import evaluate, save_report, plot_confusion_matrix
from .models_baseline import build_baseline_model, top_tokens_per_class

def train_baseline() -> Dict[str, float]:
    mlflow.set_experiment("Sentiment_Baseline")
    with mlflow.start_run(run_name="TFIDF_LogReg"):
        ds = load_dataset(str(settings.data_path))
        X_train, X_test, y_train, y_test = train_test_split(
            ds.X, ds.y, test_size=0.2, random_state=settings.random_seed, stratify=ds.y
        )

        model = build_baseline_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = evaluate(y_test, y_pred, labels=sorted(ds.y.unique()))
        save_report(report, settings.reports_dir / "baseline_report.json")
        plot_confusion_matrix(np.array(report.confusion_matrix), report.labels, settings.reports_dir / "baseline_confusion.png")

        # Save model
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, settings.model_dir / "baseline.joblib")

        # Interpretability artifact
        toks = top_tokens_per_class(model, top_k=20)
        (settings.reports_dir / "baseline_tokens.json").write_text(json.dumps(toks, indent=2), encoding="utf-8")

        # MLflow logging
        mlflow.log_params({"model_type": "baseline", "vectorizer": "tfidf", "classifier": "logistic_regression"})
        mlflow.log_metrics({"accuracy": report.accuracy, "f1_macro": report.f1_macro})
        mlflow.log_artifact(settings.reports_dir / "baseline_report.json")
        mlflow.log_artifact(settings.reports_dir / "baseline_confusion.png")
        mlflow.log_artifact(settings.reports_dir / "baseline_tokens.json")
        mlflow.sklearn.log_model(model, "model")

        return {"accuracy": report.accuracy, "f1_macro": report.f1_macro}

def train_transformer(epochs: int | None = None) -> Dict[str, float]:
    from .models_transformer import train_distilbert, LABELS as T_LABELS
    from .utils_onnx import export_to_onnx

    mlflow.set_experiment("Sentiment_Transformer")
    with mlflow.start_run(run_name="DistilBERT"):
        ds = load_dataset(str(settings.data_path))
        X_train, X_test, y_train, y_test = train_test_split(
            ds.X.tolist(), ds.y.tolist(), test_size=0.2, random_state=settings.random_seed, stratify=ds.y
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=settings.random_seed, stratify=y_train
        )

        # Use config default if epochs not provided
        epochs = epochs if epochs is not None else settings.transformer_epochs

        model, tokenizer, meta = train_distilbert(
            X_tr, y_tr, X_val, y_val,
            epochs=epochs,
            seed=settings.random_seed
        )

        # Evaluate on test
        # batched inference
        import numpy as np
        import torch
        label2id = {lab: i for i, lab in enumerate(T_LABELS)}
        id2label = {i: lab for lab, i in label2id.items()}

        def predict_texts(texts):
            preds=[]
            probs=[]
            for t in texts:
                inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=settings.transformer_max_length)
                with torch.no_grad():
                    out = model(**inputs)
                    p = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
                idx = int(np.argmax(p))
                preds.append(id2label[idx])
                probs.append(float(p[idx]))
            return preds, probs

        y_pred, y_score = predict_texts(X_test)
        # Build eval report using our evaluate helper for consistency
        from .evaluate import evaluate as eval_fn
        report = eval_fn(y_test, y_pred, labels=T_LABELS)
        save_report(report, settings.reports_dir / "transformer_report.json")
        plot_confusion_matrix(np.array(report.confusion_matrix), report.labels, settings.reports_dir / "transformer_confusion.png")

        # Save HF model
        model_dir = settings.model_dir / "distilbert"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        (settings.reports_dir / "transformer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Export to ONNX
        onnx_path = model_dir / "model.onnx"
        print("Exporting to ONNX...")
        export_to_onnx(model, tokenizer, onnx_path)

        # MLflow logging
        mlflow.log_params({"model_type": "distilbert", "epochs": epochs})
        mlflow.log_metrics({"accuracy": report.accuracy, "f1_macro": report.f1_macro})
        mlflow.log_artifact(settings.reports_dir / "transformer_report.json")
        mlflow.log_artifact(settings.reports_dir / "transformer_confusion.png")
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        
        # Log HF model using pytorch flavor
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            tokenizer=tokenizer,
        )

        return {"accuracy": report.accuracy, "f1_macro": report.f1_macro}

def choose_best(baseline_metrics: Dict[str,float], transformer_metrics: Dict[str,float]) -> None:
    best = "transformer" if transformer_metrics["f1_macro"] >= baseline_metrics["f1_macro"] else "baseline"
    out = {"model_type": best, "baseline": baseline_metrics, "transformer": transformer_metrics}
    (settings.reports_dir / "best_model.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Train sentiment models.")
    parser.add_argument("--model", choices=["baseline","transformer","both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    # Set MLflow tracking URI from config
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = {"accuracy": 0.0, "f1_macro": 0.0}
    transformer_metrics = {"accuracy": 0.0, "f1_macro": 0.0}

    if args.model in ("baseline","both"):
        baseline_metrics = train_baseline()

    if args.model in ("transformer","both"):
        transformer_metrics = train_transformer(epochs=args.epochs)

    if args.model == "both":
        choose_best(baseline_metrics, transformer_metrics)

    print("Done.")
    print("Baseline:", baseline_metrics)
    print("Transformer:", transformer_metrics)

if __name__ == "__main__":
    main()
