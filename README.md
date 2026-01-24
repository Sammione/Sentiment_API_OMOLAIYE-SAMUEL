# Sentiment Analysis API (Baseline + DistilBERT) + Streamlit Demo

This repository implements a production-style sentiment classifier with:
- **Baseline model**: TF-IDF + Logistic Regression
- **Transformer model**: DistilBERT fine-tuned for 3-class sentiment (**positive / negative / neutral**)
- **FastAPI** inference service with `/predict`, `/predict/batch`, `/health`
- **Docker + docker-compose** for one-command setup
- **Unit tests**
- **Streamlit** dashboard for live demo + metrics + interpretability

Dataset used: `data/twitter_training.csv` (Twitter sentiment training data). We **drop "Irrelevant"** and keep **Positive/Negative/Neutral**.

---

## Quickstart (recommended)

### 1) Run API + Streamlit with Docker
```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

### 2) Health check
```bash
curl http://localhost:8000/health
```

---

## Train models locally

Create a virtual env, then:

```bash
pip install -r requirements.txt
python -m app.train --model both --epochs 2
```

Artifacts written to:
- `models/` (saved models)
- `reports/` (metrics + confusion matrices + interpretability)

The best model is selected by **Macro F1** and saved in `reports/best_model.json`.

---

## API usage

### Single prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this!"}'
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love this!","This is bad","It is okay"]}'
```

---

## Tests
```bash
pytest -q
```

---

## “Extra” that helps you stand out
- **Interpretability for baseline**: `reports/baseline_tokens.json` includes top tokens driving each class.
- **Best-model auto-selection**: set `MODEL_TYPE=best` (default) and the API picks the best saved model.

You can switch explicitly:
```bash
MODEL_TYPE=baseline uvicorn app.api:app --reload
MODEL_TYPE=transformer uvicorn app.api:app --reload
```


---

## CI (GitHub Actions)

This repo includes a simple CI pipeline that:
- lints with **ruff**
- runs **pytest**
- verifies **docker build**

Workflow file: `.github/workflows/ci.yml`

---

## Handy commands

```bash
make install
make test
make train
make docker
```

---

## Bonus Features (Implemented)

### Experiment Tracking with MLflow
All training runs are now tracked with **MLflow**.
- **Metrics**: Accuracy, F1 Score
- **Parameters**: Hyperparameters
- **Artifacts**: Confusion matrices, reports, ONNX models

To view the dashboard:
```bash
mlflow ui --backend-store-uri reports/mlruns
```

### Model Optimization (ONNX)
The Transformer model (DistilBERT) is automatically exported to **ONNX** format after training.
- Location: `models/distilbert/model.onnx`
- Benefits: Faster inference on CPU, cross-platform compatibility.
