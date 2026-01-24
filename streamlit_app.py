from __future__ import annotations

import json
from pathlib import Path
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sentiment Lab", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS = PROJECT_ROOT / "reports"

st.title("Sentiment Lab — Demo + Metrics")

tab1, tab2, tab3 = st.tabs(["Live Demo", "Batch", "Model Report"])

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")
st.sidebar.caption("Run: docker compose up --build")

def call_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status":"down", "error": str(e)}

health = call_health()
st.sidebar.write("Health:", health.get("status","?"))
st.sidebar.write("Model:", health.get("model","?"), health.get("model_version",""))

with tab1:
    st.subheader("Single text")
    txt = st.text_area("Enter text", value="I love this product. It's amazing!", height=120)
    if st.button("Predict"):
        try:
            r = requests.post(f"{API_URL}/predict", json={"text": txt}, timeout=10)
            r.raise_for_status()
            data = r.json()
            st.success(f"Label: {data['prediction']['label']} | Score: {data['prediction']['score']:.3f}")
            st.caption("Cleaned text:")
            st.code(data.get("cleaned_text",""))
        except Exception as e:
            st.error(str(e))

with tab2:
    st.subheader("Batch prediction")
    batch = st.text_area("One per line", value="This is terrible\nI am okay with it\nBest thing ever", height=160)
    if st.button("Predict batch"):
        texts = [x.strip() for x in batch.splitlines() if x.strip()]
        try:
            r = requests.post(f"{API_URL}/predict/batch", json={"texts": texts}, timeout=20)
            r.raise_for_status()
            preds = r.json()["predictions"]
            st.dataframe(pd.DataFrame(preds))
        except Exception as e:
            st.error(str(e))

with tab3:
    st.subheader("Evaluation artifacts")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Baseline")
        p = REPORTS / "baseline_report.json"
        if p.exists():
            st.json(json.loads(p.read_text(encoding="utf-8")))
        img = REPORTS / "baseline_confusion.png"
        if img.exists():
            st.image(str(img), caption="Baseline Confusion Matrix")
        toks = REPORTS / "baseline_tokens.json"
        if toks.exists():
            st.markdown("**Top tokens (interpretability)**")
            st.json(json.loads(toks.read_text(encoding="utf-8")))

    with cols[1]:
        st.markdown("### Transformer")
        p = REPORTS / "transformer_report.json"
        if p.exists():
            st.json(json.loads(p.read_text(encoding="utf-8")))
        img = REPORTS / "transformer_confusion.png"
        if img.exists():
            st.image(str(img), caption="Transformer Confusion Matrix")

    best = REPORTS / "best_model.json"
    if best.exists():
        st.markdown("### Best model selection")
        st.json(json.loads(best.read_text(encoding="utf-8")))
