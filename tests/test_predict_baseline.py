from app.models_baseline import build_baseline_model
from app.predict import ModelBundle, predict_one


def test_predict_one_baseline():
    model = build_baseline_model()
    X = ["i love this", "this is awful", "it is okay"]
    y = ["positive", "negative", "neutral"]
    model.fit(X, y)
    bundle = ModelBundle(model_type="baseline", model=model, version="test")

    label, score, cleaned = predict_one(bundle, "I LOVE this!!!")
    assert label in {"positive", "negative", "neutral"}
    assert 0.0 <= score <= 1.0
    assert len(cleaned) > 0
