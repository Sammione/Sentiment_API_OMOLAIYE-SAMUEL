from app.preprocess import normalize_label, clean_text

def test_normalize_label():
    assert normalize_label("Positive") == "positive"
    assert normalize_label("negative") == "negative"
    assert normalize_label("Neutral") == "neutral"
    assert normalize_label("Irrelevant") is None

def test_clean_text_basic():
    t = clean_text("Check this https://example.com @user #Tag!!!")
    assert "http" not in t
    assert "@user" not in t
    assert "Tag" in t or "tag" in t
