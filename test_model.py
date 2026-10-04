import os
import joblib
import pytest

MODEL = "outputs/model.joblib"
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="train the model first")


@pytest.mark.parametrize("text,code", [
    ("Hello, how are you today my friend?", "en"),
    ("Bonjour, comment allez-vous aujourd'hui ?", "fr"),
    ("आप कैसे हैं आज", "hi"),
    ("Wie geht es dir heute?", "de"),
])
def test_detects_language(text, code):
    b = joblib.load(MODEL)
    assert b["model"].predict(b["vec"].transform([text.lower()]))[0] == code