import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LinguaScope", page_icon="🌍", layout="centered")

MODEL_PATH = "outputs/model.joblib"

NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


st.title("🌍 LinguaScope")
st.caption("Multilingual language identification and comparative linguistic analysis")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python linguascope.py` first, then restart this app.")
    st.stop()

bundle = load()
vec, model = bundle["vec"], bundle["model"]

st.subheader("Detect a language")
text = st.text_area("Type or paste text", height=120,
                    placeholder="Hello, how are you today?")

if st.button("Detect"):
    if not text.strip():
        st.warning("Please type some text.")
    else:
        t = re.sub(r"\s+", " ", text).strip().lower()
        probs = model.predict_proba(vec.transform([t]))[0]
        idx = np.argsort(probs)[::-1][:5]
        top = [(model.classes_[i], probs[i] * 100) for i in idx]

        code, p = top[0]
        st.success(f"Detected: **{NAMES.get(code, code)}** ({code}) with {p:.1f}% confidence")
        df = pd.DataFrame({
            "Language": [f"{NAMES.get(c, c)} ({c})" for c, _ in top],
            "Confidence %": [round(x, 1) for _, x in top],
        }).set_index("Language")
        st.bar_chart(df)

st.divider()
st.subheader("Comparative linguistics")

for title, path, note in [
    ("Language tree", "outputs/language_tree.png",
     "Languages from the same family join early (e.g. Spanish, Portuguese, Italian)."),
    ("Similarity heatmap", "outputs/similarity_heatmap.png",
     "Brighter cells mean more similar character n-gram profiles."),
    ("Confusion matrix", "outputs/confusion_matrix.png",
     "Off-diagonal cells show which languages the model mixes up."),
]:
    if os.path.exists(path):
        st.markdown(f"**{title}**")
        st.image(path, caption=note)

if os.path.exists("outputs/language_stats.csv"):
    st.markdown("**Language statistics**")
    st.dataframe(pd.read_csv("outputs/language_stats.csv"), hide_index=True)