import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="LinguaScope", page_icon="🌍", layout="wide")

MODEL_PATH = "outputs/model.joblib"

NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}

FAMILY = {
    "ar": "Afro-Asiatic (Semitic)", "bg": "Indo-European (Slavic)",
    "de": "Indo-European (Germanic)", "el": "Indo-European (Hellenic)",
    "en": "Indo-European (Germanic)", "es": "Indo-European (Romance)",
    "fr": "Indo-European (Romance)", "hi": "Indo-European (Indo-Aryan)",
    "it": "Indo-European (Romance)", "ja": "Japonic",
    "nl": "Indo-European (Germanic)", "pl": "Indo-European (Slavic)",
    "pt": "Indo-European (Romance)", "ru": "Indo-European (Slavic)",
    "sw": "Niger-Congo (Bantu)", "th": "Kra-Dai",
    "tr": "Turkic", "ur": "Indo-European (Indo-Aryan)",
    "vi": "Austroasiatic", "zh": "Sino-Tibetan",
}

SCRIPTS = [
    ("Devanagari", r"[\u0900-\u097F]"), ("Arabic", r"[\u0600-\u06FF]"),
    ("Cyrillic", r"[\u0400-\u04FF]"), ("Greek", r"[\u0370-\u03FF]"),
    ("Thai", r"[\u0E00-\u0E7F]"), ("Japanese kana", r"[\u3040-\u30FF]"),
    ("Han (Chinese characters)", r"[\u4E00-\u9FFF]"), ("Latin", r"[A-Za-z\u00C0-\u024F]"),
]

EXAMPLES = {
    "English": "The weather is really nice today, let's go for a walk.",
    "Hindi": "आज मौसम बहुत अच्छा है, चलो टहलने चलते हैं।",
    "Urdu": "آج موسم بہت اچھا ہے، چلو سیر کے لیے چلتے ہیں۔",
    "French": "Il fait très beau aujourd'hui, allons nous promener.",
    "German": "Das Wetter ist heute wirklich schön, lass uns spazieren gehen.",
    "Spanish": "Hoy hace muy buen tiempo, vamos a dar un paseo.",
    "Japanese": "今日は天気がとても良いので、散歩に行きましょう。",
}


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


def clean(t):
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def detect_script(text):
    counts = Counter()
    for name, pattern in SCRIPTS:
        counts[name] = len(re.findall(pattern, text))
    name, n = counts.most_common(1)[0]
    return name if n > 0 else "Unknown"


def top_k(text, k=5):
    probs = model.predict_proba(vec.transform([clean(text)]))[0]
    idx = np.argsort(probs)[::-1][:k]
    return [(model.classes_[i], probs[i] * 100) for i in idx]


st.title("🌍 LinguaScope")
st.caption("Multilingual language identification and comparative linguistic analysis")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python linguascope.py` first, then restart this app.")
    st.stop()

bundle = load()
vec, model = bundle["vec"], bundle["model"]

with st.sidebar:
    st.header("About")
    st.write("LinguaScope identifies the language of a text using character "
             "n-gram TF-IDF features and a linear classifier.")
    st.metric("Languages supported", len(model.classes_))
    st.write(", ".join(sorted(NAMES.get(c, c) for c in model.classes_)))
    st.caption("Dataset: papluca/language-identification")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔎 Detect", "⚖️ Compare texts", "📄 Batch CSV", "🌳 Linguistic analysis"])

# ---------------- Tab 1: Detect ----------------
with tab1:
    choice = st.selectbox("Try an example (optional)", ["-- none --"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "")
    text = st.text_area("Type or paste text", value=default, height=120,
                        placeholder="Hello, how are you today?", key=f"t1_{choice}")
    if st.button("Detect", type="primary"):
        if not text.strip():
            st.warning("Please type some text.")
        else:
            top = top_k(text)
            code, p = top[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Detected language", f"{NAMES.get(code, code)} ({code})")
            c2.metric("Confidence", f"{p:.1f}%")
            c3.metric("Script", detect_script(text))
            st.info(f"Language family: **{FAMILY.get(code, 'Unknown')}**")
            if p < 60:
                st.warning("Low confidence. The text may be too short, mixed-language, "
                           "or in a language outside the 20 supported ones.")
            df = pd.DataFrame({
                "Language": [f"{NAMES.get(c, c)} ({c})" for c, _ in top],
                "Confidence %": [round(x, 1) for _, x in top],
            }).set_index("Language")
            st.bar_chart(df)

# ---------------- Tab 2: Compare ----------------
with tab2:
    st.write("Compare two texts using their character n-gram profiles.")
    a, b = st.columns(2)
    ta = a.text_area("Text A", height=120, key="cmp_a")
    tb = b.text_area("Text B", height=120, key="cmp_b")
    if st.button("Compare"):
        if not ta.strip() or not tb.strip():
            st.warning("Please fill in both texts.")
        else:
            sim = cosine_similarity(vec.transform([clean(ta)]),
                                    vec.transform([clean(tb)]))[0, 0]
            (ca, pa), (cb, pb) = top_k(ta, 1)[0], top_k(tb, 1)[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Text A", NAMES.get(ca, ca), f"{pa:.0f}%")
            m2.metric("Text B", NAMES.get(cb, cb), f"{pb:.0f}%")
            m3.metric("Profile similarity", f"{sim:.2f}")
            st.progress(float(min(max(sim, 0.0), 1.0)))
            same_fam = FAMILY.get(ca) == FAMILY.get(cb)
            st.write("Same language family." if same_fam else "Different language families.")

# ---------------- Tab 3: Batch ----------------
with tab3:
    st.write("Upload a CSV with a column named **text**.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            data = None
        if data is not None:
            if "text" not in data.columns:
                st.error("The CSV must contain a column named 'text'.")
            else:
                data = data.head(5000).copy()
                P = model.predict_proba(vec.transform(data["text"].fillna("").map(clean)))
                data["language_code"] = model.classes_[P.argmax(axis=1)]
                data["language"] = data["language_code"].map(lambda c: NAMES.get(c, c))
                data["confidence_%"] = (P.max(axis=1) * 100).round(1)
                st.dataframe(data, hide_index=True)
                st.download_button("Download results", data.to_csv(index=False).encode("utf-8"),
                                   "linguascope_results.csv", "text/csv")

# ---------------- Tab 4: Analysis ----------------
with tab4:
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