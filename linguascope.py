"""
LinguaScope: Multilingual Language Identification + Comparative Linguistic Analysis
Run:  python linguascope.py
Needs: data/train.csv and data/test.csv (columns: labels, text)
"""

import os
import re
import random

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# ---------------------------------------------------------------
# STEP 2: Clean data
# ---------------------------------------------------------------
def clean(df):
    df = df.dropna(subset=["text", "labels"]).copy()
    df["text"] = (
        df["text"].astype(str)
        .str.replace(r"http\S+", " ", regex=True)   # remove links
        .str.replace(r"\d+", " ", regex=True)       # remove numbers
        .str.replace(r"\s+", " ", regex=True)       # collapse spaces
        .str.strip()
        .str.lower()
    )
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)

train = clean(train)
test = clean(test)

print("Train:", train.shape, "Test:", test.shape)
print(train["labels"].value_counts().head(), "\n")

# ---------------------------------------------------------------
# STEP 3: Short-text augmentation (TRAINING DATA ONLY)
# The dataset has long review-style text, so the model fails on
# short inputs like "hello how are you". We add short snippets.
# `train` stays untouched (used for stats/similarity);
# `train_aug` is used for training.
# ---------------------------------------------------------------
def make_short(df):
    rows = []
    for t, l in zip(df["text"], df["labels"]):
        words = t.split()
        if len(words) >= 4:                       # languages with spaces
            k = random.randint(1, min(6, len(words)))
            s = random.randint(0, len(words) - k)
            snippet = " ".join(words[s:s + k])
        else:                                     # ja / zh / th (no spaces)
            k = random.randint(3, 15)
            s = random.randint(0, max(0, len(t) - k))
            snippet = t[s:s + k]
        rows.append((snippet, l))
    return pd.DataFrame(rows, columns=["text", "labels"])

train_aug = pd.concat([train, make_short(train)], ignore_index=True)

# ---------------------------------------------------------------
# STEP 4: Train models (character n-gram TF-IDF)
# ---------------------------------------------------------------
vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 3),
    max_features=200000,
    sublinear_tf=True,
)
X_train = vec.fit_transform(train_aug["text"])
y_train = train_aug["labels"]
X_test = vec.transform(test["text"])
y_test = test["labels"]

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
nb = MultinomialNB(alpha=0.01)
nb.fit(X_train, y_train)

acc_lr = accuracy_score(y_test, lr.predict(X_test))
acc_nb = accuracy_score(y_test, nb.predict(X_test))
print(f"Logistic Regression accuracy: {acc_lr:.4f}")
print(f"Naive Bayes accuracy: {acc_nb:.4f}")

best_name, best = ("Logistic Regression", lr) if acc_lr >= acc_nb else ("Naive Bayes", nb)
print("\nBest model:", best_name)

# ---------------------------------------------------------------
# STEP 5: Evaluation
# ---------------------------------------------------------------
pred = best.predict(X_test)
print(classification_report(y_test, pred))

labels_sorted = sorted(y_test.unique())
cm = confusion_matrix(y_test, pred, labels=labels_sorted)
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_sorted, yticklabels=labels_sorted)
plt.title(f"Confusion Matrix ({best_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# Extra: accuracy on SHORT test snippets (evaluation only, not training)
short_test = make_short(test)
short_acc = accuracy_score(short_test["labels"],
                           best.predict(vec.transform(short_test["text"])))
print(f"Short-text accuracy (1-6 words, test only): {short_acc:.4f}\n")

# ---------------------------------------------------------------
# STEP 6: Comparative linguistics
# ---------------------------------------------------------------
# 6a. Basic statistics per language (from original, non-augmented train)
stats = (
    train.assign(
        words=train["text"].str.split(),
    )
    .assign(
        avg_word_length=lambda d: d["words"].apply(
            lambda w: np.mean([len(x) for x in w]) if w else 0),
        words_per_sentence=lambda d: d["words"].apply(len),
    )
    .groupby("labels")[["avg_word_length", "words_per_sentence"]]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={"labels": "language",
                     "words_per_sentence": "avg_words_per_sentence"})
)
stats.to_csv("outputs/language_stats.csv", index=False)
print(stats, "\n")

# 6b. Language similarity (cosine similarity of average TF-IDF profiles)
X_orig = vec.transform(train["text"])
langs = sorted(train["labels"].unique())
centroids = np.vstack([
    np.asarray(X_orig[(train["labels"] == lg).values].mean(axis=0)).ravel()
    for lg in langs
])
sim = cosine_similarity(centroids)

sim_df = pd.DataFrame(sim, index=langs, columns=langs)
plt.figure(figsize=(11, 9))
sns.heatmap(sim_df, cmap="viridis", annot=False)
plt.title("Language Similarity (cosine similarity of character n-gram profiles)")
plt.tight_layout()
plt.savefig("outputs/similarity_heatmap.png", dpi=150)
plt.close()

# 6c. Language family tree (hierarchical clustering)
dist = 1 - sim
np.fill_diagonal(dist, 0)
dist = np.clip((dist + dist.T) / 2, 0, None)
Z = linkage(squareform(dist, checks=False), method="average")

plt.figure(figsize=(12, 6))
dendrogram(Z, labels=langs, leaf_font_size=12)
plt.title("Language Tree (hierarchical clustering)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("outputs/language_tree.png", dpi=150)
plt.close()

# 6d. Most distinctive character n-grams per language (Logistic Regression)
feature_names = np.array(vec.get_feature_names_out())
for i, lg in enumerate(lr.classes_):
    top = np.argsort(lr.coef_[i])[-8:][::-1]
    print(f"{lg}: {list(feature_names[top])}")

# ---------------------------------------------------------------
# STEP 7: Save model for the web app
# ---------------------------------------------------------------
joblib.dump({"vec": vec, "model": best, "model_name": best_name},
            "outputs/model.joblib")

# ---------------------------------------------------------------
# STEP 8: Prediction interface (terminal)
# ---------------------------------------------------------------
def predict(text, top_n=3):
    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    probs = best.predict_proba(vec.transform([text]))[0]
    idx = np.argsort(probs)[::-1][:top_n]
    return [(best.classes_[i], probs[i] * 100) for i in idx]


if __name__ == "__main__":
    print("\nLinguaScope ready. Type text (or 'quit').")
    while True:
        try:
            s = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if s.strip().lower() == "quit":
            break
        if not s.strip():
            continue
        for lang, p in predict(s):
            print(f"  {lang}: {p:.1f}%")