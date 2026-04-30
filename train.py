"""
PaperSense – train.py
Run this once: python train.py
Generates model/classifier.pkl and model/vectorizer.pkl
"""

import os, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("model", exist_ok=True)

# ── Load dataset ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/abstracts.csv")
print(f"  Dataset: {len(df)} samples across {df['label'].nunique()} domains")
print(df["label"].value_counts().to_string())

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ── Vectoriser ─────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    min_df=1,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Classifier ─────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=2000, C=4.0, solver="lbfgs", random_state=42)
model.fit(X_train_vec, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
preds = model.predict(X_test_vec)
acc   = accuracy_score(y_test, preds)
print(f"\n{'─'*50}")
print(f"  PaperSense Model Training Complete")
print(f"{'─'*50}")
print(f"  Test Accuracy : {acc*100:.1f}%")
print(f"\n{classification_report(y_test, preds)}")

# ── Save ───────────────────────────────────────────────────────────────────────
pickle.dump(model,      open("model/classifier.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl",  "wb"))
print("  Saved → model/classifier.pkl")
print("  Saved → model/vectorizer.pkl")
print(f"{'─'*50}\n")