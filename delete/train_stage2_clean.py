"""
Train Stage 2 (LCPN) Models for MRO Classifier
-----------------------------------------------
Trains per-level classification models (Level1–3)
using embeddings from labeled_training.xlsx
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path

# Paths
DATA_PATH = Path("data/labeled_training.xlsx")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

print("[Stage2] Loading labeled data...")
df = pd.read_excel(DATA_PATH)

df = df.dropna(subset=["Item Description", "Level 1"]).copy()
df["Item Description"] = df["Item Description"].astype(str).str.strip()

# Load model
print("[Stage2] Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("[Stage2] Encoding item descriptions...")
embeddings = model.encode(df["Item Description"].tolist(), show_progress_bar=True)

# Helper to train per level
def train_level(level_col, X, df):
    if level_col not in df.columns:
        print(f"⚠️ Skipping {level_col} — column not found.")
        return None
    y = df[level_col].astype(str)
    classes = np.unique(y)
    if len(classes) < 2:
        print(f"⚠️ Skipping {level_col} — only one class present.")
        return None
    weights = compute_class_weight("balanced", classes=classes, y=y)
    clf = LogisticRegression(max_iter=200, class_weight=dict(zip(classes, weights)))
    clf.fit(X, y)
    print(f"[Stage2] {level_col} model trained ({len(classes)} classes).")
    joblib.dump(clf, MODEL_PATH / f"lcpn_{level_col.lower().replace(' ', '_')}.joblib")
    return clf

# Train for all levels
for col in ["Level 1", "Level 2", "Level 3"]:
    train_level(col, embeddings, df)

print("[Stage2] Training complete! Models saved in ./models/")
