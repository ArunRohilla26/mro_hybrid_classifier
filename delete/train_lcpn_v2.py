"""
Stage-2 LCPN training script (v2)
---------------------------------
Uses your manually-labeled Excel to train Level-1 and Level-2 classifiers
with balanced class weights and the same embeddings as Stage-1.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sentence_transformers import SentenceTransformer
import joblib
from src.preprocessing import normalize_text


def encode_items(model, texts):
    """Embed item descriptions in batches."""
    embs, bs = [], 32
    for i in tqdm(range(0, len(texts), bs), desc="Encoding items"):
        embs.append(model.encode(texts[i:i + bs], convert_to_numpy=True, show_progress_bar=False))
    return np.vstack(embs)


def train_weighted_classifier(X, y, label):
    """Train logistic regression with balanced class weights."""
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = dict(zip(classes, weights))

    clf = LogisticRegression(
        max_iter=300,
        n_jobs=-1,
        class_weight=class_weight,
        solver="lbfgs",
        multi_class="auto",
    )
    print(f"[Stage2-Train] Training {label} classifier ({len(classes)} classes)...")
    clf.fit(X, y)
    return clf


def main():
    input_path = Path("data/labeled_training.xlsx")
    outdir = Path("models")
    outdir.mkdir(exist_ok=True, parents=True)

    if not input_path.exists():
        raise FileNotFoundError(f"❌ Labeled file not found: {input_path}")

    print(f"[Stage2-Train] Reading labeled data: {input_path}")
    df = pd.read_excel(input_path)

    # expected columns: Item Description | Level 1 | Level 2
    df = df.dropna(subset=["Item Description"]).copy()
    df["clean_desc"] = df["Item Description"].astype(str).apply(normalize_text)

    # ---------- Stage-1 embedding model ----------
    print("[Stage2-Train] Loading embedding model (same as Stage-1)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    print("[Stage2-Train] Encoding item descriptions...")
    X_emb = encode_items(model, df["clean_desc"].tolist())

    # ---------- Level-1 ----------
    y_l1 = df["Level 1"].astype(str).tolist()
    clf_l1 = train_weighted_classifier(X_emb, y_l1, "Level-1")
    joblib.dump(clf_l1, outdir / "lcpn_level1.joblib")

    # ---------- Level-2 ----------
    if "Level 2" in df.columns:
        y_l2 = df["Level 2"].astype(str).tolist()
        clf_l2 = train_weighted_classifier(X_emb, y_l2, "Level-2")
        joblib.dump(clf_l2, outdir / "lcpn_level2.joblib")

    print(f"\n✅ Training complete. Models saved under: {outdir}\n")


if __name__ == "__main__":
    main()
