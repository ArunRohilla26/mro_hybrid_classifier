"""
Stage 2 – Train LCPN models for Levels 1, 2, and 3
--------------------------------------------------
Input: data/labeled_training.xlsx
Output: models/lcpn_level1.joblib, lcpn_level2.joblib, lcpn_level3.joblib
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import joblib

DATA_PATH = Path("data/labeled_training.xlsx")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("[Stage2-Train] Reading labeled training data...")
df = pd.read_excel(DATA_PATH)

df = df.dropna(subset=["Item Description", "Level 1"])
df["Level 2"] = df["Level 2"].fillna("Other")
df["Level 3"] = df["Level 3"].fillna("Other")

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[Stage2-Train] Encoding item descriptions...")
X = MODEL.encode(df["Item Description"].astype(str).tolist(), show_progress_bar=True)

def train_level(level_name, y):
    y = y.astype(str)
    classes = np.unique(y)
    print(f"[Stage2-Train] Training {level_name} classifier ({len(classes)} classes)...")

    weights = compute_class_weight("balanced", classes=classes, y=y)
    w_dict = dict(zip(classes, weights))

    clf = LogisticRegression(
        max_iter=2000,
        class_weight=w_dict,
        multi_class="ovr",
        solver="lbfgs",
        n_jobs=-1
    )
    clf.fit(X, y)
    joblib.dump(clf, MODEL_DIR / f"lcpn_{level_name.lower().replace(' ', '_')}.joblib")
    print(f"✅ Saved model for {level_name}")
    return clf

clf_l1 = train_level("Level1", df["Level 1"])
clf_l2 = train_level("Level2", df["Level 2"])
clf_l3 = train_level("Level3", df["Level 3"])

print("✅ All three Level models saved successfully.")
