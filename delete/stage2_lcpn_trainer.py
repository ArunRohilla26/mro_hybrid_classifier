from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .utils import load_config, load_embedding_model, logger_factory
from .preprocessing import normalize_text


def train_lcpn_models(
    labeled_path: str,
    config_path: str = "src/config.yaml",
    output_dir: str = "models"
):
    """
    Train simple LCPN-style classifiers:
    - One model for Level 1
    - One model for Level 2
    Expects labeled data with columns:
      'Item Description', 'Level 1', 'Level 2'
    """
    config = load_config(config_path)
    model = load_embedding_model(config)
    log = logger_factory(config)

    if labeled_path.lower().endswith("xlsx") or labeled_path.lower().endswith("xls"):
        df = pd.read_excel(labeled_path)
    else:
        df = pd.read_csv(labeled_path)

    for col in ["Item Description", "Level 1", "Level 2"]:
        if col not in df.columns:
            raise ValueError(f"Labeled data missing required column: {col}")

    df = df.dropna(subset=["Item Description", "Level 1", "Level 2"]).reset_index(drop=True)

    X_text = df["Item Description"].astype(str).apply(normalize_text).tolist()
    print("[Stage2-Train] Encoding item descriptions...")
    X_emb = model.encode(X_text, show_progress_bar=True)

    # Level 1 classifier
    y_l1 = df["Level 1"].astype(str).tolist()
    clf_l1 = LogisticRegression(max_iter=200, n_jobs=-1)
    print("[Stage2-Train] Training Level 1 classifier...")
    clf_l1.fit(X_emb, y_l1)

    # Level 2 classifier
    y_l2 = df["Level 2"].astype(str).tolist()
    clf_l2 = LogisticRegression(max_iter=200, n_jobs=-1)
    print("[Stage2-Train] Training Level 2 classifier...")
    clf_l2.fit(X_emb, y_l2)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_l1, out_dir / "lcpn_level1.joblib")
    joblib.dump(clf_l2, out_dir / "lcpn_level2.joblib")
    log("[Stage2-Train] Saved Level 1 and 2 classifiers to models/")
    print(f"[Stage2-Train] Models saved under: {out_dir}")
