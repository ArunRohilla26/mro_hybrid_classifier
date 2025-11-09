from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

from .utils import load_config, load_embedding_model
from .preprocessing import normalize_text


def load_lcpn_models(model_dir: str = "models"):
    base = Path(model_dir)
    l1_path = base / "lcpn_level1.joblib"
    l2_path = base / "lcpn_level2.joblib"
    if not (l1_path.exists() and l2_path.exists()):
        print("[Stage2] No LCPN models found; Stage2 will be skipped.")
        return None, None
    clf_l1 = joblib.load(l1_path)
    clf_l2 = joblib.load(l2_path)
    print("[Stage2] Loaded LCPN Level1 & Level2 classifiers.")
    return clf_l1, clf_l2


def classify_stage2_row(
    description: str,
    embed_model,
    clf_l1,
    clf_l2,
    config: Dict[str, Any]
):
    thresholds = config["thresholds"]
    override_thr = float(thresholds["stage2_override"])

    if not isinstance(description, str) or not description.strip():
        return {
            "Stage2_Level1": "",
            "Stage2_Level2": "",
            "Stage2_Conf1": 0.0,
            "Stage2_Conf2": 0.0,
            "Stage2_Use": False,
        }

    norm = normalize_text(description)
    emb = embed_model.encode(norm)

    # Level 1
    probs1 = clf_l1.predict_proba([emb])[0]
    idx1 = int(np.argmax(probs1))
    l1_label = clf_l1.classes_[idx1]
    conf1 = float(probs1[idx1])

    # Level 2
    probs2 = clf_l2.predict_proba([emb])[0]
    idx2 = int(np.argmax(probs2))
    l2_label = clf_l2.classes_[idx2]
    conf2 = float(probs2[idx2])

    use_stage2 = conf1 >= override_thr and conf2 >= override_thr

    return {
        "Stage2_Level1": str(l1_label),
        "Stage2_Level2": str(l2_label),
        "Stage2_Conf1": round(conf1 * 100, 2),
        "Stage2_Conf2": round(conf2 * 100, 2),
        "Stage2_Use": bool(use_stage2),
    }
