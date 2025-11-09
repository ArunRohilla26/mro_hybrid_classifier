"""
Hybrid Classifier v3 – Clean Output Edition (FINAL STABLE RELEASE)
------------------------------------------------------------------
• Works with Streamlit and direct Python runs.
• Fixes all relative import errors permanently.
• Handles Exact-Labeled, Stage 1 (Semantic), and Stage 2 (Supervised) classification.
• Produces clean, aligned Level1–3 taxonomy output.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import joblib

# ---------------------------------------------------------------------
#  FIXED IMPORTS (absolute-safe)
# ---------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

# these files must be in the same directory
from preprocessing import normalize_text
from utils import load_config, logger_factory, load_taxonomy, load_embedding_model


# ---------------------------------------------------------------------
#  PATH DEFINITIONS
# ---------------------------------------------------------------------
TAXONOMY_PATH = Path("data/MRO_Taxonomy_v4_Master.xlsx")
LABELED_PATH = Path("data/labeled_training.xlsx")
MODEL_DIR = Path("models")


# ---------------------------------------------------------------------
#  LOAD TAXONOMY + ENCODINGS
# ---------------------------------------------------------------------
print(f"[Init] Loading taxonomy from: {TAXONOMY_PATH}")
taxonomy_df = pd.read_excel(TAXONOMY_PATH)

taxonomy_df["Combined"] = taxonomy_df.apply(
    lambda x: " > ".join(
        [str(x.get(c, "")).strip() for c in taxonomy_df.columns
         if "Level" in c and pd.notna(x[c]) and str(x[c]).strip() != ""]
    ),
    axis=1
)
print(f"[Init] Taxonomy loaded with {len(taxonomy_df)} rows.")

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
taxonomy_emb = MODEL.encode(
    taxonomy_df["Combined"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=False
)


# ---------------------------------------------------------------------
#  LOAD LABELED LOOKUP DATA
# ---------------------------------------------------------------------
labeled_lookup = {}
if LABELED_PATH.exists():
    labeled = pd.read_excel(LABELED_PATH)
    for _, r in labeled.iterrows():
        desc = str(r.get("Item Description", "")).strip().lower()
        if desc:
            labeled_lookup[desc] = {
                "Level1": r.get("Level 1", ""),
                "Level2": r.get("Level 2", ""),
                "Level3": r.get("Level 3", "")
            }
    print(f"[Lookup] Loaded {len(labeled_lookup)} labeled reference items.")
else:
    print("[Lookup] No labeled_training.xlsx found — continuing without exact matches.")


# ---------------------------------------------------------------------
#  LOAD STAGE 2 MODELS (LCPN)
# ---------------------------------------------------------------------
clf_l1 = clf_l2 = clf_l3 = None
for lvl in ["level1", "level2", "level3"]:
    p = MODEL_DIR / f"lcpn_{lvl}.joblib"
    if p.exists():
        locals()[f"clf_{lvl[-1]}"] = joblib.load(p)
        print(f"[Stage2] Loaded {p.name}")

if any([clf_l1, clf_l2, clf_l3]):
    print("[Stage2] Classifiers ready.")
else:
    print("[Stage2] No trained Stage2 models found — will skip supervised predictions.")


# ---------------------------------------------------------------------
#  CLASSIFICATION FUNCTION
# ---------------------------------------------------------------------
def classify_single_item(description: str) -> dict:
    """
    Classify a single MRO item description.
    Returns structured taxonomy levels, confidence, and source stage.
    """

    desc = str(description).strip()
    if not desc:
        return {
            "Final_Level1": "", "Final_Level2": "", "Final_Level3": "",
            "Final_Source": "Blank", "Final_Score": "",
            "Stage2_Level1": "", "Stage2_Level2": "", "Stage2_Level3": "",
            "Stage2_Conf1": "", "Stage2_Conf2": "", "Stage2_Conf3": "",
            "Stage1_Category": "", "Stage1_Score": ""
        }

    desc_norm = normalize_text(desc)
    desc_lower = desc.lower()

    # -------------------------------------------------------------
    # 1️⃣ Exact labeled match (perfect reference)
    # -------------------------------------------------------------
    if desc_lower in labeled_lookup:
        info = labeled_lookup[desc_lower]
        return {
            "Final_Level1": info["Level1"],
            "Final_Level2": info["Level2"],
            "Final_Level3": info["Level3"],
            "Final_Source": "Exact-Labeled",
            "Final_Score": 1.0,
            "Stage2_Level1": info["Level1"],
            "Stage2_Level2": info["Level2"],
            "Stage2_Level3": info["Level3"],
            "Stage2_Conf1": 1.0, "Stage2_Conf2": 1.0, "Stage2_Conf3": 1.0,
            "Stage1_Category": "Labeled Match",
            "Stage1_Score": 1.0
        }

    # -------------------------------------------------------------
    # 2️⃣ Stage 1 – Semantic Similarity (Transformer)
    # -------------------------------------------------------------
    emb = MODEL.encode([desc_norm], convert_to_tensor=True)
    cos = util.cos_sim(emb, taxonomy_emb)[0]
    best_idx = int(np.argmax(cos))
    best_score = float(cos[best_idx])
    tax_row = taxonomy_df.iloc[best_idx]
    best_cat = tax_row["Combined"]

    # -------------------------------------------------------------
    # 3️⃣ Stage 2 – Supervised (LCPN)
    # -------------------------------------------------------------
    stage2_labels, stage2_confs = {}, {}
    flat_emb = emb.cpu().numpy().flatten()

    for lvl, clf in zip(["Level1", "Level2", "Level3"], [clf_l1, clf_l2, clf_l3]):
        if clf is not None:
            try:
                probs = clf.predict_proba([flat_emb])[0]
                idx = int(np.argmax(probs))
                stage2_labels[lvl] = clf.classes_[idx]
                stage2_confs[lvl] = float(probs[idx])
            except Exception:
                stage2_labels[lvl] = ""
                stage2_confs[lvl] = 0.0
        else:
            stage2_labels[lvl] = ""
            stage2_confs[lvl] = 0.0

    # -------------------------------------------------------------
    # 4️⃣ Decision logic (select stage)
    # -------------------------------------------------------------
    final = {"Level1": "", "Level2": "", "Level3": ""}
    source = ""
    score = 0.0

    # Prefer Stage2 if confident at Level3
    if stage2_labels.get("Level3") and stage2_confs.get("Level3", 0) >= 0.7:
        final["Level1"] = stage2_labels["Level1"]
        final["Level2"] = stage2_labels["Level2"]
        final["Level3"] = stage2_labels["Level3"]
        source = "Stage2"
        score = stage2_confs["Level3"]

    # Fallback to strong semantic match
    elif best_score >= 0.8:
        final["Level1"] = tax_row.get("Level 1", "")
        final["Level2"] = tax_row.get("Level 2", "")
        final["Level3"] = tax_row.get("Level 3", "")
        source = "Stage1"
        score = best_score

    # Weak confidence fallback
    else:
        final["Level1"] = tax_row.get("Level 1", "")
        final["Level2"] = tax_row.get("Level 2", "")
        final["Level3"] = tax_row.get("Level 3", "")
        source = "Low-Confidence"
        score = best_score

    # -------------------------------------------------------------
    # 5️⃣ Structured Output
    # -------------------------------------------------------------
    return {
        "Final_Level1": final["Level1"],
        "Final_Level2": final["Level2"],
        "Final_Level3": final["Level3"],
        "Final_Source": source,
        "Final_Score": round(score, 3),
        "Stage2_Level1": stage2_labels.get("Level1", ""),
        "Stage2_Level2": stage2_labels.get("Level2", ""),
        "Stage2_Level3": stage2_labels.get("Level3", ""),
        "Stage2_Conf1": round(stage2_confs.get("Level1", 0), 3),
        "Stage2_Conf2": round(stage2_confs.get("Level2", 0), 3),
        "Stage2_Conf3": round(stage2_confs.get("Level3", 0), 3),
        "Stage1_Category": best_cat,
        "Stage1_Score": round(best_score, 3)
    }


# ---------------------------------------------------------------------
#  TEST ENTRY (optional direct run)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("[Debug] Running a quick test...")
    test_desc = "Safety gloves nitrile coated"
    result = classify_single_item(test_desc)
    print(pd.DataFrame([result]))
