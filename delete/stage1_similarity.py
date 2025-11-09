from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm
from sentence_transformers import util

from .utils import load_config, load_embedding_model, load_taxonomy, logger_factory
from .preprocessing import normalize_text


def prepare_stage1(config_path: str):
    config = load_config(config_path)
    model = load_embedding_model(config)
    taxonomy_df = load_taxonomy(config)
    print("[Stage1] Encoding taxonomy embeddings...")
    tax_emb = model.encode(
        taxonomy_df["RefTextNorm"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return config, model, taxonomy_df, tax_emb


def classify_stage1_row(
    description: str,
    model,
    taxonomy_df: pd.DataFrame,
    tax_emb,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    thresholds = config["thresholds"]
    high = float(thresholds["stage1_high_conf"])
    low = float(thresholds["stage1_low_conf"])

    if not isinstance(description, str) or not description.strip():
        return {
            "Stage1_Level1": "",
            "Stage1_Level2": "",
            "Stage1_Level3": "",
            "Stage1_Score": 0.0,
            "Stage1_Decision": "Manual (Empty)",
        }

    norm = normalize_text(description)
    emb = model.encode(norm, convert_to_tensor=True)
    scores = util.cos_sim(emb, tax_emb)[0]
    idx = int(scores.argmax().item())
    best = float(scores[idx].item())
    row = taxonomy_df.iloc[idx]

    if best >= high:
        dec = "Auto-High"
    elif best < low:
        dec = "Manual-Low"
    else:
        dec = "Ambiguous"

    return {
        "Stage1_Level1": str(row.get("Level 1", "")),
        "Stage1_Level2": str(row.get("Level 2", "")),
        "Stage1_Level3": str(row.get("Level 3", "")),
        "Stage1_Score": round(best * 100, 2),
        "Stage1_Decision": dec,
    }


def run_stage1(df: pd.DataFrame, config_path: str = "src/config.yaml") -> pd.DataFrame:
    config, model, taxonomy_df, tax_emb = prepare_stage1(config_path)
    log = logger_factory(config)
    desc_col = config["classification"]["description_column"]
    if desc_col not in df.columns:
        raise ValueError(f"Input missing description column '{desc_col}'")

    results: List[Dict[str, Any]] = []
    print("[Stage1] Classifying items with similarity model...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        desc = row.get(desc_col, "")
        out = classify_stage1_row(str(desc), model, taxonomy_df, tax_emb, config)
        results.append(out)
        log(f"[Stage1] {desc} || {out['Stage1_Level1']} > {out['Stage1_Level2']} > {out['Stage1_Level3']} "
            f"|| {out['Stage1_Score']} || {out['Stage1_Decision']}")

    res_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)
