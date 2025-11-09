"""
Hybrid MRO Auto-Categorizer – FINAL (Level-3 ready)
--------------------------------------------------
Uses hybrid_classifier_v3_final.py to classify items
into Level 1–3 taxonomy with clean headers & live progress.
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.hybrid_classifier_v3_final import classify_single_item


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid MRO Auto-Categorizer (Level-3 ready)")
    parser.add_argument("--input", required=True, help="Input Excel file path")
    parser.add_argument("--output", required=True, help="Output Excel file path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    print(f"[App] Reading input from: {in_path}")
    df_in = pd.read_excel(in_path)
    print(f"[App] Found {len(df_in)} rows")

    results = []
    print("[App] Running hybrid classification row by row...")

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), ncols=100):
        desc = str(row.get("Item Description", "")).strip()
        try:
            res = classify_single_item(desc)
        except Exception as e:
            res = {
                "Final_Level1": "",
                "Final_Level2": "",
                "Final_Level3": "",
                "Final_Source": f"Error: {type(e).__name__}",
                "Final_Score": "",
                "Stage2_Level1": "",
                "Stage2_Level2": "",
                "Stage2_Level3": "",
                "Stage2_Conf1": "",
                "Stage2_Conf2": "",
                "Stage2_Conf3": "",
                "Stage1_Category": "",
                "Stage1_Score": ""
            }
        results.append(res)

    df_results = pd.DataFrame(results)
    df_out = pd.concat([df_in.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(out_path, index=False)

    print(f"[App] ✅ Done. Processed {len(df_out)} rows.")
    print(f"[App] Results written to: {out_path}")


if __name__ == "__main__":
    main()
