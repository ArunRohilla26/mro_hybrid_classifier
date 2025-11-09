import os
import yaml
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd


# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
def logger_factory(name: str = "mro_classifier", level=logging.INFO) -> logging.Logger:
    """Create a consistent logger for all modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


# -------------------------------------------------------------------
# Config Loader
# -------------------------------------------------------------------
def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load YAML configuration safely."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"❌ Config file not found: {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


# -------------------------------------------------------------------
# Taxonomy Loader
# -------------------------------------------------------------------
def load_taxonomy(taxonomy_path: str | Path) -> pd.DataFrame:
    """Load taxonomy Excel and combine hierarchical columns into 'Combined'."""
    taxonomy_path = Path(taxonomy_path)
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"❌ Taxonomy file not found: {taxonomy_path}")

    df = pd.read_excel(taxonomy_path)

    # Clean and concatenate hierarchical levels
    level_cols = [c for c in df.columns if "Level" in c]
    df["Combined"] = df.apply(
        lambda x: " > ".join(
            [str(x[c]).strip() for c in level_cols if pd.notna(x[c]) and str(x[c]).strip()]
        ),
        axis=1,
    )

    return df


# -------------------------------------------------------------------
# Embedding Model Loader
# -------------------------------------------------------------------
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a SentenceTransformer model safely with a log message."""
    logger = logger_factory("model_loader")
    logger.info(f"[Models] Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"⚠️ Failed to load model '{model_name}': {e}")
        raise
    return model


# -------------------------------------------------------------------
# Create directories if missing
# -------------------------------------------------------------------
def ensure_directories():
    """Ensure standard project directories exist."""
    for d in ["data", "models", "logs", "outputs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Utility for Excel Export
# -------------------------------------------------------------------
def save_dataframe_excel(df: pd.DataFrame, output_path: str | Path):
    """Save DataFrame to Excel and confirm."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"✅ Saved Excel file: {output_path}")


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    logger = logger_factory()
    ensure_directories()
    cfg = load_config("config.yaml")
    print("Loaded config keys:", list(cfg.keys()))
