import argparse

from src.stage2_lcpn_trainer import train_lcpn_models


def main():
    parser = argparse.ArgumentParser(description="Train LCPN Stage2 models for MRO classifier")
    parser.add_argument("--labeled", required=True, help="Path to labeled data (Excel/CSV with Item Description, Level 1, Level 2)")
    parser.add_argument("--config", default="src/config.yaml", help="Config file path")
    parser.add_argument("--outdir", default="models", help="Output directory for trained models")
    args = parser.parse_args()

    print(f"[Train] Training LCPN models from: {args.labeled}")
    train_lcpn_models(args.labeled, config_path=args.config, output_dir=args.outdir)


if __name__ == "__main__":
    main()
