from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .ml_pipeline import train_model
except ImportError:
    from ml_pipeline import train_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the diabetes classifier")
    parser.add_argument(
        "--data", type=Path, default=project_root / "src/data/diabetes.csv"
    )
    parser.add_argument(
        "--model", type=Path, default=project_root / "src/models/diabetes_pipeline.pkl"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=project_root / "src/models/training_metrics.json",
    )
    args = parser.parse_args()

    result = train_model(args.data, args.model)
    metrics = result["metrics"]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Saved model to {args.model}")
    print(f"Saved metrics to {args.metrics}")


if __name__ == "__main__":
    main()
