from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ads_ranking.data import encode_and_split
from ads_ranking.datasets.aliccp import load_aliccp_tabular_sample
from ads_ranking.experiments import run_experiment_suite


def load_config() -> dict:
    with (PROJECT_ROOT / "configs" / "aliccp_sample.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    config = load_config()
    dataset_config = config["dataset"]
    data_path = PROJECT_ROOT / dataset_config["path"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Ali-CCP sample not found at {data_path}. "
            "Create a preprocessed CSV or Parquet sample before running this script."
        )

    loaded = load_aliccp_tabular_sample(
        path=data_path,
        num_rows=dataset_config.get("num_rows"),
    )
    encoded = encode_and_split(
        frame=loaded.frame,
        feature_spec=loaded.feature_spec,
        test_size=config["test_size"],
        seed=config["seed"],
        add_engineered_features=False,
    )
    results = run_experiment_suite(
        model_names=config["models"],
        encoded_data=encoded,
        training_config=config["training"],
        ranking_config=config["ranking"],
        seed=config["seed"],
        verbose=False,
    )
    metric_columns = [
        "model",
        "ctr_auc",
        "ctr_logloss",
        "ctr_brier",
        "ctr_ece",
        "cvr_auc",
        "cvr_logloss",
        "cvr_brier",
        "cvr_ece",
    ]
    print("\nAli-CCP sample CTR/CVR comparison")
    print(results[metric_columns].round(4).to_string(index=False))

    output_dir = PROJECT_ROOT / "outputs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"aliccp_sample_{timestamp()}.csv"
    results[metric_columns].to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
