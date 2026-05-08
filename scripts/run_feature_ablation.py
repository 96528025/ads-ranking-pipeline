from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ads_ranking.data import FeatureSpec, encode_and_split
from ads_ranking.datasets.synthetic import generate_synthetic_ads_data
from ads_ranking.experiments import run_experiment_suite


def load_config() -> dict:
    with (PROJECT_ROOT / "configs" / "default.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def build_feature_spec(config: dict, use_engineered_features: bool) -> FeatureSpec:
    numerical = list(config["features"]["numerical"])
    if use_engineered_features:
        numerical.extend(config["engineered_features"])
    return FeatureSpec(
        categorical=config["features"]["categorical"],
        numerical=numerical,
    )


def main() -> None:
    config = load_config()
    frame = generate_synthetic_ads_data(
        num_samples=config["num_samples"],
        seed=config["seed"],
    )

    results = []
    for feature_set_name, feature_set_config in config["feature_sets"].items():
        use_engineered_features = feature_set_config["use_engineered_features"]
        print(f"\nRunning feature set: {feature_set_name}")
        feature_spec = build_feature_spec(config, use_engineered_features)
        encoded = encode_and_split(
            frame=frame,
            feature_spec=feature_spec,
            test_size=config["test_size"],
            seed=config["seed"],
            add_engineered_features=use_engineered_features,
        )
        feature_results = run_experiment_suite(
            model_names=config["ablation"]["models"],
            encoded_data=encoded,
            training_config=config["training"],
            ranking_config=config["ranking"],
            seed=config["seed"],
            verbose=False,
        )
        feature_results.insert(0, "feature_set", feature_set_name)
        results.append(feature_results)

    comparison = pd.concat(results, ignore_index=True)

    print("\nFeature ablation comparison")
    print(comparison.round(4).to_string(index=False))

    output_dir = PROJECT_ROOT / "outputs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"feature_ablation_{timestamp()}.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
