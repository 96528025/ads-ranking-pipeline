from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ads_ranking.data import FeatureSpec, encode_and_split, generate_synthetic_ads_data
from ads_ranking.experiments import run_experiment_suite


def load_config() -> dict:
    with (PROJECT_ROOT / "configs" / "default.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    config = load_config()
    feature_spec = FeatureSpec(
        categorical=config["features"]["categorical"],
        numerical=config["features"]["numerical"],
    )
    frame = generate_synthetic_ads_data(
        num_samples=config["num_samples"],
        seed=config["seed"],
    )
    encoded = encode_and_split(
        frame=frame,
        feature_spec=feature_spec,
        test_size=config["test_size"],
        seed=config["seed"],
    )

    results = run_experiment_suite(
        model_names=config["models"],
        encoded_data=encoded,
        training_config=config["training"],
        ranking_config=config["ranking"],
        seed=config["seed"],
        verbose=False,
    )
    print("\nExperiment comparison")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
