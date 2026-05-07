from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ads_ranking.data import FeatureSpec, encode_and_split, generate_synthetic_ads_data
from ads_ranking.experiments import run_model_experiment


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

    result = run_model_experiment(
        model_name="deepfm",
        encoded_data=encoded,
        training_config=config["training"],
        ranking_config=config["ranking"],
        seed=config["seed"],
        verbose=True,
    )

    print("\nDeepFM demo metrics")
    for key, value in result.items():
        if key == "model":
            print(f"{key}={value}")
        else:
            print(f"{key}={value:.4f}")


if __name__ == "__main__":
    main()
