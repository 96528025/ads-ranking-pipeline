from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ads_ranking.data import AdsDataset, FeatureSpec, encode_and_split
from ads_ranking.datasets.synthetic import generate_synthetic_ads_data
from ads_ranking.experiments import build_model_registry


def load_config() -> dict:
    with (PROJECT_ROOT / "configs" / "benchmark.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    config = load_config()
    default_config = load_default_config()
    feature_spec = FeatureSpec(
        categorical=default_config["features"]["categorical"],
        numerical=default_config["features"]["numerical"],
    )
    frame = generate_synthetic_ads_data(config["num_samples"], config["seed"])
    encoded = encode_and_split(
        frame=frame,
        feature_spec=feature_spec,
        test_size=0.2,
        seed=config["seed"],
    )
    dataset = AdsDataset(encoded.test, encoded.feature_spec, "click")
    categorical = dataset.categorical[: config["batch_size"]]
    numerical = dataset.numerical[: config["batch_size"]]

    registry = build_model_registry()
    rows = []
    for model_name in config["models"]:
        model = registry[model_name](encoded, config["training"])
        model.eval()
        params = sum(parameter.numel() for parameter in model.parameters())
        latencies_ms = benchmark_model(
            model,
            categorical,
            numerical,
            warmup_batches=config["warmup_batches"],
            benchmark_batches=config["benchmark_batches"],
        )
        avg_ms = sum(latencies_ms) / len(latencies_ms)
        p95_ms = percentile(latencies_ms, 95)
        rows.append(
            {
                "model": model_name,
                "parameters": params,
                "batch_size": len(categorical),
                "avg_latency_ms": avg_ms,
                "p95_latency_ms": p95_ms,
                "ads_per_second": len(categorical) / (avg_ms / 1000.0),
            }
        )

    results = pd.DataFrame(rows)
    print("\nInference benchmark")
    print(results.round(4).to_string(index=False))

    output_dir = PROJECT_ROOT / "outputs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"inference_benchmark_{timestamp()}.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved benchmark to {output_path}")


def load_default_config() -> dict:
    with (PROJECT_ROOT / "configs" / "default.json").open("r", encoding="utf-8") as file:
        return json.load(file)


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    categorical: torch.Tensor,
    numerical: torch.Tensor,
    warmup_batches: int,
    benchmark_batches: int,
) -> list[float]:
    for _ in range(warmup_batches):
        _ = model(categorical, numerical)

    latencies_ms = []
    for _ in range(benchmark_batches):
        start = time.perf_counter()
        _ = model(categorical, numerical)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return latencies_ms


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = int(round((percentile_value / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()

