from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ads_ranking.bigdata.spark_features import (
    add_smoothed_prior_features,
    build_default_spark_prior_specs,
)
from ads_ranking.bigdata.spark_session import create_local_spark
from ads_ranking.datasets.synthetic import generate_synthetic_ads_data


def load_config() -> dict:
    with (PROJECT_ROOT / "configs" / "spark_features.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def reset_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    config = load_config()
    raw_output = PROJECT_ROOT / config["raw_output"]
    processed_output = PROJECT_ROOT / config["processed_output"]
    sample_csv_output = PROJECT_ROOT / config["sample_csv_output"]

    reset_path(raw_output)
    reset_path(processed_output)
    reset_path(sample_csv_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    processed_output.parent.mkdir(parents=True, exist_ok=True)
    sample_csv_output.parent.mkdir(parents=True, exist_ok=True)

    pandas_frame = generate_synthetic_ads_data(
        num_samples=config["num_samples"],
        seed=config["seed"],
    )
    pandas_frame["dt"] = "2026-05-08"

    spark = create_local_spark("ads-ranking-spark-features")
    try:
        raw_frame = spark.createDataFrame(pandas_frame)
        raw_frame.write.mode("overwrite").partitionBy("dt").parquet(str(raw_output))

        loaded = spark.read.parquet(str(raw_output))
        enriched = add_smoothed_prior_features(
            loaded,
            specs=build_default_spark_prior_specs(),
            smoothing=config["smoothing"],
        )
        enriched.write.mode("overwrite").partitionBy("dt").parquet(str(processed_output))

        sample = enriched.drop("dt").toPandas()
        sample.to_csv(sample_csv_output, index=False)
    finally:
        spark.stop()

    print(f"Wrote raw partitioned Parquet: {raw_output}")
    print(f"Wrote processed partitioned Parquet: {processed_output}")
    print(f"Wrote local training sample CSV: {sample_csv_output}")


if __name__ == "__main__":
    main()
