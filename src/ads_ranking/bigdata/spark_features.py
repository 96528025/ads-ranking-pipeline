from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class SparkPriorSpec:
    name: str
    columns: list[str]
    target: str


def build_default_spark_prior_specs() -> list[SparkPriorSpec]:
    return [
        SparkPriorSpec("category_ctr_prior", ["ad_category"], "click"),
        SparkPriorSpec("creative_ctr_prior", ["creative_type"], "click"),
        SparkPriorSpec("hour_ctr_prior", ["hour_bucket"], "click"),
        SparkPriorSpec("user_segment_ctr_prior", ["user_segment"], "click"),
        SparkPriorSpec("user_ad_ctr_prior", ["user_segment", "ad_category"], "click"),
        SparkPriorSpec("category_cvr_prior", ["ad_category"], "post_click_conversion"),
        SparkPriorSpec("creative_cvr_prior", ["creative_type"], "post_click_conversion"),
        SparkPriorSpec("user_ad_cvr_prior", ["user_segment", "ad_category"], "post_click_conversion"),
    ]


def add_smoothed_prior_features(
    frame: DataFrame,
    specs: list[SparkPriorSpec],
    smoothing: float = 20.0,
) -> DataFrame:
    """Add smoothed historical prior features using Spark aggregations.

    This mirrors production offline feature engineering. In a real pipeline, priors should be fit
    only on historical/train dates and joined to future/evaluation dates to avoid leakage.
    """
    enriched = frame
    for spec in specs:
        global_rate = frame.agg(F.avg(F.col(spec.target)).alias("global_rate")).first()["global_rate"]
        stats = (
            frame.groupBy(*spec.columns)
            .agg(
                F.sum(F.col(spec.target)).alias("positive_count"),
                F.count(F.lit(1)).alias("sample_count"),
            )
            .withColumn(
                spec.name,
                (F.col("positive_count") + F.lit(smoothing * global_rate))
                / (F.col("sample_count") + F.lit(smoothing)),
            )
            .select(*spec.columns, spec.name)
        )
        enriched = enriched.join(stats, on=spec.columns, how="left")
        enriched = enriched.fillna({spec.name: float(global_rate)})

    if {"user_ad_ctr_prior", "category_ctr_prior"}.issubset(set(enriched.columns)):
        enriched = enriched.withColumn(
            "user_ad_ctr_lift",
            F.col("user_ad_ctr_prior") - F.col("category_ctr_prior"),
        )
    if {"user_ad_cvr_prior", "category_cvr_prior"}.issubset(set(enriched.columns)):
        enriched = enriched.withColumn(
            "user_ad_cvr_lift",
            F.col("user_ad_cvr_prior") - F.col("category_cvr_prior"),
        )

    return enriched

