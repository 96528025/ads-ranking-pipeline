from __future__ import annotations

from pyspark.sql import SparkSession


def create_local_spark(app_name: str = "ads-ranking-pipeline") -> SparkSession:
    """Create a local Spark session for development and small-sample preprocessing."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
