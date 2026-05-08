# Big Data Workflow

This project keeps large raw data out of the local repository. The production-inspired workflow is:

```text
raw ads logs
  -> object storage / data lake
  -> SQL or Spark feature engineering
  -> partitioned Parquet training data
  -> local or remote model training on sampled data
```

## Local Development

Local development uses PySpark with `master("local[*]")`. This is not meant to process production
scale data on a laptop. It mirrors the same transformations that would run on Databricks, EMR,
Glue, or another Spark platform.

Run:

```bash
python scripts/spark_build_features.py
```

This writes:

```text
data/synthetic/raw_ads_events/dt=2026-05-08/*.parquet
data/synthetic/processed_ads_events/dt=2026-05-08/*.parquet
data/synthetic/processed_sample.csv
```

## Cloud-Compatible Paths

The Spark code uses path strings that can point to local files or cloud object storage:

```text
data/synthetic/processed_ads_events/
s3://bucket/processed/ads_events/
gs://bucket/processed/ads_events/
```

In production, raw data would usually live in S3/GCS/OSS/HDFS and be exposed through Hive,
Databricks, Presto/Trino, BigQuery, or Snowflake.

## Leakage Rule

Historical CTR/CVR priors must be fit on historical or training dates only:

```text
fit priors on dt <= train_end_date
join priors to validation/test dates
```

Using evaluation labels to build evaluation features is data leakage.

## Ali-CCP Strategy

Ali-CCP raw files are too large for this laptop. The intended workflow is:

```text
Ali-CCP raw archive
  -> remote/cloud environment
  -> extract/sample/normalize schema
  -> sample.csv or sample.parquet
  -> local run_aliccp_sample.py
```

The local project expects a preprocessed sample with at least:

```text
click
conversion
feature columns...
```

