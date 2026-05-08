# Decision Log

This document records the project evolution and the reasoning behind major design choices.

## 1. Start With A Controlled Full-Funnel Simulation

The project started with synthetic ads data because public advertising datasets often expose only
part of the ads funnel. Many public datasets have click labels but no conversion labels, bid values,
campaign budgets, or request-level candidate sets.

The synthetic track is therefore used to validate the complete system design:

```text
impression -> click -> post-click conversion -> pCTR/pCVR prediction -> auction-aware ranking
```

The synthetic results should not be interpreted as real business impact. They are a controlled
environment for testing pipeline logic.

## 2. Separate CTR And CVR

The project separates:

```text
pCTR = P(click | impression)
pCVR = P(conversion | click)
```

instead of training only one conversion classifier. This mirrors real ads systems, where click and
conversion labels have different sparsity, delay, and feature importance.

The synthetic data keeps:

```text
click
post_click_conversion
conversion = click * post_click_conversion
```

so that final ranking can use:

```text
expected_value_score = pCTR * pCVR * bid
```

without double-counting click probability.

## 3. Add Model Baselines Before Adding Complexity

The first model version was a single embedding-based model. That was too close to a toy demo, so the
project became a model comparison framework:

```text
Logistic Regression -> Factorization Machine -> Wide & Deep -> DeepFM
```

This creates a defensible experiment ladder:

- Logistic Regression is the sanity baseline.
- Factorization Machine captures sparse pairwise feature interactions.
- Wide & Deep learns nonlinear embedding-based patterns.
- DeepFM combines FM-style interactions with deep nonlinear features.

## 4. Add Historical Prior Features With Leakage Control

Ads systems rely heavily on historical statistics such as category CTR, creative CTR, and user-ad
affinity. The project added smoothed prior features and feature ablation.

The important rule is:

```text
fit prior features on train/historical data only
join them to evaluation data
```

This avoids leakage from test labels into test features.

## 5. Add Calibration Metrics

Ads ranking uses predicted probabilities directly in value formulas. A model can have strong AUC but
poor probability calibration. The project added:

```text
Brier Score
Expected Calibration Error
calibration bins
```

This makes the evaluation more realistic for auction-aware ranking, where overestimated pCTR or pCVR
can distort bid-weighted scores.

## 6. Split The Project Into Tracks

The project now has three tracks:

```text
Synthetic full-funnel track:
  validates CTR/CVR/bid/ranking/calibration logic

Real-world benchmark track:
  validates model and feature handling on public ads data samples

Big-data preprocessing track:
  mirrors production offline feature engineering with SQL, Spark, and Parquet
```

This avoids overstating synthetic results while keeping a complete ads ranking architecture.

## 7. Do Not Process Large Raw Data On A Laptop

Ali-CCP raw files are several GB compressed and larger after extraction. A local laptop is not the
right place to process full advertising logs.

The production-inspired strategy is:

```text
large raw data -> cloud/data platform
sample/processed data -> local model development
```

This matches industrial practice: move compute to the data, not raw data to a laptop.

## 8. Add PySpark And Partitioned Parquet

The project added a local PySpark feature-engineering workflow:

```text
raw events
  -> Spark historical CTR/CVR aggregation
  -> partitioned Parquet
  -> sampled CSV for local model training
```

This simulates how ads logs are usually stored and processed in production:

```text
S3/GCS/OSS/HDFS + Hive-style partitions + Spark/Hive SQL/Presto/Databricks
```

The local Spark workflow is intentionally small but cloud-compatible.

## 9. Add Inference Benchmarking

Ads ranking systems care about serving latency, not only offline AUC. The project added an inference
benchmark reporting:

```text
parameter count
average latency
p95 latency
ads per second
```

This connects model complexity to serving constraints.

## Current Next Step

The next major step is generating a real-world Ali-CCP sample:

```text
raw Ali-CCP archive
  -> cloud/remote preprocessing
  -> sample.csv or sample.parquet
  -> local run_aliccp_sample.py
```

The local loader is ready for a preprocessed sample with:

```text
click
conversion
feature columns...
```

