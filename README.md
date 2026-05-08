# Ads Ranking Pipeline for CTR/CVR Prediction

End-to-end advertising ranking project designed for machine learning internship interviews.

The project is organized around two tracks:

1. **Synthetic full-funnel ads simulation**: a controlled environment with CTR, CVR, bid, and
   auction-aware ranking.
2. **Real-world ads benchmark**: planned extension for public advertising datasets such as Ali-CCP,
   Avazu, iPinYou, or Criteo.
3. **Big-data feature engineering**: local PySpark, SQL examples, and partitioned Parquet outputs
   that mirror production ads ML preprocessing workflows.

The current implementation focuses on the synthetic full-funnel track:

1. Generate ad impression data with user, ad, context, bid, click, and conversion signals.
2. Train separate CTR and conditional CVR prediction models.
3. Compare Logistic Regression, Factorization Machine, Wide & Deep, and DeepFM.
4. Rank candidate ads with an eCPM-style expected value score.
5. Evaluate both model quality and ranking quality.

This mirrors the core workflow used in ads recommendation systems: recall, rough ranking,
fine ranking, and auction-aware final ranking.

## Why This Project Matters

Ads ranking is not just binary classification. A practical ads system needs to estimate:

- `pCTR`: probability that a user clicks an ad.
- `pCVR`: probability that a clicked user converts.
- `bid`: advertiser willingness to pay.
- final ranking score, commonly related to expected value.

This project uses:

```text
expected_value = pCTR * pCVR * bid
```

as a simple auction-aware ranking score.

## Project Structure

```text
ads-ranking-pipeline/
  configs/
    default.json
  scripts/
    run_demo.py
    run_experiments.py
    run_feature_ablation.py
    run_aliccp_sample.py
    spark_build_features.py
    benchmark_inference.py
  src/
    ads_ranking/
      bigdata/
        spark_features.py
        spark_session.py
      data.py
      datasets/
        aliccp.py
        synthetic.py
        real_world.py
      experiments.py
      evaluation.py
      features.py
      train.py
      models/
        tabular.py
      ranking/
        auction.py
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
```

For editable package usage:

```bash
pip install -e .
```

Run the full model comparison:

```bash
python scripts/run_experiments.py
```

Run the feature engineering ablation:

```bash
python scripts/run_feature_ablation.py
```

Build Spark-based offline features and partitioned Parquet outputs:

```bash
python scripts/spark_build_features.py
```

Run an inference latency benchmark:

```bash
python scripts/benchmark_inference.py
```

The demo trains separate CTR and CVR models on synthetic ads data and prints:

- CTR AUC / LogLoss
- CVR AUC / LogLoss
- Brier Score / Expected Calibration Error
- NDCG@K for ranked ad lists
- Average expected value of selected top ads

## Resume Bullet Draft

```text
Built an end-to-end ads ranking pipeline for CTR/CVR prediction using PyTorch,
including synthetic impression generation, categorical feature embeddings,
FM/DeepFM feature interaction modeling, auction-aware ranking, and ranking quality evaluation.

Compared Logistic Regression, Factorization Machine, Wide & Deep, and DeepFM models,
then optimized final ad ordering with expected value score pCTR * pCVR * bid.
```

## Current Experiment Result

One local experiment run produced:

```text
model                  ctr_auc  ctr_logloss  ctr_brier  ctr_ece  cvr_auc  cvr_logloss  cvr_brier  cvr_ece  ndcg@5  top1_conversion_rate
logistic_regression     0.5761       0.6605     0.2145   0.1180   0.5260       0.7021     0.2339   0.1525  0.8114                0.1075
factorization_machine   0.6122       0.6694     0.2242   0.1338   0.5325       0.7612     0.2531   0.1915  0.8208                0.1122
wide_deep               0.7450       0.5075     0.1682   0.0156   0.6684       0.5500     0.1836   0.0102  0.8727                0.1345
deepfm                  0.7362       0.5154     0.1713   0.0260   0.6402       0.5656     0.1895   0.0296  0.8665                0.1322
```

## Feature Engineering Ablation

The engineered feature set adds smoothed historical prior features fitted only on the train split:

- category CTR/CVR prior
- creative CTR/CVR prior
- hour CTR prior
- user segment CTR prior
- user segment x ad category CTR/CVR prior
- user-ad CTR/CVR lift over category baseline

One local ablation run produced:

```text
feature_set   model      ctr_auc  ctr_logloss  ctr_brier  ctr_ece  cvr_auc  cvr_logloss  cvr_brier  cvr_ece  ndcg@5  top1_conversion_rate
base          wide_deep   0.7428       0.5088     0.1689   0.0126   0.6575       0.5551     0.1858   0.0079  0.8789                0.1367
base          deepfm      0.7362       0.5154     0.1713   0.0260   0.6402       0.5656     0.1895   0.0296  0.8665                0.1322
engineered    wide_deep   0.7465       0.5073     0.1681   0.0194   0.6768       0.5474     0.1827   0.0135  0.8785                0.1365
engineered    deepfm      0.7399       0.5128     0.1705   0.0224   0.6598       0.5588     0.1867   0.0308  0.8756                0.1356
```

The gains are stronger on CVR and ranking metrics than raw CTR, which is expected because historical
priors and user-ad affinity features are closer to conversion intent than immediate click appeal.
Calibration metrics do not always improve together with AUC, which motivates a separate calibration
correction step before using predicted probabilities in auction scoring.

## Real-World Benchmark Track

The real-world benchmark track is intentionally separate from the synthetic track. Public ads
datasets often provide real click labels but do not include the full funnel needed for auction-aware
ranking, such as CVR labels, bid, campaign budget, or serving constraints.

The planned benchmark will use a dataset-specific loader under `src/ads_ranking/datasets/` and reuse
the same model, feature, and calibration evaluation code where possible.

### Ali-CCP Sample Format

The first real-world loader targets a preprocessed Ali-CCP sample stored as CSV or Parquet:

```text
data/aliccp/sample.csv
```

Required label columns:

```text
click
conversion
```

All other object/string columns are treated as categorical features. Numeric non-label columns are
treated as numerical features. The real-world script focuses on CTR/CVR prediction metrics only,
because a tabular Ali-CCP sample does not include bid or request-level candidate sets for auction
ranking.

Run it with:

```bash
python scripts/run_aliccp_sample.py
```

See `docs/project_explanation.md` for the interview explanation.

## Big-Data Workflow

The project includes a local PySpark workflow that mirrors production data preprocessing:

```text
raw ads events
  -> Spark historical CTR/CVR prior aggregation
  -> partitioned Parquet
  -> sampled CSV for local model training
```

See:

- `docs/big_data_workflow.md`
- `docs/sql_feature_examples.sql`
- `docs/decision_log.md`

Experiment and benchmark outputs are written under `outputs/`, which is intentionally ignored by
Git.

For local Spark runs on this machine, use Java 17:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 17) python scripts/spark_build_features.py
```
