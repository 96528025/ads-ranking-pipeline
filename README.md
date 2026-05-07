# Ads Ranking Pipeline for CTR/CVR Prediction

End-to-end advertising ranking project designed for machine learning internship interviews.

The project simulates a modern ads delivery funnel:

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
  src/
    ads_ranking/
      data.py
      experiments.py
      evaluation.py
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

The demo trains separate CTR and CVR models on synthetic ads data and prints:

- CTR AUC / LogLoss
- CVR AUC / LogLoss
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
model                  ctr_auc  ctr_logloss  cvr_auc  cvr_logloss  ndcg@5  top1_conversion_rate
logistic_regression     0.5604       0.8008   0.4809       0.9165  0.7931                0.1001
factorization_machine   0.6122       0.6694   0.5325       0.7612  0.8208                0.1122
wide_deep               0.7450       0.5075   0.6684       0.5500  0.8727                0.1345
deepfm                  0.7362       0.5154   0.6402       0.5656  0.8665                0.1322
```

See `docs/project_explanation.md` for the interview explanation.
