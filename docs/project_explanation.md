# Project Explanation

## Goal

This project models a simplified ads ranking system. The system predicts:

- `pCTR = P(click | impression)`
- `pCVR = P(conversion | click)`

Then it ranks ads by:

```text
expected_value_score = pCTR * pCVR * bid
```

This connects machine learning metrics with the business goal of choosing ads that are likely to
create user engagement, advertiser value, and platform revenue.

## Why CTR and CVR Are Separate

CTR and CVR represent different user behaviors:

- CTR captures whether an ad is attractive enough to click.
- CVR captures whether the landing page, product, and user intent lead to conversion after click.

Training them separately makes the project closer to real ads systems, where click prediction and
conversion prediction often have different labels, delays, sparsity, and feature importance.

## Why Use Embeddings

Ads data contains many categorical features:

- user segment
- device
- hour bucket
- ad category
- creative type

Integer IDs should not be treated as continuous numbers. Embeddings allow the model to learn dense
representations for categorical values and capture interactions such as:

```text
young mobile user + video creative + gaming ad category
```

This is why the project uses an embedding-based Wide & Deep style model instead of only logistic
regression.

## Why Compare Multiple Models

The project is designed as an experiment ladder instead of a single-model demo:

```text
Logistic Regression -> Factorization Machine -> Wide & Deep -> DeepFM
```

Each model answers a different engineering question.

Logistic Regression is the sanity-check baseline. If a complex model cannot beat it, the added
complexity is not justified.

Factorization Machine adds pairwise feature interactions, which are important in ads because sparse
categorical combinations matter:

```text
user segment x ad category
device x creative type
hour bucket x ad category
```

Wide & Deep uses embeddings plus an MLP to learn nonlinear patterns from sparse and dense features.

DeepFM combines FM-style memorized pairwise interactions with a deep component for higher-order
nonlinear interactions. This makes it a natural ads CTR/CVR model to discuss in interviews.

The important point is not that the deepest model always wins. The important point is that the
project tests whether richer feature interaction modeling improves both prediction metrics and
ranking metrics.

## Why Keep Raw Bid

The model uses standardized numerical features for stable training. However, the auction score needs
the real bid value, not the standardized bid. The pipeline therefore keeps `raw_bid` for ranking and
uses scaled `bid` only as a model feature.

## How To Explain This In Interviews

Strong explanation:

```text
I built an end-to-end ads ranking pipeline that separates prediction and ranking. First, I trained
CTR and conditional CVR models with categorical embeddings and numerical features. Then I combined
the predicted probabilities with advertiser bid to compute an expected value score for final ad
ranking. I evaluated the models with AUC and LogLoss, and evaluated ranking quality with NDCG@K.
```

Possible extension ideas:

- Replace synthetic data with Criteo or Avazu CTR data.
- Add feature crosses and historical CTR features.
- Add probability calibration with Platt scaling or isotonic regression.
- Add a two-tower recall stage with FAISS.
- Add budget pacing and campaign-level constraints.
