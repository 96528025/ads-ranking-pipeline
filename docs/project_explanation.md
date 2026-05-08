# Project Explanation

## Goal

This project models a simplified ads ranking system. It has two tracks:

```text
synthetic full-funnel track -> CTR, CVR, bid, auction-aware ranking
real-world benchmark track -> public ads dataset validation
```

The current implemented track predicts:

- `pCTR = P(click | impression)`
- `pCVR = P(conversion | click)`

Then it ranks ads by:

```text
expected_value_score = pCTR * pCVR * bid
```

This connects machine learning metrics with the business goal of choosing ads that are likely to
create user engagement, advertiser value, and platform revenue.

## Why Keep Synthetic and Real-World Tracks Separate

The synthetic track is a controlled system-design environment. It lets the project model the full
ads funnel:

```text
impression -> click -> post-click conversion -> auction-aware ranking
```

Most public real-world ads datasets do not expose all of these pieces. Some have click labels but no
conversion labels. Some have conversion labels but no bid. Some have neither campaign budgets nor
serving constraints.

For that reason, the project keeps two separate goals:

```text
Synthetic track:
Validate the full-funnel pipeline design.

Real-world track:
Validate model and feature handling on public advertising logs.
```

This avoids overstating synthetic results while still preserving a complete ads ranking architecture.

## Real-World Ali-CCP Loader Design

The Ali-CCP track expects a preprocessed CSV or Parquet sample with:

```text
click
conversion
feature columns...
```

The loader maps this into the common project interface:

```text
click -> CTR target
conversion -> conversion target
post_click_conversion -> copied from conversion for CVR-style experiments
```

For the first version, the real-world track reports only prediction and calibration metrics:

```text
AUC
LogLoss
Brier Score
ECE
```

It does not report auction ranking metrics unless the dataset includes request-level candidate
groups and bid or value fields. This keeps the project honest: the synthetic track demonstrates the
full auction-aware ranking system, while the real-world track validates model behavior on public ad
logs.

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

## Why Add Historical Prior Features

Real ads systems rarely rely only on raw request features. They also use historical statistics such
as:

```text
category historical CTR
creative type historical CTR
user segment x ad category historical CTR
category historical CVR
```

These features are useful because they summarize past behavior at different granularities. For
example, if a user segment historically responds well to a certain ad category, that signal can help
the model rank similar ads higher even when an individual impression has limited context.

The project adds smoothed prior features fitted only on the training split. This avoids label
leakage: test labels are never used to create test features.

The smoothing formula is:

```text
smoothed_rate = (positive_count + alpha * global_rate) / (sample_count + alpha)
```

Smoothing matters because sparse combinations such as:

```text
user segment x ad category
```

can be noisy. A pair with one click from one impression should not be treated as a perfect 100% CTR
signal.

The project also adds lift features:

```text
user_ad_ctr_lift = user_ad_ctr_prior - category_ctr_prior
user_ad_cvr_lift = user_ad_cvr_prior - category_cvr_prior
```

These tell the model whether a user-ad pair is better or worse than the category baseline.

## Why Run Feature Ablation

Feature ablation answers whether engineered features actually help. The project compares:

```text
base features
engineered features
```

using the same data split, model architecture, training loop, auction score, and ranking metrics.
This is important because a project should show evidence, not just claim that a feature is useful.

## Why Evaluate Calibration

AUC measures ranking quality, not probability accuracy. A model can have strong AUC while still
overestimating or underestimating absolute probabilities.

This matters in ads because the final score uses predicted probabilities directly:

```text
expected_value_score = pCTR * pCVR * bid
```

If `pCTR` is overestimated by 3x, the auction score is also overestimated by 3x before even
considering `pCVR` and bid. For this reason, the project evaluates calibration in addition to AUC and
LogLoss.

The project uses Brier Score:

```text
Brier = mean((predicted_probability - label)^2)
```

and Expected Calibration Error:

```text
ECE = sum(bin_count / total_count * abs(avg_prediction - actual_rate))
```

ECE is computed by bucketing predictions into probability ranges and comparing each bucket's average
prediction with its observed positive rate.

The key takeaway is that better ranking metrics do not automatically imply better calibration. This
sets up the next project extension: fitting a calibration layer on a validation split before using
probabilities in auction scoring.

## How To Explain This In Interviews

Strong explanation:

```text
I built an end-to-end ads ranking pipeline that separates prediction and ranking. First, I trained
CTR and conditional CVR models with categorical embeddings and numerical features. Then I combined
the predicted probabilities with advertiser bid to compute an expected value score for final ad
ranking. I evaluated the models with AUC, LogLoss, Brier Score, and Expected Calibration Error, and
evaluated ranking quality with NDCG@K.
```

Possible extension ideas:

- Add an Ali-CCP, Avazu, iPinYou, or Criteo real-world benchmark loader.
- Add feature crosses and historical CTR features.
- Add probability calibration with Platt scaling or isotonic regression.
- Add a two-tower recall stage with FAISS.
- Add budget pacing and campaign-level constraints.
