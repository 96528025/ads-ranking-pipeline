-- SQL examples for offline ads feature engineering.
-- The syntax is intentionally portable across Hive SQL, Presto/Trino, BigQuery, and Snowflake
-- with minor date/function adjustments.

-- 1. Category-level historical CTR prior.
SELECT
  ad_category,
  COUNT(*) AS impressions,
  SUM(click) AS clicks,
  AVG(click) AS category_ctr_prior
FROM ads_events
WHERE dt BETWEEN '2026-05-01' AND '2026-05-07'
GROUP BY ad_category;

-- 2. User-segment x ad-category affinity prior.
SELECT
  user_segment,
  ad_category,
  COUNT(*) AS impressions,
  SUM(click) AS clicks,
  AVG(click) AS user_ad_ctr_prior
FROM ads_events
WHERE dt BETWEEN '2026-05-01' AND '2026-05-07'
GROUP BY user_segment, ad_category;

-- 3. Smoothed prior to reduce noise for sparse combinations.
WITH global_stats AS (
  SELECT AVG(click) AS global_ctr
  FROM ads_events
  WHERE dt BETWEEN '2026-05-01' AND '2026-05-07'
),
pair_stats AS (
  SELECT
    user_segment,
    ad_category,
    SUM(click) AS clicks,
    COUNT(*) AS impressions
  FROM ads_events
  WHERE dt BETWEEN '2026-05-01' AND '2026-05-07'
  GROUP BY user_segment, ad_category
)
SELECT
  pair_stats.user_segment,
  pair_stats.ad_category,
  (pair_stats.clicks + 20.0 * global_stats.global_ctr)
    / (pair_stats.impressions + 20.0) AS user_ad_ctr_prior
FROM pair_stats
CROSS JOIN global_stats;

-- 4. Join train-fitted priors to a future/evaluation split.
WITH category_prior AS (
  SELECT
    ad_category,
    AVG(click) AS category_ctr_prior
  FROM ads_events
  WHERE dt BETWEEN '2026-05-01' AND '2026-05-07'
  GROUP BY ad_category
)
SELECT
  eval.*,
  COALESCE(category_prior.category_ctr_prior, 0.0) AS category_ctr_prior
FROM ads_events AS eval
LEFT JOIN category_prior
  ON eval.ad_category = category_prior.ad_category
WHERE eval.dt = '2026-05-08';

