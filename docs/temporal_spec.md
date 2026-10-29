# Temporal specification (contract v1)

## Timestamps (all stored UTC ISO-8601; displayed in Asia/Ho_Chi_Minh)
| Field | Meaning |
|---|---|
| `published_at` | publication time reported by the source (page metadata; naive source times are read as +07:00) |
| `first_seen_at` | first time our crawler observed the URL (`document`) |
| `version_first_seen_at` | first time a specific content version (hash) was observed |
| `source_updated_at` | modification time reported by the source, if any |
| `processed_at` | when the NLP pipeline finished the article-company result |
| `forecast_cutoff` | the time a forecast is made. Daily samples use the exchange close of day t: 15:00 Asia/Ho_Chi_Minh = 08:00 UTC |

## Availability: when may a result feed a snapshot?
- **live**: `processed_at <= forecast_cutoff`. This is what the running system really knew.
- **backtest** (simulation, must be labelled as such in any report): `published_at + 30 min <= forecast_cutoff`. Assumes the article would have been seen and processed 30 minutes after publication. We do NOT claim the system observed it then. Processing time of a historical backfill is ignored on purpose.
- Rows are keyed by `availability_mode`; the two modes never mix in one table row.
- Articles without a `published_at` are not usable in backtest.

## Features (`feature_version = v1`) per `(symbol, forecast_cutoff, feature_version, nlp_model_version, availability_mode)`
- Eligible results: representative of their near-duplicate cluster, `relation != market`, available per the mode above.
- `news_count_24h`, `news_count_72h`: fixed windows. `hours_since_last_news`: over all available results (may exceed 72h).
- Sentiment/quality features (`sentiment_mean`, `sentiment_std`, `sentiment_mean_24h`, `uncertain_ratio`, `direct_news_ratio`, `unique_source_count`, `has_news`) use the last 72h (`WINDOW_H`).
- `sentiment = p_pos - p_neg`, unweighted mean. Uncertain (abstained) results are excluded from sentiment means and counted in `uncertain_ratio`; they are never treated as neutral.
- No news => `has_news=false`, counts 0, sentiment/ratios `None`. Imputation is the consumer's explicit choice (the trial forecaster uses 0 together with `has_news`).
- Lineage: `snapshot_lineage` lists the result ids that fed each snapshot.

## Aggregation unit
`one passage per (article version, company)` -> article-company result -> near-duplicate rule (only the earliest article of a cluster is representative) -> company-cutoff features. Articles linking 4+ companies are `market` (list articles) and are excluded from per-company features.

## Near-duplicate clusters
Built sequentially in `published_at` order against articles from the previous 7 days (word 5-shingle Jaccard >= 0.8). A later article never changes an earlier article's cluster.

## Trading calendar
Forecast horizons T+2/T+5 are counted in trading days of the price table (rows are exchange sessions), never calendar days.

## Leakage rules
- NLP model for an evaluation period: trained only on dev-stream labels from earlier periods; freeze the checkpoint, then evaluate forecasting later.
- Train/test split of forecast samples: train rows are purged so that no label window (t..t+h) overlaps the test period.
- Scalers, thresholds (abstention) fitted on train/validation only. Eval stream is never used for tuning.
