"""Data contract: article-company results -> company-cutoff feature snapshot.

Stdlib only. Swapping baseline -> PhoBERT changes `nlp_model_version`, nothing else.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev

FEATURE_VERSION = "v2"   # v2 adds difficult_ratio
WINDOW_H = 72   # lookback for sentiment/quality features; news_count_24h/72h are fixed windows
# ponytail: fixed backtest delay; make it per-source if timestamps prove unreliable
BACKTEST_DELAY = timedelta(minutes=30)


@dataclass(frozen=True)
class ArticleCompanyResult:
    """One row per (article version, company). Passages are already collapsed."""
    result_id: str
    symbol: str
    source: str
    published_at: datetime        # UTC, as reported by the source
    processed_at: datetime        # UTC, when the NLP pipeline finished (live mode)
    relation: str                 # direct | sector | market | ambiguous
    p_pos: float
    p_neu: float
    p_neg: float
    uncertain: bool = False       # abstained: kept, but never counted as neutral
    difficult: bool = False       # weak evidence: another company in the passage, or title-only (no body mention)
    is_representative: bool = True  # one counted article per near-duplicate cluster
    nlp_model_version: str = "baseline-v0"


@dataclass
class Snapshot:
    symbol: str
    forecast_cutoff: datetime
    feature_version: str
    nlp_model_version: str
    availability_mode: str
    features: dict = field(default_factory=dict)
    lineage: list = field(default_factory=list)  # result_ids that fed the features


def available_at(r: ArticleCompanyResult, mode: str) -> datetime:
    if mode == "live":
        return r.processed_at
    if mode == "backtest":  # simulated: assume availability from publication time
        return r.published_at + BACKTEST_DELAY
    raise ValueError(mode)


def build_snapshot(results, symbol, cutoff, mode="live", nlp_model_version="baseline-v0") -> Snapshot:
    assert cutoff.tzinfo is not None, "cutoff must be tz-aware UTC"
    avail = [r for r in results
             if r.symbol == symbol and r.nlp_model_version == nlp_model_version
             and r.is_representative and r.relation != "market"  # list/market articles: no per-stock sentiment
             and available_at(r, mode) <= cutoff]
    age = lambda r: cutoff - available_at(r, mode)
    win = [r for r in avail if age(r) <= timedelta(hours=WINDOW_H)]   # sentiment/quality features use this window
    # ponytail: uncertain results excluded from sentiment means (never counted as neutral); unweighted,
    # recency weighting is a later experiment
    def sent(rs):
        s = [r.p_pos - r.p_neg for r in rs if not r.uncertain]
        return (mean(s) if s else None), (pstdev(s) if len(s) > 1 else None)

    f = {"has_news": bool(win),
         "news_count_24h": sum(age(r) <= timedelta(hours=24) for r in avail),
         "news_count_72h": len(win),
         "unique_source_count": len({r.source for r in win}),
         "hours_since_last_news": min(age(r) for r in avail).total_seconds() / 3600 if avail else None,
         "uncertain_ratio": sum(r.uncertain for r in win) / len(win) if win else None,
         # difficult rows stay in sentiment_mean (unlike uncertain); this exposes how much of the window is
         # weak evidence so the consumer can discount it, without changing what sentiment_mean has always meant
         "difficult_ratio": sum(r.difficult for r in win) / len(win) if win else None,
         "direct_news_ratio": sum(r.relation == "direct" for r in win) / len(win) if win else None}
    f["sentiment_mean"], f["sentiment_std"] = sent(win)
    f["sentiment_mean_24h"], _ = sent([r for r in win if age(r) <= timedelta(hours=24)])
    # None = missing; imputation is the consumer's explicit policy (no news != neutral)
    return Snapshot(symbol, cutoff, FEATURE_VERSION, nlp_model_version, mode, f, [r.result_id for r in win])


def utc(*a) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)
