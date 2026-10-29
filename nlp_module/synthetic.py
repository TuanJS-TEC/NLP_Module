"""Synthetic article-company results so consumers (trial forecaster, website) can start against the contract."""
from datetime import timedelta
from .contract import ArticleCompanyResult, utc

T0 = utc(2026, 10, 19, 2, 0)


def synthetic_results():
    def mk(i, symbol, hours, ppos, pneg, **kw):
        pub = T0 + timedelta(hours=hours)
        return ArticleCompanyResult(f"r{i}", symbol, kw.pop("source", "source_a"), pub,
                                    pub + timedelta(minutes=kw.pop("lag_min", 10)),
                                    kw.pop("relation", "direct"), ppos, 1 - ppos - pneg, pneg, **kw)
    return [
        mk(1, "AAA", 0, 0.7, 0.1),
        mk(2, "AAA", 5, 0.1, 0.8, source="source_b"),
        mk(3, "AAA", 6, 0.1, 0.8, source="source_b", is_representative=False),  # repost
        mk(4, "AAA", 30, 0.4, 0.4, uncertain=True),
        mk(7, "AAA", 7, 0.6, 0.2, difficult=True),      # weak evidence: still counted in sentiment_mean
        mk(5, "AAA", 50, 0.9, 0.0),                                             # future vs. test cutoff
        mk(6, "BBB", 1, 0.2, 0.2, relation="sector"),
    ]
