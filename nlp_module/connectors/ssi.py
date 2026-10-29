"""SSI iBoard price board: live quotes, for comparing model sentiment against what the price actually did.

Research use at a polite rate, same policy as the news sources (docs/source_terms.md). Quotes are market data,
not article text, so nothing here is redistributed content - but the endpoint is undocumented and may change,
so `quotes()` fails loudly rather than returning partial rows.
"""
import json
import logging
from dataclasses import dataclass

SOURCE = "ssi-iboard"
BOARD = "https://iboard-query.ssi.com.vn/stock/exchange/{exchange}"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    """One symbol at one moment. Prices in VND (the board's own unit), not thousand VND like data/prices.csv."""
    symbol: str
    trading_date: str       # YYYYMMDD as reported
    session: str            # SSI session code: P=pre-open, O/K=continuous, A=ATC, C=closed, ...
    ref_price: float        # reference price for the day (yesterday's close, adjusted)
    matched_price: float    # last matched price; 0 before the first match -> we fall back to ref_price
    price_change: float
    price_change_pct: float
    highest: float
    lowest: float
    avg_price: float
    volume: float
    matched: bool           # False = no trade yet today, so price_change is 0 by construction


def _f(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def quotes(fetcher, symbols=None, exchange="hose"):
    """-> [Quote] for `symbols` (all listed symbols when None). One request for the whole board."""
    url = BOARD.format(exchange=exchange)
    d = json.loads(fetcher.get(url))
    rows = d.get("data") or []
    if not rows:
        raise RuntimeError(f"SSI board {exchange} returned no rows (code={d.get('code')!r}, "
                           f"message={d.get('message')!r}); the endpoint may have changed")
    want = set(symbols) if symbols else None
    out, skipped = [], 0
    for r in rows:
        sym = r.get("stockSymbol")
        if not sym or (want and sym not in want):
            continue
        ref, mp = _f(r.get("refPrice")), _f(r.get("matchedPrice"))
        matched = mp > 0
        if not matched:
            mp = ref
            skipped += 1
        out.append(Quote(sym, str(r.get("tradingDate") or ""), str(r.get("session") or r.get("exchangeSession") or ""),
                         ref, mp, _f(r.get("priceChange")), _f(r.get("priceChangePercent")),
                         _f(r.get("highest")), _f(r.get("lowest")), _f(r.get("avgPrice")),
                         _f(r.get("nmTotalTradedQty")), matched))
    log.info("ssi %s: %d rows on the board, %d kept, %d with no trade yet (price = reference)",
             exchange, len(rows), len(out), skipped)
    if want:
        missing = want - {q.symbol for q in out}
        if missing:
            log.warning("ssi %s: not on this board: %s", exchange, " ".join(sorted(missing)))
    for q in out:
        log.debug("  %-4s session=%s ref=%.0f matched=%.0f change=%+.0f (%+.2f%%) vol=%.0f%s",
                  q.symbol, q.session, q.ref_price, q.matched_price, q.price_change, q.price_change_pct,
                  q.volume, "" if q.matched else "  NO-TRADE")
    return out
