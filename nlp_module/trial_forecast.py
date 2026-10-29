"""Minimal trial forecaster: proves text features are readable by a forecaster and runs the price-only vs
price+text ablation. NOT the real fusion model. Input: OHLCV CSV with columns date,symbol,close
(rows = trading days, so T+h follows the exchange calendar by construction).
python -m nlp_module.trial_forecast [--model-version baseline-tfidf-lr-v1]"""
import argparse
from datetime import datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .contract import build_snapshot

CLOSE_UTC = time(8, 0)   # HOSE close 15:00 Asia/Ho_Chi_Minh = 08:00 UTC; forecast_cutoff = close of day t
TEXT = ["has_news", "news_count_24h", "news_count_72h", "sentiment_mean", "sentiment_mean_24h", "uncertain_ratio", "direct_news_ratio", "difficult_ratio"]
COUNTS = ["news_count_24h", "news_count_72h"]          # absolute levels: drift with corpus growth
SENT = ["sentiment_mean", "sentiment_mean_24h"]


def load_prices(path):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["symbol", "date"])
    g = df.groupby("symbol")["close"]
    df["ret1"], df["ret5"] = g.pct_change(1), g.pct_change(5)
    df["vol10"] = df.groupby("symbol")["ret1"].transform(lambda s: s.rolling(10).std())
    return df


def add_targets(df, h, target="direction"):
    """direction: the close rose over the next h sessions. excess: the ticker beat the market, proxied equal-weight
    by the mean forward return of the tickers trading that day. absmove: the ticker's |forward return| beat the
    cross-sectional median that day - a MAGNITUDE target, where news plausibly carries signal that direction does
    not (attention -> trading -> larger moves, either way). The cross-sectional comparison in excess/absmove removes
    the market-wide level, so neither target drifts, and both sit at a ~50% base rate by construction. These read
    forward returns, which is legitimate: they are the label, not a feature."""
    fwd = df.groupby("symbol")["close"].shift(-h)
    r = fwd / df["close"] - 1
    if target == "direction":
        y = fwd > df["close"]
    elif target == "excess":
        y = r > r.groupby(df["date"]).transform("mean")
    elif target == "absmove":
        y = r.abs() > r.abs().groupby(df["date"]).transform("median")
    else:
        raise ValueError(f"unknown target {target!r}")
    df[f"y{h}"] = y.astype(float)
    df.loc[fwd.isna(), f"y{h}"] = np.nan   # no label near the end of data
    return df


def add_text(df, results, mode, model_version):
    rows = []
    for r in df.itertuples():
        cut = datetime.combine(r.date.date(), CLOSE_UTC, tzinfo=timezone.utc)
        f = build_snapshot(results, r.symbol, cut, mode, model_version).features
        rows.append({k: f[k] for k in TEXT})
    t = pd.DataFrame(rows, index=df.index)
    t["has_news"] = t["has_news"].astype(float)
    # imputation policy (explicit): no news => sentiment 0 AND has_news=0, never read as a neutral article.
    # Since abstention went live, sentiment_mean is also missing when the whole window abstained (10.7% of news
    # rows): those get sentiment 0 WITH has_news=1, and `uncertain_ratio` = 1 is what marks them as imputed.
    return pd.concat([df, t.fillna(0.0)], axis=1)


def rank_cs(df, cols):
    """Replace cols by their cross-sectional rank within each date, in [0,1]. A rank is invariant to any level
    shift that hits every ticker, so corpus growth (mean news_count_72h doubles 2022->2026) stops separating
    train from test. Uses only same-day values, all available at the cutoff, so no look-ahead."""
    out = df.copy()
    for col in cols:
        out[col] = df.groupby("date")[col].rank(pct=True)
    return out


def drift(df, cols, test_start):
    """Mean of each feature before vs. from test_start: what the forecaster sees shift under its feet."""
    a, b = df[df.date < test_start], df[df.date >= test_start]
    return {c: (round(a[c].mean(), 3), round(b[c].mean(), 3)) for c in cols}


def run(df, h, test_frac=0.3, use_text=True, test_start=None, extra=()):
    """Time split with an embargo of h trading days between train and test (label windows must not overlap).
    test_start pins the first test session by date; it must be at/after the last label the NLP model was
    trained on, or the text features have seen the test period."""
    d = df.dropna(subset=["ret1", "ret5", "vol10", f"y{h}"]).sort_values("date")
    dates = sorted(d["date"].unique())
    k = (next((i for i, x in enumerate(dates) if x >= pd.Timestamp(test_start)), len(dates))
         if test_start else int(len(dates) * (1 - test_frac)))
    # purge: a train label at date i looks h days ahead, so train rows must satisfy i + h < k (no overlap with test)
    tr, te = d[d["date"].isin(dates[:max(k - h, 0)])], d[d["date"].isin(dates[k:])]
    cols = ["ret1", "ret5", "vol10"] + (TEXT if use_text else []) + list(extra)
    sc = StandardScaler().fit(tr[cols])     # scaler fit on train only
    m = LogisticRegression(max_iter=1000).fit(sc.transform(tr[cols]), tr[f"y{h}"])
    p = m.predict_proba(sc.transform(te[cols]))[:, 1]
    return {"h": h, "text": use_text, "n_train": len(tr), "n_test": len(te),
            "auc": float(roc_auc_score(te[f"y{h}"], p)) if te[f"y{h}"].nunique() > 1 else float("nan"),
            "test": pd.DataFrame({"date": te["date"].values, "y": te[f"y{h}"].values, "p": p})}


def diff_ci(text, price, boot=2000, seed=0):
    """95% CI for AUC(price+text) - AUC(price), resampling whole DATES: rows on one day share the market move."""
    t, q = text.reset_index(drop=True), price.reset_index(drop=True)
    assert (t["date"].values == q["date"].values).all() and (t["y"].values == q["y"].values).all()
    by = {k: g.index.values for k, g in t.groupby("date")}
    dates, rng, out = list(by), np.random.default_rng(seed), []
    y, pt, pq = t["y"].values, t["p"].values, q["p"].values
    for _ in range(boot):
        i = np.concatenate([by[dates[j]] for j in rng.integers(0, len(dates), len(dates))])
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], pt[i]) - roc_auc_score(y[i], pq[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main(argv=None):
    from . import fundamentals, storage
    from .pipeline import load_results
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/prices.csv")
    ap.add_argument("--model-version", default="baseline-tfidf-lr-v1")
    ap.add_argument("--start", default="2021-10-21", help="first collected news day")
    ap.add_argument("--test-start", default="2025-09-17",
                    help="first test session; default = after PhoBERT v1's training labels. '' uses --test-frac")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--target", default="direction", choices=["direction", "excess", "absmove"],
                    help="excess = beat the equal-weight market proxy; absmove = |move| beat the day's median")
    ap.add_argument("--rank", default="counts", choices=["none", "counts", "all"],
                    help="cross-sectional rank: none = raw levels (drifts), counts = news_count_*, all = + sentiment_mean_*")
    ap.add_argument("--fund", action="store_true",
                    help="add HOSE earnings-change features (fundamentals.FUND): price+fund, price+text+fund")
    a = ap.parse_args(argv)

    df = load_prices(a.prices)
    for h in (2, 5):
        df = add_targets(df, h, a.target)
    df = df[df.date >= a.start]
    c = storage.connect()
    res = load_results(c, a.model_version)
    d = pd.concat(add_text(g.copy(), [r for r in res if r.symbol == s], "backtest", a.model_version)
                  for s, g in df.groupby("symbol"))
    if a.fund:
        d = fundamentals.add_fund(d, fundamentals.load(c))
        print("has_fund share: all %.3f, test %.3f" % (d.has_fund.mean(), d[d.date >= a.test_start].has_fund.mean()))
    cols = {"none": [], "counts": COUNTS, "all": COUNTS + SENT}[a.rank]
    if cols:
        print("drift (mean before / from test_start), raw:", drift(d, cols, a.test_start))
        d = rank_cs(d, cols)
        print("drift after cross-sectional rank:        ", drift(d, cols, a.test_start))
    rows = []
    for h in (2, 5):
        kw = {"test_start": a.test_start or None, "test_frac": a.test_frac}
        pr, tx = run(d, h, use_text=False, **kw), run(d, h, use_text=True, **kw)
        lo, hi = diff_ci(tx["test"], pr["test"], a.boot)
        rows.append({"model": a.model_version, "target": a.target, "rank": a.rank, "h": h,
                     "n_train": pr["n_train"], "n_test": pr["n_test"],
                     "auc_price": pr["auc"], "auc_price_text": tx["auc"],
                     "diff": tx["auc"] - pr["auc"], "ci_lo": lo, "ci_hi": hi})
        if a.fund:
            for name, use_text in (("fund", False), ("text_fund", True)):
                r = run(d, h, use_text=use_text, extra=fundamentals.FUND, **kw)
                lo, hi = diff_ci(r["test"], pr["test"], a.boot)
                rows[-1] |= {f"auc_price_{name}": r["auc"], f"diff_{name}": r["auc"] - pr["auc"],
                             f"ci_lo_{name}": lo, f"ci_hi_{name}": hi}
        print(rows[-1])
    out = pd.DataFrame(rows).round(4)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
