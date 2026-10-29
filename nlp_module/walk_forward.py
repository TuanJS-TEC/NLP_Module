"""Walk-forward evaluation: per fold, retrain PhoBERT on labels published before the fold's validation window,
re-score all articles with it, then pick forecaster settings on validation and report once on test.
python -m nlp_module.walk_forward [--prices data/prices.csv]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import annotate, fundamentals, storage
from . import trial_forecast as tf
from .entities import Linker, load_aliases
from .pipeline import load_results, process

# (name, train_start, val_start, test_start, test_end); windows are [start, next start)
FOLDS = [("A", "2022-01-01", "2024-01-01", "2024-07-01", "2025-01-01"),
         ("B", "2022-01-01", "2025-01-01", "2025-07-01", "2026-01-01"),
         ("C", "2022-01-01", "2026-01-01", "2026-07-01", "2026-12-31")]
PRICE = ["ret1", "ret5", "vol10"]
CS = [0.01, 0.1, 1.0, 10.0]
RANK_COLS = {"none": [], "counts": tf.COUNTS, "all": tf.COUNTS + tf.SENT}   # cross-sectional de-drifting


def nlp_model(c, name, val_start, test_start, out_root="data/models"):
    """PhoBERT trained on dev labels before val_start, epoch chosen on dev labels in [val_start, test_start)."""
    from .sentiment import phobert
    X, y, rows = annotate.dataset(c, "dev")
    tr = [i for i, r in enumerate(rows) if r["published_at"] < val_start]
    va = [i for i, r in enumerate(rows) if val_start <= r["published_at"] < test_start]
    out = Path(out_root) / f"phobert-fold{name}"
    f1 = phobert.train([X[i] for i in tr], [y[i] for i in tr], [X[i] for i in va], [y[i] for i in va], out)
    print(f"fold {name}: NLP train {len(tr)} labels, val {len(va)} labels, best val macro-F1 {f1:.3f}")
    version = f"phobert-fold{name}"
    c.execute("DELETE FROM article_company_result WHERE nlp_model_version=?", (version,))  # retrain => rescore
    process(c, phobert.PhoBert(str(out), version), Linker(load_aliases()))
    return version


def _fit(d, cols, h, C):
    sc = StandardScaler().fit(d[cols])
    return sc, LogisticRegression(C=C, max_iter=1000).fit(sc.transform(d[cols]), d[f"y{h}"])


def _auc(model, d, cols, h):
    sc, m = model
    return roc_auc_score(d[f"y{h}"], m.predict_proba(sc.transform(d[cols]))[:, 1])


def _split(d, h, train_start, val_start, test_start, test_end):
    d = d.dropna(subset=PRICE + [f"y{h}"]).sort_values("date")
    dates = sorted(d.date.unique())
    pos = lambda t: next(i for i, x in enumerate(dates) if x >= pd.Timestamp(t))
    i0, iv, it = pos(train_start), pos(val_start), pos(test_start)
    # purge h sessions before each boundary: a label at day i looks h sessions ahead
    return (d[d.date.isin(dates[i0:iv - h])], d[d.date.isin(dates[iv:it])],
            d[d.date.isin(dates[i0:it - h])], d[(d.date >= dates[it]) & (d.date < pd.Timestamp(test_end))])


def forecast_fold(ds, h, train_start, val_start, test_start, test_end, extras=None):
    """Feature set, cross-sectional rank mode and C are ALL chosen on validation; test is scored once.
    ds maps rank mode -> dataframe. Rank mode is meaningless without text features, so price-only uses 'none'.
    extras {name: cols} adds price+name and price+text+name (keys 'fund', 'text+fund/<rank>'); ds must carry cols."""
    res, n = {}, None
    sets = [("price", PRICE), ("text", PRICE + tf.TEXT)]
    for g, cols in (extras or {}).items():
        sets += [(g, PRICE + cols), (f"text+{g}", PRICE + tf.TEXT + cols)]
    for fs, cols in sets:
        for rk in (RANK_COLS if fs.startswith("text") else ("none",)):
            tr, va, trva, te = _split(ds[rk], h, train_start, val_start, test_start, test_end)
            n = (len(tr), len(va), len(te))
            val = {C: _auc(_fit(tr, cols, h, C), va, cols, h) for C in CS}
            C = max(val, key=val.get)
            res[f"{fs}/{rk}" if fs.startswith("text") else fs] = {
                "C": C, "val_auc": val[C], "test_auc": _auc(_fit(trva, cols, h, C), te, cols, h)}
    pick = max(res, key=lambda k: res[k]["val_auc"])
    return res, pick, *n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default="data/prices.csv")
    ap.add_argument("--skip-nlp", action="store_true", help="reuse phobert-fold* results already in the DB")
    ap.add_argument("--target", default="direction", choices=["direction", "excess", "absmove"])
    ap.add_argument("--fund", action="store_true", help="add HOSE earnings-change features to the val search")
    ap.add_argument("--bctc", action="store_true", help="add annual audited-statement features to the val search")
    a = ap.parse_args(argv)
    c = storage.connect()
    prices = tf.load_prices(a.prices)
    for h in (2, 5):
        prices = tf.add_targets(prices, h, a.target)
    prices = prices[prices.date >= "2021-10-21"]   # first collected news day
    extras = {k: v for k, v in (("fund", fundamentals.FUND), ("bctc", fundamentals.BCTC)) if getattr(a, k)}
    rows = []
    for name, t0, v0, s0, e0 in FOLDS:
        version = f"phobert-fold{name}" if a.skip_nlp else nlp_model(c, name, v0, s0)
        res = load_results(c, version)
        d = pd.concat(tf.add_text(g.copy(), [r for r in res if r.symbol == s], "backtest", version)
                      for s, g in prices.groupby("symbol"))
        if a.fund:
            d = fundamentals.add_fund(d, fundamentals.load(c))
        if a.bctc:
            d = fundamentals.add_bctc(d, fundamentals.bctc_features())
        ds = {rk: (tf.rank_cs(d, cols) if cols else d) for rk, cols in RANK_COLS.items()}
        for h in (2, 5):
            r, pick, ntr, nva, nte = forecast_fold(ds, h, t0, v0, s0, e0, extras)
            best = max((k for k in r if k.startswith("text/")), key=lambda k: r[k]["val_auc"])
            rows.append({"fold": name, "target": a.target, "h": h, "n_train": ntr, "n_val": nva, "n_test": nte,
                         "price_val": r["price"]["val_auc"], "price_test": r["price"]["test_auc"],
                         "text_best": best, "text_val": r[best]["val_auc"], "text_test": r[best]["test_auc"],
                         "picked": pick, "picked_test": r[pick]["test_auc"]}
                        | {f"{g}_{k}": r[g][f"{k}_auc"] for g in extras for k in ("val", "test")})
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    out = pd.DataFrame(rows).round(3)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
