import numpy as np
import pandas as pd

from nlp_module import trial_forecast as tf
from nlp_module.walk_forward import PRICE, RANK_COLS, forecast_fold


def test_fold_windows_are_purged_and_disjoint():
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    rng = np.random.default_rng(0)
    d = pd.concat(pd.DataFrame({"date": dates, "symbol": s}) for s in ("AAA", "BBB")).reset_index(drop=True)
    for col in PRICE + tf.TEXT:
        d[col] = rng.normal(size=len(d))
    h = 5
    d[f"y{h}"] = rng.integers(0, 2, len(d)).astype(float)
    ds = {rk: (tf.rank_cs(d, cols) if cols else d) for rk, cols in RANK_COLS.items()}
    res, pick, n_tr, n_va, n_te = forecast_fold(ds, h, "2022-01-01", "2024-01-01", "2024-07-01", "2025-01-01")
    i_val = list(dates).index(pd.Timestamp("2024-01-01"))
    i_test = list(dates).index(pd.Timestamp("2024-07-01"))
    assert n_tr == 2 * (i_val - h)              # last h sessions before validation dropped
    assert n_va == 2 * (i_test - i_val)
    assert n_te == 2 * (len(dates) - i_test)
    assert pick in res and pick in ("price", "text/none", "text/counts", "text/all")
    assert set(res) == {"price", "text/none", "text/counts", "text/all"}   # rank mode is part of the val search
