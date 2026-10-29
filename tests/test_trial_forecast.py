import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from nlp_module import trial_forecast as tf
from nlp_module.contract import ArticleCompanyResult


class TrialForecastTest(unittest.TestCase):
    def test_text_features_help_when_signal_exists_and_purge_holds(self):
        rng = np.random.default_rng(0)
        days = pd.bdate_range("2025-01-01", periods=400)
        sent = rng.choice([-1, 0, 1], size=len(days))
        # price at t+1 moves with the sentiment of articles published on day t (signal), plus noise
        ret = np.r_[0, 0.012 * sent[:-1]] + rng.normal(0, 0.006, len(days))
        close = 100 * np.cumprod(1 + ret)
        path = tempfile.mktemp(suffix=".csv")
        pd.DataFrame({"date": days, "symbol": "AAA", "close": close}).to_csv(path, index=False)
        results = []
        for i, (d, s) in enumerate(zip(days, sent)):
            if s:
                pub = datetime(d.year, d.month, d.day, 2, 0, tzinfo=timezone.utc)   # 09:00 ICT, before 15:00 close
                p = (0.8, 0.1, 0.1) if s > 0 else (0.1, 0.1, 0.8)
                results.append(ArticleCompanyResult(f"r{i}", "AAA", "s", pub, pub, "direct", *p))
        df = tf.add_targets(tf.load_prices(path), 2)
        df = tf.add_text(df, results, "backtest", "baseline-v0")
        with_text, price_only = tf.run(df, 2, use_text=True), tf.run(df, 2, use_text=False)
        self.assertGreater(with_text["auc"], price_only["auc"] + 0.05)
        self.assertLess(with_text["n_train"] + with_text["n_test"], len(days))   # embargo/purge removed rows


    def test_cross_sectional_rank_removes_a_level_shift(self):
        # same cross-sectional ORDER every day, but the level doubles in the second half (corpus growth)
        days = pd.bdate_range("2025-01-01", periods=20)
        d = pd.DataFrame([{"date": day, "symbol": s, "news_count_72h": (i + 1) * (1 + (k >= 10))}
                          for k, day in enumerate(days) for i, s in enumerate("ABC")])
        half = days[10]
        before, after = tf.drift(d, ["news_count_72h"], half)["news_count_72h"]
        self.assertAlmostEqual(after, before * 2)                      # the drift the forecaster trips on
        r = tf.rank_cs(d, ["news_count_72h"])
        lo, hi = tf.drift(r, ["news_count_72h"], half)["news_count_72h"]
        self.assertEqual(lo, hi)                                       # rank is invariant to the level shift
        self.assertEqual(sorted(r[r.date == days[0]]["news_count_72h"].round(3)), [0.333, 0.667, 1.0])


    def test_excess_target_strips_the_market_move(self):
        # every ticker rises, so raw direction is all 1s and unlearnable; AAA rises fastest, so only it beats the mean
        days = pd.bdate_range("2025-01-01", periods=10)
        rows = [{"date": d, "symbol": s, "close": 100 * (1 + g) ** i}
                for i, d in enumerate(days) for s, g in (("AAA", 0.03), ("BBB", 0.01), ("CCC", 0.01))]
        df = pd.DataFrame(rows).sort_values(["symbol", "date"])
        raw = tf.add_targets(df.copy(), 2)["y2"].dropna()
        exc = tf.add_targets(df.copy(), 2, "excess")
        self.assertEqual(set(raw), {1.0})                                      # no information left in direction
        got = exc.dropna(subset=["y2"]).groupby("symbol")["y2"].mean().to_dict()
        self.assertEqual(got, {"AAA": 1.0, "BBB": 0.0, "CCC": 0.0})


    def test_absmove_target_ignores_sign_and_ranks_magnitude(self):
        # AAA swings hardest, BBB falls hard, CCC barely moves: magnitude must flag AAA and BBB, not CCC
        days = pd.bdate_range("2025-01-01", periods=10)
        rows = [{"date": d, "symbol": s, "close": 100 * (1 + g) ** i}
                for i, d in enumerate(days) for s, g in (("AAA", 0.05), ("BBB", -0.04), ("CCC", 0.001))]
        df = pd.DataFrame(rows).sort_values(["symbol", "date"])
        got = tf.add_targets(df.copy(), 2, "absmove").dropna(subset=["y2"]).groupby("symbol")["y2"].mean().to_dict()
        self.assertEqual(got, {"AAA": 1.0, "BBB": 0.0, "CCC": 0.0})   # median of 3 => only the largest clears it
        self.assertEqual(tf.add_targets(df.copy(), 2, "excess").dropna(subset=["y2"])
                         .groupby("symbol")["y2"].mean()["BBB"], 0.0)   # BBB loses on direction, so the two differ


if __name__ == "__main__":
    unittest.main()
