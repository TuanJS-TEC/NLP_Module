import unittest
from nlp_module.contract import build_snapshot, utc
from nlp_module.synthetic import synthetic_results

R = synthetic_results()
CUTOFF = utc(2026, 10, 20, 9, 0)  # 31h after T0: r5 (T0+50h) must be invisible


class SnapshotTest(unittest.TestCase):
    def test_no_future_articles_and_reposts_dropped(self):
        s = build_snapshot(R, "AAA", CUTOFF)
        self.assertEqual(s.lineage, ["r1", "r2", "r4", "r7"])  # no r5 (future), no r3 (repost)

    def test_uncertain_not_neutral(self):
        f = build_snapshot(R, "AAA", CUTOFF).features
        self.assertAlmostEqual(f["uncertain_ratio"], 1 / 4)
        self.assertAlmostEqual(f["sentiment_mean"],
                               ((0.7 - 0.1) + (0.1 - 0.8) + (0.6 - 0.2)) / 3)  # r4 excluded, r7 kept

    def test_difficult_is_reported_but_still_counted(self):
        f = build_snapshot(R, "AAA", CUTOFF).features
        self.assertAlmostEqual(f["difficult_ratio"], 1 / 4)        # r7 of r1,r2,r4,r7
        self.assertIsNone(build_snapshot(R, "CCC", CUTOFF).features["difficult_ratio"])   # no news => missing

    def test_no_news_is_missing_not_neutral(self):
        f = build_snapshot(R, "CCC", CUTOFF).features
        self.assertFalse(f["has_news"])
        self.assertEqual(f["news_count_24h"], 0)
        self.assertIsNone(f["sentiment_mean"])

    def test_live_requires_processing_before_cutoff(self):
        # r1 published at T0, processed T0+10min; cutoff 5 min after publication => not yet usable live
        s = build_snapshot(R, "AAA", utc(2026, 10, 19, 2, 5))
        self.assertEqual(s.lineage, [])


if __name__ == "__main__":
    unittest.main()


class WindowTest(unittest.TestCase):
    def test_old_articles_leave_sentiment_window_but_not_recency(self):
        s = build_snapshot(R, "AAA", utc(2026, 10, 25, 0, 0))   # everything is > 72h old except nothing
        f = s.features
        self.assertFalse(f["has_news"])
        self.assertIsNone(f["sentiment_mean"])
        self.assertGreater(f["hours_since_last_news"], 72)
        self.assertEqual(s.lineage, [])
