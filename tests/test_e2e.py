import tempfile
import unittest
from datetime import datetime, timezone

from nlp_module import pipeline, storage
from nlp_module.connectors.base import ParsedDoc
from nlp_module.entities import Linker, load_aliases
from nlp_module.sentiment import lexicon


def doc(url, title, body, pub):
    return ParsedDoc("cafef", url, "news", title, body, pub, None, b"<html/>")


class E2ETest(unittest.TestCase):
    def test_article_to_cutoff_feature_without_leakage(self):
        c, raw = storage.connect(":memory:"), tempfile.mkdtemp()
        storage.ingest(c, doc("https://x.vn/1", "VCB báo lãi kỷ lục", "Vietcombank công bố lợi nhuận tăng mạnh, vượt kế hoạch.", "2026-10-20T02:00:00Z"), raw)
        storage.ingest(c, doc("https://x.vn/2", "FPT bị xử phạt", "Công ty FPT bị xử phạt vì vi phạm công bố thông tin.", "2026-10-22T02:00:00Z"), raw)  # after cutoff
        n = pipeline.process(c, lexicon, Linker(load_aliases()))
        self.assertEqual(n, 2)
        cut = datetime(2026, 10, 20, 8, 0, tzinfo=timezone.utc)
        snaps = {s.symbol: s for s in pipeline.snapshots(c, ["VCB", "FPT"], [cut], "backtest", lexicon.version)}
        self.assertTrue(snaps["VCB"].features["has_news"])
        self.assertGreater(snaps["VCB"].features["sentiment_mean"], 0.3)
        self.assertFalse(snaps["FPT"].features["has_news"])        # published after the cutoff
        self.assertEqual(snaps["FPT"].lineage, [])
        self.assertEqual(c.execute("SELECT COUNT(*) FROM snapshot_lineage").fetchone()[0], 1)
        ev = pipeline.evidence(c, "VCB")
        self.assertEqual(ev[0]["url"], "https://x.vn/1")


if __name__ == "__main__":
    unittest.main()
