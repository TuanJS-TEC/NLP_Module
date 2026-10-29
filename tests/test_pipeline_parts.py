import tempfile
import unittest
from datetime import date

from nlp_module import dedup, storage
from nlp_module.connectors import cafef, vietstock
from nlp_module.connectors.base import ParsedDoc, sitemap_entries, to_utc_iso
from nlp_module.entities import Linker, load_aliases
from nlp_module.passages import select_passage
from nlp_module.text import fold, normalize, split_sentences

L = Linker(load_aliases())


class TextTest(unittest.TestCase):
    def test_sentences_offsets_and_numbers(self):
        t = normalize("SSI lãi 3.219 tỷ đồng. Doanh thu giảm 24%.  VPS không đổi.")
        s = split_sentences(t)
        self.assertEqual(len(s), 3)
        for _, a, b, txt in s:
            self.assertEqual(t[a:b], txt)

    def test_keeps_negation_and_diacritics(self):
        self.assertIn("không", normalize("Lợi nhuận không tăng"))
        self.assertEqual(fold("Đầu tư Hòa Phát"), "dau tu hoa phat")


class EntityTest(unittest.TestCase):
    def test_direct_title_and_brand(self):
        ls = L.link("SSI báo lãi quý 3", "Chứng khoán SSI ghi nhận lãi. Vinhomes không liên quan.")
        d = {l.symbol: l for l in ls}
        self.assertEqual(d["SSI"].relation, "direct")
        self.assertEqual(d["VHM"].relation, "ambiguous")  # one body mention only

    def test_currency_and_city_not_company(self):
        ls = L.link("", "Tổng vốn 5.000 tỷ VND tại TP.HCM, cổ phiếu HCM tăng.")
        self.assertEqual([l.symbol for l in ls], ["HCM"])  # VND currency and TP.HCM place skipped

    def test_alias_validity_window(self):
        rows = [dict(entity_id="X", symbol="X", legal_name="", alias="OldName", alias_type="brand",
                     valid_from="2020-01-01", valid_to="2021-12-31")]
        lk = Linker(rows)
        self.assertTrue(lk.find("OldName tăng", "2021-06-01"))
        self.assertFalse(lk.find("OldName tăng", "2023-06-01"))


class PassageTest(unittest.TestCase):
    def test_two_companies_get_own_passages_and_difficult_flag(self):
        body = normalize("Công ty VCB báo lãi tăng mạnh trong quý. Trong khi đó BID ghi nhận lỗ do chi phí tài chính.")
        sents = split_sentences(body)
        links = {l.symbol: l for l in L.link("", body)}
        out = {}
        for sym, l in links.items():
            others = [m for o in links.values() if o is not l for m in o.mentions]
            out[sym] = select_passage("", sents, l, others)
        self.assertEqual(out["VCB"]["sentence_ids"], [0])
        self.assertEqual(out["BID"]["sentence_ids"], [1])
        self.assertFalse(out["VCB"]["difficult"])


class StorageDedupTest(unittest.TestCase):
    def setUp(self):
        self.c = storage.connect(":memory:")
        self.raw = tempfile.mkdtemp()

    def doc(self, url, body, pub, title="SSI lãi"):
        return ParsedDoc("t", url, "news", title, body, pub, None, b"<html/>")

    def test_same_content_is_one_version_and_change_is_new_version(self):
        v1, new1 = storage.ingest(self.c, self.doc("https://a.vn/x?utm=1", "SSI lãi lớn.", "2026-01-01T00:00:00Z"), self.raw)
        v2, new2 = storage.ingest(self.c, self.doc("https://a.vn/x", "SSI lãi lớn.", "2026-01-01T00:00:00Z"), self.raw)
        v3, new3 = storage.ingest(self.c, self.doc("https://a.vn/x", "SSI lãi lớn hơn dự kiến.", "2026-01-01T00:00:00Z"), self.raw)
        self.assertEqual((new1, new2, new3), (True, False, True))
        self.assertEqual(v1, v2)
        self.assertEqual(self.c.execute("SELECT COUNT(*) FROM document").fetchone()[0], 1)

    def test_repost_is_clustered_sequentially(self):
        body = "Công ty cổ phần chứng khoán SSI công bố kết quả kinh doanh quý ba với lợi nhuận tăng mạnh so với cùng kỳ năm ngoái nhờ mảng tự doanh"
        storage.ingest(self.c, self.doc("https://a.vn/1", body, "2026-01-01T01:00:00Z"), self.raw)
        storage.ingest(self.c, self.doc("https://b.vn/2", body + " .", "2026-01-01T05:00:00Z"), self.raw)
        storage.ingest(self.c, self.doc("https://c.vn/3", "Thị trường hôm nay giảm điểm mạnh do áp lực bán", "2026-01-01T06:00:00Z", "VN-Index"), self.raw)
        dedup.assign_clusters(self.c)
        rep = {r[0]: r[1] for r in self.c.execute("SELECT url_normalized, is_representative FROM document d JOIN document_version v USING(document_id)")}
        self.assertEqual(rep, {"https://a.vn/1": 1, "https://b.vn/2": 0, "https://c.vn/3": 1})


class ConnectorTest(unittest.TestCase):
    def test_sitemap_and_timestamps(self):
        xml = "<urlset><url><loc>https://cafef.vn/a-1882610202045.chn</loc><lastmod>2026-10-20T20:45:58+07:00</lastmod><image:image><image:title>&lt;![CDATA[Tiêu đề A]]&gt;</image:title></image:image></url></urlset>"
        self.assertEqual(list(sitemap_entries(xml)), [("https://cafef.vn/a-1882610202045.chn", "2026-10-20T20:45:58+07:00", "Tiêu đề A")])
        self.assertEqual(to_utc_iso("2026-10-20T18:14:00"), "2026-10-20T11:14:00Z")  # naive = +07:00

    def test_cafef_bucket_selection(self):
        idx = "".join(f"<sitemap><loc>https://cafef.vn/sitemaps/sitemaps-2026-{m}-{a}-{b}.xml</loc></sitemap>" for m, a, b in [(10, 1, 5), (10, 6, 10), (9, 26, 30)])
        self.assertEqual(len(list(cafef.sitemap_files(idx, date(2026, 10, 7), date(2026, 10, 8)))), 1)

    def test_parsers_on_minimal_html(self):
        h = b'<html><head><meta property="article:published_time" content="2026-10-20T18:14:00"/></head><body><h1>T</h1><div class="sapo">S.</div><div class="detail-content"><p>Dong 1.</p><div class="relatednews"><p>TIN LIEN QUAN</p></div><p>Dong 2.</p></div></body></html>'
        d = cafef.parse(h, "u")
        self.assertEqual((d.title, d.body_text, d.published_at), ("T", "S.\nDong 1.\nDong 2.", "2026-10-20T11:14:00Z"))


if __name__ == "__main__":
    unittest.main()


class ListArticleTest(unittest.TestCase):
    def test_many_companies_means_market_level(self):
        ls = L.link("Lãi suất ngân hàng hôm nay", "Vietcombank 4,6%, BIDV 4,7%, VietinBank 4,7%, Techcombank 4,8%, MB Bank 4,9%.")
        self.assertTrue(len(ls) >= 4 and all(l.relation == "market" for l in ls))


class SsiQuotesTest(unittest.TestCase):
    def test_board_parsing_and_no_trade_fallback(self):
        import json
        from nlp_module.connectors import ssi

        board = {"code": "SUCCESS", "data": [
            {"stockSymbol": "VIC", "tradingDate": "20261026", "session": "C", "refPrice": 236500,
             "matchedPrice": 237000, "priceChange": 500, "priceChangePercent": 0.21,
             "highest": 241500, "lowest": 233900, "avgPrice": 238062.36, "nmTotalTradedQty": 2218600},
            {"stockSymbol": "ZZZ", "tradingDate": "20261026", "session": "P", "refPrice": 10000,
             "matchedPrice": 0},                                   # pre-open: no trade yet
            {"stockSymbol": "OTHER", "refPrice": 1, "matchedPrice": 1, "tradingDate": "20261026"},
        ]}

        class FakeFetcher:
            def get(self, url):
                assert "hose" in url, url
                return json.dumps(board).encode()

        qs = {q.symbol: q for q in ssi.quotes(FakeFetcher(), ["VIC", "ZZZ", "NOTLISTED"])}
        self.assertEqual(set(qs), {"VIC", "ZZZ"})                  # OTHER filtered out, NOTLISTED absent
        self.assertTrue(qs["VIC"].matched)
        self.assertAlmostEqual(qs["VIC"].price_change_pct, 0.21)
        self.assertFalse(qs["ZZZ"].matched)                        # no trade
        self.assertEqual(qs["ZZZ"].matched_price, 10000)           # falls back to the reference price
        self.assertEqual(qs["ZZZ"].price_change, 0.0)

    def test_empty_board_fails_loudly(self):
        import json
        from nlp_module.connectors import ssi

        class FakeFetcher:
            def get(self, url):
                return json.dumps({"code": "X", "message": "nope", "data": []}).encode()

        with self.assertRaises(RuntimeError):      # a silently empty board would look like "no price data"
            ssi.quotes(FakeFetcher(), ["VIC"])


class HoseFilingTest(unittest.TestCase):
    """Issuer profit-explanation filings: keep the narrative, score it for the title-prefix ticker."""
    PAGES = ["CÔNG BỐ THÔNG TIN BẤT THƯỜNG\nKính gửi: Sở Giao dịch Chứng khoán\nMã chứng khoán: FPT",
             "Về việc giải trình biến động lợi nhuận\nCông ty xin giải trình như sau:\n"
             "Báo cáo tài chính | 96.349.615.557 | 60.065.383.211| 36.284.232.346 60,41%\n"
             "Lợi nhuận sau thuế quý 3 năm 2026 tăng 60,41% so với cùng kỳ năm trước\n"
             "do doanh thu thuần bán hàng tăng mạnh.\nTrân trọng.\nNơi nhận: như trên"]

    def test_narrative_drops_cover_table_and_signoff(self):
        from nlp_module.connectors.hose import narrative
        t = narrative(self.PAGES)
        self.assertTrue(t.startswith("Lợi nhuận sau thuế quý 3"), t)
        self.assertIn("doanh thu thuần bán hàng tăng mạnh.", t)   # OCR line break re-joined
        for junk in ("Kính gửi", "96.349", "Trân trọng", "Nơi nhận"):
            self.assertNotIn(junk, t)

    def test_narrative_keeps_lowercase_cbtt_page_and_drops_english(self):
        from nlp_module.connectors.hose import narrative
        t = narrative(["V/v giải trình. Về việc công bố thông tin, Ngân hàng xin giải trình như sau:\n"
                       "Lợi nhuận sau thuế riêng của Ngân hàng tăng 45% so với cùng kỳ năm trước.\n"
                       "Net profit after tax of the Bank increased by 45% compared to last year.\n"
                       "Digitally signed by NGÂN HÀNG TMCP VIỆT NAM THỊNH VƯỢNG", "PHỤ LỤC 1"])
        self.assertEqual(t, "Lợi nhuận sau thuế riêng của Ngân hàng tăng 45% so với cùng kỳ năm trước.")

    def test_narrative_two_page_cover_without_nhu_sau(self):
        from nlp_module.connectors.hose import narrative
        t = narrative(["CÔNG BỐ THÔNG TIN ĐỊNH KỲ\nKính gửi: Ủy ban Chứng khoán Nhà nước",
                       "Chúng tôi xin cam kết thông tin trên là đúng sự thật.\nTrân trọng./.",
                       "GIẢI TRÌNH MỘT SỐ CHỈ TIÊU\nLợi nhuận sau thuế quý 2 giảm 30% do sản lượng điện giảm.\n"
                       "Trân trọng."])
        self.assertEqual(t, "Lợi nhuận sau thuế quý 2 giảm 30% do sản lượng điện giảm.")

    def test_process_scores_filing_for_title_ticker(self):
        from nlp_module.connectors.hose import narrative
        from nlp_module.pipeline import process
        from nlp_module.sentiment import lexicon
        c = storage.connect(":memory:")
        storage.ingest(c, ParsedDoc("hose", "https://api.hsx.vn/n/api/v1/1/news/1", "financial_narrative",
                                    "FPT: Giải trình biến động LNST quý 3/2026", narrative(self.PAGES),
                                    "2026-10-23T03:00:00Z", None, b""), tempfile.mkdtemp())
        self.assertEqual(process(c, lexicon, L), 1)
        r = c.execute("SELECT symbol, relation, difficult, passage_text FROM article_company_result").fetchone()
        self.assertEqual((r[0], r[1], r[2]), ("FPT", "direct", 0))
        self.assertIn("doanh thu thuần", r[3])   # the whole narrative, though it never names FPT
