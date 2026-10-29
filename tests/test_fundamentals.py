import pandas as pd
from pytest import approx

from nlp_module.fundamentals import WINDOW_D, add_fund, profit_change


def test_profit_change_reads_sign_and_skips_other_profit_lines():
    # phrasings from real filings (ACB, VPB, HSG mid-2026); direction word is the last tăng/giảm before the %
    assert profit_change("Lợi nhuận sau thuế hợp nhất Quý II năm 2026 giảm 588.681 triệu đông, tương ứng giảm 12,1% so") \
        == approx(-0.121)
    assert profit_change("lợi nhuận sau thuế riêng đạt 12.453.436 triệu đồng, tăng 3.855.205 triệu đồng, "
                         "tương ứng mức tăng 44,84%") == approx(0.4484)
    # HSG: the only % after the LNST mention belongs to gross profit -> not an LNST change
    assert profit_change("Lợi nhuận sau thuế riêng đạt 124 tỷ đồng, tăng 61 tỷ đồng. Lợi nhuận gộp tăng 26 tỷ, "
                         "tương ứng với mức tăng 3%") is None
    assert profit_change("Doanh thu tăng 20%. Không nêu lợi nhuận.") is None
    # POW Q2/2026: LNST sentence lost its numbers to OCR; the next % is revenue, in the English half
    assert profit_change("lợi nhuận sau thuế thu nhập doanh nghiệp quý 02 năm 2026 tăng lần lượt là với cùng kỳ. "
                         "Total revenue of the Parent Company increased 245,70%") is None


def test_filing_counts_only_after_its_backtest_availability_and_expires():
    # filed 2025-07-20 03:00 UTC (10:00 ICT): usable at that day's close (08:00 UTC), not the session before
    fund = pd.DataFrame({"symbol": ["AAA"], "published_at": pd.to_datetime(["2025-07-20 03:00"]), "np_yoy": [-0.3]})
    days = pd.to_datetime(["2025-07-18", "2025-07-20", "2025-07-21", "2025-10-30"]) + pd.Timedelta(hours=7)
    df = pd.DataFrame({"date": days, "symbol": "AAA"}, index=[10, 11, 12, 13])
    got = add_fund(df, fund)
    assert list(got.index) == [10, 11, 12, 13]                       # caller's row order and index kept
    assert list(got.has_fund) == [0.0, 1.0, 1.0, 0.0]                # before availability / live / live / expired
    assert list(got.np_yoy) == [0.0, -0.3, -0.3, 0.0]
    assert got.fund_age.iloc[0] == got.fund_age.iloc[3] == WINDOW_D
    # a filing posted after the close counts from the next session only
    late = fund.assign(published_at=pd.to_datetime(["2025-07-20 08:10"]))
    assert list(add_fund(df, late).has_fund) == [0.0, 0.0, 1.0, 0.0]


def test_annual_wide_pivots_items_and_dates_availability_after_audit_deadline():
    from nlp_module.fundamentals import annual_wide
    long = pd.DataFrame({"ticker": ["VCB", "FPT", "FPT"], "year": [2024, 2024, 2024],
                         "item_code": ["is_loi_nhuan_sau_thue", "is_lai_lo_thuan_sau_thue", "is_doanh_so_thuan"],
                         "value": [3.4e13, 9.4e12, 6.3e13]})
    w = annual_wide(long).set_index("symbol")
    assert w.loc["FPT", "is_doanh_so_thuan"] == 6.3e13 and pd.isna(w.loc["VCB", "is_doanh_so_thuan"])
    assert (w["available_at"] == pd.Timestamp("2025-04-01")).all()     # FY2024 unknown until after Q1 2025


def test_profit_change_prefers_consolidated_over_separate():
    # PLX FY2025 / HSG Q1 FY2025-26: separate stated first, consolidated later and with the opposite sign
    assert profit_change("a/ Theo báo cáo tài chính riêng: Lợi nhuận sau thuế TNDN năm 2025 đạt 1.903,8 tỷ đồng, "
                         "tăng 231,4 tỷ đồng (13,8%) so với năm 2024. b/ Theo báo cáo tài chính hợp nhất: Lợi nhuận "
                         "sau thuế năm 2025 biến động giảm 4%") == approx(-0.04)
    assert profit_change("lợi nhuận sau thuế riêng Công ty mẹ đạt 43,2 tỷ đồng, tăng 37,2 tỷ đồng, tương ứng tăng 615%. "
                         "Trong Quý I, lợi nhuận sau thuế hợp nhất đạt 62,4 tỷ đồng, giảm 103 tỷ đồng, tương ứng "
                         "giảm 62%") == approx(-0.62)


def test_bctc_features_and_availability(tmp_path):
    from nlp_module.fundamentals import add_bctc, bctc_features
    rows = [("AAA", y, code, v) for y, np_, eq, ta in ((2023, -50.0, 500.0, 1000.0), (2024, 100.0, 400.0, 1000.0))
            for code, v in (("is_lai_lo_thuan_sau_thue", np_), ("bs_von_chu_so_huu_4d280b22", eq), ("bs_tong_tai_san", ta))]
    pd.DataFrame(rows, columns=["ticker", "year", "item_code", "value"]).to_parquet(tmp_path / "b.parquet")
    f = bctc_features(tmp_path / "b.parquet")
    r = f[f.available_at == pd.Timestamp("2025-04-01")].iloc[0]
    assert r.bc_np_yoy == approx(2.0) and r.bc_roe == approx(0.25) and r.bc_lev == approx(0.6)   # -50 -> 100: +300%, clipped
    days = pd.to_datetime(["2025-03-31", "2025-04-01", "2026-06-01"]) + pd.Timedelta(hours=7)
    got = add_bctc(pd.DataFrame({"date": days, "symbol": "AAA"}), f)
    # FY2023 has no prior year -> no growth -> imputed; FY2024 from 2025-04-01; stale after ~13 months
    assert list(got.has_bctc) == [0.0, 1.0, 0.0] and list(got.bc_lev) == [0.0, 0.6, 0.0]
