"""Earnings-change features from the issuers' own HOSE profit explanations (connectors/hose.py).

A company files one when quarterly/half-year after-tax profit moves >= 10% vs the same period last year, so a filing
is itself an earnings event; the stated % change is its size and sign. Timestamps are the exchange's real posting
times, so there is no publication-lag guesswork. Coverage is whatever the HOSE API serves (from ~2025-04) and only
quarters with a >= 10% swing: no filing means "small change OR not collected", which is why absence is has_fund=0,
never np_yoy=0 read as a measured flat quarter.
python -m nlp_module.fundamentals     # prints the parsed filings
"""
import re
from datetime import datetime

import numpy as np
import pandas as pd

from .contract import BACKTEST_DELAY
from .trial_forecast import CLOSE_UTC

FUND = ["has_fund", "np_yoy", "fund_age"]
WINDOW_D = 90      # one reporting cycle: an older filing describes a superseded period
CLIP = 2.0         # +360% (VIC H1/2026) and the like: a size cap, the sign carries most of it
# annual statements (vnfinancialdata): banks and non-banks file the after-tax-profit line under different codes
NP_CODES = ["is_lai_lo_thuan_sau_thue", "is_loi_nhuan_sau_thue"]
EQ_CODES = ["bs_von_chu_so_huu_4d280b22", "bs_von_chu_so_huu_6cda78ae"]   # non-bank / bank equity
BCTC = ["has_bctc", "bc_np_yoy", "bc_roe", "bc_lev", "bc_size"]

_LNST = re.compile(r"lợi\s+nhuận[^.%]{0,40}?sau\s+thu|LNST", re.I)
_PCT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_DIR = re.compile(r"tăng|giảm", re.I)
# another line item between the LNST mention and the %: the % is that item's (gross profit, revenue in POW's
# bilingual tables). ponytail: amount-only filings (STB "là 1.319,2 tỷ, giảm 1.689,5 tỷ") are dropped, not
# derived - OCR mixes . and , as separators; derive the % from the two amounts if coverage matters
_OTHER = re.compile(r"lợi\s+nhuận|doanh\s+thu|chi\s+phí|thu\s+nhập\s+lãi|revenue|expense", re.I)
# many filings state the separate (parent-only) LNST first; it can differ from the consolidated one in sign
# (PLX FY2025 +13.8% vs -4%) or tenfold (VRE +928% vs +57%). The last scope word before the % decides.
_SCOPE = re.compile(r"hợp\s+nhất|consolidated|riêng|công\s+ty\s+mẹ|separate|parent", re.I)


def profit_change(text):
    """Signed after-tax-profit change vs the same period last year as a fraction (-0.121 = down 12.1%): the first
    consolidated LNST mention that states one, else the first of any scope; None when the filing gives only amounts.
    The direction word is the last tăng/giảm before the %, since filings write 'giảm 588.681 triệu đồng, tương ứng
    giảm 12,1%'. Checked against audited FY2025 statements by `python -m nlp_module.fundamentals --check`."""
    first = None
    for a in _LNST.finditer(text):
        m = _PCT.search(text, a.end(), a.end() + 300)
        if not m or _OTHER.search(text[a.end():m.start()]):
            continue
        d = _DIR.findall(text, a.end(), m.start())
        if d:
            v = float(m.group(1).replace(",", ".")) / 100      # 44,84% and 47.42% both occur: % never has thousands
            v = v if d[-1].lower() == "tăng" else -v
            s = _SCOPE.findall(text, max(0, a.start() - 200), m.start())
            if s and s[-1].lower()[:3] in ("hợp", "con"):
                return v
            first = v if first is None else first
    return first


def _filings(c):
    return c.execute("""SELECT v.title, v.published_at, v.normalized_text FROM document d
                        JOIN document_version v USING(document_id)
                        WHERE d.source='hose' AND v.is_representative=1""").fetchall()


def load(c):
    """-> DataFrame symbol, published_at (naive UTC), np_yoy; filings with no stated % are dropped."""
    rows = []
    for title, pub, text in _filings(c):
        if (y := profit_change(text)) is not None:
            rows.append({"symbol": title.split(":")[0].strip(), "np_yoy": max(-CLIP, min(CLIP, y)),
                         "published_at": datetime.fromisoformat(pub.replace("Z", ""))})
    out = pd.DataFrame(rows, columns=["symbol", "published_at", "np_yoy"])
    out["published_at"] = pd.to_datetime(out["published_at"])   # typed even when empty, for merge_asof
    return out


def add_fund(df, fund):
    """Latest filing available by each session's close (backtest: published_at + BACKTEST_DELAY, as for articles),
    if no older than WINDOW_D days. Imputation (explicit): none => has_fund 0, np_yoy 0, fund_age WINDOW_D."""
    d = df.assign(_cut=df["date"].dt.normalize() + pd.Timedelta(hours=CLOSE_UTC.hour), _i=range(len(df)))
    f = fund.assign(_avail=fund["published_at"] + BACKTEST_DELAY).drop(columns="published_at").sort_values("_avail")
    m = pd.merge_asof(d.sort_values("_cut"), f, left_on="_cut", right_on="_avail", by="symbol", direction="backward")
    age = (m["_cut"] - m["_avail"]).dt.total_seconds() / 86400
    live = age <= WINDOW_D                                           # NaN age (no filing yet) compares False
    m["has_fund"] = live.astype(float)
    m["np_yoy"] = m["np_yoy"].where(live, 0.0)
    m["fund_age"] = age.where(live, WINDOW_D)
    return m.sort_values("_i").drop(columns=["_cut", "_avail", "_i"]).set_index(df.index)


def annual_wide(long):
    """vnfinancialdata long rows -> one row per (symbol, year), one column per item_code, plus available_at.
    The dataset carries no publication dates, so year Y counts as known from April 1 of Y+1: audited annual reports are
    due 90 days after the Dec 31 year-end. ponytail: assumes calendar fiscal years; take real filing dates if a
    non-December filer matters. Never feed year Y into a snapshot cut before available_at."""
    w = long.pivot_table(index=["ticker", "year"], columns="item_code", values="value", aggfunc="first")
    w = w.reset_index().rename(columns={"ticker": "symbol"}).rename_axis(columns=None)
    w["symbol"] = w["symbol"].astype(str)   # the dataset's arrow string dtype will not merge_asof with CSV symbols
    w["available_at"] = pd.to_datetime((w["year"] + 1).astype(str) + "-04-01")
    return w


def annual(symbols, items=NP_CODES, statement="income_statement"):
    """Yearly statement items (2009-2025, HSX+HNX; no securities firms such as SSI/VND) from the public HF dataset behind
    vnfinancialdata, cached in ~/.cache/huggingface after the first download. See annual_wide for the columns.
    Item codes: vnf.search_items("lợi nhuận", statement="income_statement")."""
    import vnfinancialdata as vnf
    return annual_wide(pd.concat([vnf.load(exchange=x, statement=statement, ticker=symbols, item_code=items)
                                  for x in ("HSX", "HNX")]))


def bctc_features(path="data/bctc.parquet"):
    """Audited annual statements (README: data/bctc.parquet) -> symbol, available_at, BCTC values, one row per fiscal
    year: after-tax-profit growth vs the prior year (over |prior| so a loss base keeps the sign), ROE, leverage
    (1 - equity/assets) and size (log total assets). Banks and non-banks file profit and equity under different codes."""
    w = annual_wide(pd.read_parquet(path)).sort_values(["symbol", "year"])
    first = lambda codes: w.reindex(columns=codes).bfill(axis=1).iloc[:, 0]
    np_, eq, ta = first(NP_CODES), first(EQ_CODES), w["bs_tong_tai_san"]
    prev = np_.groupby(w["symbol"]).shift().where(w.groupby("symbol")["year"].diff() == 1)   # a gap year is no base
    w["bc_np_yoy"] = ((np_ - prev) / prev.abs()).clip(-CLIP, CLIP)
    w["bc_roe"] = (np_ / eq).clip(-1, 1)
    w["bc_lev"] = 1 - eq / ta
    w["bc_size"] = np.log(ta)
    return w[["symbol", "available_at"] + BCTC[1:]].dropna(subset=["available_at"])


def add_bctc(df, feats):
    """Latest fiscal year known by each session's close (available_at = April 1 of the next year, see annual_wide),
    if at most ~13 months old: a missed filing must not carry the year before forward. Imputation (explicit): no
    statement (SSI and other brokers are not in the dataset) or a missing ratio => has_bctc 0, every bc_* 0."""
    d = df.assign(_cut=df["date"].dt.normalize() + pd.Timedelta(hours=CLOSE_UTC.hour), _i=range(len(df)))
    feats = feats.astype({"available_at": d["_cut"].dtype})   # merge_asof refuses mixed ns/us resolutions
    m = pd.merge_asof(d.sort_values("_cut"), feats.sort_values("available_at"), left_on="_cut",
                      right_on="available_at", by="symbol", direction="backward", tolerance=pd.Timedelta(days=400))
    m["has_bctc"] = m[BCTC[1:]].notna().all(axis=1).astype(float)
    m[BCTC[1:]] = m[BCTC[1:]].where(m["has_bctc"] == 1, 0.0)
    return m.sort_values("_i").drop(columns=["_cut", "_i", "available_at"]).set_index(df.index)


def check_annual(c):
    """Full-year filings' parsed % vs the audited consolidated LNST change in vnfinancialdata (None = not covered):
    the only ground truth for profit_change, since the dataset has no quarters."""
    f = pd.DataFrame([{"symbol": t.split(":")[0].strip(), "year": int(y.group(1)), "parsed": profit_change(x)}
                      for t, _, x in _filings(c)
                      if (y := re.search(r"năm (20\d\d)", t)) and not re.search(r"quý|niên|tháng", t, re.I)])
    w = annual(sorted(set(f["symbol"]))).sort_values(["symbol", "year"])
    w["np"] = w.reindex(columns=NP_CODES).bfill(axis=1).iloc[:, 0]
    w["bctc"] = w.groupby("symbol")["np"].pct_change(fill_method=None)
    return f.merge(w[["symbol", "year", "bctc"]], how="left").sort_values(["year", "symbol"])


if __name__ == "__main__":
    import sys
    from . import storage
    c = storage.connect()
    if "--check" in sys.argv:
        print(check_annual(c).round(4).to_string(index=False))
    else:
        print(load(c).sort_values("published_at").to_string(index=False))
