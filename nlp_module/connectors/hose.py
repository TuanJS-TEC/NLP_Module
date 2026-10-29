"""HOSE listed-company disclosures: the issuer's own profit-change explanation ("Giải trình biến động LNST/KQKD").

Listed companies must file one when quarterly after-tax profit moves >= 10% vs the same quarter last year; it is the
issuer's narrative of WHY, which is what we score. The listing and attachment endpoints are the JSON API behind
hsx.vn (undocumented, see docs/source_terms.md); the PDFs are mostly scans, so text comes from OCR when the PDF has
no real text layer.
"""
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

from .base import ParsedDoc

SOURCE = "hose"
NEWS = "https://api.hsx.vn/n/api/v1/1/news"              # /1 = Vietnamese
MEDIA = "https://api.hsx.vn/m/api/v1/1/mediafiles/1"     # /1 = attachments of a news item
STATIC = "https://staticfile.hsx.vn"                     # filePath "~/Uploads/..." is relative to this host
CATE = "tin-tuc/tin-to-chuc-niem-yet"
PREFILTERED = True        # discover() already keeps only tracked tickers; the title matcher would drop VND/HCM/GAS
MAX_PAGES = 6             # ponytail: an explanation is 1-4 pages; a full BCTC attached by mistake would cost minutes of OCR
TESSDATA = Path(__file__).resolve().parents[2] / "data" / "tessdata"

_DOC = re.compile(r"giải trình.*(kqkd|kết quả kinh doanh|lnst|lợi nhuận|bctc|báo cáo tài chính)", re.I)
# the cover page's uppercase heading; the explanation itself often says "Về việc công bố thông tin" in lowercase
_COVER = re.compile(r"C[ÔÓO]NG\s+B[ỐÓO]\s+TH[ÔO]NG\s+TIN")
_START = re.compile(r"như\s+sau\s*:", re.I)
_PROFIT = re.compile(r"lợi\s+nhuận", re.I)
_END = re.compile(r"Trân\s+trọng|Nơi\s+nhận", re.I)
_SIGNATURE = re.compile(r"Digitally signed|Reason:|Location:|Foxit|DN: ?C=VN|Signature Not Verified", re.I)


def discover(fetcher, start, end):
    """-> (detail_url, None, title) for profit-explanation filings of tracked tickers published in [start, end]."""
    from ..entities import load_aliases
    syms = {r["symbol"] for r in load_aliases()}
    page = 1
    while True:
        q = urlencode({"pageIndex": page, "pageSize": 100, "startDate": start.isoformat(),
                       "endDate": end.isoformat(), "aliasCate": CATE})
        data = json.loads(fetcher.get(f"{NEWS}/cate?{q}"))["data"]
        for x in data["list"]:
            title = x.get("title") or ""
            if title.split(":")[0].strip() in syms and _DOC.search(title):
                yield f"{NEWS}/{x['id']}", None, title
        if page >= (data["paging"]["totalPages"] or 0):
            return
        page += 1


def fetch_parse(fetcher, url) -> ParsedDoc:
    """Detail JSON + every attached PDF -> one document whose body is the explanation narrative only."""
    d = json.loads(fetcher.get(url))["data"]
    files = json.loads(fetcher.get(f"{MEDIA}/{url.rsplit('/', 1)[-1]}?pageIndex=1&pageSize=10&year=0"))["data"]["list"]
    pdfs = [fetcher.get(STATIC + quote(f["filePath"].lstrip("~"))) for f in files
            if (f.get("fileType") or "").lower() == ".pdf"]
    body = "\n".join(filter(None, (narrative(pdf_pages(p)) for p in pdfs)))
    published = datetime.fromtimestamp(int(d["postedDate"]), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # ponytail: only the first PDF is kept as the raw object; a filing with separate consolidated/parent files
    # still has both narratives in `body`
    return ParsedDoc(SOURCE, url, "financial_narrative", d.get("title") or "", body, published, None,
                     pdfs[0] if pdfs else b"")


def pdf_pages(pdf: bytes) -> list:
    """Page texts: the PDF's own text layer when it has one, OCR (Vietnamese) when it is a scan."""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "doc.pdf"
        f.write_bytes(pdf)
        run = lambda *a: subprocess.run(a, capture_output=True, text=True, timeout=180, check=True).stdout
        pages = run("pdftotext", "-l", str(MAX_PAGES), str(f), "-").split("\f")
        # a scan's only text layer is the digital-signature stamp (~300 letters); a typed page has thousands.
        # A layer with a broken font map ("cONc NGHTEP") has the letters but no Vietnamese diacritics: OCR it too.
        letters = "".join(ch for ch in "".join(pages) if ch.isalpha())
        if len(letters) >= 800 * max(1, len([p for p in pages if p.strip()])) and _vi_share(letters) >= 0.08:
            return pages
        run("pdftoppm", "-r", "300", "-gray", "-png", "-l", str(MAX_PAGES), str(f), f"{tmp}/p")
        env = {**os.environ, "TESSDATA_PREFIX": str(TESSDATA)} if (TESSDATA / "vie.traineddata").exists() else None
        return [subprocess.run(["tesseract", str(png), "-", "-l", "vie", "--psm", "6"], capture_output=True,
                               text=True, timeout=180, check=True, env=env).stdout
                for png in sorted(Path(tmp).glob("p-*.png"))]


def narrative(pages: list) -> str:
    """Keep the explanation itself: drop the CBTT cover letter, the letterhead, figure tables and the sign-off."""
    if len(pages) > 1 and _COVER.search(pages[0][:1500]) and not _START.search(pages[0]):
        pages = pages[1:]          # cover page is boilerplate addressed to the regulators
    text = "\n".join(pages)
    if m := _START.search(text):
        text = text[m.end():]
    elif m := _PROFIT.search(text):   # no "như sau:": start at the first line about profit, not the letterhead
        text = text[text.rfind("\n", 0, m.start()) + 1:]
    if m := _END.search(text):        # searched only after the start: a 2-page cover has its own "Trân trọng"
        text = text[:m.start()]
    keep = [ln.strip() for ln in text.splitlines()
            if len(ln.strip()) >= 12 and ln.count("|") < 2 and not _SIGNATURE.search(ln)
            and sum(ch.isalpha() for ch in ln) / len(ln.strip()) >= 0.5      # table rows are digits and pipes
            and (len(ln.strip()) < 30 or _vi_share(ln) >= 0.03)]             # English half of bilingual filings
    return re.sub(r"\s+", " ", " ".join(keep)).strip()   # OCR breaks lines mid-sentence


def _vi_share(text: str) -> float:
    """Share of letters outside ASCII: ~0.2 in Vietnamese, ~0 in English or in a broken font map."""
    letters = [ch for ch in text if ch.isalpha()]
    return sum(not ch.isascii() for ch in letters) / len(letters) if letters else 0.0
