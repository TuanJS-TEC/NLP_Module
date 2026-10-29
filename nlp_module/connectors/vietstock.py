import re
from datetime import date
from selectolax.lexbor import LexborHTMLParser as P
from .base import ParsedDoc, sitemap_entries, to_utc_iso

SOURCE = "vietstock"
INDEX = "https://vietstock.vn/sitemap.xml"
_ART = re.compile(r"vietstock\.vn/(\d{4})/(\d{1,2})/([^/]+?)-\d+-\d+\.htm$")


def discover(fetcher, start: date, end: date):
    """-> (url, None, slug_title). Sitemap is one big file; month comes from the URL path, exact date from the page."""
    for loc, _, _ in sitemap_entries(fetcher.get(INDEX).decode("utf-8-sig")):
        m = _ART.search(loc)
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if (start.year, start.month) <= (y, mo) <= (end.year, end.month):
            yield loc, None, m.group(3).replace("-", " ")


def parse(raw: bytes, url: str) -> ParsedDoc:
    t = P(raw.decode("utf-8", "replace"))
    h1 = t.css_first("h1.article-title") or t.css_first("h1")
    head = t.css_first("p.pHead")
    paras = [p.text(strip=True) for p in t.css("#vst_detail p.pBody, #vst_detail p.pHead")]
    if head and not paras:
        paras = [head.text(strip=True)]
    pub = t.css_first('meta[property="article:published_time"]')
    mod = t.css_first('meta[property="article:modified_time"]')
    return ParsedDoc(SOURCE, url, "news", h1.text(strip=True) if h1 else "", "\n".join(p for p in paras if p),
                     to_utc_iso(pub.attributes.get("content")) if pub else None,
                     to_utc_iso(mod.attributes.get("content")) if mod else None, raw)
