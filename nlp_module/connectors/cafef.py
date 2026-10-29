import logging
import re
from datetime import date
from selectolax.lexbor import LexborHTMLParser as P
from .base import ParsedDoc, sitemap_entries, to_utc_iso

SOURCE = "cafef"
INDEX = "https://cafef.vn/sitemap.xml"
_FILE = re.compile(r"sitemaps-(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})\.xml$")
log = logging.getLogger(__name__)


def sitemap_files(index_xml: str, start: date, end: date):
    """Date-bucketed sitemap files (5-day buckets back to 2016) overlapping [start, end]."""
    for loc, _, _ in sitemap_entries(index_xml.replace("<sitemap>", "<url>").replace("</sitemap>", "</url>")):
        m = _FILE.search(loc)
        if m:
            y, mo, d1, d2 = map(int, m.groups())
            if date(y, mo, d1) <= end and date(y, mo, d2) >= start:
                yield loc


def discover(fetcher, start: date, end: date):
    """-> (url, lastmod, title) for every article listed in sitemap buckets overlapping the range."""
    files = list(sitemap_files(fetcher.get(INDEX).decode(), start, end))
    log.info("cafef: %d sitemap bucket(s) overlap %s..%s: %s", len(files), start, end,
             [f.rsplit("/", 1)[-1] for f in files])
    for f in files:
        kept = out_of_range = not_article = 0
        for loc, lm, title in sitemap_entries(fetcher.get(f).decode()):
            if lm and not (start.isoformat() <= lm[:10] <= end.isoformat()):
                out_of_range += 1
                continue
            if not loc.endswith(".chn"):
                not_article += 1
                continue
            kept += 1
            yield loc, lm, title
        log.info("cafef: %s -> %d in range, %d outside %s..%s, %d non-article",
                 f.rsplit("/", 1)[-1], kept, out_of_range, start, end, not_article)


def _noisy(node) -> bool:
    n = node
    while n is not None:
        cls = (n.attributes.get("class") or "").lower()
        if any(k in cls for k in ("related", "caption", "photo", "tin-lien-quan", "box-", "sharing", "newsrelate")):
            return True
        n = n.parent
    return False


def parse(raw: bytes, url: str) -> ParsedDoc:
    t = P(raw.decode("utf-8", "replace"))
    h1 = t.css_first("h1")
    sapo = t.css_first(".sapo")
    paras = [p.text(strip=True) for p in t.css(".detail-content p") if not _noisy(p)]
    body = "\n".join(([sapo.text(strip=True)] if sapo else []) + [p for p in paras if p])
    pub = t.css_first('meta[property="article:published_time"]')
    mod = t.css_first('meta[property="article:modified_time"]')
    return ParsedDoc(SOURCE, url, "news", h1.text(strip=True) if h1 else "", body,
                     to_utc_iso(pub.attributes.get("content")) if pub else None,
                     to_utc_iso(mod.attributes.get("content")) if mod else None, raw)
