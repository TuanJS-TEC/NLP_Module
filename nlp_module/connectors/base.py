import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


@dataclass
class ParsedDoc:
    source: str
    url: str
    document_type: str
    title: str
    body_text: str
    published_at: str | None        # UTC ISO
    source_updated_at: str | None   # UTC ISO
    raw: bytes


def normalize_url(u: str) -> str:
    p = urlsplit(u.strip())
    return urlunsplit((p.scheme or "https", p.netloc.lower(), p.path.rstrip("/") or "/", "", ""))


def to_utc_iso(s: str | None) -> str | None:
    """Source timestamps are Asia/Ho_Chi_Minh (+07:00) when no offset is given."""
    if not s:
        return None
    d = datetime.fromisoformat(s.strip())
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone(__import__("datetime").timedelta(hours=7)))
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def content_hash(title: str, body: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", f"{title}\n{body}").strip().encode()).hexdigest()


def sitemap_entries(xml: str):
    """-> [(loc, lastmod|None, title|None)] from a sitemap (regex: the files are huge and CDATA-escaped)."""
    for blk in re.findall(r"<url>(.*?)</url>", xml, re.S):
        loc = re.search(r"<loc>\s*([^<\s]+)\s*</loc>", blk)
        if not loc:
            continue
        lm = re.search(r"<lastmod>([^<]+)</lastmod>", blk)
        t = re.search(r"<image:title>(?:&lt;!\[CDATA\[)?(.*?)(?:\]\]&gt;)?</image:title>", blk, re.S)
        yield loc.group(1), lm.group(1) if lm else None, t.group(1).strip() if t else None
