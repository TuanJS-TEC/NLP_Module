"""Collection with resume (skip URLs already stored), error logging, and structure-change detection."""
import logging

from . import storage
from .connectors import cafef, hose, vietstock

SOURCES = {"cafef": cafef, "vietstock": vietstock, "hose": hose}
MAX_CONSECUTIVE_EMPTY = 5   # parser probably broken by a page-structure change
log = logging.getLogger(__name__)


def collect(c, fetcher, source, start, end, matcher=None, symbols=None, limit=None):
    mod, stored, empty = SOURCES[source], 0, 0
    if getattr(mod, "PREFILTERED", False):
        matcher = None   # the connector filters by ticker itself
    want = set(symbols or (matcher_symbols(matcher) if matcher else []))
    log.info("collect %s %s..%s, prefilter=%s, symbols=%s, limit=%s",
             source, start, end, bool(matcher), sorted(want) or "all", limit)
    seen = dup = filtered = failed = 0
    for url, lm, title in mod.discover(fetcher, start, end):
        seen += 1
        if limit and stored >= limit:
            log.info("limit %d reached after %d listed urls", limit, seen)
            break
        if storage.seen(c, url):
            dup += 1
            log.debug("skip, already stored: %s", url)
            continue
        if matcher:
            hit = matcher.match(title) & want
            if not hit:
                filtered += 1
                log.debug("skip, no ticker in title: %r", (title or "")[:90])
                continue
            log.info("match %s | %s", ",".join(sorted(hit)), (title or "")[:90])
        try:
            # a connector that needs more than one request per document (attachments) fetches for itself
            doc = mod.fetch_parse(fetcher, url) if hasattr(mod, "fetch_parse") else mod.parse(fetcher.get(url), url)
        except Exception as e:
            storage.log_error(c, url, e)
            failed += 1
            log.warning("parse/fetch failed (%s): %s", e, url)
            continue
        if not doc.body_text or not doc.published_at:
            storage.log_error(c, url, "empty body or missing published_at")
            empty += 1
            log.warning("empty body (%d chars) or no published_at (%s): %s",
                        len(doc.body_text or ""), doc.published_at, url)
            if empty >= MAX_CONSECUTIVE_EMPTY:
                raise RuntimeError(f"{source}: {empty} consecutive unparsable pages; page structure may have changed")
            continue
        empty = 0
        _, new = storage.ingest(c, doc)
        stored += new
        log.info("%s %s | %d chars | %s | %s", "stored" if new else "dup-content",
                 doc.published_at, len(doc.body_text), (doc.title or "")[:70], url)
    log.info("collect done: %d listed, %d stored, %d already-seen, %d title-filtered, %d failed",
             seen, stored, dup, filtered, failed)
    return stored


def matcher_symbols(m):
    return {s for s, _ in m.pats}
