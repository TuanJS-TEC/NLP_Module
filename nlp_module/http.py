"""Polite fetcher: robots.txt, fixed min interval per host, bounded retries with exponential backoff."""
import logging
import time
import urllib.request
import urllib.robotparser
from urllib.error import HTTPError
from urllib.parse import urlparse

UA = "NLPModuleResearchBot/0.1 (duckmeme13@gmail.com)"  # identifiable, per docs/source_terms.md
log = logging.getLogger(__name__)


class FetchError(Exception):
    pass


class Fetcher:
    def __init__(self, min_interval=1.0, retries=3, timeout=20, ua=UA):
        self.min_interval, self.retries, self.timeout, self.ua = min_interval, retries, timeout, ua
        self._last = {}
        self._robots = {}

    def allowed(self, url) -> bool:
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.parse(self._get_raw(f"https://{host}/robots.txt").decode("utf-8-sig").splitlines())
                log.info("robots.txt loaded for %s", host)
            except HTTPError as e:
                if e.code in (401, 403):      # RFC 9309: access denied -> treat the whole host as disallowed
                    rp.parse(["User-agent: *", "Disallow: /"])
                    log.warning("robots.txt forbidden (HTTP %d) for %s: denying everything on this host", e.code, host)
                elif 400 <= e.code < 500:     # RFC 9309: "unavailable" (404 et al.) -> no restrictions
                    rp.parse(["User-agent: *", "Disallow:"])
                    log.info("no robots.txt for %s (HTTP %d): no restrictions", host, e.code)
                else:                         # 5xx and anything else -> be conservative
                    rp.parse(["User-agent: *", "Disallow: /"])
                    log.warning("robots.txt unavailable (HTTP %d) for %s: denying everything on this host", e.code, host)
            except Exception as e:
                rp.parse(["User-agent: *", "Disallow: /"])  # can't read robots -> be conservative
                log.warning("robots.txt unreadable for %s (%s): denying everything on this host", host, e)
            self._robots[host] = rp
        return self._robots[host].can_fetch(self.ua, url)

    def _get_raw(self, url) -> bytes:
        host = urlparse(url).netloc
        wait = self._last.get(host, 0) + self.min_interval - time.monotonic()
        if wait > 0:
            log.debug("rate limit: sleeping %.2fs before %s", wait, host)
            time.sleep(wait)
        self._last[host] = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept-Encoding": "identity"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = r.read()
        log.debug("GET %s -> %d bytes in %.0f ms", url, len(body), 1000 * (time.monotonic() - t0))
        return body

    def get(self, url) -> bytes:
        if not self.allowed(url):
            log.warning("blocked by robots.txt: %s", url)
            raise FetchError(f"blocked by robots.txt: {url}")
        last = None
        for i in range(self.retries + 1):
            try:
                return self._get_raw(url)
            except HTTPError as e:
                if e.code in (401, 403, 404, 410):  # don't hammer, don't retry client errors
                    log.warning("HTTP %d (not retried): %s", e.code, url)
                    raise FetchError(f"HTTP {e.code}: {url}") from e
                last = e
            except Exception as e:
                last = e
            log.warning("fetch failed (attempt %d/%d, %s): %s", i + 1, self.retries + 1, last, url)
            time.sleep(2 ** i)
        raise FetchError(f"{type(last).__name__}: {last}: {url}")
