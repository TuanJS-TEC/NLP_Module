"""Historical coverage survey from sitemap titles only (no article fetches). Counts are LOWER BOUNDS:
articles that mention a company only in the body are invisible here."""
import csv
import re
from collections import defaultdict

from .entities import AMBIGUOUS_SYMBOLS
from .text import fold


class TitleMatcher:
    def __init__(self, rows):
        self.pats = []
        for r in rows:
            if r["alias_type"] == "symbol" and r["alias"] in AMBIGUOUS_SYMBOLS:
                continue   # too noisy on accent-folded lowercase titles
            self.pats.append((r["symbol"], re.compile(rf"(?<!\w){re.escape(fold(r['alias']))}(?!\w)")))

    def match(self, title: str) -> set:
        f = fold(title or "")
        return {s for s, rx in self.pats if rx.search(f)}


def run(fetcher, discover, matcher, symbols, start, end, out_csv, source):
    total = defaultdict(int)
    cnt = defaultdict(int)
    days = defaultdict(set)
    for url, lm, title in discover(fetcher, start, end):
        mon = lm[:7] if lm else "{}-{:02d}".format(*map(int, re.search(r"/(\d{4})/(\d{1,2})/", url).groups()))
        total[mon] += 1
        for s in matcher.match(title) & set(symbols):
            cnt[(s, mon)] += 1
            if lm:
                days[(s, mon)].add(lm[:10])
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "symbol", "month", "title_matches", "days_with_match(lastmod)", "articles_in_month"])
        for (s, mon), n in sorted(cnt.items()):
            w.writerow([source, s, mon, n, len(days[(s, mon)]) or "", total[mon]])
    return sum(total.values()), sum(cnt.values())
