"""Entity linking: document -> company. Rule-based; scores are matching scores, NOT probabilities."""
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

COMPANIES_CSV = Path(__file__).resolve().parent.parent / "data" / "companies.csv"
# Symbols that are also ordinary words/places/currency: need "cổ phiếu/mã/CP" right before them.
LIST_ARTICLE_MIN = 4      # articles linking >= this many companies are treated as market-level
AMBIGUOUS_SYMBOLS = {"VND", "HCM", "GAS"}
_CTX_BEFORE = re.compile(r"(cổ phiếu|mã|\bcp|chứng khoán)\s*$", re.I)
_STOCK_CTX = re.compile(r"cổ phiếu|\bmã\b|\bcp\b|hose|hnx|upcom|công ty|ngân hàng|tập đoàn", re.I)


@dataclass(frozen=True)
class Mention:
    entity_id: str
    start: int
    end: int
    alias: str


@dataclass
class Link:
    entity_id: str
    symbol: str
    relation: str          # direct | ambiguous | market (list article); 'sector' not implemented
    matching_score: float
    in_title: bool
    mentions: list = field(default_factory=list)  # body mentions (offsets into body)


def load_aliases(path=COMPANIES_CSV):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class Linker:
    def __init__(self, rows):
        self.symbol = {r["entity_id"]: r["symbol"] for r in rows}
        self.pats = []  # (entity_id, alias, regex, valid_from, valid_to, is_ambiguous_symbol)
        for r in rows:
            a = r["alias"]
            sym = r["alias_type"] == "symbol"
            flags = 0 if sym else re.I
            rx = re.compile(rf"(?<![\w]){re.escape(a)}(?![\w])", flags)
            self.pats.append((r["entity_id"], a, rx, r["valid_from"] or "", r["valid_to"] or "9999",
                              sym and a in AMBIGUOUS_SYMBOLS))

    def find(self, text: str, on_date: str = "") -> list:
        out = []
        for eid, alias, rx, vf, vt, amb in self.pats:
            if on_date and not (vf <= on_date[:10] <= vt):
                continue
            for m in rx.finditer(text):
                if amb:
                    before = text[max(0, m.start() - 25):m.start()]
                    if not _CTX_BEFORE.search(before) or re.search(r"tp\.?\s*$", before, re.I):
                        continue
                out.append(Mention(eid, m.start(), m.end(), alias))
        return sorted(set(out), key=lambda m: (m.start, m.end))

    def link(self, title: str, body: str, published_at: str = "") -> list:
        tm = self.find(title, published_at)
        bm = self.find(body, published_at)
        out = []
        for eid in {m.entity_id for m in tm + bm}:
            t = [m for m in tm if m.entity_id == eid]
            b = [m for m in bm if m.entity_id == eid]
            ctx = any(_STOCK_CTX.search(body[max(0, m.start - 40):m.end + 40]) for m in b) or bool(t)
            n = len(b)
            score = 0.5 * bool(t) + 0.3 * min(n, 3) / 3 + 0.2 * ctx
            out.append(Link(eid, self.symbol[eid], "direct" if (t or n >= 2) else "ambiguous",
                            round(score, 3), bool(t), b))
        if len(out) >= LIST_ARTICLE_MIN:   # rate tables, VN30 lists, roundups: not about any single company
            for l in out:
                l.relation = "market"
        return sorted(out, key=lambda l: -l.matching_score)
