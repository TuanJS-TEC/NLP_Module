"""SQLite storage (portable SQL; see schema.sql) + raw HTML object store."""
import gzip
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .connectors.base import ParsedDoc, content_hash, normalize_url
from .text import normalize

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "nlp.db"
RAW_DIR = ROOT / "data" / "raw"
SCHEMA = Path(__file__).with_name("schema.sql")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path=DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True) if str(path) != ":memory:" else None
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='document'").fetchone():
        c.executescript(SCHEMA.read_text())  # ponytail: no migration tool yet; add one before schema v2
    # ponytail: tables added after the first schema are created idempotently here so existing DBs pick them up.
    # The DDL is duplicated in schema.sql for fresh DBs; collapse both into a migration tool at the third table.
    c.execute("""CREATE TABLE IF NOT EXISTS market_quote (
                 symbol TEXT NOT NULL, fetched_at TEXT NOT NULL, trading_date TEXT NOT NULL, session TEXT,
                 ref_price REAL, matched_price REAL, price_change REAL, price_change_pct REAL,
                 highest REAL, lowest REAL, avg_price REAL, volume REAL,
                 matched INTEGER NOT NULL DEFAULT 1, source TEXT NOT NULL,
                 PRIMARY KEY (symbol, fetched_at))""")
    c.execute("""CREATE TABLE IF NOT EXISTS vol_forecast (
                 symbol TEXT NOT NULL, base_date TEXT NOT NULL, h INTEGER NOT NULL, p_strong REAL NOT NULL,
                 call TEXT, model_version TEXT NOT NULL, created_at TEXT NOT NULL,
                 PRIMARY KEY (symbol, base_date, h, model_version))""")
    return c


def save_quotes(c, quotes, source) -> int:
    rows = [(q.symbol, now(), q.trading_date, q.session, q.ref_price, q.matched_price, q.price_change,
             q.price_change_pct, q.highest, q.lowest, q.avg_price, q.volume, int(q.matched), source)
            for q in quotes]
    c.executemany("INSERT OR REPLACE INTO market_quote VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit()
    return len(rows)


def save_raw(doc: ParsedDoc, chash: str, raw_dir=RAW_DIR) -> str:
    key = f"{doc.source}/{chash[:16]}.html.gz"
    p = Path(raw_dir) / key
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(gzip.compress(doc.raw))
    return key


def ingest(c, doc: ParsedDoc, raw_dir=RAW_DIR):
    """Store a parsed document. Returns (document_version_id, is_new_version).
    Same URL + same content hash = another observation of the same version (not a new row)."""
    url = normalize_url(doc.url)
    did = hashlib.sha1(url.encode()).hexdigest()[:16]
    t = now()
    c.execute("INSERT OR IGNORE INTO document VALUES (?,?,?,?,?)", (did, doc.source, url, doc.document_type, t))
    body = normalize(doc.body_text)
    title = normalize(doc.title)
    h = content_hash(title, body)
    row = c.execute("SELECT document_version_id FROM document_version WHERE document_id=? AND content_hash=?",
                    (did, h)).fetchone()
    if row:
        return row[0], False
    vid = f"{did}-{h[:8]}"
    c.execute("""INSERT INTO document_version (document_version_id, document_id, content_hash, title, original_text,
                 normalized_text, published_at, source_updated_at, version_first_seen_at, raw_object_key)
                 VALUES (?,?,?,?,?,?,?,?,?,?)""",
              (vid, did, h, title, doc.body_text, body, doc.published_at, doc.source_updated_at, t,
               save_raw(doc, h, raw_dir)))
    c.commit()
    return vid, True


def log_error(c, url, err):
    c.execute("INSERT INTO fetch_error VALUES (?,?,?)", (url, str(err), now()))
    c.commit()


def seen(c, url) -> bool:
    return c.execute("SELECT 1 FROM document WHERE url_normalized=?", (normalize_url(url),)).fetchone() is not None
