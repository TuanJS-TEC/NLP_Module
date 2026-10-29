-- Portable SQL (no SQLite-only features) so a later PostgreSQL move is a migration, not a rewrite.
-- All timestamps UTC ISO-8601 text; display in Asia/Ho_Chi_Minh at the edge.

CREATE TABLE document (
    document_id        TEXT PRIMARY KEY,
    source             TEXT NOT NULL,
    url_normalized     TEXT NOT NULL UNIQUE,
    document_type      TEXT NOT NULL,           -- news | disclosure | financial_narrative
    first_seen_at      TEXT NOT NULL
);

CREATE TABLE document_version (
    document_version_id    TEXT PRIMARY KEY,
    document_id            TEXT NOT NULL REFERENCES document(document_id),
    content_hash           TEXT NOT NULL,
    title                  TEXT,
    original_text          TEXT NOT NULL,
    normalized_text        TEXT NOT NULL,
    published_at           TEXT,                -- as reported by the source
    source_updated_at      TEXT,
    version_first_seen_at  TEXT NOT NULL,
    raw_object_key         TEXT NOT NULL,
    duplicate_cluster_id   TEXT,
    is_representative      INTEGER NOT NULL DEFAULT 1,
    UNIQUE (document_id, content_hash)
);

CREATE TABLE article_company_result (
    result_id              TEXT PRIMARY KEY,
    document_version_id    TEXT NOT NULL REFERENCES document_version(document_version_id),
    target_entity_id       TEXT NOT NULL,
    symbol                 TEXT NOT NULL,
    relation               TEXT NOT NULL,       -- direct | sector | market | ambiguous
    matching_score         REAL,                -- rule score, NOT a calibrated probability
    sentence_ids           TEXT NOT NULL,       -- JSON array
    passage_text           TEXT NOT NULL,
    difficult              INTEGER NOT NULL DEFAULT 0,
    p_pos REAL, p_neu REAL, p_neg REAL,
    entropy                REAL,
    uncertain              INTEGER NOT NULL DEFAULT 0,
    needs_review           INTEGER NOT NULL DEFAULT 0,
    nlp_model_version      TEXT NOT NULL,
    processed_at           TEXT NOT NULL,
    UNIQUE (document_version_id, target_entity_id, nlp_model_version)
);

CREATE TABLE feature_snapshot (
    symbol                 TEXT NOT NULL,
    forecast_cutoff        TEXT NOT NULL,
    feature_version        TEXT NOT NULL,
    nlp_model_version      TEXT NOT NULL,
    availability_mode      TEXT NOT NULL,       -- live | backtest
    features_json          TEXT NOT NULL,
    PRIMARY KEY (symbol, forecast_cutoff, feature_version, nlp_model_version, availability_mode)
);

CREATE TABLE snapshot_lineage (
    symbol TEXT NOT NULL, forecast_cutoff TEXT NOT NULL, feature_version TEXT NOT NULL,
    nlp_model_version TEXT NOT NULL, availability_mode TEXT NOT NULL,
    result_id TEXT NOT NULL REFERENCES article_company_result(result_id),
    PRIMARY KEY (symbol, forecast_cutoff, feature_version, nlp_model_version, availability_mode, result_id)
);

CREATE TABLE company_alias (
    entity_id    TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    legal_name   TEXT,
    alias        TEXT NOT NULL,
    alias_type   TEXT NOT NULL,                 -- symbol | legal | brand | abbreviation
    valid_from   TEXT,
    valid_to     TEXT,
    PRIMARY KEY (entity_id, alias)
);

CREATE TABLE crawl_state (
    key    TEXT PRIMARY KEY,                    -- e.g. cafef:sitemap:sitemaps-2026-10-16-20.xml
    value  TEXT NOT NULL
);

CREATE TABLE fetch_error (
    url        TEXT NOT NULL,
    error      TEXT NOT NULL,
    at         TEXT NOT NULL
);

CREATE TABLE annotation_sample (
    sample_id            TEXT PRIMARY KEY,
    document_version_id  TEXT NOT NULL REFERENCES document_version(document_version_id),
    target_entity_id     TEXT NOT NULL,
    passage_text         TEXT NOT NULL,
    sentence_ids         TEXT NOT NULL,
    difficult            INTEGER NOT NULL DEFAULT 0,
    stream               TEXT NOT NULL,         -- dev | eval
    sentiment_label      TEXT,                  -- positive | neutral | negative | NULL (not yet labeled)
    annotation_status    TEXT NOT NULL DEFAULT 'pending',   -- pending | labeled | skipped | unclear
    annotator_id         TEXT,
    guideline_version    TEXT,
    UNIQUE (document_version_id, target_entity_id)
);

CREATE TABLE IF NOT EXISTS market_quote (
    symbol            TEXT NOT NULL,
    fetched_at        TEXT NOT NULL,          -- UTC ISO, when we read the board
    trading_date      TEXT NOT NULL,          -- YYYYMMDD as the board reports it
    session           TEXT,                   -- SSI session code (P/O/K/A/C...)
    ref_price         REAL, matched_price REAL, price_change REAL, price_change_pct REAL,
    highest           REAL, lowest REAL, avg_price REAL, volume REAL,
    matched           INTEGER NOT NULL DEFAULT 1,   -- 0 = no trade yet, price = reference
    source            TEXT NOT NULL,
    PRIMARY KEY (symbol, fetched_at)
);

CREATE TABLE IF NOT EXISTS vol_forecast (
    symbol         TEXT NOT NULL,
    base_date      TEXT NOT NULL,             -- YYYY-MM-DD, the symbol's last session with features
    h              INTEGER NOT NULL,          -- horizon in sessions
    p_strong       REAL NOT NULL,             -- P(|forward return| beats the day's cross-sectional median)
    call           TEXT,                      -- strong | weak | NULL (abstain)
    model_version  TEXT NOT NULL,
    created_at     TEXT NOT NULL,             -- UTC ISO
    PRIMARY KEY (symbol, base_date, h, model_version)
);
