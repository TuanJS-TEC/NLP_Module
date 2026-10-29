"""Processing + snapshot stages over the DB."""
import json
import logging

from . import storage
from .contract import ArticleCompanyResult, FEATURE_VERSION, build_snapshot
from .dedup import assign_clusters
from .entities import Link
from .passages import MAX_CHARS, select_passage
from .sentiment.base import decide
from .text import split_sentences

log = logging.getLogger(__name__)


def _utc(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def process(c, model, linker, abstain_entropy=None, batch=64):
    """Link -> select passage -> sentiment -> article-company results, for versions not yet done by this model.
    One passage per (article version, company), so a long article cannot outweigh a short one."""
    assign_clusters(c)
    rows = c.execute("""SELECT v.*, d.document_type FROM document_version v JOIN document d USING(document_id)
                        WHERE NOT EXISTS (SELECT 1 FROM article_company_result r
                        WHERE r.document_version_id=v.document_version_id AND r.nlp_model_version=?)""",
                     (model.version,)).fetchall()
    log.info("process: %d article versions not yet scored by %s", len(rows), model.version)
    todo, no_link, no_passage = [], 0, 0
    issuer = {s: e for e, s in linker.symbol.items()}
    for v in rows:
        sents = split_sentences(v["normalized_text"])
        filing = v["document_type"] == "financial_narrative"
        if filing:
            # an issuer's own filing ("SYM: Giải trình ..."): the issuer is the title prefix, and every sentence of the
            # narrative is about it even though few name it, so the linker and sentence selection would miss it
            eid = issuer.get((v["title"] or "").split(":")[0].strip())
            links = [Link(eid, linker.symbol[eid], "direct", 1.0, True, [])] if eid else []
        else:
            links = linker.link(v["title"] or "", v["normalized_text"], v["published_at"] or "")
        log.debug("version %s | %d chars, %d sentences | %r", v["document_version_id"],
                  len(v["normalized_text"] or ""), len(sents), (v["title"] or "")[:80])
        if not links:
            no_link += 1
            log.debug("  no company linked")
        for l in links:
            others = [m for o in links if o is not l for m in o.mentions]
            p = ({"passage_text": f"{v['title']}. {v['normalized_text']}"[:MAX_CHARS],
                  "sentence_ids": [s[0] for s in sents], "difficult": False} if filing
                 else select_passage(v["title"] or "", sents, l, others))
            log.debug("  link %s relation=%s score=%.3f in_title=%s mentions=%d -> passage %d chars%s",
                      l.symbol, l.relation, l.matching_score, l.in_title, len(l.mentions),
                      len(p["passage_text"]), " DIFFICULT" if p["difficult"] else "")
            if p["passage_text"]:
                todo.append((v["document_version_id"], l, p))
            else:
                no_passage += 1
    log.info("process: %d (article, company) passages to score; %d versions linked nothing, %d links gave no passage",
             len(todo), no_link, no_passage)
    # ponytail: a version that links nothing writes no result row, so it is re-linked on every process run
    # (79 of 17,062 as of 2026-10-26, ~0.5%). Store a sentinel row if that ever stops being free.
    n = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        log.info("scoring batch %d-%d of %d", i, i + len(chunk), len(todo))
        probs = model.predict([p["passage_text"] for _, _, p in chunk])
        for (vid, l, p), pr in zip(chunk, probs):
            unc, h = decide(pr, abstain_entropy)
            review = unc or p["difficult"] or l.relation == "ambiguous"
            log.debug("  %s %s pos=%.3f neu=%.3f neg=%.3f H=%.3f%s%s", l.symbol, l.relation, *pr, h,
                      " UNCERTAIN" if unc else "", " NEEDS_REVIEW" if review else "")
            c.execute("""INSERT OR IGNORE INTO article_company_result (result_id, document_version_id, target_entity_id,
                         symbol, relation, matching_score, sentence_ids, passage_text, difficult, p_pos, p_neu, p_neg,
                         entropy, uncertain, needs_review, nlp_model_version, processed_at)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (f"{vid}:{l.entity_id}:{model.version}", vid, l.entity_id, l.symbol, l.relation, l.matching_score,
                       json.dumps(p["sentence_ids"]), p["passage_text"], int(p["difficult"]), *pr, h, int(unc),
                       int(review), model.version, storage.now()))
            n += 1
    c.commit()
    log.info("process: %d results written for %s", n, model.version)
    return n


def load_results(c, model_version):
    out = []
    for r in c.execute("""SELECT r.*, v.published_at, v.is_representative, d.source FROM article_company_result r
                          JOIN document_version v USING(document_version_id) JOIN document d USING(document_id)
                          WHERE r.nlp_model_version=? AND v.published_at IS NOT NULL""", (model_version,)):
        out.append(ArticleCompanyResult(r["result_id"], r["symbol"], r["source"], _utc(r["published_at"]),
                                        _utc(r["processed_at"]), r["relation"], r["p_pos"], r["p_neu"], r["p_neg"],
                                        bool(r["uncertain"]), bool(r["difficult"]),
                                        bool(r["is_representative"]), r["nlp_model_version"]))
    return out


def snapshots(c, symbols, cutoffs, mode, model_version):
    """Build + persist snapshots for symbols x cutoffs. Returns list of Snapshot."""
    results = load_results(c, model_version)
    out = []
    for s in symbols:
        mine = [r for r in results if r.symbol == s]   # build_snapshot scans its input per cutoff
        for cut in cutoffs:
            snap = build_snapshot(mine, s, cut, mode, model_version)
            key = (s, cut.strftime("%Y-%m-%dT%H:%M:%SZ"), FEATURE_VERSION, model_version, mode)
            c.execute("INSERT OR REPLACE INTO feature_snapshot VALUES (?,?,?,?,?,?)", (*key, json.dumps(snap.features)))
            c.execute("DELETE FROM snapshot_lineage WHERE symbol=? AND forecast_cutoff=? AND feature_version=? AND nlp_model_version=? AND availability_mode=?", key)
            c.executemany("INSERT INTO snapshot_lineage VALUES (?,?,?,?,?,?)", [(*key, rid) for rid in snap.lineage])
            out.append(snap)
    c.commit()
    return out


def snapshots_to_parquet(snaps, path):
    import pandas as pd
    df = pd.DataFrame([{"symbol": s.symbol, "forecast_cutoff": s.forecast_cutoff, "feature_version": s.feature_version,
                        "nlp_model_version": s.nlp_model_version, "availability_mode": s.availability_mode,
                        **s.features, "n_inputs": len(s.lineage)} for s in snaps])
    df.to_parquet(path, index=False)
    return df


def sentiment_vs_price(c, quotes, model_version, cutoff=None):
    """Join each quoted symbol's live sentiment snapshot with what its price actually did this session.

    A MONITORING view, not a backtest. The model has no measured ability to predict either the sign or the
    magnitude of these moves (docs/model_report.md: every text ablation AUC sits within noise of 0.5), so
    `same_sign` is co-movement to watch, never an accuracy score.
    """
    from datetime import datetime, timezone
    cutoff = cutoff or datetime.now(timezone.utc)
    results = load_results(c, model_version)
    # equal-weight market proxy over the quoted symbols: on a day when the whole board moves together, the raw
    # sign carries almost no per-ticker information (same proxy as trial_forecast --target excess)
    traded = [q.price_change_pct for q in quotes if q.matched]
    mkt = sum(traded) / len(traded) if traded else 0.0
    log.info("market proxy (equal-weight over %d traded symbols): %+.2f%%", len(traded), mkt)
    out = []
    for q in quotes:
        mine = [r for r in results if r.symbol == q.symbol]
        f = build_snapshot(mine, q.symbol, cutoff, "live", model_version).features
        sent = f["sentiment_mean"]
        excess = q.price_change_pct - mkt if q.matched else None
        cmp_ok = sent is not None and sent != 0 and q.matched
        same = (sent > 0) == (q.price_change_pct > 0) if cmp_ok and q.price_change_pct != 0 else None
        same_ex = (sent > 0) == (excess > 0) if cmp_ok and excess else None
        out.append({"symbol": q.symbol, "sentiment_mean": sent, "sentiment_mean_24h": f["sentiment_mean_24h"],
                    "news_count_72h": f["news_count_72h"], "news_count_24h": f["news_count_24h"],
                    "uncertain_ratio": f["uncertain_ratio"], "difficult_ratio": f["difficult_ratio"],
                    "hours_since_last_news": f["hours_since_last_news"],
                    "trading_date": q.trading_date, "session": q.session, "matched": q.matched,
                    "ref_price": q.ref_price, "matched_price": q.matched_price,
                    "price_change": q.price_change, "price_change_pct": q.price_change_pct,
                    "market_change_pct": mkt, "excess_change_pct": excess,
                    "volume": q.volume, "same_sign": same, "same_sign_excess": same_ex})
        log.info("%-4s sentiment %s (n72=%2d, abstain %s, weak %s) | price %9.0f %+6.2f%% | %s",
                 q.symbol, f"{sent:+.3f}" if sent is not None else "  none",
                 f["news_count_72h"],
                 f"{f['uncertain_ratio']:.0%}" if f["uncertain_ratio"] is not None else " n/a",
                 f"{f['difficult_ratio']:.0%}" if f["difficult_ratio"] is not None else " n/a",
                 q.matched_price, q.price_change_pct,
                 ("vs market %+6.2f%% %s" % (excess, "same sign" if same_ex else
                                             ("opposite" if same_ex is False else "n/a"))
                  if excess is not None else "no trade"))
    for key, what in (("same_sign", "raw move"), ("same_sign_excess", "move vs market")):
        cmp_ = [r for r in out if r[key] is not None]
        if cmp_:
            log.info("%-15s comparable %d/%d, same sign %d (%.0f%%) - co-movement only, not accuracy",
                     what, len(cmp_), len(out), sum(r[key] for r in cmp_),
                     100 * sum(r[key] for r in cmp_) / len(cmp_))
    return out


def evidence(c, symbol, limit=20, model_version=None):
    """Per-article evidence rows for the website: what was said about the company, sentiment, and the flags.
    Without model_version the rows of every model are returned, which is rarely what a caller wants."""
    return [dict(r) for r in c.execute("""
        SELECT d.url_normalized AS url, v.title, v.published_at, r.symbol, r.relation, r.passage_text,
               r.p_pos, r.p_neu, r.p_neg, r.entropy, r.uncertain, r.difficult, r.needs_review,
               r.nlp_model_version
        FROM article_company_result r JOIN document_version v USING(document_version_id) JOIN document d USING(document_id)
        WHERE r.symbol=? AND v.is_representative=1 AND (? IS NULL OR r.nlp_model_version=?)
        ORDER BY v.published_at DESC LIMIT ?""", (symbol, model_version, model_version, limit))]
