"""python -m nlp_module.cli <command> ... (see --help)"""
import argparse
import json
from datetime import date, datetime, timedelta, timezone

from . import annotate, evaluate, storage
from .collect import collect
from .entities import Linker, load_aliases
from .http import Fetcher


def d(s):
    return date.fromisoformat(s)


def load_model(spec):
    kind, _, arg = spec.partition(":")
    if kind == "lexicon":
        from .sentiment import lexicon
        return lexicon
    if kind == "baseline":
        from .sentiment.baseline import Baseline
        return Baseline.load(arg)
    if kind == "phobert":
        from .sentiment.phobert import PhoBert
        return PhoBert(arg)
    raise SystemExit(f"unknown model {spec!r}: use lexicon | baseline:<file> | phobert:<dir>")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nlp_module")
    ap.add_argument("--db", default=str(storage.DB_PATH))
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v: what was fetched, matched and scored; -vv: every url, link, passage and probability")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        p = sub.add_parser(name, **kw)
        return p

    p = add("survey"); p.add_argument("source"); p.add_argument("start", type=d); p.add_argument("end", type=d)
    p.add_argument("--out", default="data/coverage.csv")
    p = add("collect"); p.add_argument("source"); p.add_argument("start", type=d); p.add_argument("end", type=d)
    p.add_argument("--symbols", nargs="*"); p.add_argument("--limit", type=int); p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--no-prefilter", action="store_true", help="fetch every article (many more requests)")
    p = add("process"); p.add_argument("--model", default="lexicon"); p.add_argument("--abstain-entropy", type=float)
    p = add("snapshot"); p.add_argument("--model", default="lexicon"); p.add_argument("--mode", choices=["live", "backtest"], default="backtest")
    p.add_argument("--symbols", nargs="+", required=True); p.add_argument("--start", type=d, required=True)
    p.add_argument("--end", type=d, required=True); p.add_argument("--out", default="data/snapshots.parquet")
    p = add("annotate-sample"); p.add_argument("--model", default="lexicon"); p.add_argument("-n", type=int, default=100)
    p.add_argument("--eval-after", required=True, help="YYYY-MM-DD: samples published on/after go to the frozen eval stream")
    p = add("annotate-export"); p.add_argument("path"); p.add_argument("--stream", choices=["dev", "eval"])
    p = add("annotate-import"); p.add_argument("path"); p.add_argument("--annotator", required=True); p.add_argument("--guideline", default="v0")
    add("leak-check")
    p = add("train-baseline"); p.add_argument("--out", default="data/models/baseline.joblib"); p.add_argument("--version", default="baseline-tfidf-lr-v1")
    p.add_argument("--val-frac", type=float, default=0.0, help="hold out the last fraction by published_at (for calibration)")
    p = add("train-phobert"); p.add_argument("--out", default="data/models/phobert-v1"); p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p = add("abstain", help="choose the abstention entropy threshold on dev validation; --apply sets `uncertain`")
    p.add_argument("--model", required=True); p.add_argument("--min-acc", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.2); p.add_argument("--apply", action="store_true")
    p.add_argument("--apply-to", help="nlp_model_version to update (default: the model's own); use this when the "
                                      "threshold is picked on a calibration twin of the production model")
    p = add("eval"); p.add_argument("--model", required=True); p.add_argument("--stream", choices=["dev", "eval"], default="dev")
    p = add("quotes", help="SSI iBoard quotes next to the model's live sentiment (monitoring, not a backtest)")
    p.add_argument("--symbols", nargs="*"); p.add_argument("--exchange", default="hose")
    p.add_argument("--model", default="baseline:data/models/baseline.joblib")
    p.add_argument("--no-store", action="store_true"); p.add_argument("--interval", type=float, default=1.0)
    p = add("evidence"); p.add_argument("symbol"); p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model", help="nlp_model_version to report (default: every model's rows)")
    a = ap.parse_args(argv)
    if a.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG if a.verbose > 1 else logging.INFO,
                            format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")

    c = storage.connect(a.db)
    from .entities import Linker
    rows = load_aliases()
    linker = Linker(rows)

    if a.cmd in ("survey", "collect"):
        from .survey import TitleMatcher, run
        from .connectors import cafef, vietstock
        f = Fetcher(min_interval=getattr(a, "interval", 1.0))
        matcher = TitleMatcher(rows)
        if a.cmd == "survey":
            mod = {"cafef": cafef, "vietstock": vietstock}[a.source]
            symbols = sorted({r["symbol"] for r in rows})
            print("articles listed, title matches:", run(f, mod.discover, matcher, symbols, a.start, a.end, a.out, a.source))
        else:
            n = collect(c, f, a.source, a.start, a.end, None if a.no_prefilter else matcher, a.symbols, a.limit)
            print("stored", n)
    elif a.cmd == "process":
        from .pipeline import process
        print("results:", process(c, load_model(a.model), linker, a.abstain_entropy))
    elif a.cmd == "snapshot":
        from .pipeline import snapshots, snapshots_to_parquet
        cuts = [datetime.combine(a.start + timedelta(days=i), datetime.min.time().replace(hour=8), timezone.utc)
                for i in range((a.end - a.start).days + 1)]
        m = load_model(a.model)
        snaps = snapshots(c, a.symbols, cuts, a.mode, m.version)
        snapshots_to_parquet(snaps, a.out)
        print(f"{len(snaps)} snapshots -> {a.out}; with news: {sum(s.features['has_news'] for s in snaps)}")
    elif a.cmd == "annotate-sample":
        print("sampled", annotate.sample(c, load_model(a.model).version, a.n, a.eval_after))
    elif a.cmd == "annotate-export":
        print("exported", annotate.export_csv(c, a.path, a.stream))
    elif a.cmd == "annotate-import":
        print("imported, bad labels:", annotate.import_csv(c, a.path, a.annotator, a.guideline))
    elif a.cmd == "leak-check":
        hits = annotate.leak_check(c)
        print("near-duplicate dev/eval pairs:", len(hits), hits[:10])
    elif a.cmd == "train-baseline":
        from pathlib import Path
        from .sentiment.baseline import train
        X, y, rows_ = annotate.dataset(c, "dev")   # dev stream only; eval stream stays untouched
        if a.val_frac:
            order = sorted(range(len(X)), key=lambda i: rows_[i]["published_at"])
            tr = order[:int(len(X) * (1 - a.val_frac))]
            X, y = [X[i] for i in tr], [y[i] for i in tr]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        train(X, y, a.version).save(a.out)
        print("trained on", len(X), "dev samples ->", a.out)
    elif a.cmd == "train-phobert":
        from .sentiment import phobert
        X, y, rows_ = annotate.dataset(c, "dev")
        k = int(len(X) * (1 - a.val_frac))   # dev rows are ordered by sample_id: use time order for a time-based val
        order = sorted(range(len(X)), key=lambda i: rows_[i]["published_at"])
        tr, va = order[:k], order[k:]
        print("best val macro-F1:", phobert.train([X[i] for i in tr], [y[i] for i in tr], [X[i] for i in va],
                                                  [y[i] for i in va], a.out, epochs=a.epochs))
    elif a.cmd == "abstain":
        X, y, rows_ = annotate.dataset(c, "dev")   # never the eval stream
        order = sorted(range(len(X)), key=lambda i: rows_[i]["published_at"])
        va = order[int(len(X) * (1 - a.val_frac)):]
        m = load_model(a.model)
        probs, labels = m.predict([X[i] for i in va]), [y[i] for i in va]
        curve = evaluate.abstention_curve(labels, probs)
        print(f"dev validation: {len(va)} rows from {rows_[va[0]]['published_at'][:10]}, "
              f"full-coverage accuracy {curve[-1][2]:.3f}")
        step = max(1, len(curve) // 12)
        for t, cov, acc in curve[::step] + [curve[-1]]:
            print(f"  abstain if H > {t:.3f} -> coverage {cov:.3f}, accuracy on kept {acc:.3f}")
        th = evaluate.choose_abstain(labels, probs, a.min_acc)
        if th is None:
            raise SystemExit(f"no threshold reaches accuracy {a.min_acc} on validation; lower --min-acc")
        cov = next(c_ for t, c_, _ in curve if t == th)
        print(f"chosen: --abstain-entropy {th:.3f}  (coverage {cov:.3f} at accuracy >= {a.min_acc})")
        if a.apply:
            # `uncertain` is a pure function of the stored entropy, so changing the threshold needs no re-scoring
            target = a.apply_to or m.version
            n = c.execute("UPDATE article_company_result SET uncertain = (entropy > ?) WHERE nlp_model_version = ?",
                          (th, target)).rowcount
            c.commit()
            print(f"applied to {n} stored results of {target}; rebuild snapshots to pick it up")
    elif a.cmd == "eval":
        X, y, _ = annotate.dataset(c, a.stream)
        m = load_model(a.model)
        r = evaluate.metrics(y, m.predict(X))
        print(json.dumps({k: v for k, v in r.items() if k != "report"}, indent=1))
    elif a.cmd == "quotes":
        from .connectors import ssi
        from .pipeline import sentiment_vs_price
        syms = a.symbols or sorted({r["symbol"] for r in rows})
        qs = ssi.quotes(Fetcher(min_interval=a.interval), syms, a.exchange)
        if not a.no_store:
            print("quotes stored:", storage.save_quotes(c, qs, ssi.SOURCE))
        cmp_ = sentiment_vs_price(c, qs, load_model(a.model).version)
        out = {"symbols": len(cmp_), "market_change_pct": round(cmp_[0]["market_change_pct"], 3) if cmp_ else None}
        for k in ("same_sign", "same_sign_excess"):
            both = [r for r in cmp_ if r[k] is not None]
            out[k] = {"comparable": len(both), "same": sum(r[k] for r in both)}
        out["note"] = "co-movement only; text features show no measured forecasting power"
        print(json.dumps(out, indent=1))
    elif a.cmd == "evidence":
        from .pipeline import evidence
        mv = load_model(a.model).version if a.model else None
        print(json.dumps(evidence(c, a.symbol, a.limit, mv), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
