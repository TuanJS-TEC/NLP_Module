# NLP_Module

Financial-news NLP subsystem: news -> company -> target-specific passage -> sentiment -> point-in-time text features
for T+2/T+5 forecasting. Design/plan: `docs/`, spec of timestamps and features: `docs/temporal_spec.md`.
Usage rights of sources: `docs/source_terms.md` (not cleared for redistribution; research use, low rate, no sharing of article text).

```
.venv/bin/python -m pytest -q tests                                # tests
CLI="python -m nlp_module.cli"
$CLI survey cafef 2026-07-01 2026-10-20 --out data/coverage_cafef.csv   # sitemap-only coverage
$CLI collect cafef 2026-10-01 2026-10-20 --symbols VCB FPT --limit 100  # polite fetch (1 req/s, robots.txt)
$CLI process --model lexicon                                       # link + passage + sentiment (placeholder model)
$CLI collect hose 2026-07-01 2026-10-26                            # issuer profit explanations (PDF, OCR'd)
$CLI annotate-sample --eval-after 2026-10-10 -n 100                # draw samples; stream by time
$CLI annotate-export data/pilot.csv --stream dev                   # label in a spreadsheet: column `label`
$CLI annotate-import data/pilot.csv --annotator me --guideline v0
$CLI train-baseline ; $CLI process --model baseline:data/models/baseline.joblib
$CLI train-baseline --val-frac 0.2 --version baseline-tfidf-lr-v1-cal --out data/models/baseline-cal.joblib
$CLI abstain --model baseline:data/models/baseline-cal.joblib --min-acc 0.85 --apply --apply-to baseline-tfidf-lr-v1
$CLI train-phobert --epochs 5 ; $CLI eval --model phobert:data/models/phobert-v1 --stream dev
$CLI snapshot --model baseline:... --mode backtest --symbols VCB FPT --start 2026-10-01 --end 2026-10-20
$CLI evidence VCB --model baseline:data/models/baseline.joblib      # per-article evidence JSON for the website
$CLI -v quotes --symbols VCB FPT   # SSI iBoard price next to live sentiment; stores a row in market_quote
scripts/live_update.sh             # collect today's news -> baseline + PhoBERT -> SSI quotes (Mon-Fri 9:00-15:15)
# cron, every 15 min:  */15 * * * * "<repo>/scripts/live_update.sh" >> "<repo>/data/live.log" 2>&1
python -m nlp_module.walk_forward                                 # 3-fold walk-forward: per-fold PhoBERT + forecaster (needs data/prices.csv)
python -m nlp_module.fundamentals                                 # LNST % change parsed from the HOSE profit explanations
python -c "import pandas as pd, vnfinancialdata as vnf; pd.concat([vnf.load(e, s, start_year=2019) for e in ('HSX','HNX') for s in ('balance_sheet','cash_flow','income_statement')], ignore_index=True).to_parquet('data/bctc.parquet', index=False)"   # all BCTC 2019+ (2020 feeds 2021 growth; long format)
python -m nlp_module.fundamentals --check                         # parsed full-year % vs audited BCTC (vnfinancialdata)
python -c "from nlp_module.fundamentals import annual; print(annual(['VCB','FPT']))"   # yearly BCTC (vnfinancialdata, HF, public)
python -m nlp_module.trial_forecast --fund ; python -m nlp_module.walk_forward --skip-nlp --fund   # + earnings features
python -m nlp_module.volcast train ; python -m nlp_module.volcast predict   # T+2/T+5 volatility forecast (confidence-gated) -> tab "Dự báo biến động"
```

Model contract: any object with `.version` and `.predict(texts) -> [(p_pos, p_neu, p_neg)]`; swapping models changes
only `nlp_model_version`. Trial forecaster: `nlp_module/trial_forecast.py` (needs an OHLCV CSV: date,symbol,close).
Word segmentation for PhoBERT uses the VnCoreNLP jar in `data/vncorenlp` (JRE only).
HOSE filings (`connectors/hose.py`): scanned PDFs are OCR'd with tesseract (`pdftoppm` + `tesseract`, Vietnamese
model in `data/tessdata/vie.traineddata`); only the explanation narrative is kept and scored for the title ticker.
Abstention: `abstain` picks the entropy threshold on dev validation (coverage vs accuracy); the production baseline is
trained on all dev labels, so calibrate a `--val-frac 0.2` twin and `--apply-to` the production version. Current
thresholds: baseline H > 0.888, PhoBERT H > 0.571, both ~73% coverage at 0.85 accuracy on kept.
Add `-v` / `-vv` to any command for fetch, link, passage and probability logs.
`quotes` logs one line per symbol - sentiment, abstain/weak share, matched price, move, move vs an equal-weight
market proxy - and is the monitoring view for sentiment against real prices. It measures CO-MOVEMENT, not accuracy:
the text features have no measured forecasting power (see `docs/model_report.md`). UI: `streamlit run app.py`,
tabs "Theo mã CK", "Sentiment vs Giá SSI" and "Dự báo biến động".
Never tune on the `eval` stream.
