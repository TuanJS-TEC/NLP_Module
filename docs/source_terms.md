# Source terms check (week 1 gate)

Checked 2026-10-21. Status: **NOT CLEARED for bulk collection or redistribution.** Technical access is permitted by robots.txt; usage rights are unresolved. This is a factual record, not legal advice.

## CafeF (cafef.vn, operator: VCCorp)
- robots.txt (verified with curl): `User-agent: *` / `Allow: /`, no Disallow, no Crawl-delay. Sitemaps: `/sitemap.xml`, `/google-news-sitemap.xml`.
- Footer: "© Copyright 2007 - 2026 - Công ty Cổ phần VCCorp." Only footer legal link found: privacy policy (`/static/chinh-sach-bao-mat.html`), which says nothing on copying, scraping or reuse.
- No terms-of-use page found. Contact: info@cafef.vn; advertising: doanhnghiep@admicro.vn.

## Vietstock (vietstock.vn, operator: CTCP Tài Việt)
- robots.txt (verified with curl): `Disallow: /*.js`, `/*.css`, `/manager`, `/export`, `/cache`. No Crawl-delay. Sitemap: `/sitemap.xml`.
- Footer: "© Bản quyền thuộc về Vietstock". Privacy policy says nothing on copying, scraping or reuse. No terms-of-use page found.
- Sells licensed data (DataFeed API, VietstockXLS, VietstockFinance) via dichvu.vietstock.vn; no licensing terms published. Contact: kinhdoanh@vietstock.vn.

## What this means
- Both sites claim copyright; neither publishes terms I could find that either allow or forbid automated access or reuse. Absence of terms is not permission.
- robots.txt allows fetching article pages on both (Vietstock: avoid the Disallowed paths).
- Redistributing full article text (e.g. on the website) is the highest-risk use.

## Rules adopted until permission is clarified
1. Collect only article pages (use sitemaps), honor robots.txt, a fixed low request rate (proposal: 1 req/s max, off-peak), identifiable User-Agent with contact email, no login/paywall pages, no circumvention of blocks.
2. Use stored text only for internal research (training, features). The website shows our derived outputs (sentiment, passages-as-short-evidence at most, link to the original), not full article copies.
3. Do not publish or share the raw corpus or the labeled dataset with article text; share ids/URLs/labels only.
4. Ask both publishers in writing whether research use is acceptable (addresses above); record replies here.

## Decision (2026-10-21, project owner)
Proceed without waiting for publisher replies, for **non-commercial research and learning only**. Rules 1-3 above stay in force (low rate, identifiable UA, robots.txt, no sharing of article text). Any commercial use or redistribution requires written permission first.

## Open
- Neither site's full legal terms were located by automated lookup; check the sites manually for a "Điều khoản" page I may have missed (the fetch tool summarizes pages and may miss footer links).
- Tool caveat: page summaries came from a fetch tool, robots.txt files were verified directly.

## SSI iBoard (added 2026-10-26)
`https://iboard-query.ssi.com.vn/stock/exchange/{exchange}` - undocumented JSON endpoint behind the public iBoard
price board. Used for market quotes only (last matched price, reference price, change, volume), never article text,
so nothing from it is redistributed content.

- `https://iboard.ssi.com.vn/robots.txt` is `User-agent: * / Disallow:` - no restrictions.
- The query host returns 404 for `/robots.txt`. Per RFC 9309 an unavailable robots.txt means no restrictions, which
  is what `http.Fetcher` now implements (404 and other 4xx except 401/403 -> allow; 401/403 and 5xx -> deny).
- One request returns the whole exchange, so a full universe refresh costs a single call at the standard 1 req/s.
- The endpoint is undocumented and may change without notice: `ssi.quotes()` raises on an empty `data` array rather
  than returning silently empty rows, so a breakage shows up as an error instead of as "no price data".

## HOSE disclosures (added 2026-10-26)
Issuer filings published by the exchange itself (Sở GDCK TP.HCM), i.e. mandatory public disclosures. Only the
profit-change explanations ("Giải trình biến động LNST/KQKD", `nlp_module/connectors/hose.py`) of tracked tickers.

- `https://www.hsx.vn/robots.txt` is `User-agent: * / Disallow:` - no restrictions. `api.hsx.vn` and
  `staticfile.hsx.vn` return 404 for `/robots.txt` (RFC 9309: no restrictions).
- HOSE advertises an RSS feed (`api.hsx.vn/n/api/v1/News/NewsFeed`, footer link) but it carries only the 10 latest
  items and no attachments, so the connector uses the JSON API the hsx.vn site itself calls: `/n/api/v1/1/news/cate`
  (listing), `/n/api/v1/1/news/{id}` (detail), `/m/api/v1/1/mediafiles/1/{id}` (attachments). Undocumented, may change.
- Cost: listing is ~120 filings/day across all issuers at 100 per request, plus 3 requests per kept filing
  (~5 per month for the tracked tickers). Same 1 req/s and identifiable User-Agent as the other sources.
- PDFs are mostly scans: text is OCR'd locally (tesseract, `data/tessdata/vie.traineddata` from the official
  tesseract-ocr/tessdata_best repository). Same storage rules as articles: internal research use, no redistribution.
