# Historical coverage survey (2026-07-01 .. 2026-10-20, 112 days)

Method: sitemap titles only (no article fetched), accent-folded alias match from `data/companies.csv`; ambiguous symbols (VND, HCM, GAS) skipped. **All counts are lower bounds**: an article naming a company only in its body is invisible. CafeF "days" use sitemap `lastmod` (modification time, not publication time). Raw numbers: `data/coverage_cafef.csv`, `data/coverage_vietstock.csv`.

Sources: CafeF lists ~11-12k articles/month back to 2016 in 5-day sitemap buckets; Vietstock ~2k/month in this window (single sitemap, month from URL, exact time from page metadata).

CafeF, ticker ranking by share of days with a matching title: VIC .88, VCB .95, CTG .96, BID .93, STB .96, HDB .88, TCB .69, VPB .65, FPT .62, ACB .48, VHM .47, HPG .46, NVL .36, MWG .31, MSN .29; then SSI .24, PLX .15, VNM .14, below .11: HSG, MBB, GAS, VRE, HCM, SAB, VND, GVR, POW.

## Decision (proposal, revisit after the longer-history survey)
- First study universe = the 15 tickers with >= ~30% of days having a title mention on CafeF (above). Rarely mentioned tickers would mostly yield "no news" days.
- Caveat: banks (VCB/CTG/BID/STB...) appear very often partly via interest-rate roundups; those are list articles, now classed `market` and excluded from per-company features, so real per-company coverage is lower than these ratios.
- Study window: not fixed. Run the survey over 2024-07 .. 2026-06 and check that coverage is stable before choosing the training period.

## Long-history survey (2024-01-01 .. 2026-10-20, 34 months; run 2026-10-21)
Raw: `data/coverage_cafef_2024.csv`, `data/coverage_vietstock_2024.csv`. Same method and lower-bound caveat as above.
- CafeF lists 331,579 articles (7.1k-12.6k/month, rising ~+35% from 2024 to 2026); 12,936 title matches. Vietstock lists 64,153 (1.4k-2.3k/month); 2,998 matches. No month is missing in either source.
- Title-match ratio is stable: CafeF 2.7-3.9% in 2024, 3.5-4.3% in 2025, 4.0-5.2% in 2026 (a step up from 2026-03/04); Vietstock 3.6-7.0%, no trend. Feb is always low (Tet).
- Per-ticker stability (CafeF, title matches per half-year) is uneven. Roughly flat: FPT, HPG, NVL, TCB, MSN, VNM, SSI, HCM. Rising: VCB, BID, CTG, STB, HDB (bank coverage grew). Falling: MWG (151 -> ~70/half), VND, VRE, HSG. 2026H2 is only 3.7 months.
- Implication: a study window starting 2025-01 gives 21 months with stable volume and fewer drift changes in ticker coverage than 2024; starting 2024-01 gives 34 months at the cost of more shift. Pick the window before `collect`, and do not tune on the last months (eval stream).
