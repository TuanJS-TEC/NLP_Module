# Annotation guideline v1 (v0 + pilot rules, 2026-10-21)

**Unit**: one target company + one passage (the sentences about that company, plus the headline if it is about the company).
**Question**: what does this passage say about the *target company's own* prospects or results? Not about the market, not about other companies, not about what the price did afterwards.

## Labels
- `positive`: favorable development for the target (profit up, contract won, debt reduced, loss narrowed, dividend paid...).
- `negative`: unfavorable development (loss, profit down, penalty, default, insider selling stated as negative news, lawsuit lost...).
- `neutral`: informational, no clear direction (AGM scheduled, leadership list, address change, price table).
- `unclear`: you cannot decide from the passage alone (mixed, needs outside knowledge). Not a training class; kept to measure ambiguity.
- `skip`: broken passage (extraction error, wrong company, not about the company).

Never label from what the stock price did later. If the passage is about someone else (e.g. "BID lỗ" in a passage about VCB), label `skip` or fix the link, do not guess.

## Decision rules (add examples as you meet them)
| Case | Label |
|---|---|
| "Lợi nhuận giảm 20%" | negative |
| "Lỗ giảm 50%" / "thu hẹp lỗ" | positive (still a loss, but improving; note it) |
| Growth below expectations ("tăng 5%, thấp hơn kế hoạch") | negative if the passage frames it as a miss, else positive; mark `difficult` in the note |
| Planned/expected result ("dự kiến lãi") vs. reported | planned = weaker; label direction, note "planned" |
| Quote of someone's optimism, not the author's finding | neutral unless facts follow |
| Legal matter without a ruling | neutral; with sanction/fine = negative |
| Share issuance, insider buy/sell, dividend | direction depends on the passage's framing; do not auto-label by event type |
| Old event mentioned again as background | neutral |
| Two companies compared in one sentence | label for the target only; add `note=multi` |
| Target only appears as a holding in someone else's portfolio, as a partner/customer of the article's subject, or as a landmark ("next to Vinhomes...") | skip (linking error, not about the target) |
| Price move, trading volume, foreign/fund buying or selling the stock | neutral (it is what the price did, not news about the company) |
| Product launch, promotion, charity, sponsorship, deposit-rate table, customer notice | neutral |
| News about a subsidiary (Bảo hiểm BIDV, VPBankS, Masan Consumer) | label for the parent only if the passage states an effect on the parent; else neutral |
| New large project / partnership / contract won / patent | positive; a project only "proposed" or "studied" (MoU) = neutral |
| Treasury-share buyback, insider/major-holder buying | positive; insider selling linked to debt trouble = negative |

## Process
1. Pilot: ~100 samples, `dev` stream. Write every doubt into `note`; update this file; bump `guideline_version`.
2. Then grow `dev`; the `eval` stream (time-based, frozen) is labeled but never used for tuning.
3. Seek an independent reviewer with financial knowledge for a subset; with a single annotator, report that limitation and claim no agreement statistic.
4. Re-label pilot items under the final guideline before training.
