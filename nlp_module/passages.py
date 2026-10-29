"""Target-specific passage selection: sentences mentioning the target (+ context), plus title if it concerns the target."""
INPUT_FORMAT_VERSION = "v0"   # change only together with re-annotation and retraining
MAX_CHARS = 1200              # coarse cap; the tokenizer truncates at 256 tokens anyway
SHORT_WORDS = 8


def select_passage(title, sents, link, others):
    """sents: [(id, start, end, text)]; link: entities.Link; others: [Mention] of OTHER companies in the body.
    -> dict(passage_text, sentence_ids, difficult)."""
    hit = [i for i, s, e, _ in sents if any(s <= m.start < e for m in link.mentions)]
    keep = set(hit)
    for i in hit:   # extra context for fragments like "Lợi nhuận tăng 20%."
        if len(sents[i][3].split()) < SHORT_WORDS and i > 0:
            keep.add(i - 1)
    ids = sorted(keep)
    difficult = any(s <= m.start < e for i, s, e, _ in sents if i in keep for m in others)
    text = " ".join(sents[i][3] for i in ids)[:MAX_CHARS]
    if link.in_title:
        text = f"{title}. {text}".strip(". ") if text else title
    return {"passage_text": text, "sentence_ids": ids, "difficult": difficult or len(link.mentions) == 0 and not ids}
