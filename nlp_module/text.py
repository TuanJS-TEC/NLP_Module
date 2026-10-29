"""Normalization, sentence splitting (with offsets), accent folding."""
import re
import unicodedata

_SENT = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ"“(\d])')


def normalize(s: str) -> str:
    """NFC + whitespace. Keeps diacritics, digits, %, units and negations untouched (no stop-word removal)."""
    s = unicodedata.normalize("NFC", s).replace("\xa0", " ").replace("​", "")
    return "\n".join(re.sub(r"[ \t]+", " ", p).strip() for p in s.splitlines() if p.strip())


def split_sentences(text: str):
    """-> [(sentence_id, start, end, sentence)] with offsets into `text` (the normalized text)."""
    out, i, pos = [], 0, 0
    for para in text.split("\n"):
        start = text.index(para, pos)
        pos = start + len(para)
        cursor = 0
        for m in list(_SENT.finditer(para)) + [None]:
            end = m.start() if m else len(para)
            seg = para[cursor:end].strip()
            if seg:
                s = start + cursor + para[cursor:end].index(seg)
                out.append((i, s, s + len(seg), seg))
                i += 1
            cursor = m.end() if m else end
    return out


def fold(s: str) -> str:
    """Lowercase, strip diacritics (đ->d). Used only for matching slugs/titles, never for model input."""
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")
