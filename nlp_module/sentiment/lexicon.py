"""PLACEHOLDER model so the chain runs before any labels exist. Keyword counts + a few negation/inversion
phrases. Not calibrated, not a research result: replace with baseline/PhoBERT (same predict() contract)."""
import re

version = "lexicon-v0"
POS = ["tăng", "lãi", "kỷ lục", "vượt", "tích cực", "cải thiện", "hoàn thành", "mua ròng", "cổ tức", "khởi sắc", "đột phá"]
NEG = ["giảm", "lỗ", "sụt", "thua lỗ", "vi phạm", "xử phạt", "nợ xấu", "bán ròng", "tiêu cực", "đình chỉ", "khởi tố", "suy yếu"]
INVERT = [(r"giảm lỗ|lỗ giảm|thu hẹp lỗ|lỗ thu hẹp", +1), (r"không (?:tăng|có lãi)|chưa (?:tăng|có lãi)", -1),
          (r"không giảm|không lỗ", +1)]


def predict(texts):
    out = []
    for t in texts:
        t = re.sub(r"lãi suất|lãi vay", " ", t.lower())   # interest rate is not profit
        sp = sum(len(re.findall(rf"\b{w}\b", t)) for w in POS)
        sn = sum(len(re.findall(rf"\b{w}\b", t)) for w in NEG)
        for pat, sign in INVERT:
            for _ in re.findall(pat, t):
                sp, sn = (sp + 2, max(0, sn - 2)) if sign > 0 else (max(0, sp - 1), sn + 2)
        z = [sp + 0.3, 1.0, sn + 0.3]
        s = sum(z)
        out.append(tuple(x / s for x in z))
    return out
