"""PhoBERT fine-tuning + inference. Input: word-segmented passage, max 256 tokens, head on <s> pooling (HF default)."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import LABELS
from .segment import segment

BASE = "vinai/phobert-base-v2"
MAX_LEN = 256


def _enc(tok, texts):
    return tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


def train(train_texts, train_labels, val_texts, val_labels, out_dir, epochs=5, lr=2e-5, bs=16, seed=42):
    """Plain loop (small data): class-weighted CE, bf16 autocast, early-stop on validation macro-F1."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3).to(dev)
    xs, xv = segment(train_texts), segment(val_texts)
    ys = np.array([LABELS.index(l) for l in train_labels])
    yv = np.array([LABELS.index(l) for l in val_labels])
    w = torch.tensor(len(ys) / (3 * np.maximum(np.bincount(ys, minlength=3), 1)), dtype=torch.float, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, out = -1.0, Path(out_dir)
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(len(xs))
        for s in range(0, len(xs), bs):
            b = perm[s:s + bs]
            enc = {k: v.to(dev) for k, v in _enc(tok, [xs[j] for j in b]).items()}
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(ys[b], device=dev), weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        f1 = f1_score(yv, _predict_ids(model, tok, xv, dev, bs), average="macro")
        print(f"epoch {ep + 1}: val macro-F1 {f1:.3f}")
        if f1 >= best:  # ties go to the later (more trained) epoch
            best = f1
            model.save_pretrained(out)
            tok.save_pretrained(out)
            (out / "meta.json").write_text(json.dumps({"base": BASE, "val_macro_f1": f1, "epoch": ep + 1,
                                                       "labels": LABELS}))
    return best


@torch.no_grad()
def _logits(model, tok, segmented, dev, bs=32):
    model.eval()
    res = []
    for s in range(0, len(segmented), bs):
        enc = {k: v.to(dev) for k, v in _enc(tok, segmented[s:s + bs]).items()}
        res.append(model(**enc).logits.float().cpu())
    return torch.cat(res) if res else torch.zeros(0, 3)


def _predict_ids(model, tok, segmented, dev, bs=32):
    return _logits(model, tok, segmented, dev, bs).argmax(-1).numpy()


class PhoBert:
    def __init__(self, path, version=None):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.dev)
        self.version = version or f"phobert-{Path(path).name}"

    def predict(self, texts):
        p = torch.softmax(_logits(self.model, self.tok, segment(texts), self.dev), -1).numpy()
        return [tuple(map(float, r)) for r in p]


def report(model_path, texts, labels):
    m = PhoBert(model_path)
    pred = [LABELS[int(np.argmax(p))] for p in m.predict(texts)]
    return classification_report(labels, pred, labels=LABELS, zero_division=0, output_dict=True)
