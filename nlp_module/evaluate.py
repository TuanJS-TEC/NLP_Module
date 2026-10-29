import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .sentiment.base import LABELS, entropy


def metrics(labels, probs):
    pred = [LABELS[int(np.argmax(p))] for p in probs]
    return {"n": len(labels), "macro_f1": f1_score(labels, pred, labels=LABELS, average="macro", zero_division=0),
            "report": classification_report(labels, pred, labels=LABELS, zero_division=0, output_dict=True),
            "confusion[pos,neu,neg]": confusion_matrix(labels, pred, labels=LABELS).tolist()}


def abstention_curve(labels, probs):
    """-> [(entropy_threshold, coverage, accuracy_on_kept)] to choose the abstention threshold on VALIDATION data."""
    ent = np.array([entropy(p) for p in probs])
    ok = np.array([LABELS[int(np.argmax(p))] == l for p, l in zip(probs, labels)])
    out = []
    for t in np.unique(np.round(ent, 3)):
        keep = ent <= t
        out.append((float(t), float(keep.mean()), float(ok[keep].mean()) if keep.any() else float("nan")))
    return out


def choose_abstain(labels, probs, min_acc):
    """Largest coverage whose accuracy on kept items >= min_acc; None if the target is unreachable."""
    ok = [(t, cov) for t, cov, acc in abstention_curve(labels, probs) if acc >= min_acc]
    return max(ok, key=lambda x: x[1])[0] if ok else None
