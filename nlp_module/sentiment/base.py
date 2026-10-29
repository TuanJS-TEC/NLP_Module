import math

LABELS = ["positive", "neutral", "negative"]   # fixed order everywhere (training, inference, export)


def entropy(p) -> float:
    """Normalized to [0,1] by ln(3)."""
    return -sum(x * math.log(x) for x in p if x > 0) / math.log(3)


def decide(p, abstain_entropy):
    """-> (uncertain, entropy). `uncertain` is an operational status: it is never turned into neutral.
    abstain_entropy must come from validation data; None = never abstain."""
    h = entropy(p)
    return (abstain_entropy is not None and h > abstain_entropy), h
