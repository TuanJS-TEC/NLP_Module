"""TF-IDF + Logistic Regression baseline, trained on labeled annotation samples."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .base import LABELS


def train(texts, labels, version):
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(texts, labels)
    return Baseline(m, version)


class Baseline:
    def __init__(self, model, version):
        self.model, self.version = model, version

    def predict(self, texts):
        proba = self.model.predict_proba(texts)
        idx = {c: i for i, c in enumerate(self.model.classes_)}
        return [tuple(float(row[idx[l]]) if l in idx else 0.0 for l in LABELS) for row in proba]

    def save(self, path):
        joblib.dump((self.model, self.version), path)

    @classmethod
    def load(cls, path):
        return cls(*joblib.load(path))
