from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from joblib import dump, load
import io
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve

from .pca_gmm import PCAGMMModel

@dataclass
class Calibrator:
    model_type: str
    feature_mode: str
    n_pca_used: int
    threshold_fpr_1pct: float
    threshold_fpr_5pct: float
    clf_bytes: bytes  # pickle via joblib

    def save(self, path: str):
        dump(asdict(self), path)

    @staticmethod
    def load(path: str) -> "Calibrator":
        d = load(path)
        c = Calibrator(**d)
        return c

    def predict_proba(self, pga: PCAGMMModel, X: np.ndarray) -> np.ndarray:
        # reconstruct classifier and features
        # load classifier from in-memory bytes
        obj = load(io.BytesIO(self.clf_bytes))
        Z = pga.transform(X)
        scores = pga.score(X)
        # build supervised feature vector: [buggy_post, llr, Z[:,:n]]
        n = self.n_pca_used
        F = np.concatenate([scores["buggy_post"].reshape(-1,1),
                            scores["llr"].reshape(-1,1),
                            Z[:, :n]], axis=1)
        # return probability of UC=1
        if hasattr(obj, "predict_proba"):
            return obj.predict_proba(F)[:,1]
        else:
            # DecisionTree might not have calibrate; use 0/1
            return obj.predict(F).astype(float)

def _choose_threshold_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """
    Robust threshold at target FPR using negative-score quantile:
      P(score >= thr | y=0) ~= target_fpr  => thr = Quantile_{1-target_fpr}(neg_scores)
    """
    neg_scores = y_score[(np.asarray(y_true) == 0)]
    if neg_scores.size == 0:
        return float("inf")
    q = 1.0 - float(target_fpr)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(neg_scores, q))

def fit_calibrator(
    pga: PCAGMMModel,
    X: np.ndarray,
    y: np.ndarray,
    n_pca_used: int = 3,
    model_type: str = "logreg", # or "tree"
    C: float = 1.0,
    random_state: int = 0
) -> Calibrator:
    # Build supervised feature table from unsupervised outputs
    Z = pga.transform(X)
    scores = pga.score(X)
    n = min(n_pca_used, Z.shape[1])
    F = np.concatenate([scores["buggy_post"].reshape(-1,1),
                        scores["llr"].reshape(-1,1),
                        Z[:, :n]], axis=1)

    if model_type == "logreg":
        clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", random_state=random_state)
    else:
        clf = DecisionTreeClassifier(max_depth=4, random_state=random_state, class_weight=None)

    clf.fit(F, y)
    # compute thresholds on LR probability output
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(F)[:,1]
    else:
        prob = clf.predict(F).astype(float)

    thr1 = _choose_threshold_at_fpr(y, prob, 0.01)
    thr5 = _choose_threshold_at_fpr(y, prob, 0.05)

    # persist classifier inside the object
    import io
    buf = io.BytesIO()
    dump(clf, buf)
    clf_bytes = buf.getvalue()

    return Calibrator(
        model_type=model_type,
        feature_mode="buggy_post+llr+pca",
        n_pca_used=n,
        threshold_fpr_1pct=float(thr1),
        threshold_fpr_5pct=float(thr5),
        clf_bytes=clf_bytes,
    )
