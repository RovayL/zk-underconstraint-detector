from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    # take max TPR s.t. FPR <= target
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok].max()) if ok.size else 0.0

def bootstrap_ci(metric_fn: Callable[[np.ndarray,np.ndarray], float],
                 y_true: np.ndarray, y_score: np.ndarray,
                 B: int = 200, seed: int = 0) -> Tuple[float,float,float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    arr = np.sort(np.array(vals))
    lo = float(np.percentile(arr, 2.5))
    hi = float(np.percentile(arr, 97.5))
    return float(arr.mean()), lo, hi

def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    tpr1 = tpr_at_fpr(y_true, y_score, 0.01)
    tpr5 = tpr_at_fpr(y_true, y_score, 0.05)
    # calibration curve (10 bins)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    return {
        "auroc": float(auroc),
        "average_precision": float(aupr),
        "tpr_at_fpr_1pct": float(tpr1),
        "tpr_at_fpr_5pct": float(tpr5),
        "calibration_curve": {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()},
    }
