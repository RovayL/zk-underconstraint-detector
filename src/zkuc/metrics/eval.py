from __future__ import annotations
from typing import Dict, Any, Tuple, Callable
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def save_plots(y_true: np.ndarray, y_score: np.ndarray, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC", color=(1.0, 0.0, 0.0))
    plt.plot([0,1],[0,1], linestyle="--", color=(0.7, 0.0, 1.0))
    plt.xlabel("False Positive Rate", color = (0.6, 0.0, 0.0))
    plt.ylabel("True Positive Rate", color = (0.6, 0.0, 0.0))
    plt.title("ROC Curve", color = (0.6, 0.0, 0.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=150)
    plt.close()
    # PR
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec, label=f"PR (AP={ap:.3f})", color=(1.0, 0.0, 0.0))
    plt.xlabel("Recall", color = (0.6, 0.0, 0.0))
    plt.ylabel("Precision", color = (0.6, 0.0, 0.0))
    plt.title("Precision-Recall Curve", color = (0.6, 0.0, 0.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pr.png", dpi=150)
    plt.close()
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration", color=(1.0, 0.0, 0.0))
    plt.plot([0,1],[0,1], linestyle="--", color=(0.7, 0.0, 1.0))
    plt.xlabel("Predicted probability", color = (0.6, 0.0, 0.0))
    plt.ylabel("Empirical probability", color = (0.6, 0.0, 0.0))
    plt.title("Calibration Curve", color = (0.6, 0.0, 0.0))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "calibration.png", dpi=150)
    plt.close()

def eval_with_thresholds(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, Any]:
    yhat = (y_score >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, yhat)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
    }

def save_overlays(models_curves: Dict[str, Dict[str, np.ndarray]], out_dir: str):
    """
    models_curves: name -> {"fpr":..., "tpr":..., "rec":..., "prec":...}
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # ROC
    plt.figure()
    for name, c in models_curves.items():
        plt.plot(c["fpr"], c["tpr"], label=name)
    plt.plot([0,1],[0,1], "--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC (overlay)")
    plt.legend(); plt.tight_layout(); plt.savefig(out/"roc_overlay.png", dpi=150); plt.close()
    # PR
    plt.figure()
    for name, c in models_curves.items():
        plt.plot(c["rec"], c["prec"], label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (overlay)")
    plt.legend(); plt.tight_layout(); plt.savefig(out/"pr_overlay.png", dpi=150); plt.close()

