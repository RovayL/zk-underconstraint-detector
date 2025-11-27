# src/zkuc/experiments/ablations.py
from __future__ import annotations
import json, math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from zkuc.model.pca_gmm import dataset_from_jsonl, fit_pca_gmm
from zkuc.model.calibrate import fit_calibrator
from zkuc.model.baselines import Featurizer, gaussian_nb, logreg, linear_svm, rbf_svm, adaboost
from zkuc.metrics.eval import evaluate_scores, eval_with_thresholds, save_overlays

ARROWS = {"auroc":"↑","average_precision":"↑","tpr_at_fpr_1pct":"↑","tpr_at_fpr_5pct":"↑",
          "accuracy":"↑","precision":"↑","recall":"↑","f1":"↑","brier":"↓"}

# ----- feature masks ----------------------------------------------------------
PROBE_KEYS = {"rank_mean","rank_std","nullity_mean","nullity_std",
              "dead_rows_frac_mean","wiggle_rate_mean"}
STRUCT_KEYS = None  # computed dynamically as (all - PROBE_KEYS - meta)

META_KEYS = {"circuit_id","trials","subset","n_constraints","n_vars","n_inputs","n_outputs",
             "ratio_c_v","avg_fanin","avg_fanout","mult_share","dup_rows_frac","deg_proxy"}

def split_dataset(jsonl_path: str, test_size: float=0.3, seed: int=0):
    X, feat_names, y_list, ids, parents = dataset_from_jsonl(jsonl_path, return_ids=True, return_parents=True)
    mask = np.array([yy is not None for yy in y_list])
    X, y = X[mask], np.array([yy for yy in y_list if yy is not None], dtype=int)
    ids = [ids[i] for i,m in enumerate(mask) if m]
    groups = np.array([parents[i] for i,m in enumerate(mask) if m])

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return (X[tr_idx], y[tr_idx], [ids[i] for i in tr_idx]), (X[te_idx], y[te_idx], [ids[i] for i in te_idx]), feat_names

def select_features(X: np.ndarray, feat_names: List[str], drop_probe: bool=False) -> Tuple[np.ndarray, List[str]]:
    global STRUCT_KEYS
    if STRUCT_KEYS is None:
        STRUCT_KEYS = set([n for n in feat_names]) - PROBE_KEYS
    keep = []
    for i,n in enumerate(feat_names):
        if n in PROBE_KEYS and drop_probe:
            continue
        keep.append(i)
    return X[:, keep], [feat_names[i] for i in keep]

def run_gmm_pipeline(train, test, feat_names, use_pca=True, n_pca=6, drop_probe=False) -> Dict[str,Any]:
    (Xtr,ytr,_), (Xte,yte,_) = train, test
    Xtr2, names_tr = select_features(Xtr, feat_names, drop_probe=drop_probe)
    Xte2, _        = select_features(Xte, feat_names, drop_probe=drop_probe)

    # unsupervised + calibrator
    n_pca_eff = n_pca if use_pca else Xtr2.shape[1]  # raw space uses full dimension
    pga = fit_pca_gmm(Xtr2, names_tr, n_pca=n_pca_eff, random_state=0)
    cal = fit_calibrator(pga, Xtr2, ytr, n_pca_used=min(3, pga.n_pca), model_type="logreg", random_state=0)
    proba = cal.predict_proba(pga, Xte2)
    curves = {}
    fpr, tpr, _ = roc_curve(yte, proba)
    rec, prec, _ = precision_recall_curve(yte, proba)
    curves["GMM+Cal"] = {"fpr":fpr, "tpr":tpr, "rec":rec, "prec":prec}

    metrics = evaluate_scores(yte, proba)
    # also compute thresholded metrics at 1% FP using negative-score quantile
    thr = np.quantile(proba[yte==0], 0.99) if np.any(yte==0) else float("inf")
    metrics.update(eval_with_thresholds(yte, proba, thr))
    return {"name": f"GMM({'PCA' if use_pca else 'raw'}){'-noProbe' if drop_probe else ''}",
            "metrics":metrics, "curves":curves}

def run_supervised_baseline(train, test, feat_names, model_name:str, use_pca=True, n_pca=6, drop_probe=False):
    (Xtr,ytr,_), (Xte,yte,_) = train, test
    Xtr2, names_tr = select_features(Xtr, feat_names, drop_probe=drop_probe)
    Xte2, _        = select_features(Xte, feat_names, drop_probe=drop_probe)

    fz = Featurizer(use_pca=use_pca, n_pca=n_pca).fit(Xtr2)
    Ztr, Zte = fz.transform(Xtr2), fz.transform(Xte2)

    if model_name=="gnb": clf = gaussian_nb()
    elif model_name=="logreg": clf = logreg()
    elif model_name=="linsvm": clf = linear_svm()
    elif model_name=="rbfsvm": clf = rbf_svm()
    elif model_name=="ada": clf = adaboost()
    else: raise ValueError(model_name)

    clf.fit(Ztr, ytr)
    # decision scores with graceful fallback to predict_proba
    if hasattr(clf, "decision_function"):
        proba = clf.decision_function(Zte)
        # map scores to [0,1] via logistic for plotting fairness
        proba = 1/(1+np.exp(-proba))
    else:
        proba = clf.predict_proba(Zte)[:,1]

    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(yte, proba)
    rec, prec, _ = precision_recall_curve(yte, proba)
    curves = {f"{model_name.upper()}{'(PCA)' if use_pca else '(raw)'}": {"fpr":fpr, "tpr":tpr, "rec":rec, "prec":prec}}

    from zkuc.metrics.eval import evaluate_scores, eval_with_thresholds
    metrics = evaluate_scores(yte, proba)
    thr = np.quantile(proba[yte==0], 0.99) if np.any(yte==0) else float("inf")
    metrics.update(eval_with_thresholds(yte, proba, thr))
    return {"name": f"{model_name.upper()}({'PCA' if use_pca else 'raw'}){'-noProbe' if drop_probe else ''}",
            "metrics":metrics, "curves":curves}

def write_table(rows: List[Dict[str,Any]], out_md: str):
    from tabulate import tabulate
    cols = ["Model","AUROC","AP","TPR@1%","TPR@5%","Acc","Prec","Recall","F1","Brier"]
    table = []
    for r in rows:
        m = r["metrics"]
        table.append([
            r["name"],
            f"{m['auroc']:.3f} ↑", f"{m['average_precision']:.3f} ↑",
            f"{m['tpr_at_fpr_1pct']:.3f} ↑", f"{m['tpr_at_fpr_5pct']:.3f} ↑",
            f"{m['accuracy']:.3f} ↑", f"{m['precision']:.3f} ↑",
            f"{m['recall']:.3f} ↑", f"{m['f1']:.3f} ↑",
            f"{m['brier']:.3f} ↓",
        ])
    md = tabulate(table, headers=cols, tablefmt="github")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("# Ablations & Robustness\n\n" + md + "\n")


def leave_family_out(jsonl_path: str, use_pca: bool=True, n_pca: int=6, drop_probe: bool=False):
    """
    Train on all families except one and test on the held-out family.
    Returns list of dicts with metrics per family.
    """
    X, feat_names, y_list, ids, parents = dataset_from_jsonl(jsonl_path, return_ids=True, return_parents=True)
    mask = np.array([yy is not None for yy in y_list])
    X, y = X[mask], np.array([yy for yy in y_list if yy is not None], dtype=int)
    ids = [ids[i] for i,m in enumerate(mask) if m]
    parents = np.array([parents[i] for i,m in enumerate(mask) if m])

    families = sorted(set(parents))
    results = []
    for fam in families:
        te_mask = parents == fam
        tr_mask = ~te_mask
        train = (X[tr_mask], y[tr_mask], [ids[i] for i,m in enumerate(tr_mask) if m])
        test  = (X[te_mask], y[te_mask], [ids[i] for i,m in enumerate(te_mask) if m])
        r = run_gmm_pipeline(train, test, feat_names, use_pca=use_pca, n_pca=n_pca, drop_probe=drop_probe)
        r["family"] = fam
        results.append(r)
    return results


def write_family_table(rows: List[Dict[str,Any]], out_md: str, title: str="Family Hold-out (LOFO)"):
    from tabulate import tabulate
    cols = ["Family","AUROC","AP","TPR@1%","TPR@5%"]
    table = []
    for r in rows:
        m = r["metrics"]
        table.append([
            r.get("family",""),
            f"{m['auroc']:.3f} ↑", f"{m['average_precision']:.3f} ↑",
            f"{m['tpr_at_fpr_1pct']:.3f} ↑", f"{m['tpr_at_fpr_5pct']:.3f} ↑",
        ])
    md = tabulate(table, headers=cols, tablefmt="github")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(f"# {title}\n\n" + md + "\n")


def rank_mode_compare(jsonl_modp: str, jsonl_struct: str, seed: int=0, use_pca: bool=False, drop_probe: bool=False) -> List[Dict[str,Any]]:
    """
    Compare mod-p vs structural(float/SVD) rank features on otherwise identical splits.
    """
    rows = []
    for tag, path in [("mod-p", jsonl_modp), ("float-SVD", jsonl_struct)]:
        train, test, feat_names = split_dataset(path, test_size=0.3, seed=seed)
        r = run_gmm_pipeline(train, test, feat_names, use_pca=use_pca, n_pca=6, drop_probe=drop_probe)
        r["rank_mode"] = tag
        rows.append(r)
    return rows


def write_rank_table(rows: List[Dict[str,Any]], out_md: str):
    from tabulate import tabulate
    cols = ["Rank mode","AUROC","AP","TPR@1%","TPR@5%"]
    table = []
    for r in rows:
        m = r["metrics"]
        table.append([
            r.get("rank_mode",""),
            f"{m['auroc']:.3f} ↑", f"{m['average_precision']:.3f} ↑",
            f"{m['tpr_at_fpr_1pct']:.3f} ↑", f"{m['tpr_at_fpr_5pct']:.3f} ↑",
        ])
    md = tabulate(table, headers=cols, tablefmt="github")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("# Rank-mode ablation\n\n" + md + "\n")
