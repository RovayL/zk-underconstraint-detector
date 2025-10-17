from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import json
import math
import numpy as np
from pathlib import Path
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

NUMERIC_FEATURES_DEFAULT = [
    # structural
    "ratio_c_v","avg_fanin","avg_fanout","mult_share","dup_rows_frac",
    # probe
    "rank_mean","rank_std","nullity_mean","nullity_std",
    "dead_rows_frac_mean","wiggle_rate_mean",
    # sizes (often useful)
    "n_constraints","n_vars","n_inputs","n_outputs",
]

def _read_jsonl(jsonl_path: str) -> List[Dict[str,Any]]:
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def dataset_from_jsonl(jsonl_path: str, feature_keys: Optional[List[str]]=None) -> Tuple[np.ndarray, List[str], List[Optional[int]]]:
    rows = _read_jsonl(jsonl_path)
    X_list, y_list = [], []
    if feature_keys is None:
        feature_keys = NUMERIC_FEATURES_DEFAULT
    for r in rows:
        feats = r.get("features", {})
        x = []
        for k in feature_keys:
            v = feats.get(k, None)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                v = 0.0
            x.append(float(v))
        X_list.append(x)
        y_list.append(r.get("label_uc", None))
    X = np.asarray(X_list, dtype=float)
    return X, feature_keys, y_list

@dataclass
class PCAGMMModel:
    feature_names: List[str]
    mean_: np.ndarray
    scale_: np.ndarray
    pca_components_: np.ndarray
    pca_mean_: np.ndarray
    pca_explained_variance_: np.ndarray
    gmm_weights_: np.ndarray
    gmm_means_: np.ndarray
    gmm_covariances_: np.ndarray
    buggy_component: int
    n_components: int
    n_pca: int
    version: str = "pca_gmm.v1"

    # ------ runtime helpers (not persisted) ------
    _pca: Optional[PCA] = None
    _gmm: Optional[GaussianMixture] = None

    def _rebuild(self):
        pca = PCA(n_components=self.n_pca)
        pca.components_ = self.pca_components_
        pca.mean_ = self.pca_mean_
        pca.explained_variance_ = self.pca_explained_variance_
        pca.n_features_in_ = len(self.feature_names)
        self._pca = pca

        gmm = GaussianMixture(n_components=self.n_components, covariance_type="full")
        gmm.weights_ = self.gmm_weights_
        gmm.means_ = self.gmm_means_
        gmm.covariances_ = self.gmm_covariances_
        # sklearn requires precisions_ derived
        from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
        gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, "full")
        self._gmm = gmm

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None: self._rebuild()
        Z = (X - self.mean_) / np.where(self.scale_ == 0, 1, self.scale_)
        return self._pca.transform(Z)

    def score(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self._pca is None or self._gmm is None: self._rebuild()
        Z = self.transform(X)
        log_prob = self._gmm.score_samples(Z)                   # total log-likelihood
        resp = self._gmm.predict_proba(Z)                       # responsibilities
        buggy_post = resp[:, self.buggy_component]
        normal_comp = 1 - self.buggy_component
        # log-likelihood ratio in favor of buggy vs normal (using responsibilities as a proxy)
        eps = 1e-12
        llr = np.log(buggy_post + eps) - np.log(resp[:, normal_comp] + eps)
        return {"loglik": log_prob, "buggy_post": buggy_post, "llr": llr, "resp": resp, "Z": Z}

    def save(self, path: str):
        d = asdict(self)
        # strip runtime
        d.pop("_pca", None); d.pop("_gmm", None)
        dump(d, path)

    @staticmethod
    def load(path: str) -> "PCAGMMModel":
        d = load(path)
        m = PCAGMMModel(**d)
        m._rebuild()
        return m

def fit_pca_gmm(
    X: np.ndarray,
    feature_names: List[str],
    n_pca: int = 6,
    n_components: int = 2,
    random_state: int = 0,
    buggy_rule: str = "smallest_weight" # or "lowest_mean_loglik"
) -> PCAGMMModel:
    # standardize
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    Xz = (X - mean) / scale

    # PCA
    n_pca = int(min(n_pca, Xz.shape[1], max(2, min(16, Xz.shape[0]))))
    pca = PCA(n_components=n_pca, random_state=random_state)
    Z = pca.fit_transform(Xz)

    # GMM
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=random_state, n_init=5, reg_covar=1e-6
    ).fit(Z)

    # decide which mixture is "buggy"
    if buggy_rule == "smallest_weight":
        buggy_component = int(np.argmin(gmm.weights_))
    elif buggy_rule == "lowest_mean_loglik":
        lp = gmm.score_samples(Z)
        resp = gmm.predict_proba(Z)
        comp_lp_means = []
        for k in range(n_components):
            # weighted mean loglik per comp
            w = resp[:, k]
            comp_lp_means.append((w * lp).sum() / (w.sum() + 1e-9))
        buggy_component = int(np.argmin(comp_lp_means))
    else:
        buggy_component = int(np.argmin(gmm.weights_))

    model = PCAGMMModel(
        feature_names=list(feature_names),
        mean_=mean, scale_=scale,
        pca_components_=pca.components_, pca_mean_=pca.mean_, pca_explained_variance_=pca.explained_variance_,
        gmm_weights_=gmm.weights_, gmm_means_=gmm.means_, gmm_covariances_=gmm.covariances_,
        buggy_component=buggy_component,
        n_components=n_components, n_pca=n_pca,
    )
    model._pca = pca
    model._gmm = gmm
    return model
